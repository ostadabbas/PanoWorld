"""Single-process batch driver for PanoWorld V3 pers2pano.

Loads the diffusion pipeline + SigLIP2 ONCE and loops over master.csv clips,
saving ~95s of model-reload per clip vs the subprocess pattern in
``infer_panoworld.py``. For 150 clips on H100 this reclaims ~4.5h.

Output layout is identical to the subprocess runner so eval downstream is
unchanged:

    <results>/panoworld_pers/<clip_id_dir>/video.mp4
    <results>/panoworld_pers/<clip_id_dir>/run.json
    <results>/_logs/panoworld_pers/<clip_id>.log

Usage (full 150-clip run):

    python test_set_pkg/eval/runners/infer_panoworld_batch.py \\
        --master $PANO_DATA_ROOT/test/master.csv \\
        --results eval_results_panoworld \\
        --finetune_checkpoint checkpoints/v5_geo_long/iter_000001000/model_ema_bf16.pt

The V3 setup + per-clip steps are copied from ``generate_pano.py``'s
``run_v3_pipeline``; this driver intentionally duplicates that code rather
than refactoring the upstream CLI to keep ``generate_pano.py`` untouched.
Any future edit to the V3 path needs to be mirrored here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Make the cosmos repo importable.
REPO = Path(__file__).resolve().parents[3]   # repo root (e.g. /path/to/PanoWorld)
sys.path.insert(0, str(REPO))

# ───────────────────────────── runner helpers ─────────────────────────
RUNNERS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNNERS_DIR))
from _common import (                                                # noqa: E402
    MASTER_DEFAULT,
    clip_id_to_dirname,
    erp_to_perspective_image,
    have_output,
    load_caption_with_style,
    make_static_video_from_image,
    read_first_frame_rgb_uint8,
    read_master,
    write_run_json,
)

# ───────────────────────────── cosmos imports ─────────────────────────
from cosmos_predict2._src.imaginaire.utils import log              # noqa: E402
from cosmos_predict2._src.imaginaire.visualize.video import (       # noqa: E402
    save_img_or_video,
)
from cosmos_predict2._src.predict2.inference.video2world import (   # noqa: E402
    Video2WorldInference,
)
from cosmos_predict2.config import (                                 # noqa: E402
    DEFAULT_NEGATIVE_PROMPT,
    MODEL_CHECKPOINTS,
    MODEL_KEYS,
)

# Reuse the V3 helpers we don't need to re-implement.
from generate_pano import read_perspective_video                    # noqa: E402

SIGLIP2_MODEL_ID = "google/siglip2-so400m-patch14-384"


# ────────────────────────── one-time pipe + siglip ─────────────────────
def setup_v3_pipe(args) -> Video2WorldInference:
    """Build Video2WorldInference, load fine-tuned weights, switch to
    equirectangular RoPE, and install the circular-decode patch.

    Mirrors lines 619-752 of ``generate_pano.run_v3_pipeline`` *except* the
    perspective-input I2V branch (we always use ``--pers_input``)."""
    model_key = MODEL_KEYS[args.model]
    checkpoint = MODEL_CHECKPOINTS[model_key]
    ckpt_path = checkpoint.path
    experiment = checkpoint.experiment
    config_file = "cosmos_predict2/_src/predict2/configs/video2world/config.py"

    target_T = args.num_frames
    target_T_lat = (target_T - 1) // 4 + 1
    experiment_opts = [
        "++model.config.net.extra_image_context_dim=1152",
        "++model.config.net.reference_channels=16",
        f"++model.config.state_t={target_T_lat}",
    ]

    log.info("=" * 60)
    log.info("V3 batch driver: building pipeline (one-time)")
    log.info(f"  model      = {args.model}")
    log.info(f"  ft_ckpt    = {args.finetune_checkpoint}")
    log.info(f"  num_frames = {target_T}  (state_t={target_T_lat})")
    log.info(f"  resolution = {args.resolution}")
    log.info("=" * 60)

    pipe = Video2WorldInference(
        experiment_name=experiment,
        ckpt_path=ckpt_path,
        s3_credential_path="",
        context_parallel_size=1,
        config_file=config_file,
        experiment_opts=experiment_opts,
    )

    log.info(f"Loading fine-tuned weights from {args.finetune_checkpoint} ...")
    ft_sd = torch.load(args.finetune_checkpoint, map_location="cpu",
                       weights_only=False)
    net_sd = {k.replace("net.", ""): v for k, v in ft_sd.items()
              if k.startswith("net.")}
    load_info = pipe.model.net.load_state_dict(net_sd, strict=False)
    log.info(f"  missing={len(load_info.missing_keys)}  "
             f"unexpected={len(load_info.unexpected_keys)}")
    del ft_sd, net_sd
    torch.cuda.empty_cache()

    net = pipe.model.net
    if args.equirect_rope:
        if hasattr(net, "pos_embedder"):
            net.pos_embedder.grid_type = "equirectangular"
        elif hasattr(net, "rope_position_embedding"):
            net.rope_position_embedding.grid_type = "equirectangular"
            net.rope_position_embedding._is_initialized = False
        log.info("Switched to equirectangular RoPE")

    if args.latent_padding_size > 0:
        _orig_decode = pipe.model.decode
        _pad = args.latent_padding_size

        @torch.no_grad()
        def _circular_decode(latent):
            left_p  = latent[..., -_pad:]
            right_p = latent[..., :_pad]
            padded  = torch.cat([left_p, latent, right_p], dim=-1)
            decoded = _orig_decode(padded)
            px_p    = _pad * 8
            return decoded[..., px_p:-px_p]

        pipe.model.decode = _circular_decode
        log.info(f"Installed circular latent decode (pad={_pad} cols/side)")

    return pipe


def setup_siglip(model_id: str = SIGLIP2_MODEL_ID):
    """Load SigLIP2 once, kept on GPU. Returns (model, image_size, hidden_size)."""
    from transformers import SiglipVisionModel

    log.info(f"Loading SigLIP2 from {model_id} ...")
    clip_model = SiglipVisionModel.from_pretrained(model_id)
    clip_model = clip_model.to(device="cuda", dtype=torch.bfloat16).eval()
    return clip_model, clip_model.config.image_size, clip_model.config.hidden_size


def encode_temporal_clip_with(siglip_model, image_size: int,
                              hidden_size: int,
                              video_frames_01: torch.Tensor) -> torch.Tensor:
    """Like ``generate_pano.encode_temporal_clip`` but uses a pre-loaded model.

    Returns (1, T*729, 1152) bf16. Frames are (T, C, H, W) in [0, 1]."""
    from torchvision.transforms.functional import normalize

    T = video_frames_01.shape[0]
    frames = F.interpolate(video_frames_01, size=(image_size, image_size),
                           mode="bicubic", align_corners=False)
    frames = normalize(frames, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    frames = frames.to(dtype=torch.bfloat16, device="cuda")

    feats = []
    bsz = 4
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, T, bsz):
            f = siglip_model(pixel_values=frames[i:i + bsz],
                             output_hidden_states=False).last_hidden_state
            feats.append(f)
    out = torch.cat(feats, dim=0)                       # (T, 729, hidden)
    tok = out.shape[1]
    out = out.reshape(1, T * tok, hidden_size).to(dtype=torch.bfloat16)
    del frames
    return out


# ────────────────────────────── per-clip V3 ────────────────────────────
def run_v3_one_clip(pipe: Video2WorldInference,
                    siglip_model, siglip_image_size, siglip_hidden_size,
                    args, *,
                    prompt: str,
                    pers_input_mp4: str,
                    output_dir: str,
                    seed: int) -> tuple[str, dict]:
    """Steps 1+3+4+5 of the V3 pipeline using a pre-loaded ``pipe``.

    Returns (output_mp4_path, timing_info)."""
    os.makedirs(output_dir, exist_ok=True)
    h_pano, w_pano = [int(x) for x in args.resolution.split(",")]
    target_T = args.num_frames

    timing: dict[str, float] = {}

    # ── Step 1: read pers video ────────────────────────────────────────
    t = time.time()
    pers_frames_m1p1 = read_perspective_video(pers_input_mp4, target_T).to("cuda")
    pers_frames_01 = (pers_frames_m1p1 + 1.0) / 2.0
    timing["t_step1_read"] = time.time() - t

    # ── Step 3: temporal CLIP ──────────────────────────────────────────
    t = time.time()
    temporal_clip_emb = encode_temporal_clip_with(
        siglip_model, siglip_image_size, siglip_hidden_size,
        pers_frames_01.to("cuda"),
    )
    timing["t_step3_clip"] = time.time() - t

    # ── Step 4: pers→equi + VAE encode ────────────────────────────────
    t = time.time()
    from cosmos_predict2._src.predict2.utils.pano_conditioning import (
        pers2equi_simple,
    )
    T_in   = pers_frames_m1p1.shape[0]
    device = torch.device("cuda")
    Hp, Wp = pers_frames_m1p1.shape[2], pers_frames_m1p1.shape[3]
    hw_ratio = Hp / Wp

    roll_t  = torch.zeros(T_in, device=device)
    pitch_t = torch.zeros(T_in, device=device)
    yaw_t   = torch.full((T_in,), args.yaw, device=device)

    equi_proj, pixel_mask = pers2equi_simple(
        pers_frames_m1p1.to(device), fov_x=args.fov_x,
        roll=roll_t, pitch=pitch_t, yaw=yaw_t,
        equi_h=h_pano, equi_w=w_pano, hw_ratio=hw_ratio,
    )
    coverage = pixel_mask.mean().item()
    log.info(f"Step 4: coverage {coverage:.1%}")

    equi_BCTHW   = equi_proj.permute(1, 0, 2, 3).unsqueeze(0)
    mask_11THW   = pixel_mask[:, :1].permute(1, 0, 2, 3).unsqueeze(0)

    noise_fill   = torch.randn_like(equi_BCTHW) * 0.5
    masked_equi  = torch.where(mask_11THW.expand_as(equi_BCTHW) > 0.5,
                                equi_BCTHW, noise_fill)

    tokenizer = pipe.model.tokenizer
    if hasattr(tokenizer, "encoder") and tokenizer.encoder is not None:
        if next(tokenizer.encoder.parameters()).device.type == "cpu":
            tokenizer.encoder = tokenizer.encoder.to(device)

    with torch.no_grad():
        guided_image     = tokenizer.encode(equi_BCTHW.float().to(device))
        reference_latent = tokenizer.encode(masked_equi.float().to(device))

    _, _, T_lat, H_lat, W_lat = guided_image.shape
    guided_mask = F.interpolate(
        mask_11THW.float().to(device),
        size=(T_lat, H_lat, W_lat), mode="nearest",
    )
    guided_mask = (guided_mask > 0.5).float()
    log.info(f"Step 4: guided_image={tuple(guided_image.shape)}  "
             f"latent_mask_coverage={guided_mask.mean().item():.1%}")

    if hasattr(tokenizer, "encoder") and tokenizer.encoder is not None:
        tokenizer.encoder = tokenizer.encoder.to("cpu")
        torch.cuda.empty_cache()

    del pers_frames_01, pers_frames_m1p1, equi_proj, pixel_mask
    del equi_BCTHW, mask_11THW, noise_fill, masked_equi
    torch.cuda.empty_cache()
    timing["t_step4_proj"] = time.time() - t

    # ── Step 5: monkey-patch + diffusion ───────────────────────────────
    t = time.time()
    net = pipe.model.net
    _original_forward  = net.forward
    _clip_emb     = temporal_clip_emb.to(dtype=torch.bfloat16)
    _ref_lat      = reference_latent.to(dtype=torch.bfloat16)
    _spatial_mask = guided_mask.to(dtype=torch.bfloat16)
    _cfg_call_counter = [0]

    def _patched_forward(*a, **kw):
        dev = a[0].device if a else "cuda"
        kw["reference_latent_B_C_T_H_W"]            = _ref_lat.to(device=dev,
                                                                    dtype=torch.bfloat16)
        kw["condition_video_input_mask_B_C_T_H_W"]  = _spatial_mask.to(device=dev,
                                                                       dtype=torch.bfloat16)
        is_cond = (_cfg_call_counter[0] % 2 == 0)
        _cfg_call_counter[0] += 1
        kw["img_context_emb"] = (_clip_emb.to(device=dev, dtype=torch.bfloat16)
                                 if is_cond else None)
        a = tuple(
            x.to(dtype=torch.bfloat16) if isinstance(x, torch.Tensor) and x.is_floating_point() else x
            for x in a
        )
        for k, v in kw.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                kw[k] = v.to(dtype=torch.bfloat16)
        return _original_forward(*a, **kw)

    net.forward = _patched_forward

    model = pipe.model
    original_generate = model.generate_samples_from_batch
    _g_img  = guided_image.to(dtype=torch.bfloat16)
    _g_mask = guided_mask.to(dtype=torch.bfloat16)

    @torch.no_grad()
    def patched_generate_v3(data_batch, guidance=1.5, seed=1, state_shape=None,
                            n_sample=None, is_negative_prompt=False,
                            num_steps=35, shift=5.0, **kwargs):
        import tqdm as tqdm_mod
        from cosmos_predict2._src.imaginaire.utils import misc

        model._normalize_video_databatch_inplace(data_batch)
        model._augment_image_dim_inplace(data_batch)
        is_image_batch = model.is_image_batch(data_batch)
        input_key = (model.input_image_key if is_image_batch
                     else model.input_data_key)
        if n_sample is None:
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            _T, _H, _W = data_batch[input_key].shape[-3:]
            state_shape = [
                model.config.state_ch,
                model.tokenizer.get_latent_num_frames(_T),
                _H // model.tokenizer.spatial_compression_factor,
                _W // model.tokenizer.spatial_compression_factor,
            ]

        noise = misc.arch_invariant_rand(
            (n_sample,) + tuple(state_shape),
            torch.float32, model.tensor_kwargs["device"], seed,
        )
        seed_g = torch.Generator(device=model.tensor_kwargs["device"])
        seed_g.manual_seed(seed)

        model.sample_scheduler.set_timesteps(
            num_steps, device=model.tensor_kwargs["device"], shift=shift,
            use_kerras_sigma=getattr(model.config,
                                     "use_kerras_sigma_at_inference", False),
        )
        timesteps = model.sample_scheduler.timesteps

        velocity_fn = model.get_velocity_fn_from_batch(
            data_batch, guidance, is_negative_prompt=is_negative_prompt,
        )
        latents = noise

        g_img  = _g_img.to(device=latents.device, dtype=latents.dtype)
        g_mask = _g_mask.to(device=latents.device, dtype=latents.dtype)
        if g_mask.shape[2:] != latents.shape[2:]:
            g_mask = F.interpolate(g_mask.float(), size=latents.shape[2:],
                                   mode="nearest").to(dtype=latents.dtype)
        if g_img.shape[2:] != latents.shape[2:]:
            g_img = F.interpolate(g_img.float(), size=latents.shape[2:],
                                  mode="nearest").to(dtype=latents.dtype)

        for step_idx, ts in enumerate(tqdm_mod.tqdm(timesteps, desc="V3 panoramic")):
            latents_blended = g_mask * g_img + (1.0 - g_mask) * latents
            _cfg_call_counter[0] = 0
            timestep = torch.stack([ts])
            velocity_pred = velocity_fn(noise, latents_blended,
                                         timestep.unsqueeze(0))
            temp_x0 = model.sample_scheduler.step(
                velocity_pred.unsqueeze(0), ts,
                latents_blended[0].unsqueeze(0),
                return_dict=False, generator=seed_g,
            )[0]
            latents = temp_x0.squeeze(0)

        latents = g_mask * g_img + (1.0 - g_mask) * latents
        return latents

    model.generate_samples_from_batch = patched_generate_v3

    try:
        video = pipe.generate_vid2world(
            prompt=prompt,
            input_path=None,
            guidance=args.guidance,
            num_video_frames=args.num_frames,
            num_latent_conditional_frames=0,
            resolution=args.resolution,
            seed=seed,
            negative_prompt=args.negative_prompt,
            num_steps=args.num_steps,
        )
    finally:
        model.generate_samples_from_batch = original_generate
        net.forward = _original_forward

    timing["t_step5_diffuse"] = time.time() - t

    # ── save ───────────────────────────────────────────────────────────
    t = time.time()
    video_out = (1.0 + video[0]) / 2.0
    rope_tag  = "equirect" if args.equirect_rope else "linear"
    base      = f"pano_v3_{h_pano}x{w_pano}_{rope_tag}_s{seed}"
    output_path = os.path.join(output_dir, base)
    save_img_or_video(video_out, output_path, fps=16)
    timing["t_save"] = time.time() - t

    return f"{output_path}.mp4", timing


# ────────────────────────────── main loop ──────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master", default=str(MASTER_DEFAULT))
    ap.add_argument("--results", required=True,
                    help="results_dir; output → <results>/panoworld_pers/<clip>/")
    ap.add_argument("--finetune_checkpoint", required=True,
                    help="Path to fine-tuned panoramic weights "
                         "(e.g. checkpoints/v5_geo_long/iter_000001000/model_ema_bf16.pt)")
    ap.add_argument("--model", default="2B/post-trained")
    ap.add_argument("--num_frames", type=int, default=93)
    ap.add_argument("--guidance", type=int, default=7)
    ap.add_argument("--num_steps", type=int, default=35)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resolution", default="512,1024")
    ap.add_argument("--fov_x", type=float, default=90.0)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--pers_h", type=int, default=480)
    ap.add_argument("--pers_w", type=int, default=640)
    ap.add_argument("--latent_padding_size", type=int, default=2)
    ap.add_argument("--equirect_rope", action="store_true", default=True)
    ap.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)
    ap.add_argument("--gen_fps", type=int, default=16,
                    help="static-replicate fps for the synthetic pers_input.mp4")
    ap.add_argument("--caption_style", default="long",
                    help="Preferred caption style (long/medium/short); falls back if missing.")
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--only_clips", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip_existing", action="store_true", default=True)
    ap.add_argument("--no_skip_existing", dest="skip_existing",
                    action="store_false")
    return ap.parse_args()


def main():
    args = parse_args()

    # Filter master.csv
    rows = read_master(args.master)
    if args.splits:
        rows = [r for r in rows if r.get("split") in set(args.splits)]
    if args.only_clips:
        rows = [r for r in rows if r.get("clip_id") in set(args.only_clips)]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("[batch] no clips selected; exiting")
        return 0

    method_id = "panoworld_pers"
    results   = Path(args.results)
    method_dir = results / method_id
    log_dir    = results / "_logs" / method_id
    method_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup pipe + SigLIP ONCE.
    t_setup_start = time.time()
    pipe = setup_v3_pipe(args)
    siglip_model, siglip_image_size, siglip_hidden_size = setup_siglip()
    setup_s = time.time() - t_setup_start
    print(f"[batch] setup completed in {setup_s:.1f}s "
          f"({len(rows)} clip(s) to run)")

    n_ok = n_skip = n_err = 0
    t_loop_start = time.time()

    for idx, row in enumerate(rows):
        clip_id  = row["clip_id"]
        out_dir  = method_dir / clip_id_to_dirname(clip_id)
        final_mp4 = out_dir / "video.mp4"
        clip_log  = log_dir / f"{clip_id_to_dirname(clip_id)}.log"

        if args.skip_existing and have_output(out_dir):
            n_skip += 1
            print(f"[{idx + 1}/{len(rows)}] {clip_id} skip (have output)")
            continue

        try:
            # caption
            cap_text, cap_style = load_caption_with_style(
                row.get("caption_path", ""), style=args.caption_style,
            )
            if not cap_text:
                raise RuntimeError(f"no caption (path={row.get('caption_path')})")

            gt_video = row.get("video_path", "")
            if not gt_video or not Path(gt_video).is_file():
                raise RuntimeError(f"GT video missing: {gt_video}")

            # Pers crop + static-repeat → pers_input.mp4
            work_dir = out_dir / "_work"
            work_dir.mkdir(parents=True, exist_ok=True)
            erp0 = read_first_frame_rgb_uint8(gt_video)
            pers_arr = erp_to_perspective_image(
                erp0, fov_x_deg=args.fov_x, yaw_rad=0.0, pitch_rad=0.0,
                out_h=args.pers_h, out_w=args.pers_w,
            )
            pers_input_mp4 = work_dir / "pers_input.mp4"
            make_static_video_from_image(
                pers_arr, pers_input_mp4,
                n_frames=args.num_frames, fps=args.gen_fps,
            )

            # Provenance BEFORE generation.
            write_run_json(out_dir, {
                "method_id":    method_id,
                "clip_id":      clip_id,
                "stage":        "pers",
                "gt_video":     gt_video,
                "caption":      cap_text,
                "caption_style": cap_style,
                "status":       "running",
                "ts_start":     int(time.time()),
                "num_frames":   args.num_frames,
                "num_steps":    args.num_steps,
                "guidance":     args.guidance,
                "resolution":   args.resolution,
                "seed":         args.seed,
                "finetune_checkpoint": args.finetune_checkpoint,
                "driver":       "infer_panoworld_batch",
            })

            # Generate.
            t_clip = time.time()
            gen_mp4_path, timing = run_v3_one_clip(
                pipe,
                siglip_model, siglip_image_size, siglip_hidden_size,
                args,
                prompt=cap_text,
                pers_input_mp4=str(pers_input_mp4),
                output_dir=str(work_dir),
                seed=args.seed,
            )
            elapsed_s = time.time() - t_clip

            # Move final mp4 into <out_dir>/video.mp4 and clean work_dir.
            os.replace(gen_mp4_path, final_mp4)

            write_run_json(out_dir, {
                "method_id":    method_id,
                "clip_id":      clip_id,
                "stage":        "pers",
                "gt_video":     gt_video,
                "caption":      cap_text,
                "caption_style": cap_style,
                "status":       "ok",
                "elapsed_s":    elapsed_s,
                "timing":       timing,
                "num_frames":   args.num_frames,
                "num_steps":    args.num_steps,
                "guidance":     args.guidance,
                "resolution":   args.resolution,
                "seed":         args.seed,
                "finetune_checkpoint": args.finetune_checkpoint,
                "driver":       "infer_panoworld_batch",
            })

            n_ok += 1
            avg = (time.time() - t_loop_start) / max(n_ok, 1)
            remaining = len(rows) - idx - 1
            eta_min = remaining * avg / 60.0
            print(f"[{idx + 1}/{len(rows)}] {clip_id} ok "
                  f"({elapsed_s:.1f}s) avg={avg:.1f}s/clip "
                  f"ETA {eta_min:.1f} min")

        except Exception as e:
            n_err += 1
            tb = traceback.format_exc()
            print(f"[{idx + 1}/{len(rows)}] {clip_id} ERROR: {e}")
            print(tb)
            try:
                clip_log.parent.mkdir(parents=True, exist_ok=True)
                clip_log.write_text(tb)
                write_run_json(out_dir, {
                    "method_id":    method_id,
                    "clip_id":      clip_id,
                    "stage":        "pers",
                    "status":       "failed",
                    "error":        str(e),
                    "log":          str(clip_log),
                    "driver":       "infer_panoworld_batch",
                })
            except Exception:
                pass

    total_s = time.time() - t_loop_start
    print(f"[batch] === DONE ===  ok={n_ok}  skip={n_skip}  err={n_err}  "
          f"setup={setup_s:.1f}s  loop={total_s:.1f}s  "
          f"({total_s / max(n_ok, 1):.1f}s/ok-clip)")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
