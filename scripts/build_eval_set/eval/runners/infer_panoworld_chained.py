"""Single-process Round-1 + Round-2 chained driver for PanoWorld.

This is the **methodologically correct** PanoWorld eval pipeline:

    pers_crop  -- Round1 (V3 pers→pano) -->  ERP video  -- frame[0] -->
    erp_first_frame.png  -- Round2 (image-init V2W, no FOV mask) -->  EVAL video

Round 1 has the FOV-locked spatial mask blending (necessary to bootstrap a
plausible 360° first frame from a 90° pers crop). Round 2 takes only the
*first frame* of Round 1's output and runs **standard image-conditioned
video2world** — no spatial mask, no static-repeat, no img_context_emb,
no reference_latent — so the model is free to generate motion in *all*
360° regions including the original FOV center.

The previous driver (``infer_panoworld_batch.py``) only ran Round 1 and
saved its output as the eval video, which produced FOV-center static
videos because of the per-step ``g_mask * g_img + (1-g_mask) * latents``
blending. Those outputs are now archived under
``panoworld_round1_only/`` for ablation; this driver writes its Round-2
videos to ``panoworld_pers/`` (the canonical method id in eval_config.yaml,
defined as ``inference: round1_then_round2``).

Output layout
-------------
    <results>/panoworld_pers/<clip_id_dir>/video.mp4         # Round-2 EVAL video
    <results>/panoworld_pers/<clip_id_dir>/run.json
    <results>/panoworld_pers/<clip_id_dir>/_work/
        pers_input.mp4         (Round-1 input — static-repeat pers crop)
        round1.mp4             (Round-1 output — FOV-locked ERP video)
        erp_first_frame.png    (Round-1 frame 0 — Round-2 input)
    <results>/_logs/panoworld_pers/<clip>.log

Round-1 reuse
-------------
If ``--round1_reuse_dir`` is given (default
``panoworld_round1_only``), and that directory contains a completed
``<clip>/video.mp4`` from a previous Round-1-only run, the driver
extracts its frame 0 and **skips** Round 1 for that clip. This saves
~7min/clip when re-running after the round1-only batch.

Usage::

    python test_set_pkg/eval/runners/infer_panoworld_chained.py \\
        --master $PANO_DATA_ROOT/test/master.csv \\
        --results eval_results_panoworld \\
        --finetune_checkpoint checkpoints/v5_geo_long/iter_000001000/model_ema_bf16.pt
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

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

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
    save_image_png,
    write_run_json,
)

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

from generate_pano import read_perspective_video                    # noqa: E402

SIGLIP2_MODEL_ID = "google/siglip2-so400m-patch14-384"


# ────────────────────────── one-time pipe + siglip ─────────────────────
def setup_v3_pipe(args) -> Video2WorldInference:
    """Identical to infer_panoworld_batch.setup_v3_pipe."""
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
    log.info("Chained driver: building pipeline (one-time)")
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
            left_p = latent[..., -_pad:]
            right_p = latent[..., :_pad]
            padded = torch.cat([left_p, latent, right_p], dim=-1)
            decoded = _orig_decode(padded)
            px_p = _pad * 8
            return decoded[..., px_p:-px_p]

        pipe.model.decode = _circular_decode
        log.info(f"Installed circular latent decode (pad={_pad} cols/side)")

    return pipe


def setup_siglip(model_id: str = SIGLIP2_MODEL_ID):
    from transformers import SiglipVisionModel
    log.info(f"Loading SigLIP2 from {model_id} ...")
    clip_model = SiglipVisionModel.from_pretrained(model_id)
    clip_model = clip_model.to(device="cuda", dtype=torch.bfloat16).eval()
    return clip_model, clip_model.config.image_size, clip_model.config.hidden_size


def encode_temporal_clip_with(siglip_model, image_size: int,
                              hidden_size: int,
                              video_frames_01: torch.Tensor) -> torch.Tensor:
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
    out = torch.cat(feats, dim=0)
    tok = out.shape[1]
    out = out.reshape(1, T * tok, hidden_size).to(dtype=torch.bfloat16)
    del frames
    return out


# ────────────────────────────── Round 1 ────────────────────────────────
def run_round1(pipe: Video2WorldInference,
               siglip_model, siglip_image_size, siglip_hidden_size,
               args, *,
               prompt: str,
               pers_input_mp4: str,
               output_dir: str,
               seed: int) -> tuple[str, dict, torch.Tensor, torch.Tensor]:
    """V3 pers→pano (FOV-locked). Returns (mp4_path, timing, video, temporal_clip_emb).

    The returned ``temporal_clip_emb`` is the SigLIP cond computed over the
    perspective input (93-frame static repeat). Round-2 reuses it verbatim:
    SigLIP was trained on perspective frames and our fine-tune saw exactly
    this static-pers cond, so reusing it keeps Round-2's img_context path
    in-distribution.
    """
    os.makedirs(output_dir, exist_ok=True)
    h_pano, w_pano = [int(x) for x in args.resolution.split(",")]
    target_T = args.num_frames

    timing: dict[str, float] = {}

    t = time.time()
    pers_frames_m1p1 = read_perspective_video(pers_input_mp4, target_T).to("cuda")
    pers_frames_01 = (pers_frames_m1p1 + 1.0) / 2.0
    timing["t_step1_read"] = time.time() - t

    t = time.time()
    temporal_clip_emb = encode_temporal_clip_with(
        siglip_model, siglip_image_size, siglip_hidden_size,
        pers_frames_01.to("cuda"),
    )
    timing["t_step3_clip"] = time.time() - t

    t = time.time()
    from cosmos_predict2._src.predict2.utils.pano_conditioning import (
        pers2equi_simple,
    )
    T_in = pers_frames_m1p1.shape[0]
    device = torch.device("cuda")
    Hp, Wp = pers_frames_m1p1.shape[2], pers_frames_m1p1.shape[3]
    hw_ratio = Hp / Wp

    roll_t = torch.zeros(T_in, device=device)
    pitch_t = torch.zeros(T_in, device=device)
    yaw_t = torch.full((T_in,), args.yaw, device=device)

    # === FOV-orientation fix ===
    # _common.erp_to_perspective_image() uses cosmos's equi2pers (z=down convention)
    # which produces vertically-flipped pers crops relative to a "normal upright photo".
    # pers2equi_simple uses z=up convention. Round-trip without correction = FOV center
    # vertically flipped in the projected ERP (verified: red-top/blue-bottom test ERP →
    # round-trip ERP center has BLUE on top). That makes ref_latent disagree with what
    # training sees (training uses raw upright GT inside FOV). The model dutifully
    # generates a flipped FOV center.
    # Training keeps SigLIP cond as equi2pers output (flipped), so we must NOT touch
    # pers_frames_m1p1 in the SigLIP path above. We only flip the copy fed to
    # pers2equi_simple, restoring upright orientation for the projection only.
    pers_for_proj = pers_frames_m1p1.to(device).flip(dims=[2])  # vertical flip (H axis)

    equi_proj, pixel_mask = pers2equi_simple(
        pers_for_proj, fov_x=args.fov_x,
        roll=roll_t, pitch=pitch_t, yaw=yaw_t,
        equi_h=h_pano, equi_w=w_pano, hw_ratio=hw_ratio,
    )
    coverage = pixel_mask.mean().item()
    log.info(f"[Round1] coverage {coverage:.1%} (pers vertical-flipped before projection to fix FOV-orientation)")

    equi_BCTHW = equi_proj.permute(1, 0, 2, 3).unsqueeze(0)
    mask_11THW = pixel_mask[:, :1].permute(1, 0, 2, 3).unsqueeze(0)

    noise_fill = torch.randn_like(equi_BCTHW) * 0.5
    masked_equi = torch.where(mask_11THW.expand_as(equi_BCTHW) > 0.5,
                              equi_BCTHW, noise_fill)

    tokenizer = pipe.model.tokenizer
    if hasattr(tokenizer, "encoder") and tokenizer.encoder is not None:
        if next(tokenizer.encoder.parameters()).device.type == "cpu":
            tokenizer.encoder = tokenizer.encoder.to(device)

    with torch.no_grad():
        guided_image = tokenizer.encode(equi_BCTHW.float().to(device))
        reference_latent = tokenizer.encode(masked_equi.float().to(device))

    _, _, T_lat, H_lat, W_lat = guided_image.shape
    guided_mask = F.interpolate(
        mask_11THW.float().to(device),
        size=(T_lat, H_lat, W_lat), mode="nearest",
    )
    guided_mask = (guided_mask > 0.5).float()
    log.info(f"[Round1] guided_image={tuple(guided_image.shape)}  "
             f"latent_mask_coverage={guided_mask.mean().item():.1%}")

    if hasattr(tokenizer, "encoder") and tokenizer.encoder is not None:
        tokenizer.encoder = tokenizer.encoder.to("cpu")
        torch.cuda.empty_cache()

    del pers_frames_01, pers_frames_m1p1, pers_for_proj, equi_proj, pixel_mask
    del equi_BCTHW, mask_11THW, noise_fill, masked_equi
    torch.cuda.empty_cache()
    timing["t_step4_proj"] = time.time() - t

    t = time.time()
    net = pipe.model.net
    _original_forward = net.forward
    _clip_emb = temporal_clip_emb.to(dtype=torch.bfloat16)
    _ref_lat = reference_latent.to(dtype=torch.bfloat16)
    _spatial_mask = guided_mask.to(dtype=torch.bfloat16)
    _cfg_call_counter = [0]

    def _patched_forward(*a, **kw):
        dev = a[0].device if a else "cuda"
        kw["reference_latent_B_C_T_H_W"] = _ref_lat.to(device=dev, dtype=torch.bfloat16)
        kw["condition_video_input_mask_B_C_T_H_W"] = _spatial_mask.to(device=dev, dtype=torch.bfloat16)
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
    _g_img = guided_image.to(dtype=torch.bfloat16)
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

        g_img = _g_img.to(device=latents.device, dtype=latents.dtype)
        g_mask = _g_mask.to(device=latents.device, dtype=latents.dtype)
        if g_mask.shape[2:] != latents.shape[2:]:
            g_mask = F.interpolate(g_mask.float(), size=latents.shape[2:],
                                   mode="nearest").to(dtype=latents.dtype)
        if g_img.shape[2:] != latents.shape[2:]:
            g_img = F.interpolate(g_img.float(), size=latents.shape[2:],
                                  mode="nearest").to(dtype=latents.dtype)

        for step_idx, ts in enumerate(tqdm_mod.tqdm(timesteps, desc="Round1 V3")):
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

    t = time.time()
    video_out = (1.0 + video[0]) / 2.0          # (T, C, H, W) in [0, 1]
    base = "round1"
    output_path = os.path.join(output_dir, base)
    save_img_or_video(video_out, output_path, fps=16)
    timing["t_save_round1"] = time.time() - t

    # Detach + move to CPU so the caller can keep a small reference across
    # rounds without holding GPU memory while Round-2 sets up its own ref_lat.
    temporal_clip_emb_cpu = temporal_clip_emb.detach().to("cpu")
    return f"{output_path}.mp4", timing, video, temporal_clip_emb_cpu


# ────────────────────────────── Round 2 ────────────────────────────────
def run_round2(pipe: Video2WorldInference, args, *,
               prompt: str,
               erp_first_frame_png: str,
               round1_mp4_path: str,
               temporal_clip_emb: torch.Tensor,
               output_dir: str,
               seed: int) -> tuple[str, dict]:
    """ERP-image-init video2world with panorama-aware conditioning ("C-prime").

    The fine-tuned model was trained to *require* ``reference_latent_B_C_T_H_W``
    and ``img_context_emb`` (28 cross_attn k_img/v_img layers + img_context_proj
    + 16 extra reference channels in x_embedder). A bare ``generate_vid2world``
    feeds zeros into those weights, which is severely OOD; the visible quality
    drop in our first chained outputs traces back to that. Here we reuse the
    same conditioning interface as Round 1 but without blended diffusion (which
    would re-lock the FOV center and undo the temporal motion Round 2 is
    supposed to unlock):

      • ref_latent      = VAE-encode(round1.mp4)        (93-frame ERP context)
      • img_context_emb = temporal_clip_emb (from R1)   (static pers SigLIP)
      • spatial_mask    = 1 on frame-0 latent, 0 elsewhere  (anchor frame 0)
      • generate_vid2world(input_path=erp_frame0, num_latent_conditional_frames=1)

    The pipe-level i2v anchor on frame 0 still holds; the patched forward
    additionally injects ref/img-context into the model's cross-attn so the
    fine-tuned weights see in-distribution inputs.
    """
    os.makedirs(output_dir, exist_ok=True)
    timing: dict[str, float] = {}
    target_T = args.num_frames
    device = torch.device("cuda")

    # ── 1. Read Round-1 ERP video → VAE encode → reference_latent ─────────
    t = time.time()
    r1_frames_m1p1 = read_perspective_video(round1_mp4_path, target_T).to(device)
    r1_BCTHW = r1_frames_m1p1.permute(1, 0, 2, 3).unsqueeze(0)  # (1,3,T,H,W)

    tokenizer = pipe.model.tokenizer
    if hasattr(tokenizer, "encoder") and tokenizer.encoder is not None:
        if next(tokenizer.encoder.parameters()).device.type == "cpu":
            tokenizer.encoder = tokenizer.encoder.to(device)

    with torch.no_grad():
        reference_latent = tokenizer.encode(r1_BCTHW.float().to(device))

    if hasattr(tokenizer, "encoder") and tokenizer.encoder is not None:
        tokenizer.encoder = tokenizer.encoder.to("cpu")
        torch.cuda.empty_cache()

    _, _, T_lat, H_lat, W_lat = reference_latent.shape

    # Frame-0-only spatial mask: model is told "trust frame 0, free elsewhere"
    spatial_mask = torch.zeros(
        (1, 1, T_lat, H_lat, W_lat), device=device, dtype=torch.float32,
    )
    spatial_mask[:, :, 0:1, :, :] = 1.0

    log.info(f"[Round2] ref_latent={tuple(reference_latent.shape)}  "
             f"frame0_only_mask  clip_emb={tuple(temporal_clip_emb.shape)}")
    del r1_frames_m1p1, r1_BCTHW
    torch.cuda.empty_cache()
    timing["t_round2_setup"] = time.time() - t

    # ── 2. Patch net.forward (CFG-aware: img_context only on cond branch) ──
    net = pipe.model.net
    _original_forward = net.forward
    _clip_emb = temporal_clip_emb.to(device=device, dtype=torch.bfloat16)
    _ref_lat = reference_latent.to(dtype=torch.bfloat16)
    _spatial_mask = spatial_mask.to(dtype=torch.bfloat16)
    _cfg_call_counter = [0]

    def _patched_forward(*a, **kw):
        dev = a[0].device if a else "cuda"
        kw["reference_latent_B_C_T_H_W"] = _ref_lat.to(device=dev, dtype=torch.bfloat16)
        kw["condition_video_input_mask_B_C_T_H_W"] = _spatial_mask.to(
            device=dev, dtype=torch.bfloat16,
        )
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

    # ── 3. Diffuse (vanilla generate_vid2world; NO blended diffusion) ──────
    t = time.time()
    try:
        video = pipe.generate_vid2world(
            prompt=prompt,
            input_path=erp_first_frame_png,
            guidance=args.guidance,
            num_video_frames=args.num_frames,
            num_latent_conditional_frames=1,
            resolution=args.resolution,
            seed=seed,
            negative_prompt=args.negative_prompt,
            num_steps=args.num_steps,
        )
    finally:
        net.forward = _original_forward
    timing["t_round2_diffuse"] = time.time() - t

    # ── 4. Save ────────────────────────────────────────────────────────────
    t = time.time()
    video_out = (1.0 + video[0]) / 2.0
    base = "round2"
    output_path = os.path.join(output_dir, base)
    save_img_or_video(video_out, output_path, fps=16)
    timing["t_save_round2"] = time.time() - t

    del reference_latent, spatial_mask
    torch.cuda.empty_cache()

    return f"{output_path}.mp4", timing


# ────────────────────────────── helpers ────────────────────────────────
def extract_frame0_to_png(video_tensor_BTCHW_or_TCHW: torch.Tensor,
                          out_png: str) -> str:
    """Save frame 0 of an (B,T,C,H,W) or (T,C,H,W) [-1,1] tensor as PNG."""
    v = video_tensor_BTCHW_or_TCHW
    if v.dim() == 5:
        v = v[0]
    f0 = v[0]
    f0 = (f0 + 1.0) / 2.0
    f0 = (f0.clamp(0, 1).permute(1, 2, 0).cpu().float().numpy() * 255.0).astype(np.uint8)
    return save_image_png(f0, out_png)


def extract_frame0_from_mp4(mp4_path: str, out_png: str) -> str:
    """Decode an mp4 from disk and save its first frame as PNG."""
    arr = read_first_frame_rgb_uint8(mp4_path)
    return save_image_png(arr, out_png)


# ────────────────────────────── main loop ──────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master", default=str(MASTER_DEFAULT))
    ap.add_argument("--results", required=True,
                    help="results_dir; output → <results>/panoworld_pers/<clip>/")
    ap.add_argument("--finetune_checkpoint", required=True)
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
    ap.add_argument("--gen_fps", type=int, default=16)
    ap.add_argument("--caption_style", default="long")
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--only_clips", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip_existing", action="store_true", default=True)
    ap.add_argument("--no_skip_existing", dest="skip_existing",
                    action="store_false")
    ap.add_argument("--method_id", default="panoworld_main",
                    help="Output method-dir name under <results>/. Default "
                         "'panoworld_main' matches the public PanoWorld "
                         "checkpoint. Use a versioned name to keep multiple "
                         "runs side-by-side without overwriting.")
    ap.add_argument("--round1_reuse_dir", default="panoworld_round1_only",
                    help="Method dirname under <results>/ to reuse Round-1 "
                         "videos from (extract frame 0, skip Round 1). "
                         "Pass empty string to disable.")
    ap.add_argument("--keep_work_dir", action="store_true", default=True,
                    help="Retain _work/ (round1.mp4 + erp_first_frame.png + pers_input.mp4)")
    ap.add_argument("--scene_first", action="store_true", default=False,
                    help="Reorder clips so every scene contributes its 1st clip "
                         "before any 2nd clip is touched (round-robin by scene). "
                         "Useful for getting wide scene coverage early. Scene id "
                         "is derived from clip_id by stripping the trailing "
                         "'__clipNNN' suffix.")
    return ap.parse_args()


def main():
    args = parse_args()

    rows = read_master(args.master)
    if args.splits:
        rows = [r for r in rows if r.get("split") in set(args.splits)]
    if args.only_clips:
        rows = [r for r in rows if r.get("clip_id") in set(args.only_clips)]
    if args.scene_first:
        # Group by scene (clip_id stripped of trailing "__clipNNN") preserving
        # original order within a scene; then interleave scenes round-robin so
        # wave-0 = every scene's 1st clip, wave-1 = every scene's 2nd clip, ...
        import re
        from collections import OrderedDict
        clip_re = re.compile(r"__clip\d+$")
        groups = OrderedDict()  # scene_id -> [rows...] in master.csv order
        for r in rows:
            scene = clip_re.sub("", r.get("clip_id", ""))
            groups.setdefault(scene, []).append(r)
        max_w = max(len(v) for v in groups.values())
        reordered = []
        for w in range(max_w):
            for scene, rs in groups.items():
                if w < len(rs):
                    reordered.append(rs[w])
        rows = reordered
        print(f"[chained] --scene_first ON: {len(groups)} scene(s), "
              f"max {max_w} clip(s)/scene → reordered {len(rows)} row(s); "
              f"first {len(groups)} = 1-per-scene wave")
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("[chained] no clips selected; exiting")
        return 0

    method_id = args.method_id
    results = Path(args.results)
    method_dir = results / method_id
    log_dir = results / "_logs" / method_id
    method_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    reuse_dir = None
    if args.round1_reuse_dir:
        reuse_dir = results / args.round1_reuse_dir
        if not reuse_dir.is_dir():
            print(f"[chained] round1_reuse_dir does not exist: {reuse_dir} (will run Round 1 fresh for all clips)")
            reuse_dir = None
        else:
            print(f"[chained] Round-1 reuse enabled: {reuse_dir}")

    t_setup_start = time.time()
    pipe = setup_v3_pipe(args)
    siglip_model, siglip_image_size, siglip_hidden_size = setup_siglip()
    setup_s = time.time() - t_setup_start
    print(f"[chained] setup completed in {setup_s:.1f}s "
          f"({len(rows)} clip(s) to run)")

    n_ok = n_skip = n_err = n_reuse = 0
    t_loop_start = time.time()

    for idx, row in enumerate(rows):
        clip_id = row["clip_id"]
        out_dir = method_dir / clip_id_to_dirname(clip_id)
        final_mp4 = out_dir / "video.mp4"
        clip_log = log_dir / f"{clip_id_to_dirname(clip_id)}.log"

        if args.skip_existing and have_output(out_dir):
            n_skip += 1
            print(f"[{idx + 1}/{len(rows)}] {clip_id} skip (have output)")
            continue

        try:
            cap_text, cap_style = load_caption_with_style(
                row.get("caption_path", ""), style=args.caption_style,
            )
            if not cap_text:
                raise RuntimeError(f"no caption (path={row.get('caption_path')})")

            gt_video = row.get("video_path", "")
            if not gt_video or not Path(gt_video).is_file():
                raise RuntimeError(f"GT video missing: {gt_video}")

            work_dir = out_dir / "_work"
            work_dir.mkdir(parents=True, exist_ok=True)

            erp_png = work_dir / "erp_first_frame.png"
            timing_total: dict = {}
            round1_source = None  # 'reuse' or 'fresh'
            temporal_clip_emb = None  # populated by both fresh + reuse paths

            # ── Round 1 (or reuse) ───────────────────────────────────────
            reuse_video = None
            if reuse_dir is not None:
                cand = reuse_dir / clip_id_to_dirname(clip_id) / "video.mp4"
                if cand.is_file() and cand.stat().st_size > 0:
                    reuse_video = cand

            t_clip = time.time()
            if reuse_video is not None:
                t = time.time()
                extract_frame0_from_mp4(str(reuse_video), str(erp_png))
                # Symlink (or copy) reuse video into _work/round1.mp4 for provenance.
                round1_mp4 = work_dir / "round1.mp4"
                if round1_mp4.exists() or round1_mp4.is_symlink():
                    round1_mp4.unlink()
                try:
                    os.symlink(reuse_video.resolve(), round1_mp4)
                except OSError:
                    import shutil
                    shutil.copy2(reuse_video, round1_mp4)
                timing_total["t_round1_reuse"] = time.time() - t
                round1_source = "reuse"
                n_reuse += 1
                print(f"[{idx + 1}/{len(rows)}] {clip_id} reuse Round-1 "
                      f"from {reuse_video}")

                # Round 2 needs a SigLIP cond. Recompute it from a static-pers
                # cropped from GT (same recipe used by the fresh path).
                t = time.time()
                erp0 = read_first_frame_rgb_uint8(gt_video)
                pers_arr = erp_to_perspective_image(
                    erp0, fov_x_deg=args.fov_x, yaw_rad=0.0, pitch_rad=0.0,
                    out_h=args.pers_h, out_w=args.pers_w,
                )
                pers_static = (
                    torch.from_numpy(pers_arr).float() / 127.5 - 1.0
                ).permute(2, 0, 1).unsqueeze(0).repeat(args.num_frames, 1, 1, 1)
                pers_static_01 = (pers_static + 1.0) / 2.0
                temporal_clip_emb = encode_temporal_clip_with(
                    siglip_model, siglip_image_size, siglip_hidden_size,
                    pers_static_01.to("cuda"),
                ).detach().to("cpu")
                del pers_static, pers_static_01
                torch.cuda.empty_cache()
                timing_total["t_round1_reuse_clip_emb"] = time.time() - t
            else:
                # Fresh Round 1: prepare pers_input.mp4, run V3.
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

                write_run_json(out_dir, {
                    "method_id": method_id,
                    "clip_id": clip_id,
                    "stage": "round1_then_round2",
                    "phase": "round1",
                    "gt_video": gt_video,
                    "caption": cap_text,
                    "caption_style": cap_style,
                    "status": "running",
                    "ts_start": int(time.time()),
                    "num_frames": args.num_frames,
                    "num_steps": args.num_steps,
                    "guidance": args.guidance,
                    "resolution": args.resolution,
                    "seed": args.seed,
                    "finetune_checkpoint": args.finetune_checkpoint,
                    "driver": "infer_panoworld_chained",
                })

                round1_mp4_path, t1, video1, temporal_clip_emb = run_round1(
                    pipe,
                    siglip_model, siglip_image_size, siglip_hidden_size,
                    args,
                    prompt=cap_text,
                    pers_input_mp4=str(pers_input_mp4),
                    output_dir=str(work_dir),
                    seed=args.seed,
                )
                timing_total.update({f"round1__{k}": v for k, v in t1.items()})
                # Frame 0 of Round-1 output → ERP PNG (Round-2 input).
                # Read from the saved mp4 (robust to tensor-shape variants returned by pipe).
                del video1
                extract_frame0_from_mp4(round1_mp4_path, str(erp_png))
                torch.cuda.empty_cache()
                round1_source = "fresh"

            # ── Round 2 ──────────────────────────────────────────────────
            write_run_json(out_dir, {
                "method_id": method_id,
                "clip_id": clip_id,
                "stage": "round1_then_round2",
                "phase": "round2",
                "gt_video": gt_video,
                "caption": cap_text,
                "caption_style": cap_style,
                "status": "running",
                "ts_start": int(time.time()),
                "round1_source": round1_source,
                "num_frames": args.num_frames,
                "num_steps": args.num_steps,
                "guidance": args.guidance,
                "resolution": args.resolution,
                "seed": args.seed,
                "finetune_checkpoint": args.finetune_checkpoint,
                "driver": "infer_panoworld_chained",
            })

            round2_mp4_path, t2 = run_round2(
                pipe, args,
                prompt=cap_text,
                erp_first_frame_png=str(erp_png),
                round1_mp4_path=str(work_dir / "round1.mp4"),
                temporal_clip_emb=temporal_clip_emb,
                output_dir=str(work_dir),
                seed=args.seed,
            )
            timing_total.update({f"round2__{k}": v for k, v in t2.items()})

            # The Round-2 output is the EVAL video.
            os.replace(round2_mp4_path, final_mp4)
            elapsed_s = time.time() - t_clip

            write_run_json(out_dir, {
                "method_id": method_id,
                "clip_id": clip_id,
                "stage": "round1_then_round2",
                "phase": "done",
                "gt_video": gt_video,
                "caption": cap_text,
                "caption_style": cap_style,
                "status": "ok",
                "elapsed_s": elapsed_s,
                "round1_source": round1_source,
                "timing": timing_total,
                "num_frames": args.num_frames,
                "num_steps": args.num_steps,
                "guidance": args.guidance,
                "resolution": args.resolution,
                "seed": args.seed,
                "finetune_checkpoint": args.finetune_checkpoint,
                "driver": "infer_panoworld_chained",
            })

            n_ok += 1
            avg = (time.time() - t_loop_start) / max(n_ok, 1)
            remaining = len(rows) - idx - 1
            eta_h = remaining * avg / 3600.0
            print(f"[{idx + 1}/{len(rows)}] {clip_id} ok "
                  f"({elapsed_s:.1f}s, src={round1_source}) "
                  f"avg={avg:.1f}s/clip ETA {eta_h:.1f}h")

        except Exception as e:
            n_err += 1
            tb = traceback.format_exc()
            print(f"[{idx + 1}/{len(rows)}] {clip_id} ERROR: {e}")
            print(tb)
            try:
                clip_log.parent.mkdir(parents=True, exist_ok=True)
                clip_log.write_text(tb)
                write_run_json(out_dir, {
                    "method_id": method_id,
                    "clip_id": clip_id,
                    "stage": "round1_then_round2",
                    "status": "failed",
                    "error": str(e),
                    "log": str(clip_log),
                    "driver": "infer_panoworld_chained",
                })
            except Exception:
                pass

    total_s = time.time() - t_loop_start
    print(f"[chained] === DONE ===  ok={n_ok}  reuse={n_reuse}  "
          f"skip={n_skip}  err={n_err}  setup={setup_s:.1f}s  "
          f"loop={total_s:.1f}s  ({total_s / max(n_ok, 1):.1f}s/ok-clip)")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
