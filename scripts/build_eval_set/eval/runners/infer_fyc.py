#!/usr/bin/env python3
"""Follow-Your-Canvas inference runner (perspective video → ERP outpaint).

Follow-Your-Canvas (Chen et al., 2024) does generic spatial outpainting on
short videos. It is NOT panoramic-aware (no spherical motion module, no
ERP geometry priors), but it is the strongest published video-outpainting
baseline; for our paper we drive it with a 2:1 (ERP-shaped) target canvas
and let it outpaint left/right around the GT first ERP frame's perspective
crop.

Setup (run once on A100):
    cd $HOME/Le/FollowYourCanvas
    bash install.sh                 # creates conda env `fyc`
    # ckpts under pretrained_models/follow-your-canvas/checkpoint-40000.ckpt
    # SD-2.1 base under HF cache (downloaded with PanoWorld pipeline).
    # SAM-vit-b in $HOME/Le/_baseline_shared/sam/sam_vit_b_01ec64.pth
    # Qwen-VL-Chat (LMM) — same model Imagine360 needs; share download.

Per-clip flow:
    GT first ERP frame --(pers crop, 90 deg)--> pers image
    pers image --(static-repeat 64f @ 8fps)--> pers_input.mp4
    write per-clip yaml override (target_size = 512x1024, 1 video in dir,
    prompts_input = [caption])
    inference_outpainting-dir-with-prompt.py --config <yaml>  --> ERP-ish video

FYC writes its final smoothed result to {output_dir}/result/<video_name>.

Output:
    results_dir/follow_your_canvas/<clip_id_dir>/video.mp4

Usage:
    python -m test_set_pkg.eval.runners.infer_fyc \\
        --results $PANO_DATA_ROOT/eval_results \\
        --conda_env fyc \\
        --splits self_iid argus_ood habitat_ood --limit 1
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _common import (  # noqa: E402
    MASTER_DEFAULT,
    erp_to_perspective_image,
    find_first_mp4,
    have_output,
    load_caption_with_style,
    make_static_video_from_image,
    normalize_output_video,
    per_clip_outdir,
    read_first_frame_rgb_uint8,
    read_master,
    run_subprocess,
    select_clips,
    wrap_with_conda,
    write_run_json,
)

FYC_REPO = Path("$HOME/Le/FollowYourCanvas")
BASE_YAML = FYC_REPO / "infer-configs/prompt-panda.yaml"
METHOD_ID = "follow_your_canvas"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=str(MASTER_DEFAULT))
    ap.add_argument("--results", required=True)
    ap.add_argument("--logs_dir", default=None)
    ap.add_argument("--conda_env", default="fyc")
    ap.add_argument("--repo", default=str(FYC_REPO))
    ap.add_argument("--base_yaml", default=str(BASE_YAML))
    ap.add_argument("--ckpt", default="pretrained_models/follow-your-canvas/checkpoint-40000.ckpt")
    _hf_cache = os.path.expanduser(
        os.environ.get("HF_HOME", os.path.join("~", ".cache", "huggingface"))
    )
    _baseline_shared = os.environ.get(
        "BASELINE_SHARED_DIR",
        os.path.join(os.path.expanduser("~"), "_baseline_shared"),
    )
    ap.add_argument("--sd21_path",
                    default=os.path.join(_hf_cache, "hub", "models--Manojb--stable-diffusion-2-1-base", "snapshots", "0094d483a120f3f33dafbd187ea4aa60d10de75c"),
                    help="SD-2.1 / 2.1-base local path. FYC's UNet/VAE/text-encoder load from here.")
    ap.add_argument("--lmm_path",
                    default=os.path.join(_hf_cache, "hub", "models--Qwen--Qwen-VL-Chat", "snapshots", "HEAD"),
                    help="Qwen-VL-Chat local path (LMM used by FYC's caption helper)")
    ap.add_argument("--sam_path",
                    default=os.path.join(_baseline_shared, "sam", "sam_vit_b_01ec64.pth"))
    ap.add_argument("--num_frames_input", type=int, default=64)
    ap.add_argument("--input_fps", type=int, default=8)
    ap.add_argument("--target_h", type=int, default=512)
    ap.add_argument("--target_w", type=int, default=1024)
    ap.add_argument("--fov_x", type=float, default=90.0)
    ap.add_argument("--pers_h", type=int, default=512)
    ap.add_argument("--pers_w", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--only_clips", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip_existing", action="store_true", default=True)
    ap.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    ap.add_argument("--dry_run", action="store_true", default=False)
    return ap.parse_args()


def write_per_clip_yaml(args, *, prompt: str, video_dir: str, output_dir: str,
                        out_yaml: Path):
    """Load the base FYC yaml and override the runtime fields cleanly with
    PyYAML (avoids the indentation pitfalls of string-level editing)."""
    import yaml
    cfg = yaml.safe_load(Path(args.base_yaml).read_text())
    cfg["output_dir"] = output_dir
    cfg["pretrained_model_path"] = args.sd21_path
    cfg["motion_pretrained_model_path"] = args.ckpt
    cfg["lmm_path"] = args.lmm_path
    cfg["image_pretrained_model_path"] = args.sam_path
    cfg["video_dir"] = video_dir
    # FYC iterates by index over prompts_input AND negative_prompt_input,
    # so both must be lists of length matching #videos in video_dir (we have 1).
    cfg["prompts_input"] = [prompt]
    cfg["negative_prompt_input"] = ["noisy, ugly, nude, watermark"]
    cfg["global_seed"] = int(args.seed)
    # ERP-shaped target canvas: 2:1 aspect (height x width).
    cfg["target_size"] = [int(args.target_h), int(args.target_w)]
    # Reasonable overlap defaults; FYC is robust to these.
    cfg["min_overlap"] = [int(min(96, args.target_h // 4)),
                          int(min(96, args.target_w // 4))]
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False,
                                       default_flow_style=False))


def process_clip(args, row: dict) -> dict:
    clip_id = row["clip_id"]
    out_dir = per_clip_outdir(args.results, METHOD_ID, clip_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_mp4 = out_dir / "video.mp4"
    if args.skip_existing and have_output(out_dir):
        return {"clip_id": clip_id, "status": "skipped"}

    caption, caption_style = load_caption_with_style(
        row.get("caption_path", ""), style="long")
    if not caption:
        return {"clip_id": clip_id, "status": "error", "error": "no caption"}

    gt_video = row.get("video_path", "")
    if not gt_video or not Path(gt_video).is_file():
        return {"clip_id": clip_id, "status": "error",
                "error": f"GT video missing: {gt_video}"}

    work_dir = out_dir / "_work"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    erp0 = read_first_frame_rgb_uint8(gt_video)
    pers_arr = erp_to_perspective_image(
        erp0, fov_x_deg=args.fov_x, yaw_rad=0.0, pitch_rad=0.0,
        out_h=args.pers_h, out_w=args.pers_w,
    )
    video_dir = work_dir / "input_videos"
    video_dir.mkdir()
    pers_mp4 = video_dir / "input.mp4"
    make_static_video_from_image(pers_arr, pers_mp4,
                                 n_frames=args.num_frames_input,
                                 fps=args.input_fps)

    cfg = work_dir / "config.yaml"
    output_dir = work_dir / "output"
    output_dir.mkdir()
    write_per_clip_yaml(args, prompt=caption, video_dir=str(video_dir),
                        output_dir=str(output_dir), out_yaml=cfg)

    cmd_inner = [
        "python", "inference_outpainting-dir-with-prompt.py",
        "--config", str(cfg),
    ]
    cmd = wrap_with_conda(cmd_inner, args.conda_env)

    write_run_json(out_dir, {
        "method_id": METHOD_ID, "clip_id": clip_id, "status": "running",
        "caption": caption, "caption_style": caption_style,
        "cmd": cmd, "gt_video": gt_video,
        "ts_start": int(time.time()),
    })

    if args.dry_run:
        return {"clip_id": clip_id, "status": "dry_run", "cmd": cmd}

    log_dir = Path(args.logs_dir) if args.logs_dir else (
        Path(args.results) / "_logs" / METHOD_ID
    )
    log_path = log_dir / f"{clip_id.replace('::', '__')}.log"
    t0 = time.time()
    rc = run_subprocess(cmd, log_path=log_path, cwd=str(args.repo))
    dt = time.time() - t0
    if rc != 0:
        write_run_json(out_dir, {
            "method_id": METHOD_ID, "clip_id": clip_id, "status": "failed",
            "rc": rc, "elapsed_s": dt, "log": str(log_path), "cmd": cmd,
        })
        return {"clip_id": clip_id, "status": "error",
                "error": f"rc={rc}", "log": str(log_path)}

    # FYC writes its final smoothed result under {output_dir}/<runname>/result/<video_name>
    result_dirs = list(output_dir.rglob("result"))
    src_mp4 = None
    for rd in result_dirs:
        cands = list(rd.glob("*.mp4"))
        if cands:
            src_mp4 = cands[0]
            break
    if src_mp4 is None:
        # fallback: any non-grid mp4 under output_dir
        src_mp4 = find_first_mp4(output_dir,
                                 exclude_substrings=("input", "pers", "grid",
                                                     "overlap", "original",
                                                     "replace", "smooth",
                                                     "samples"))
    if src_mp4 is None:
        write_run_json(out_dir, {
            "method_id": METHOD_ID, "clip_id": clip_id, "status": "failed",
            "elapsed_s": dt, "error": "no mp4", "log": str(log_path),
            "cmd": cmd,
        })
        return {"clip_id": clip_id, "status": "error",
                "error": "no mp4; see " + str(log_path)}

    # FYC outputs at target_size already (target_h x target_w).
    normalize_output_video(src_mp4, final_mp4)

    write_run_json(out_dir, {
        "method_id": METHOD_ID, "clip_id": clip_id, "status": "ok",
        "caption": caption, "caption_style": caption_style,
        "elapsed_s": dt, "video": str(final_mp4),
        "log": str(log_path), "cmd": cmd, "gt_video": gt_video,
    })
    try:
        shutil.rmtree(work_dir)
    except Exception:
        pass
    return {"clip_id": clip_id, "status": "ok", "elapsed_s": dt,
            "video": str(final_mp4)}


def main():
    args = parse_args()
    rows = select_clips(read_master(args.master),
                        splits=args.splits, only_clips=args.only_clips,
                        limit=args.limit)
    print(f"[fyc_runner] {len(rows)} clip(s)"
          f"{' DRY RUN' if args.dry_run else ''}")
    n_ok = n_skip = n_err = 0
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        cid = row["clip_id"]
        try:
            res = process_clip(args, row)
        except Exception as e:
            import traceback; traceback.print_exc()
            res = {"clip_id": cid, "status": "error", "error": str(e)}
        st = res.get("status")
        if st == "ok":
            n_ok += 1; extra = f" ({res.get('elapsed_s', 0):.1f}s)"
        elif st == "skipped":
            n_skip += 1; extra = " skip"
        elif st == "dry_run":
            extra = " dry"
        else:
            n_err += 1; extra = f" FAIL: {res.get('error', '')}"
        print(f"[{i}/{len(rows)}] {cid}{extra}", flush=True)
    print(f"\n=== Done ===  ok={n_ok}  skip={n_skip}  err={n_err}  "
          f"elapsed={time.time()-t0:.1f}s")
    if n_err > 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
