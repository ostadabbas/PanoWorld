#!/usr/bin/env python3
"""Argus inference runner (perspective-video → panoramic-video).

Argus (Bansal et al., 2024) takes a perspective video clip and emits an
ERP panoramic video using a custom Stable Video Diffusion pipeline with
MASt3R-derived camera params. For input parity with PanoWorld, we use
the static-repeat policy on the GT first ERP frame's perspective crop:

    GT first ERP frame --(pers crop)--> single pers image
    pers image --(static-repeat 25f @ 8fps)--> pers_input.mp4

Setup (run once on A100):
    cd $HOME/Le/argus-code
    bash install.sh                 # creates conda env `argus`
    # ckpts already at checkpoints/pretrained-weights/unet (UNet only).
    # Argus also auto-downloads the MASt3R weights from HF on first run
    # (naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric, ~5G).
    # Argus also requires SVD base from HF:
    #   stabilityai/stable-video-diffusion-img2vid (auto-downloads).

Per-clip flow:
    inference.py --val_base_folder <work_dir/clips>
                 --unet_path checkpoints/pretrained-weights
                 --num_frames 25 --width 1024 --height 512 ...

Argus emits multiple files under val_save_folder; we pick the panoramic
mp4 (filename mirrors the input filename).

Output:
    results_dir/argus/<clip_id_dir>/video.mp4

Usage:
    python -m test_set_pkg.eval.runners.infer_argus \\
        --results $PANO_DATA_ROOT/eval_results \\
        --conda_env argus \\
        --splits self_iid argus_ood habitat_ood --limit 1
"""
from __future__ import annotations

import argparse
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

ARGUS_REPO = Path("$HOME/Le/argus-code")
METHOD_ID = "argus"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=str(MASTER_DEFAULT))
    ap.add_argument("--results", required=True)
    ap.add_argument("--logs_dir", default=None)
    ap.add_argument("--conda_env", default="argus")
    ap.add_argument("--repo", default=str(ARGUS_REPO))
    ap.add_argument("--unet_path", default="checkpoints/pretrained-weights",
                    help="Relative to --repo")
    ap.add_argument("--svd_base",
                    default="stabilityai/stable-video-diffusion-img2vid",
                    help="HF repo id for SVD; Argus passes this to "
                         "--pretrained_model_name_or_path")
    ap.add_argument("--num_frames_input", type=int, default=25,
                    help="Argus's default sampler length")
    ap.add_argument("--input_fps", type=int, default=8)
    ap.add_argument("--width", type=int, default=1024,
                    help="Output ERP width (Argus also uses this for input pers W)")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--fov_x", type=float, default=90.0)
    ap.add_argument("--pers_h", type=int, default=480)
    ap.add_argument("--pers_w", type=int, default=640)
    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--guidance_scale", type=float, default=1.0)
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--only_clips", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip_existing", action="store_true", default=True)
    ap.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    ap.add_argument("--dry_run", action="store_true", default=False)
    return ap.parse_args()


def process_clip(args, row: dict) -> dict:
    clip_id = row["clip_id"]
    out_dir = per_clip_outdir(args.results, METHOD_ID, clip_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_mp4 = out_dir / "video.mp4"
    if args.skip_existing and have_output(out_dir):
        return {"clip_id": clip_id, "status": "skipped"}

    gt_video = row.get("video_path", "")
    if not gt_video or not Path(gt_video).is_file():
        return {"clip_id": clip_id, "status": "error",
                "error": f"GT video missing: {gt_video}"}

    work_dir = out_dir / "_work"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    # Build a single-clip "val_base_folder": Argus expects a directory of
    # input videos. We give it just our static-repeat pers mp4.
    val_base = work_dir / "clips"
    val_base.mkdir()
    erp0 = read_first_frame_rgb_uint8(gt_video)
    pers_arr = erp_to_perspective_image(
        erp0, fov_x_deg=args.fov_x, yaw_rad=0.0, pitch_rad=0.0,
        out_h=args.pers_h, out_w=args.pers_w,
    )
    pers_mp4 = val_base / "input.mp4"
    make_static_video_from_image(pers_arr, pers_mp4,
                                 n_frames=args.num_frames_input,
                                 fps=args.input_fps)

    val_save = work_dir / "out"
    val_save.mkdir()
    cmd_inner = [
        "python", "inference.py",
        "--val_base_folder", str(val_base),
        "--val_save_folder", str(val_save),
        "--unet_path", args.unet_path,
        "--pretrained_model_name_or_path", args.svd_base,
        "--num_frames", str(args.num_frames_input),
        "--width", str(args.width),
        "--height", str(args.height),
        "--num_inference_steps", str(args.num_inference_steps),
        "--guidance_scale", str(args.guidance_scale),
        "--fixed_fov", str(args.fov_x),
        "--full_sampling",
    ]
    cmd = wrap_with_conda(cmd_inner, args.conda_env)

    write_run_json(out_dir, {
        "method_id": METHOD_ID, "clip_id": clip_id, "status": "running",
        "cmd": cmd, "gt_video": gt_video, "ts_start": int(time.time()),
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

    src_mp4 = find_first_mp4(val_save,
                             exclude_substrings=("input", "pers", "calib"))
    if src_mp4 is None:
        write_run_json(out_dir, {
            "method_id": METHOD_ID, "clip_id": clip_id, "status": "failed",
            "elapsed_s": dt, "error": "no mp4", "log": str(log_path),
            "cmd": cmd,
        })
        return {"clip_id": clip_id, "status": "error",
                "error": "no mp4; see " + str(log_path)}

    normalize_output_video(src_mp4, final_mp4)

    write_run_json(out_dir, {
        "method_id": METHOD_ID, "clip_id": clip_id, "status": "ok",
        "elapsed_s": dt, "video": str(final_mp4), "log": str(log_path),
        "gt_video": gt_video, "cmd": cmd,
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
    print(f"[argus_runner] {len(rows)} clip(s)"
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
