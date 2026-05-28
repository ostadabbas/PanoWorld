#!/usr/bin/env python3
"""Imagine360 inference runner (perspective-video → panoramic-video).

Imagine360 (Wang et al., 2024) takes a perspective video clip + a caption
and outpaints it into an ERP panoramic video using two diffusion stages
(pers UNet + pano UNet with motion lora). For input parity with PanoWorld,
we feed it our static-repeat policy: replicate the GT first ERP frame's
perspective crop into a static perspective video matching Imagine360's
native length (default 64 frames @ 8 fps = 8 s).

Setup (run once on A100):
    cd $HOME/Le/Imagine360
    bash install.sh                 # creates conda env `imagine360`
    # ckpts already downloaded under _ckpt/imagine360_checkpoints; we must
    # symlink them to the paths the prompt-dual.yaml references:
    mkdir -p ~/.cache/imagine360
    ln -sf $HOME/Le/Imagine360/_ckpt/imagine360_checkpoints \\
           ~/.cache/imagine360/imagine360_checkpoints
    ln -sf $HOME/Le/_baseline_shared/sam/sam_vit_b_01ec64.pth \\
           ~/.cache/imagine360/sam_vit_b_01ec64.pth
    # Plus: hf-download Qwen-VL-Chat (LMM, ~10G):
    huggingface-cli download Qwen/Qwen-VL-Chat \\
        --local-dir ~/.cache/huggingface/hub/models--Qwen--Qwen-VL-Chat/snapshots/HEAD

Per-clip flow:
    GT first ERP frame --(pers crop, fov=90)--> pers image
    pers image --(static-repeat 64f @ 8fps)--> pers_input.mp4
    write per-clip yaml overriding video_path → work_dir w/ pers_input.mp4
    inference_dual_p2e.py --config <yaml>  --> ERP video.mp4

Output:
    results_dir/imagine360/<clip_id_dir>/video.mp4   (64f @ 8fps, 512x1024)

Usage:
    python -m test_set_pkg.eval.runners.infer_imagine360 \\
        --results $PANO_DATA_ROOT/eval_results \\
        --conda_env imagine360 \\
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

IMG360_REPO = Path("$HOME/Le/Imagine360")
BASE_YAML = IMG360_REPO / "configs/prompt-dual.yaml"

METHOD_ID = "imagine360"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=str(MASTER_DEFAULT))
    ap.add_argument("--results", required=True)
    ap.add_argument("--logs_dir", default=None)
    ap.add_argument("--conda_env", default="imagine360")
    ap.add_argument("--repo", default=str(IMG360_REPO))
    ap.add_argument("--base_yaml", default=str(BASE_YAML),
                    help="Imagine360 prompt-dual.yaml; we copy + override per clip")
    ap.add_argument("--num_frames_input", type=int, default=64,
                    help="Imagine360 video_sample_length (8 fps -> 8 s)")
    ap.add_argument("--input_fps", type=int, default=8)
    ap.add_argument("--fov_x", type=float, default=90.0)
    ap.add_argument("--pers_h", type=int, default=480)
    ap.add_argument("--pers_w", type=int, default=640)
    ap.add_argument("--pano_h", type=int, default=512)
    ap.add_argument("--pano_w", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=996995)
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--only_clips", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip_existing", action="store_true", default=True)
    ap.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    ap.add_argument("--dry_run", action="store_true", default=False)
    ap.add_argument("--no_pers_vflip", action="store_true", default=False,
                    help="Disable the FOV-orientation vertical flip on the "
                         "perspective input. Default keeps the flip ON to "
                         "match imagine360's z=up pers→ERP convention. "
                         "Pass this flag only to reproduce the legacy "
                         "(inverted-FOV) outputs.")
    return ap.parse_args()


def write_per_clip_yaml(args, *, prompt: str, video_dir: str, output_dir: str,
                        out_yaml: Path):
    """Load Imagine360's base prompt-dual.yaml and override only the runtime
    fields. Cleanly via PyYAML to preserve nested structures."""
    import yaml
    cfg = yaml.safe_load(Path(args.base_yaml).read_text())
    cfg["video_path"] = video_dir
    cfg["output_dir"] = output_dir
    cfg["prompt"] = prompt
    cfg["global_seed"] = int(args.seed)
    cfg["video_sample_length"] = int(args.num_frames_input)
    cfg["pano_H"] = int(args.pano_h)
    cfg["pano_W"] = int(args.pano_w)
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
    # === FOV-orientation fix (mirrors infer_panoworld_chained.run_round1) ===
    # cosmos's equi2pers uses z=down convention → returns a vertically
    # flipped pers crop relative to a "normal upright photo". Imagine360's
    # internal pers→ERP projection assumes z=up (upright). Without this
    # flip, the resulting ERP video has its FOV center inverted top-to-bottom
    # (sky on the floor, etc.). Verified empirically: a red-top/blue-bottom
    # synthetic pers produces blue-top/red-bottom inside the FOV cone of
    # imagine360's output. So we flip the pers vertically before saving the
    # static-repeat mp4 that imagine360 reads.
    if not getattr(args, "no_pers_vflip", False):
        pers_arr = pers_arr[::-1, :, :].copy()
    video_dir = work_dir / "input_videos"
    video_dir.mkdir()
    pers_mp4 = video_dir / "input.mp4"
    make_static_video_from_image(pers_arr, pers_mp4,
                                 n_frames=args.num_frames_input,
                                 fps=args.input_fps)
    # Imagine360 reads <video>.txt as the prompt iff it exists, skipping
    # its built-in Qwen-VL-Chat caption generation. We feed our own
    # gpt-5-mini caption to bypass the LMM entirely (saves ~17 GB GPU RAM).
    pers_mp4.with_suffix(".txt").write_text(caption.strip())

    cfg = work_dir / "config.yaml"
    output_dir = work_dir / "output"
    output_dir.mkdir()
    write_per_clip_yaml(args, prompt=caption, video_dir=str(video_dir),
                        output_dir=str(output_dir), out_yaml=cfg)

    cmd_inner = [
        "python", "inference_dual_p2e.py",
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

    # Imagine360 dual_p2e writes outputs under multiple subdirs:
    #   <output_dir>/input_vid/<prompt>.mp4     ← input pers
    #   <output_dir>/mask/<prompt>.mp4          ← mask video
    #   <output_dir>/mask/color_<prompt>.mp4    ← naive pers→ERP stitch (NO pano-UNet)
    #   <output_dir>/output_vid/<prompt>.mp4    ← FINAL diffused panoramic video ✅
    # Always prefer output_vid/. find_first_mp4 walks alphabetically, so
    # without an explicit preference it picks mask/color_*.mp4 first which
    # *looks perspective* (FOV cone with black borders elsewhere).
    output_vid_dir = Path(output_dir) / "output_vid"
    src_mp4 = None
    if output_vid_dir.is_dir():
        candidates = [p for p in sorted(output_vid_dir.glob("*.mp4"))
                      if p.stat().st_size > 0]
        if candidates:
            src_mp4 = candidates[0]
    if src_mp4 is None:
        # Fallback: legacy heuristic (older imagine360 forks)
        src_mp4 = find_first_mp4(
            output_dir,
            exclude_substrings=("input", "pers", "grid", "mask", "color_"),
        )
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
    print(f"[imagine360_runner] {len(rows)} clip(s)"
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
