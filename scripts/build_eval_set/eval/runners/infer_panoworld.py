#!/usr/bin/env python3
"""PanoWorld inference runner.

Implements the two-stage inference protocol from eval_config.yaml:

  Stage 1 (panoworld_pers, "Round 1 + Round 2"):
      pers image (90deg yaw=0 crop of GT first ERP frame) + caption
        --[V3 pers2pano: project+CLIP+blended diffusion]-->  ERP video

  Stage 2 (panoworld_erp, "Round 2 only"):
      real ERP first frame (from GT clip) + caption
        --[video2pano with num_input_frames=1]-->  ERP video

Both stages emit:
    results_dir/<method_id>/<clip_id_dir>/video.mp4
    results_dir/<method_id>/<clip_id_dir>/run.json   (provenance)

Each clip is a separate `python generate_pano.py` subprocess. This is
slow (~60-90s model-load + ~30-60s inference per clip) but parallelizes
trivially across GPUs and keeps the runner free of monkey-patching
state.

Frame-rate / length convention (Scheme A):
    PanoWorld was trained on 93 frames @ 16 fps  (state_t = 24, ~5.81s).
    We generate at the model's native length: 93 frames @ 16 fps = 5.81s.
    GT clips live at heterogeneous fps (self_iid 25 fps, argus_ood 5 fps,
    habitat_ood 25 fps, all ~5s). The metrics layer resamples both pred
    and GT to a common eval grid (80 frames @ 16 fps = 5.0s) before
    computing any per-frame metric. See eval_config.yaml -> eval_grid.

Usage:
    cd $REPO_ROOT  # path to your PanoWorld clone
    python -m test_set_pkg.eval.runners.infer_panoworld \
        --stage pers \
        --results $PANO_DATA_ROOT/eval_results \
        --finetune_checkpoint checkpoints/v5_geo_final/model_ema_bf16.pt \
        --num_frames 93 --resolution 512,1024 \
        --splits self_iid argus_ood \
        --limit 1
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

# allow `python -m test_set_pkg.eval.runners.infer_panoworld`
HERE = Path(__file__).resolve().parent           # .../test_set_pkg/eval/runners
REPO = HERE.parent.parent.parent                  # .../cosmos-predict2.5
sys.path.insert(0, str(HERE))

from _common import (  # noqa: E402
    MASTER_DEFAULT,
    erp_to_perspective_image,
    have_output,
    load_caption_with_style,
    make_static_video_from_image,
    per_clip_outdir,
    read_first_frame_rgb_uint8,
    read_master,
    run_subprocess,
    save_image_png,
    select_clips,
    write_run_json,
)

GENERATE_PANO_PY = REPO / "generate_pano.py"


# ─────────────────────────── arg parsing ──────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pers", "erp"], required=True,
                    help="pers = Stage1 (Round1+Round2 via V3 pers2pano); "
                         "erp = Stage2 (Round2 only, video2pano on GT first ERP frame)")
    ap.add_argument("--master", default=str(MASTER_DEFAULT))
    ap.add_argument("--results", required=True,
                    help="results_dir; outputs go to <results>/panoworld_{pers,erp}/<clip_id_dir>/")
    ap.add_argument("--logs_dir", default=None,
                    help="Per-clip log dir. Default: <results>/_logs/panoworld_{pers,erp}")
    ap.add_argument("--finetune_checkpoint",
                    default="checkpoints/v5_geo_final/model_ema_bf16.pt",
                    help="Relative or absolute path to fine-tuned panoramic weights")
    ap.add_argument("--model", default="2B/post-trained")
    ap.add_argument("--num_frames", type=int, default=93,
                    help="Frames to generate. Default 93 = PanoWorld's native "
                         "training length (state_t=24, 16 fps, ~5.81s). The metrics "
                         "layer resamples both pred (16 fps) and GT (heterogeneous "
                         "fps per split) to a common 80f @ 16fps = 5.0s eval grid.")
    ap.add_argument("--gen_fps", type=int, default=16,
                    help="Generated video FPS. generate_pano.py writes mp4 at this "
                         "rate; Cosmos is trained at 16 fps so leave at 16.")
    ap.add_argument("--resolution", default="512,1024", help="ERP resolution H,W")
    ap.add_argument("--guidance", type=int, default=7)
    ap.add_argument("--num_steps", type=int, default=35)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fov_x", type=float, default=90.0,
                    help="Stage 1 only: FOV used both for cropping pers image from "
                         "the GT ERP frame AND for the V3 pers→equi projection.")
    ap.add_argument("--pers_h", type=int, default=480, help="Stage 1: pers crop H")
    ap.add_argument("--pers_w", type=int, default=640, help="Stage 1: pers crop W")
    ap.add_argument("--latent_padding_size", type=int, default=2)
    ap.add_argument("--equirect_rope", action="store_true", default=True)
    ap.add_argument("--use_clip", action="store_true", default=True)
    # selection
    ap.add_argument("--splits", nargs="*", default=None,
                    help="Filter by master.csv split column "
                         "(e.g. self_iid argus_ood habitat_ood)")
    ap.add_argument("--only_clips", nargs="*", default=None,
                    help="Filter by exact clip_id values")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip_existing", action="store_true", default=True)
    ap.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    ap.add_argument("--dry_run", action="store_true", default=False,
                    help="Print commands without executing")
    return ap.parse_args()


# ─────────────────────────── per-stage cmd builders ───────────────────
def build_pers_cmd(args, *, prompt: str, pers_input_mp4: str, output_dir: str) -> list[str]:
    """Stage 1: V3 pers2pano. CLIP + reference-latent + blended diffusion."""
    cmd = [
        "python", str(GENERATE_PANO_PY),
        "--prompt", prompt,
        "--pers_input", pers_input_mp4,
        "--output_dir", output_dir,
        "--resolution", args.resolution,
        "--num_frames", str(args.num_frames),
        "--guidance", str(args.guidance),
        "--num_steps", str(args.num_steps),
        "--seed", str(args.seed),
        "--fov_x", str(args.fov_x),
        "--yaw", "0", "--pitch", "0", "--roll", "0",
        "--model", args.model,
        "--finetune_checkpoint", args.finetune_checkpoint,
        "--latent_padding_size", str(args.latent_padding_size),
        "--v3",
    ]
    if args.equirect_rope:
        cmd.append("--equirect_rope")
    if args.use_clip:
        cmd.append("--use_clip")
    return cmd


def build_erp_cmd(args, *, prompt: str, erp_input_png: str, output_dir: str) -> list[str]:
    """Stage 2: video2pano with num_input_frames=1 on the real ERP first frame."""
    cmd = [
        "python", str(GENERATE_PANO_PY),
        "--prompt", prompt,
        "--input_path", erp_input_png,
        "--output_dir", output_dir,
        "--resolution", args.resolution,
        "--num_frames", str(args.num_frames),
        "--num_input_frames", "1",
        "--guidance", str(args.guidance),
        "--num_steps", str(args.num_steps),
        "--seed", str(args.seed),
        "--model", args.model,
        "--finetune_checkpoint", args.finetune_checkpoint,
        "--latent_padding_size", str(args.latent_padding_size),
    ]
    if args.equirect_rope:
        cmd.append("--equirect_rope")
    if args.use_clip:
        cmd.append("--use_clip")
    return cmd


# ─────────────────────────── output normalization ────────────────────
def find_generated_mp4(work_dir: Path) -> Path | None:
    """generate_pano.py writes pano_v3_<H>x<W>_<ropetag>_s<seed>.mp4 (V3 path)
    or pano_<H>x<W>_<ropetag>_s<seed>.mp4 (Phase-2/3 paths). We pick whichever
    main mp4 exists, ignoring the v3_step1_perspective.mp4 / round{N} extras.
    """
    candidates = []
    for p in sorted(work_dir.glob("*.mp4")):
        name = p.name
        if name.startswith("v3_step") or "_round" in name or "_cube_" in name:
            continue
        if "pers2equi" in name:
            continue
        candidates.append(p)
    return candidates[0] if candidates else None


# ─────────────────────────── per-clip orchestration ──────────────────
def process_clip(args, row: dict, method_id: str) -> dict:
    clip_id = row["clip_id"]
    out_dir = per_clip_outdir(args.results, method_id, clip_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_mp4 = out_dir / "video.mp4"

    if args.skip_existing and have_output(out_dir):
        return {"clip_id": clip_id, "status": "skipped", "video": str(final_mp4)}

    caption, caption_style = load_caption_with_style(
        row.get("caption_path", ""), style="long")
    if not caption:
        return {"clip_id": clip_id, "status": "error",
                "error": "no caption available"}

    gt_video = row.get("video_path", "")
    if not gt_video or not Path(gt_video).is_file():
        return {"clip_id": clip_id, "status": "error",
                "error": f"GT video missing: {gt_video}"}

    # Per-clip working dir for generate_pano.py outputs (we'll move final mp4 out).
    work_dir = out_dir / "_work"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Build inputs
    erp0 = read_first_frame_rgb_uint8(gt_video)  # (H, W, 3) uint8
    if args.stage == "pers":
        # Crop perspective view from the GT ERP first frame.
        pers_arr = erp_to_perspective_image(
            erp0, fov_x_deg=args.fov_x, yaw_rad=0.0, pitch_rad=0.0,
            out_h=args.pers_h, out_w=args.pers_w,
        )
        # generate_pano.py's pers2pano path expects an mp4. Static-replicate at
        # the model's native fps so the conditioner sees fps=16 (training value).
        pers_mp4 = work_dir / "pers_input.mp4"
        # generate_pano.py's read_perspective_video subsamples uniformly to
        # `model_required_frames` frames; replicating to num_frames is plenty.
        make_static_video_from_image(pers_arr, pers_mp4,
                                     n_frames=args.num_frames, fps=args.gen_fps)
        cmd = build_pers_cmd(args, prompt=caption,
                             pers_input_mp4=str(pers_mp4),
                             output_dir=str(work_dir))
    else:  # erp
        erp_png = work_dir / "erp_first_frame.png"
        save_image_png(erp0, erp_png)
        cmd = build_erp_cmd(args, prompt=caption,
                            erp_input_png=str(erp_png),
                            output_dir=str(work_dir))

    # Provenance BEFORE running so even if the run crashes we know what we tried
    write_run_json(out_dir, {
        "method_id": method_id,
        "clip_id": clip_id,
        "stage": args.stage,
        "gt_video": gt_video,
        "caption": caption,
        "caption_style": caption_style,
        "cmd": cmd,
        "status": "running",
        "ts_start": int(time.time()),
    })

    if args.dry_run:
        return {"clip_id": clip_id, "status": "dry_run", "cmd": cmd}

    log_dir = Path(args.logs_dir) if args.logs_dir else (
        Path(args.results) / "_logs" / method_id
    )
    log_path = log_dir / f"{clip_id.replace('::', '__')}.log"

    t0 = time.time()
    rc = run_subprocess(cmd, log_path=log_path, cwd=str(REPO))
    dt = time.time() - t0
    if rc != 0:
        write_run_json(out_dir, {
            "method_id": method_id,
            "clip_id": clip_id,
            "stage": args.stage,
            "gt_video": gt_video,
            "caption": caption,
            "caption_style": caption_style,
            "cmd": cmd,
            "status": "failed",
            "rc": rc,
            "elapsed_s": dt,
            "log": str(log_path),
        })
        return {"clip_id": clip_id, "status": "error",
                "error": f"subprocess rc={rc}; see {log_path}", "log": str(log_path)}

    # Locate produced mp4 and move into place.
    src_mp4 = find_generated_mp4(work_dir)
    if src_mp4 is None or not src_mp4.is_file():
        write_run_json(out_dir, {
            "method_id": method_id, "clip_id": clip_id, "stage": args.stage,
            "status": "failed", "error": "no mp4 produced",
            "elapsed_s": dt, "cmd": cmd, "log": str(log_path),
        })
        return {"clip_id": clip_id, "status": "error",
                "error": "no mp4 produced; see " + str(log_path)}

    shutil.move(str(src_mp4), str(final_mp4))
    write_run_json(out_dir, {
        "method_id": method_id,
        "clip_id": clip_id,
        "stage": args.stage,
        "gt_video": gt_video,
        "caption": caption,
        "caption_style": caption_style,
        "cmd": cmd,
        "status": "ok",
        "elapsed_s": dt,
        "video": str(final_mp4),
        "log": str(log_path),
    })

    # Best-effort cleanup of the work dir, keep nothing heavy.
    try:
        shutil.rmtree(work_dir)
    except Exception:
        pass

    return {"clip_id": clip_id, "status": "ok",
            "video": str(final_mp4), "elapsed_s": dt}


def main():
    args = parse_args()
    method_id = "panoworld_pers" if args.stage == "pers" else "panoworld_erp"

    rows = read_master(args.master)
    rows = select_clips(rows, splits=args.splits, only_clips=args.only_clips,
                        limit=args.limit)
    print(f"[panoworld_runner] stage={args.stage} method={method_id}")
    print(f"[panoworld_runner] results={args.results}")
    print(f"[panoworld_runner] {len(rows)} clip(s) to run")
    if args.dry_run:
        print(f"[panoworld_runner] DRY RUN")

    n_ok, n_skip, n_err = 0, 0, 0
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        cid = row["clip_id"]
        try:
            res = process_clip(args, row, method_id=method_id)
        except Exception as e:
            import traceback; traceback.print_exc()
            res = {"clip_id": cid, "status": "error", "error": str(e)}
        st = res.get("status")
        if st == "ok":
            n_ok += 1
            extra = f" ({res.get('elapsed_s', 0):.1f}s)"
        elif st == "skipped":
            n_skip += 1; extra = " (skipped)"
        elif st == "dry_run":
            extra = " (dry)"
        else:
            n_err += 1
            extra = f" FAIL: {res.get('error', 'unknown')}"
        elapsed = time.time() - t0
        rate = i / max(elapsed, 1e-3)
        eta = (len(rows) - i) / max(rate, 1e-3)
        print(f"[{i}/{len(rows)}] {cid}{extra} | {rate:.2f} clip/s | ETA {eta/60:.1f} min",
              flush=True)

    print(f"\n=== Done ===  ok={n_ok}  skip={n_skip}  err={n_err}  "
          f"elapsed={time.time()-t0:.1f}s")
    if n_err > 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
