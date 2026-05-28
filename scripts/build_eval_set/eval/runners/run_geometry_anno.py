#!/usr/bin/env python3
"""Run DAP depth + CoTracker3 tracks on every method's predicted video.

Mirrors the same pipeline used to build GT annotations
(``generate_annotations.py``) so that the depth-aligned and tracks-aligned
metrics in ``metrics/depth.py`` / ``metrics/tracks.py`` can be computed for
arbitrary methods, not only for ours.

Layout expected:

    <results_dir>/<method>/<clip>/video.mp4

After this script runs, each clip dir gains:

    <results_dir>/<method>/<clip>/
        depth/0000.npy ... 0019.npy   # (H, W) float16 depth maps
        tracks_2d.npy                  # (N, T, 2) float16
        tracks_3d.npy                  # (N, T, 3) float16
        visibility.npy                 # (N, T) float16
        meta.json
        [camera_poses.npy]             # for non-static clips

Then ``run_eval.py`` will pick these up automatically via
``_resolve_input_paths`` and compute Depth-AbsRel / δ<1.25 / TAP-AJ / OA / δ_avg.

Usage:
    python runners/run_geometry_anno.py \\
        --results $HOME/Le/eval_results_combined \\
        --methods panoworld_main argus imagine360 follow_your_canvas_merged \\
        --gpu 0 --resume

Each clip takes ~10–20s on H100 (DAP forward + CoTracker3 forward + 3D lift).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Resolve project structure
HERE = Path(__file__).resolve().parent
EVAL_ROOT = HERE.parent
PROJ_ROOT = EVAL_ROOT.parent.parent          # repo root (e.g. /path/to/PanoWorld)
DAP_ROOT = PROJ_ROOT / "DAP"

# Reuse the existing GT generator helpers verbatim
sys.path.insert(0, str(PROJ_ROOT))
sys.path.insert(0, str(DAP_ROOT))
sys.path.insert(0, str(DAP_ROOT / "test"))


def _load_models(gpu: str):
    """Load DAP + CoTracker3 once. Mirrors generate_annotations.py."""
    from generate_annotations import load_dap_model, load_cotracker_model
    dap_config = str(DAP_ROOT / "config" / "infer.yaml")
    dap_model, dap_device = load_dap_model(dap_config, gpu=gpu)
    ct_model = load_cotracker_model(device=dap_device)
    return dap_model, dap_device, ct_model


def _process_one(video_path: Path, out_dir: Path, dap_model, dap_device,
                 ct_model, grid_size: int, max_frames: int, max_depth: float):
    from generate_annotations import process_single_video
    return process_single_video(
        str(video_path), str(out_dir),
        dap_model, dap_device, ct_model,
        grid_size=grid_size, max_frames=max_frames,
        max_depth=max_depth,
    )


def _list_clips(method_dir: Path) -> list[Path]:
    """Return all clip directories under a method that contain video.mp4."""
    if not method_dir.is_dir():
        return []
    out = []
    for c in sorted(method_dir.iterdir()):
        if not c.is_dir():
            continue
        if (c / "video.mp4").is_file():
            out.append(c)
    return out


def _is_done(clip_dir: Path) -> bool:
    """Resume check: a clip is done if meta.json AND tracks_2d.npy exist."""
    return (clip_dir / "meta.json").is_file() and (clip_dir / "tracks_2d.npy").is_file()


def _process_method(method_dir: Path, models, args) -> dict:
    dap_model, dap_device, ct_model = models
    clips = _list_clips(method_dir)
    if args.shard_id is not None and args.total_shards is not None and args.total_shards > 1:
        clips = [c for i, c in enumerate(clips) if i % args.total_shards == args.shard_id]

    stats = {"processed": 0, "skipped": 0, "failed": 0, "static": 0, "dynamic": 0}
    failed = []
    t0 = time.time()

    for idx, clip in enumerate(clips):
        if args.resume and _is_done(clip):
            stats["skipped"] += 1
            continue
        try:
            video_path = clip / "video.mp4"
            real = video_path.resolve() if not video_path.is_file() else video_path
            if not real.is_file():
                stats["failed"] += 1
                failed.append(clip.name)
                continue
            meta = _process_one(
                real, clip, dap_model, dap_device, ct_model,
                grid_size=args.grid_size, max_frames=args.max_frames,
                max_depth=args.max_depth,
            )
            if meta is None:
                stats["failed"] += 1
                failed.append(clip.name)
                continue
            stats["processed"] += 1
            if meta.get("camera_motion", {}).get("is_static"):
                stats["static"] += 1
            else:
                stats["dynamic"] += 1
            if stats["processed"] % 10 == 0:
                rate = stats["processed"] / max(1.0, time.time() - t0)
                eta_s = (len(clips) - idx - 1) / rate if rate > 0 else 0
                print(f"  [{method_dir.name}] {stats['processed']}/{len(clips)}  "
                      f"{rate:.2f} clip/s  ETA {eta_s/60:.1f} min",
                      flush=True)
        except Exception as e:
            print(f"  [{method_dir.name}] FAIL {clip.name}: {e}", flush=True)
            stats["failed"] += 1
            failed.append(clip.name)

    elapsed = time.time() - t0
    print(f"  [{method_dir.name}] done. processed={stats['processed']}  "
          f"skipped={stats['skipped']}  failed={stats['failed']}  "
          f"static={stats['static']}  dynamic={stats['dynamic']}  "
          f"elapsed={elapsed/60:.1f}min", flush=True)
    return {**stats, "elapsed_s": elapsed, "failed_clips": failed}


def main():
    _default_results = os.environ.get(
        "PANO_EVAL_RESULTS_DIR",
        os.path.join(os.path.expanduser("~"), "eval_results"),
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=str,
                    default=_default_results,
                    help="Root containing <method>/<clip>/video.mp4 layout")
    ap.add_argument("--methods", nargs="+", required=True,
                    help="Method dir names to process")
    ap.add_argument("--gpu", type=str, default="0")
    ap.add_argument("--max_frames", type=int, default=20,
                    help="Frames sampled per clip (matches GT annotation = 20)")
    ap.add_argument("--grid_size", type=int, default=30,
                    help="CoTracker grid (30 → 900 tracks, matches GT)")
    ap.add_argument("--max_depth", type=float, default=100.0)
    ap.add_argument("--resume", action="store_true",
                    help="Skip clips already containing meta.json + tracks_2d.npy")
    ap.add_argument("--total_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    args = ap.parse_args()

    results_root = Path(args.results)
    if not results_root.is_dir():
        print(f"FATAL: results root not found: {results_root}", flush=True)
        return 1

    method_dirs = []
    for m in args.methods:
        d = results_root / m
        # follow symlinks
        if d.is_symlink():
            d = d.resolve()
        if not d.is_dir():
            print(f"WARN: method dir not found: {results_root / m}", flush=True)
            continue
        method_dirs.append(d)

    if not method_dirs:
        print("FATAL: no valid method dirs", flush=True)
        return 1

    n_total = sum(len(_list_clips(m)) for m in method_dirs)
    print(f"[geom] results root  = {results_root}")
    print(f"[geom] method dirs   = {[str(d) for d in method_dirs]}")
    print(f"[geom] total clips   = {n_total}")
    print(f"[geom] grid_size={args.grid_size}  max_frames={args.max_frames}",
          flush=True)

    print("[geom] loading DAP + CoTracker3 ...", flush=True)
    models = _load_models(args.gpu)
    print("[geom] models loaded", flush=True)

    summary: dict = {}
    for m in method_dirs:
        summary[m.name] = _process_method(m, models, args)

    # Persist a top-level summary next to results
    out_summary = results_root / "geometry_anno_summary.json"
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[geom] summary -> {out_summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
