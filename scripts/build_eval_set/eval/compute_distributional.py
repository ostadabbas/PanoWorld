#!/usr/bin/env python3
"""Compute distributional video/image quality metrics (FVD, FAED, FID).

Distributional metrics differ from per-clip metrics in that they need a *set*
of pred videos and a *set* of GT videos: for each (method, bucket) pair we fit
a Gaussian to each set's embeddings and report Fréchet distance between them.

This script runs **after** ``run_eval.py`` finishes the per-clip pass and
**appends** rows to the same ``results_long.csv`` (so ``aggregate.py`` picks
them up alongside per-clip metrics).

Output rows have:
    clip_id    = ""                         (empty → bucket-level row)
    split      = bucket-name (see below)    e.g. "all" | "self_iid" | ...
    method     = method_id                  e.g. "panoworld_pers"
    metric_name= "vq_fvd" | "vq_faed" | "vq_fid"
    value      = float
    strata_json= JSON describing the bucket dimension
                 (e.g. {"bucket":"all"} or {"bucket":"scene","scene_type":"outdoor_with_sky"})
    run_id     = same convention as run_eval.py

Usage:
    python compute_distributional.py \\
        --master  /path/to/master.csv \\
        --strata  /path/to/strata.csv \\
        --results /path/to/eval_results \\
        --out     /path/to/eval_results/results_long.csv \\
        [--methods panoworld_pers omniroam_pers ...] \\
        [--device cuda] [--min-samples 8]

Notes
-----
* GT embeddings are cached under ``<results_dir>/_dist_cache/gt/<model>/<sig>.npy``
* Pred embeddings under          ``<results_dir>/_dist_cache/pred/<method>/<model>/<sig>.npy``
  Cache key is (filename + size + mtime) — re-running after only adding new
  methods reuses every existing GT embedding for free.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from runners._common import clip_id_to_dirname  # noqa: E402

from metrics.distribution import (  # noqa: E402
    compute_bucket_metrics,
    extract_set_embeddings,
)


# ---------- master / strata loaders (mirroring run_eval.py) -------------------
def _load_master(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out[row["clip_id"]] = row
    return out


def _load_strata(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    out: dict[str, dict] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out[row["clip_id"]] = row
    return out


# ---------- method discovery --------------------------------------------------
def _discover_methods(results_dir: Path) -> list[str]:
    if not results_dir.is_dir():
        return []
    excluded = {"_dist_cache", "_embeddings_cache"}
    return sorted(
        d.name for d in results_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".") and d.name not in excluded
    )


# Matches run_eval.py::method_fps_overrides one-to-one. Methods that emit at
# fps != target_fps must be listed here so distributional encoders resample
# correctly. Semantic fps, NOT container fps (see metrics/_common.py docstring).
METHOD_FPS = {
    # ours
    "panoworld_main":                  16.0,
    "panoworld_main_erp":              16.0,
    # OmniRoam (Wan-2.1 backbone) — semantic 16 fps, container says 30
    "omniroam_pers":                    16.0,
    "omniroam_erp":                     16.0,
    "omniroam_erp_full_steps":          16.0,
    "omniroam_erp_merged":              16.0,
    # 8-fps generators
    "dvd_360":                           8.0,
    "imagine360":                        8.0,
    "argus":                             8.0,
    "follow_your_canvas":                8.0,
    "follow_your_canvas_full_steps":     8.0,
    "follow_your_canvas_merged":         8.0,
}


# ---------- bucket build ------------------------------------------------------
def _build_buckets(master: dict[str, dict],
                   strata: dict[str, dict]) -> dict[tuple, dict]:
    """Return dict[bucket_key] -> {"clip_ids": [...], "strata_json": {...}}.

    Buckets:
      ("all",)
      ("split",  <split>)
      ("scene",  <scene_type>)
    """
    buckets: dict[tuple, dict] = {}

    def push(key: tuple, clip_id: str, strata_payload: dict):
        if key not in buckets:
            buckets[key] = {"clip_ids": [], "strata_json": strata_payload}
        buckets[key]["clip_ids"].append(clip_id)

    for clip_id, row in master.items():
        push(("all",), clip_id, {"bucket": "all"})
        split = row.get("split", "")
        if split:
            push(("split", split), clip_id,
                 {"bucket": "split", "split": split})
        scene = (strata.get(clip_id) or {}).get("scene_type", "")
        if scene:
            push(("scene", scene), clip_id,
                 {"bucket": "scene", "scene_type": scene})

    return buckets


def _bucket_split_label(key: tuple) -> str:
    """Mapped to the ``split`` csv column for downstream aggregator joins.

    aggregate.py groups by strata_json fields so any of these labels work;
    we choose human-readable strings.
    """
    if key == ("all",):
        return "all"
    if key[0] == "split":
        return key[1]
    if key[0] == "scene":
        return f"scene::{key[1]}"
    return "/".join(key)


# ---------- main --------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "eval_config.yaml"))
    ap.add_argument("--master", default=None)
    ap.add_argument("--strata", default=None)
    ap.add_argument("--results", default=None)
    ap.add_argument("--out", default=None,
                    help="results_long.csv to append to. Default: from config.")
    ap.add_argument("--methods", nargs="*", default=None,
                    help="Only compute for these method ids. "
                         "Default: every directory under results_dir/.")
    ap.add_argument("--buckets", nargs="*",
                    default=["all", "split", "scene"],
                    help="Which bucket levels to emit. Subset of "
                         "{all, split, scene}.")
    ap.add_argument("--models", nargs="*",
                    default=["r3d18", "swin3d_t", "inception"],
                    help="Backbones to use. Subset of "
                         "{r3d18, swin3d_t, inception}.")
    ap.add_argument("--min-samples", type=int, default=8,
                    help="Min #clips required in a bucket to compute Frechet "
                         "distance (otherwise NaN). Default 8.")
    ap.add_argument("--device", default="cuda",
                    help="Torch device (e.g. 'cuda:3' for sharding across "
                         "multiple methods on a multi-GPU node).")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Batch size for backbone forward passes. Default 8.")
    ap.add_argument("--gt-only", action="store_true",
                    help="Pre-pass: extract & cache GT embeddings only, "
                         "compute no Frechet distances, emit no rows. "
                         "Run this ONCE on a single GPU before launching N "
                         "parallel per-method workers (which share the GT "
                         "cache and would otherwise race on cold cache).")
    args = ap.parse_args()

    # ---- config + paths ----
    cfg = {}
    if Path(args.config).is_file():
        try:
            import yaml
            cfg = yaml.safe_load(Path(args.config).read_text()) or {}
        except Exception:
            cfg = {}
    paths = (cfg.get("paths") or {})
    master_path = Path(args.master  or paths.get("master_csv"))
    strata_path = Path(args.strata  or paths.get("strata_csv"))
    results_dir = Path(args.results or paths.get("results_dir"))
    out_path    = Path(args.out     or paths.get("output_csv"))

    grid = (cfg.get("eval_grid") or {})
    target_fps    = int(grid.get("target_fps", 16))
    target_secs   = float(grid.get("target_secs", 5.0))

    print(f"[dist] master  = {master_path}")
    print(f"[dist] strata  = {strata_path}")
    print(f"[dist] results = {results_dir}")
    print(f"[dist] out     = {out_path}")

    if not results_dir.is_dir():
        print(f"[dist] FATAL: results dir not found: {results_dir}")
        return 1

    master = _load_master(master_path)
    strata = _load_strata(strata_path)
    print(f"[dist] {len(master)} clips, {len(strata)} strata rows")

    # ---- methods ----
    if args.methods:
        methods = list(args.methods)
    else:
        methods = _discover_methods(results_dir)
    print(f"[dist] methods: {methods}")
    if not methods:
        print("[dist] no methods found, nothing to do.")
        return 0

    # ---- bucket layout ----
    buckets_all = _build_buckets(master, strata)
    bucket_filter = set(args.buckets)
    buckets = {k: v for k, v in buckets_all.items() if k[0] in bucket_filter}
    print(f"[dist] {len(buckets)} buckets in scope: "
          f"{[(k, len(v['clip_ids'])) for k, v in buckets.items()]}")

    # ---- cache root ----
    cache_root = results_dir / "_dist_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    # ---- gt-only mode: just warm the GT cache and exit ----
    if args.gt_only:
        gt_paths_all: list[str] = []
        gt_fps_all:   list[float] = []
        for cid, row in master.items():
            gt = row.get("video_path")
            if gt and Path(gt).is_file():
                gt_paths_all.append(gt)
                try:
                    gt_fps_all.append(float(row.get("fps") or target_fps))
                except (TypeError, ValueError):
                    gt_fps_all.append(float(target_fps))
        print(f"[dist][gt-only] warming cache for {len(gt_paths_all)} GT clips "
              f"× {len(args.models)} backbones on {args.device}")
        for mk in args.models:
            extract_set_embeddings(
                gt_paths_all, gt_fps_all,
                model_key=mk,
                cache_root=cache_root, cache_tag="gt",
                device=args.device, name="gt", batch_size=args.batch_size,
            )
        print("[dist][gt-only] done.")
        return 0

    # ---- output writer ----
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.is_file()
    f_out = out_path.open("a", newline="")
    w = csv.writer(f_out)
    if write_header:
        w.writerow(["clip_id", "split", "method", "input_kind",
                    "metric_name", "value", "strata_json", "run_id"])

    run_id = f"{int(time.time())}-{os.getpid()}-dist"
    rows_emitted = 0

    # ---- iterate (method × bucket) ----
    for method_id in methods:
        method_dir = results_dir / method_id
        pred_fps = METHOD_FPS.get(method_id, float(target_fps))
        print(f"[dist] === method {method_id} (pred_fps={pred_fps}) ===")

        for bkey, bdata in buckets.items():
            bucket_label = _bucket_split_label(bkey)
            clip_ids: list[str] = bdata["clip_ids"]

            pred_paths: list[str] = []
            gt_paths:   list[str] = []
            gt_fps_list:list[float] = []
            for cid in clip_ids:
                # master.csv clip_id is "<split>::<basename>", on disk we
                # store under "<split>__<basename>" (filesystems hate "::").
                pred_mp4 = method_dir / clip_id_to_dirname(cid) / "video.mp4"
                if not pred_mp4.is_file():
                    continue
                row = master.get(cid)
                if not row or not row.get("video_path"):
                    continue
                gt_video = row["video_path"]
                if not Path(gt_video).is_file():
                    continue
                pred_paths.append(str(pred_mp4))
                gt_paths.append(gt_video)
                try:
                    gt_fps_list.append(float(row.get("fps") or target_fps))
                except (TypeError, ValueError):
                    gt_fps_list.append(float(target_fps))

            n = len(pred_paths)
            if n < args.min_samples:
                print(f"[dist]   bucket {bucket_label}: {n} clips < "
                      f"min_samples={args.min_samples}, skip", flush=True)
                continue
            print(f"[dist]   bucket {bucket_label}: {n} (pred,gt) pairs",
                  flush=True)

            metrics = compute_bucket_metrics(
                pred_paths, [pred_fps] * n,
                gt_paths,   gt_fps_list,
                cache_root=cache_root,
                pred_cache_tag=f"pred/{method_id}",
                gt_cache_tag="gt",
                device=args.device,
                min_samples=args.min_samples,
                models=tuple(args.models),
                batch_size=args.batch_size,
            )

            strata_json = json.dumps(bdata["strata_json"],
                                     sort_keys=True, separators=(",", ":"))
            for metric_name, value in metrics.items():
                if value != value:
                    val_str = "nan"
                else:
                    val_str = f"{value:.6f}"
                w.writerow([
                    "",                       # clip_id (bucket-level row)
                    bucket_label,             # split
                    method_id,                # method
                    "",                       # input_kind
                    metric_name,
                    val_str,
                    strata_json,
                    run_id,
                ])
                rows_emitted += 1
            f_out.flush()

    f_out.close()
    print(f"[dist] done. {rows_emitted} rows appended to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
