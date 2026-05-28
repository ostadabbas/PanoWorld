#!/usr/bin/env python3
"""Compute every enabled metric for every (clip × method) found under
results_dir, append rows to results_long.csv (the *fat* csv).

Expected layout under `results_dir`:

    results_dir/
      <method_id>/
        <clip_id>/
          video.mp4              ← method's generated panoramic video
          depth/0000.npy ...     ← per-frame depth, optional
          tracks_2d.npy          ← optional
          tracks_3d.npy          ← optional
          visibility.npy         ← optional
          camera_poses.npy       ← optional, (T,4,4) | (T,12) | (T,3)

If a method directory or per-clip directory is missing, that combination is
silently skipped; we never raise on missing methods so partial pipelines
work end-to-end.

Output columns: see README.md → results_long.csv schema.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

# allow `python run_eval.py` from this dir
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from metrics import get as get_metric  # noqa: E402
from runners._common import clip_id_to_dirname  # noqa: E402


def _load_yaml(path: Path) -> dict:
    try:
        import yaml

        return yaml.safe_load(path.read_text())
    except ModuleNotFoundError:
        text = path.read_text()
        out: dict = {}
        # extremely tiny fallback: only handles `key: value` lines, no nested.
        for line in text.splitlines():
            if ":" in line and not line.lstrip().startswith("#"):
                k, _, v = line.partition(":")
                out[k.strip()] = v.strip()
        return out


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


def _load_caption(caption_path: str) -> str:
    if not caption_path or not Path(caption_path).is_file():
        return ""
    try:
        data = json.loads(Path(caption_path).read_text())
    except Exception:
        return ""

    def _walk(node):
        if isinstance(node, str):
            return node
        if isinstance(node, dict):
            for k in ("medium", "long", "short"):
                if k in node:
                    r = _walk(node[k])
                    if r:
                        return r
            for v in node.values():
                r = _walk(v)
                if r:
                    return r
        if isinstance(node, list) and node:
            return _walk(node[0])
        return ""

    return _walk(data) or ""


def _resolve_input_paths(
    method_dir: Path, clip_id: str, master_row: dict
) -> dict[str, str | None]:
    # clip_id in master.csv is "<split>::<clip_basename>", but on disk the
    # runners write to "<split>__<clip_basename>" (filesystems hate "::").
    pred_clip_dir = method_dir / clip_id_to_dirname(clip_id)
    if not pred_clip_dir.is_dir():
        return {"present": False}
    pred_video = pred_clip_dir / "video.mp4"
    out: dict[str, str | None] = {
        "present": pred_video.is_file(),
        "pred_video": str(pred_video) if pred_video.is_file() else None,
        "pred_depth_dir": str(pred_clip_dir / "depth")
        if (pred_clip_dir / "depth").is_dir()
        else None,
        "pred_tracks_dir": str(pred_clip_dir)
        if (pred_clip_dir / "tracks_2d.npy").is_file()
        else None,
        "pred_camera_poses": str(pred_clip_dir / "camera_poses.npy")
        if (pred_clip_dir / "camera_poses.npy").is_file()
        else None,
    }
    out["gt_video"] = master_row.get("video_path")
    out["gt_anno_dir"] = master_row.get("annotation_dir")
    out["gt_depth_dir"] = (
        str(Path(master_row["annotation_dir"]) / "depth")
        if master_row.get("annotation_dir")
        else None
    )
    out["gt_camera_poses"] = (
        str(Path(master_row["annotation_dir"]) / "camera_poses.npy")
        if master_row.get("annotation_dir")
        and (Path(master_row["annotation_dir"]) / "camera_poses.npy").is_file()
        else None
    )
    return out


def _emit(
    writer: csv.writer,
    *,
    clip_id: str,
    split: str,
    method: str,
    input_kind: str,
    metric_name: str,
    value: float,
    strata: dict,
    run_id: str,
):
    if value is None:
        value = float("nan")
    writer.writerow([
        clip_id,
        split,
        method,
        input_kind,
        metric_name,
        f"{value:.6f}" if value == value else "nan",
        json.dumps(strata, sort_keys=True, separators=(",", ":")),
        run_id,
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "eval_config.yaml"))
    ap.add_argument("--master", default=None)
    ap.add_argument("--strata", default=None)
    ap.add_argument("--results", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Only evaluate these method ids (default: all in config)",
    )
    ap.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="for quick dry-runs",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = _load_yaml(cfg_path) if cfg_path.is_file() else {}

    paths = cfg.get("paths", {}) or {}
    master_path = Path(args.master or paths.get("master_csv"))
    strata_path = Path(args.strata or paths.get("strata_csv"))
    results_dir = Path(args.results or paths.get("results_dir"))
    out_path = Path(args.out or paths.get("output_csv"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.is_file()

    print(f"[eval] master  = {master_path}")
    print(f"[eval] strata  = {strata_path}")
    print(f"[eval] results = {results_dir}")
    print(f"[eval] out     = {out_path}")

    master = _load_master(master_path)
    strata_map = _load_strata(strata_path)
    print(f"[eval] {len(master)} clips, {len(strata_map)} strata rows")

    metrics_cfg: dict[str, bool] = cfg.get("metrics", {}) or {}
    methods_cfg: list[dict] = cfg.get("methods", []) or []
    eval_grid_cfg: dict = cfg.get("eval_grid", {}) or {}
    eval_target_fps    = int(eval_grid_cfg.get("target_fps", 16))
    eval_target_secs   = float(eval_grid_cfg.get("target_secs", 5.0))
    eval_target_frames = int(eval_grid_cfg.get(
        "target_frames", round(eval_target_fps * eval_target_secs)
    ))
    print(f"[eval] eval grid: {eval_target_frames}f @ {eval_target_fps}fps "
          f"= {eval_target_secs:.2f}s")
    if args.methods:
        methods_cfg = [m for m in methods_cfg if m.get("id") in set(args.methods)]
    print(f"[eval] methods to scan: {[m.get('id') for m in methods_cfg]}")
    enabled_metrics = [k for k, v in metrics_cfg.items() if v]
    print(f"[eval] metric groups : {enabled_metrics}")

    run_id = f"{int(time.time())}-{os.getpid()}"
    rows_emitted = 0

    f_out = out_path.open("a", newline="")
    w = csv.writer(f_out)
    if write_header:
        w.writerow([
            "clip_id",
            "split",
            "method",
            "input_kind",
            "metric_name",
            "value",
            "strata_json",
            "run_id",
        ])

    metric_callables = {name: get_metric(name) for name in enabled_metrics}

    for m_idx, m in enumerate(methods_cfg):
        method_id = m["id"]
        input_kind = m.get("input_kind", "")
        method_dir = results_dir / method_id
        if not method_dir.is_dir():
            print(f"[eval] [{method_id}] no results dir, skip ({method_dir})")
            continue

        print(f"[eval] === method {method_id} ({m_idx + 1}/{len(methods_cfg)}) ===")
        seen_clips = 0
        for cidx, (clip_id, master_row) in enumerate(master.items()):
            if args.max_clips and seen_clips >= args.max_clips:
                break
            paths_d = _resolve_input_paths(method_dir, clip_id, master_row)
            if not paths_d.get("present"):
                continue

            seen_clips += 1
            split = master_row["split"]
            strata = strata_map.get(clip_id, {})
            caption = _load_caption(master_row.get("caption_path", ""))

            # FPS bookkeeping for eval-grid resampling.
            # - GT fps comes from master.csv (heterogeneous: 25 / 5 / 25 across splits).
            # - Pred fps is the method's native output fps; for any Cosmos-family
            #   runner (PanoWorld / OmniRoam) generate_pano.py writes mp4 @ 16 fps,
            #   so 16 is the right default. Methods that emit at different fps
            #   should set "pred_fps" in their per-clip run.json (future hook).
            try:
                gt_fps = float(master_row.get("fps") or eval_target_fps)
            except (TypeError, ValueError):
                gt_fps = float(eval_target_fps)
            # Native (semantic) fps of each method. For OmniRoam variants the
            # mp4 container hardcodes fps=30 but the underlying Wan-2.1 backbone
            # samples at 16 fps; we pass the semantic value, not the container
            # tag (see metrics/_common.py docstring).
            method_fps_overrides = {
                # ours — Cosmos backbone @ 16 fps
                "panoworld_main":                  16.0,
                "panoworld_main_erp":              16.0,
                # OmniRoam variants — Wan-2.1 sample fps = 16 (container says 30)
                "omniroam_pers":                    16.0,
                "omniroam_erp":                     16.0,
                "omniroam_erp_full_steps":          16.0,
                "omniroam_erp_merged":              16.0,
                # 8 fps generators
                "dvd_360":                           8.0,
                "imagine360":                        8.0,
                "argus":                             8.0,
                "follow_your_canvas":                8.0,
                "follow_your_canvas_full_steps":     8.0,
                "follow_your_canvas_merged":         8.0,
            }
            pred_fps = method_fps_overrides.get(method_id, float(eval_target_fps))

            # build ctx common to all metrics
            common_ctx = {
                "depth_kind": master_row.get("depth_kind", "dap_estimated"),
                "rgb_video_path": paths_d.get("pred_video"),
                "gt_video_path": paths_d.get("gt_video"),
                "caption": caption,
                "windows": cfg.get("trajectory", {}).get("windows"),
                "pred_camera_poses_path": paths_d.get("pred_camera_poses"),
                "gt_camera_poses_path": paths_d.get("gt_camera_poses"),
                "H": int(master_row.get("resolution", "1024x512").split("x")[1]) if "x" in master_row.get("resolution", "") else 512,
                "W": int(master_row.get("resolution", "1024x512").split("x")[0]) if "x" in master_row.get("resolution", "") else 1024,
                # fps + eval-grid plumbing (Scheme A)
                "pred_fps": pred_fps,
                "gt_fps":   gt_fps,
                "eval_target_fps":    eval_target_fps,
                "eval_target_secs":   eval_target_secs,
                "eval_target_frames": eval_target_frames,
            }

            for group_name, fn in metric_callables.items():
                # decide what (pred, gt) to pass
                if group_name == "visual":
                    pred = paths_d.get("pred_video")
                    gt = paths_d.get("gt_video")
                elif group_name == "depth":
                    pred = paths_d.get("pred_depth_dir")
                    gt = paths_d.get("gt_depth_dir")
                elif group_name == "trajectory":
                    pred = paths_d.get("pred_video")
                    gt = paths_d.get("gt_video")
                elif group_name == "tracks":
                    pred = paths_d.get("pred_tracks_dir")
                    gt = paths_d.get("gt_anno_dir")
                elif group_name == "self_consistency":
                    # pred_clip_dir is reused via pred_tracks_dir (same path)
                    pred = paths_d.get("pred_tracks_dir")
                    # GT not needed; pass any non-None to bypass the skip
                    gt = pred
                else:
                    pred = paths_d.get("pred_video")
                    gt = paths_d.get("gt_video")

                # Some methods don't emit depth/tracks/etc. Skip the group
                # silently when either side is missing — emit no rows so the
                # metric just shows up as NaN in aggregation.
                if pred is None or gt is None:
                    continue

                try:
                    metric_dict = fn(pred, gt, **common_ctx)
                except Exception as e:
                    traceback.print_exc()
                    print(f"[eval]   {method_id} :: {clip_id} :: {group_name} FAILED: {e}", flush=True)
                    metric_dict = {}

                for metric_name, value in metric_dict.items():
                    _emit(
                        w,
                        clip_id=clip_id,
                        split=split,
                        method=method_id,
                        input_kind=input_kind,
                        metric_name=metric_name,
                        value=float(value) if value is not None else float("nan"),
                        strata=strata,
                        run_id=run_id,
                    )
                    rows_emitted += 1

            if seen_clips % 25 == 0:
                print(f"[eval]   {method_id}: {seen_clips} clips done", flush=True)
                f_out.flush()
        print(f"[eval] [{method_id}] {seen_clips} clips evaluated")

    f_out.flush()
    f_out.close()
    print(f"[eval] done. {rows_emitted} rows appended to {out_path}")


if __name__ == "__main__":
    sys.exit(main())
