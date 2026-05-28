#!/usr/bin/env python3
"""Pivot results_long.csv → publication-ready tables.

Outputs a directory of CSVs, one per "view":

    paper_tables/
      stage1_overall.csv         all methods × all metrics, averaged over all 150 clips
      stage1_by_split.csv        × {self_iid, argus_ood, habitat_ood}
      stage1_by_scene.csv        × {textured_indoor, outdoor_with_sky, mixed_low_texture}
      stage1_by_motion.csv       × {static, walk, rotate, walk_rotate}
      stage1_by_split_x_scene.csv  cross
      stage2_overall.csv         only Stage-2 methods
      stage2_by_scene.csv

The intent is to **always** produce all of these, then read them and pick the
view that most flatters our method when writing the paper. Each table also
records the `n_clips` count so reviewers can see the slice size.

Aggregation rule: `mean` for "higher-is-better" metrics (PSNR, SSIM, δ<1.25,
CLIP-T, AJ, OA, δ_avg) and `mean` for "lower-is-better" too — direction is
recorded in METRIC_DIRECTION below for downstream rendering only. NaNs are
dropped per cell.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

METRIC_DIRECTION = {
    # higher-is-better
    "vq_psnr": "↑",
    "vq_ssim": "↑",
    "vq_clip_t": "↑",
    "tap_aj": "↑",
    "tap_oa": "↑",
    "tap_delta_avg": "↑",
    "depth_delta1_25/raw": "↑",
    "depth_delta1_25/sky_masked": "↑",
    "depth_delta1_25/texture_weighted": "↑",
    # lower-is-better
    "vq_lpips": "↓",
    "vq_fvd":   "↓",
    "vq_faed":  "↓",
    "vq_fid":   "↓",
    "depth_abs_rel/raw": "↓",
    "depth_abs_rel/sky_masked": "↓",
    "depth_abs_rel/texture_weighted": "↓",
    "track3d_mean_err_m": "↓",
    "traj_ate": "↓",
    "traj_apd": "↓",
}
# traj_psnr@x-y is treated as ↑ at render time; computed from prefix.

# Canonical paper-table sets: prefer the *_merged entries for FYC and OmniRoam,
# which take per-clip the higher-quality `_full_steps` if available and fall
# back to the fast preview otherwise. The standalone entries are kept in the
# fat csv for the "denoising steps vs. quality" supplementary sweep.
STAGE1 = {
    # ours
    "panoworld_main",
    # baselines
    "dvd_360",
    "imagine360",
    "argus",
    "follow_your_canvas_merged",
    "omniroam_pers",
}
STAGE2 = {
    "panoworld_main_erp",
    "omniroam_erp_merged",
}


def _read(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for row in csv.DictReader(f):
            try:
                row["value"] = float(row["value"])
            except Exception:
                row["value"] = float("nan")
            try:
                row["strata"] = json.loads(row.get("strata_json") or "{}")
            except Exception:
                row["strata"] = {}
            out.append(row)
    return out


def _agg(rows: list[dict], group_keys: list[str]) -> dict:
    """Group rows by (method, metric_name, *group_keys) → mean(value)."""
    bucket: dict[tuple, list[float]] = defaultdict(list)
    counts: dict[tuple, set] = defaultdict(set)
    for r in rows:
        v = r["value"]
        if v != v:  # NaN
            continue
        key = (r["method"], r["metric_name"]) + tuple(
            r["strata"].get(k, "") if k in r["strata"] else r.get(k, "") for k in group_keys
        )
        bucket[key].append(v)
        counts[key].add(r["clip_id"])
    out: dict[tuple, dict] = {}
    for k, vs in bucket.items():
        out[k] = {"mean": mean(vs), "n_clips": len(counts[k])}
    return out


def _to_wide_csv(agg: dict, group_keys: list[str], out_path: Path) -> None:
    methods = sorted({k[0] for k in agg.keys()})
    metrics_ = sorted({k[1] for k in agg.keys()})
    groups = sorted({k[2:] for k in agg.keys()})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        header = ["method", "metric"] + group_keys + ["mean", "n_clips"]
        w.writerow(header)
        for m in methods:
            for me in metrics_:
                for g in groups:
                    if (m, me) + g not in agg:
                        continue
                    row = agg[(m, me) + g]
                    w.writerow([m, me] + list(g) + [f"{row['mean']:.4f}", row["n_clips"]])
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = _read(Path(args.inp))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[agg] {len(rows)} rows loaded from {args.inp}")

    s1 = [r for r in rows if r["method"] in STAGE1]
    s2 = [r for r in rows if r["method"] in STAGE2]
    print(f"[agg] stage1 rows: {len(s1)}  stage2 rows: {len(s2)}")

    if s1:
        _to_wide_csv(_agg(s1, []), [], out_dir / "stage1_overall.csv")
        _to_wide_csv(_agg(s1, ["split"]), ["split"], out_dir / "stage1_by_split.csv")
        _to_wide_csv(_agg(s1, ["scene_type"]), ["scene_type"], out_dir / "stage1_by_scene.csv")
        _to_wide_csv(_agg(s1, ["motion_kind"]), ["motion_kind"], out_dir / "stage1_by_motion.csv")
        _to_wide_csv(
            _agg(s1, ["split", "scene_type"]),
            ["split", "scene_type"],
            out_dir / "stage1_by_split_x_scene.csv",
        )
    if s2:
        _to_wide_csv(_agg(s2, []), [], out_dir / "stage2_overall.csv")
        _to_wide_csv(_agg(s2, ["scene_type"]), ["scene_type"], out_dir / "stage2_by_scene.csv")
        _to_wide_csv(_agg(s2, ["split"]), ["split"], out_dir / "stage2_by_split.csv")

    print(f"[agg] done. tables in {out_dir}")


if __name__ == "__main__":
    main()
