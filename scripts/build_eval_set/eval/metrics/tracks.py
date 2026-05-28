"""Track metrics: TAP-Vid AJ / OA / δ_avg + 3D-track consistency.

For each generated video we run CoTracker3 (the same tracker we use for
ground-truth annotations) and compare against the GT tracks_2d / tracks_3d /
visibility tensors stored in master.csv's annotation_dir.

Skeleton: real CoTracker3 wrapper is in ../runners/cotracker_eval.py and is
deliberately not invoked here to keep this module CPU-only / fast. When
`pred_tracks_dir` is provided in ctx we compare those directly; otherwise we
return NaN for tracker metrics.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


THRESHOLDS_PX = (1, 2, 4, 8, 16)


def _aj_oa_delta(
    pred_xy: np.ndarray,
    pred_vis: np.ndarray,
    gt_xy: np.ndarray,
    gt_vis: np.ndarray,
    img_size: tuple[int, int] = (512, 1024),
) -> dict[str, float]:
    """Standard TAP-Vid metrics.

    pred_xy / gt_xy : (N, T, 2)  -- pixel coords
    pred_vis / gt_vis: (N, T)    -- {0,1}
    """
    out = {
        "tap_aj": float("nan"),
        "tap_oa": float("nan"),
        "tap_delta_avg": float("nan"),
    }
    if pred_xy.shape != gt_xy.shape or pred_xy.size == 0:
        return out

    H, W = img_size
    diag = math.hypot(H, W)
    scale = 256.0 / diag  # tap-vid normalisation

    err = np.linalg.norm(pred_xy - gt_xy, axis=-1) * scale  # (N, T)
    valid = (gt_vis > 0.5)

    deltas = []
    correct_per_thresh = []
    for thr in THRESHOLDS_PX:
        ok = (err < thr) & valid
        if valid.sum() < 1:
            deltas.append(float("nan"))
            correct_per_thresh.append(np.zeros_like(err, dtype=bool))
            continue
        deltas.append(float(ok.sum()) / float(valid.sum()))
        correct_per_thresh.append(ok)

    out["tap_delta_avg"] = float(np.nanmean(deltas))

    pred_v = (pred_vis > 0.5)
    out["tap_oa"] = float(((pred_v == valid).sum()) / max(valid.size, 1))

    aj_vals = []
    for ok in correct_per_thresh:
        tp = (ok & valid).sum()
        fp = (pred_v & ~valid).sum()
        fn = (~pred_v & valid).sum() + (valid & ~ok).sum()
        denom = tp + fp + fn
        aj_vals.append(float(tp) / float(denom) if denom > 0 else float("nan"))
    out["tap_aj"] = float(np.nanmean(aj_vals))
    return out


def _track3d_consistency(
    pred_xyz: np.ndarray, gt_xyz: np.ndarray, vis: np.ndarray
) -> float:
    if pred_xyz.shape != gt_xyz.shape:
        return float("nan")
    valid = vis > 0.5
    if valid.sum() < 16:
        return float("nan")
    diff = pred_xyz - gt_xyz  # (N, T, 3)
    err = np.linalg.norm(diff, axis=-1)  # (N, T)
    return float(err[valid].mean())


def eval_tracks(pred_path: str, gt_path: str, **ctx) -> dict[str, float]:
    """
    pred_path : optional dir / .npy of pre-computed tracks for the prediction.
                If absent, we return NaN for everything (CoTracker3 will be run
                in a separate batch step, not from inside this function).
    gt_path   : annotation_dir of the GT clip (contains tracks_2d.npy, etc.)
    ctx       : H, W of the images
    """
    out = {
        "tap_aj": float("nan"),
        "tap_oa": float("nan"),
        "tap_delta_avg": float("nan"),
        "track3d_mean_err_m": float("nan"),
    }
    if not pred_path:
        return out
    pred_dir = Path(pred_path)
    gt_dir = Path(gt_path) if gt_path else None
    if not gt_dir or not gt_dir.is_dir():
        return out

    gt_xy = gt_dir / "tracks_2d.npy"
    gt_xyz = gt_dir / "tracks_3d.npy"
    gt_vis = gt_dir / "visibility.npy"
    if not gt_xy.is_file() or not gt_vis.is_file():
        return out

    try:
        gt_xy_arr = np.load(gt_xy).astype(np.float32)
        gt_vis_arr = np.load(gt_vis).astype(np.float32)
    except Exception:
        return out

    pred_xy_path = pred_dir / "tracks_2d.npy"
    pred_vis_path = pred_dir / "visibility.npy"
    if not pred_xy_path.is_file() or not pred_vis_path.is_file():
        return out

    try:
        pred_xy_arr = np.load(pred_xy_path).astype(np.float32)
        pred_vis_arr = np.load(pred_vis_path).astype(np.float32)
    except Exception:
        return out

    H = int(ctx.get("H", 512))
    W = int(ctx.get("W", 1024))
    out.update(_aj_oa_delta(pred_xy_arr, pred_vis_arr, gt_xy_arr, gt_vis_arr, (H, W)))

    pred_xyz_path = pred_dir / "tracks_3d.npy"
    if pred_xyz_path.is_file() and gt_xyz.is_file():
        try:
            pred_xyz_arr = np.load(pred_xyz_path).astype(np.float32)
            gt_xyz_arr = np.load(gt_xyz).astype(np.float32)
            out["track3d_mean_err_m"] = _track3d_consistency(pred_xyz_arr, gt_xyz_arr, gt_vis_arr)
        except Exception:
            pass
    return out
