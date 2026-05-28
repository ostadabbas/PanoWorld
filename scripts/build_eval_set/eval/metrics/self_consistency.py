"""Self-consistency / temporal-stability metrics computed on the predicted
video alone (no GT correspondence required).

These metrics are appropriate for free-form generation (Stage 1) where the
hallucinated camera path differs across methods, making any GT-aligned metric
(PSNR / LPIPS / Depth-AbsRel / TAP-AJ) noisy. They probe whether the generated
video is internally geometrically consistent — a hard requirement for any
downstream 3D reconstruction (3DGS) pipeline.

Inputs
------
``pred_path`` (a.k.a. ``pred_clip_dir``) is expected to contain artefacts
already produced by ``runners/run_geometry_anno.py``:

    pred_clip_dir/
        depth/<frame>.npy            per-frame predicted depth (DAP)
        tracks_2d.npy                (N, T, 2) CoTracker3 tracks
        tracks_3d.npy                (N, T, 3) deprojected tracks
        visibility.npy               (N, T)   {0,1} track visibility
        meta.json                    {smoothness_median_m, rigidity_median_m,
                                      reproj_error_median_px,
                                      camera_motion: {inlier_ratio, ...}, ...}

Reported metrics
----------------
geom_depth_temporal_std        lower better, per-pixel std of depth over time
                                (median over pixels, normalised by mean depth).
                                Probes whether the generator produces a
                                temporally stable scene depth.

geom_track_lifetime_frac       higher better, mean fraction of frames a track
                                stays visible. Long-lived tracks ⇔ temporally
                                coherent imagery.

geom_smoothness_median_m       lower better, frame-to-frame 3D-track jitter
                                (cm). Read from meta.json.

geom_rigidity_median_m         lower better, departure-from-rigid-pairwise
                                distance over time. Read from meta.json.

geom_reproj_error_median_px    lower better, median 2-D→3-D→2-D reprojection
                                error of cached tracks. Read from meta.json.

geom_camera_inlier_ratio       higher better, fraction of points consistent
                                with a single rigid camera motion estimate.
                                Read from meta.json.

geom_motion_2d_end_p95_px      higher better (within reason), 95th-percentile
                                of per-track end-to-end 2-D displacement (px).
                                A video extender given a static-repeat input
                                returns ~3 px; a true panoramic generator
                                produces >=10 px. Distinguishes "actually
                                moving" generation from frame interpolation.

geom_motion_3d_end_p95_m       higher better (within reason), same idea on
                                stabilised 3-D tracks (m). Residual scene
                                motion proxy.

geom_motion_aware_lifetime     higher better, ``track_lifetime_frac``
                                multiplied by ``min(motion_2d_end_p95_px / 20,
                                1.0)``. This composite penalises static-video
                                generation: a method that makes long tracks via
                                no motion at all scores low.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def _read_meta(clip_dir: Path) -> dict:
    p = clip_dir / "meta.json"
    if not p.is_file():
        return {}
    try:
        with open(p) as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _depth_temporal_std(depth_dir: Path) -> float:
    """Median-over-pixels of (per-pixel std over time) / mean.
    Returns NaN if fewer than 4 frames or no valid depth.
    """
    if not depth_dir.is_dir():
        return float("nan")
    files = sorted(depth_dir.glob("*.npy"))
    if len(files) < 4:
        return float("nan")
    try:
        d = np.stack([np.load(f).astype(np.float32) for f in files], axis=0)
    except Exception:
        return float("nan")
    if d.ndim != 3:
        return float("nan")
    valid = np.isfinite(d) & (d > 1e-4)
    valid_pix = valid.all(axis=0)  # (H, W)
    if valid_pix.sum() < 64:
        return float("nan")
    mu = d.mean(axis=0)             # (H, W)
    std = d.std(axis=0)             # (H, W)
    norm = std / np.maximum(mu, 1e-3)
    return float(np.median(norm[valid_pix]))


def _track_lifetime(vis_path: Path) -> float:
    """Mean fraction of frames each track is visible."""
    if not vis_path.is_file():
        return float("nan")
    try:
        v = np.load(vis_path).astype(np.float32)
    except Exception:
        return float("nan")
    if v.ndim != 2 or v.size == 0:
        return float("nan")
    per_track = (v > 0.5).mean(axis=1)  # (N,)
    return float(per_track.mean())


def _motion_2d_end_p95_px(tr2d_path: Path, vis_path: Path) -> float:
    """95th-percentile of |xy[T-1] - xy[0]| across tracks (px). More robust to
    float16 quantisation in saved tracks than the per-frame-median variant.
    Captures the magnitude of motion on the most-moving 5 % of points, which
    is what distinguishes a video extender given a static-repeat input
    (returns ~3 px) from a true panoramic video generator (returns >= 10 px).

    Uses end-to-end displacement (not per-frame), since panoramic generators
    typically accumulate motion gradually."""
    if not tr2d_path.is_file():
        return float("nan")
    try:
        xy = np.load(tr2d_path).astype(np.float32)  # (N, T, 2)
    except Exception:
        return float("nan")
    if xy.ndim != 3 or xy.shape[1] < 2:
        return float("nan")
    end_disp = np.linalg.norm(xy[:, -1] - xy[:, 0], axis=-1)  # (N,)
    valid = np.ones_like(end_disp, dtype=bool)
    if vis_path.is_file():
        try:
            v = np.load(vis_path).astype(np.float32)
            if v.shape == xy.shape[:2]:
                # require visibility at both endpoints
                valid = (v[:, 0] > 0.5) & (v[:, -1] > 0.5)
        except Exception:
            pass
    vals = end_disp[valid]
    if vals.size == 0:
        return float("nan")
    return float(np.percentile(vals, 95))


def _motion_3d_end_p95_m(tr3d_path: Path, vis_path: Path) -> float:
    """95th-percentile of end-to-end 3D displacement in metres on the
    *stabilised* (camera-motion-compensated) tracks. Residual scene motion /
    non-rigidity proxy."""
    if not tr3d_path.is_file():
        return float("nan")
    try:
        xyz = np.load(tr3d_path).astype(np.float32)
    except Exception:
        return float("nan")
    if xyz.ndim != 3 or xyz.shape[1] < 2:
        return float("nan")
    end_disp = np.linalg.norm(xyz[:, -1] - xyz[:, 0], axis=-1)
    valid = np.ones_like(end_disp, dtype=bool)
    if vis_path.is_file():
        try:
            v = np.load(vis_path).astype(np.float32)
            if v.shape == xyz.shape[:2]:
                valid = (v[:, 0] > 0.5) & (v[:, -1] > 0.5)
        except Exception:
            pass
    vals = end_disp[valid]
    if vals.size == 0:
        return float("nan")
    return float(np.percentile(vals, 95))


def eval_self_consistency(pred_path: str, gt_path: str | None = None,
                           **ctx) -> dict[str, float]:
    out: dict[str, float] = {
        "geom_depth_temporal_std":   float("nan"),
        "geom_track_lifetime_frac":  float("nan"),
        "geom_smoothness_median_m":  float("nan"),
        "geom_rigidity_median_m":    float("nan"),
        "geom_reproj_error_median_px": float("nan"),
        "geom_camera_inlier_ratio":  float("nan"),
        "geom_motion_2d_end_p95_px": float("nan"),
        "geom_motion_3d_end_p95_m":  float("nan"),
        "geom_motion_aware_lifetime": float("nan"),
    }
    if not pred_path:
        return out
    pred_dir = Path(pred_path)
    if not pred_dir.is_dir():
        return out

    out["geom_depth_temporal_std"] = _depth_temporal_std(pred_dir / "depth")
    out["geom_track_lifetime_frac"] = _track_lifetime(pred_dir / "visibility.npy")
    out["geom_motion_2d_end_p95_px"] = _motion_2d_end_p95_px(
        pred_dir / "tracks_2d.npy", pred_dir / "visibility.npy")
    out["geom_motion_3d_end_p95_m"] = _motion_3d_end_p95_m(
        pred_dir / "tracks_3d.npy", pred_dir / "visibility.npy")
    # composite: long-lived tracks gated by motion magnitude. Threshold of 20 px
    # marks "actually moving" generation; below that we cap the gate so that
    # static-repeat extenders (~3 px end-disp) pay a heavy penalty.
    if (out["geom_track_lifetime_frac"] == out["geom_track_lifetime_frac"]
            and out["geom_motion_2d_end_p95_px"] ==
                out["geom_motion_2d_end_p95_px"]):
        gate = min(1.0, out["geom_motion_2d_end_p95_px"] / 20.0)
        out["geom_motion_aware_lifetime"] = float(
            out["geom_track_lifetime_frac"] * gate)

    meta = _read_meta(pred_dir)
    if meta:
        for src, dst in (("smoothness_median_m",       "geom_smoothness_median_m"),
                         ("rigidity_median_m",         "geom_rigidity_median_m"),
                         ("reproj_error_median_px",    "geom_reproj_error_median_px")):
            v = meta.get(src)
            if v is not None and isinstance(v, (int, float)) and math.isfinite(v):
                out[dst] = float(v)
        cam = meta.get("camera_motion") or {}
        ir = cam.get("inlier_ratio")
        if ir is not None and isinstance(ir, (int, float)) and math.isfinite(ir):
            out["geom_camera_inlier_ratio"] = float(ir)

    for k, v in list(out.items()):
        if v is None or (isinstance(v, float) and (v != v or math.isinf(v))):
            out[k] = float("nan")
    return out
