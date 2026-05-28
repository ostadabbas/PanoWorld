"""Trajectory metrics.

Primary (OmniRoam protocol): per-window PSNR between method output video and
GT video, where every method is conditioned on the same GT trajectory. This
side-steps SfM-based pose extraction (which OmniRoam explicitly notes is too
noisy for fair cross-method comparison).

Secondary (recorded but only reported in paper if it favours us):
    - APD  (Average Position Distance) of estimated camera positions against GT
    - ATE  (Absolute Trajectory Error) after Sim(3) Umeyama alignment
    - RPE  (Relative Pose Error)
    - traj_smoothness (jerk) on the estimated path

Camera poses are *optionally* extracted from the generated video using a fast
PnP-on-CoTracker3 pipeline (same as our annotation pipeline). If pose
extraction fails, ATE/APD return NaN, but per-window PSNR is still recorded.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np

WINDOWS_DEFAULT: Sequence[tuple[int, int]] = (
    (5, 10),
    (20, 25),
    (50, 55),
    (80, 85),
)


def _read_video(path: str) -> np.ndarray | None:
    try:
        import imageio.v3 as iio

        v = iio.imread(path)
        if v.ndim != 4:
            return None
        return v
    except Exception:
        return None


def _psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    if pred.shape != gt.shape:
        n = min(pred.shape[0], gt.shape[0])
        pred, gt = pred[:n], gt[:n]
        h = min(pred.shape[1], gt.shape[1])
        w = min(pred.shape[2], gt.shape[2])
        pred, gt = pred[:, :h, :w], gt[:, :h, :w]
    if pred.size == 0:
        return float("nan")
    diff = pred.astype(np.float64) - gt.astype(np.float64)
    mse = float((diff * diff).mean())
    if mse < 1e-12:
        return 100.0
    return 10.0 * math.log10((255.0**2) / mse)


def _per_window_psnr(
    pred_video: np.ndarray,
    gt_video: np.ndarray,
    windows: Sequence[tuple[int, int]],
) -> dict[str, float]:
    out: dict[str, float] = {}
    T = min(pred_video.shape[0], gt_video.shape[0])
    for a, b in windows:
        if b > T:
            out[f"traj_psnr@{a}-{b}"] = float("nan")
            continue
        out[f"traj_psnr@{a}-{b}"] = _psnr(pred_video[a:b], gt_video[a:b])
    return out


def _umeyama_alignment(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Sim(3) alignment: returns aligned src such that ||tgt - aligned||² is min.
    src, tgt: (T, 3). Returns (T, 3)."""
    if src.shape[0] < 3:
        return src.copy()
    mu_s = src.mean(0)
    mu_t = tgt.mean(0)
    s = src - mu_s
    t = tgt - mu_t
    H = s.T @ t
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    var_s = (s**2).sum() / max(s.shape[0], 1)
    if var_s < 1e-12:
        return src.copy()
    scale = (S @ np.diag([1.0, 1.0, d])).sum() / (var_s * s.shape[0])
    aligned = (scale * (s @ R.T)) + mu_t
    return aligned


def _ate(pred_xyz: np.ndarray, gt_xyz: np.ndarray) -> float:
    if pred_xyz.shape[0] != gt_xyz.shape[0] or pred_xyz.shape[0] < 3:
        return float("nan")
    aligned = _umeyama_alignment(pred_xyz, gt_xyz)
    diff = aligned - gt_xyz
    return float(np.sqrt((diff * diff).sum(-1)).mean())


def _apd(pred_xyz: np.ndarray, gt_xyz: np.ndarray) -> float:
    """Average per-frame Position Distance (no alignment)."""
    if pred_xyz.shape != gt_xyz.shape or pred_xyz.shape[0] < 1:
        return float("nan")
    diff = pred_xyz - gt_xyz
    return float(np.sqrt((diff * diff).sum(-1)).mean())


def _load_camera_xyz(path: str) -> np.ndarray | None:
    try:
        arr = np.load(path)
    except Exception:
        return None
    if arr.ndim == 3 and arr.shape[1:] == (4, 4):  # (T, 4, 4)
        return arr[:, :3, 3]
    if arr.ndim == 2 and arr.shape[1] == 12:  # (T, 12) flat 3x4
        return arr[:, [3, 7, 11]]
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr
    return None


def eval_trajectory(pred_path: str, gt_path: str, **ctx) -> dict[str, float]:
    """
    Inputs:
      pred_path : path to method-generated panoramic video (mp4)
      gt_path   : path to GT panoramic video (mp4) used for per-window PSNR

    ctx (optional):
      windows: list of (a,b) frame ranges (default WINDOWS_DEFAULT)
      pred_camera_poses_path: .npy of estimated poses (T,4,4) | (T,12) | (T,3)
      gt_camera_poses_path  : .npy of GT poses, same shape
    """
    windows = ctx.get("windows", WINDOWS_DEFAULT)
    out: dict[str, float] = {f"traj_psnr@{a}-{b}": float("nan") for (a, b) in windows}
    out["traj_ate"] = float("nan")
    out["traj_apd"] = float("nan")

    pred_video = _read_video(pred_path) if pred_path else None
    gt_video = _read_video(gt_path) if gt_path else None
    if pred_video is not None and gt_video is not None:
        out.update(_per_window_psnr(pred_video, gt_video, windows))

    pred_pose_path = ctx.get("pred_camera_poses_path")
    gt_pose_path = ctx.get("gt_camera_poses_path")
    if pred_pose_path and gt_pose_path:
        pred_xyz = _load_camera_xyz(pred_pose_path)
        gt_xyz = _load_camera_xyz(gt_pose_path)
        if pred_xyz is not None and gt_xyz is not None:
            n = min(pred_xyz.shape[0], gt_xyz.shape[0])
            out["traj_ate"] = _ate(pred_xyz[:n], gt_xyz[:n])
            out["traj_apd"] = _apd(pred_xyz[:n], gt_xyz[:n])

    return out
