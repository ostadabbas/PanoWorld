"""Depth metrics: AbsRel, δ<1.25, with raw / sky-masked / texture-weighted variants.

Inputs
------
pred_depth_dir : directory of per-frame .npy depth predictions, shape (H, W) float
                 (the method should write one .npy per frame, sorted by name)
gt_depth_dir   : same layout, ground truth (habitat = metric meters; self/argus =
                 DAP estimate stored in [0, 1] scaled by MAX_DEPTH_SCALE = 100)

Variants reported
-----------------
  depth_abs_rel/raw            classic, no masking
  depth_abs_rel/sky_masked     pixels classified as 'sky' by a fast detector are dropped
  depth_abs_rel/texture_weighted   per-pixel weight = local image gradient L1 (normalized)
  depth_delta1_25/<variant>    fraction of pixels with max(p/g, g/p) < 1.25

Notes
-----
* For depth_kind='dap_estimated', errors should be interpreted as
  *consistency-with-DAP*, not absolute correctness. For habitat (true GT)
  errors are absolute.
* Sky detection here is intentionally very cheap (color-based). For paper-final
  numbers we'll swap in a real semantic segmenter; the API stays the same.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

MAX_DEPTH_SCALE = 100.0  # convention from pano_trajectory_pipeline.py


def _load_pred_gt(pred_dir: Path, gt_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    pred_files = sorted(pred_dir.glob("*.npy"))
    gt_files = sorted(gt_dir.glob("*.npy"))
    if not pred_files or not gt_files:
        return np.zeros(0), np.zeros(0)
    n = min(len(pred_files), len(gt_files))
    pred = np.stack([np.load(f).astype(np.float32) for f in pred_files[:n]], axis=0)
    gt = np.stack([np.load(f).astype(np.float32) for f in gt_files[:n]], axis=0)
    return pred, gt


def _normalize_gt(gt: np.ndarray, depth_kind: str) -> np.ndarray:
    if depth_kind == "dap_estimated":
        return gt * MAX_DEPTH_SCALE
    return gt


def _sky_mask_color(rgb_video: np.ndarray) -> np.ndarray:
    """Cheap sky mask. rgb_video: (T, H, W, 3) uint8. Returns (T, H, W) bool."""
    if rgb_video.ndim != 4 or rgb_video.shape[-1] < 3:
        return np.zeros(rgb_video.shape[:-1], dtype=bool) if rgb_video.ndim >= 3 else np.zeros(0, dtype=bool)
    r = rgb_video[..., 0].astype(np.int16)
    g = rgb_video[..., 1].astype(np.int16)
    b = rgb_video[..., 2].astype(np.int16)
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    sky = (b > r) & (b > g - 5) & (luma > 130)
    h = sky.shape[1]
    upper = np.zeros_like(sky)
    upper[:, : int(h * 0.45)] = True
    return sky & upper


def _gradient_weight(rgb_video: np.ndarray) -> np.ndarray:
    if rgb_video.ndim != 4:
        return np.ones(rgb_video.shape[:3], dtype=np.float32)
    g = rgb_video.astype(np.float32).mean(-1)
    gx = np.abs(np.diff(g, axis=2, prepend=g[:, :, :1]))
    gy = np.abs(np.diff(g, axis=1, prepend=g[:, :1, :]))
    w = gx + gy
    p99 = np.percentile(w, 99)
    if p99 < 1e-6:
        return np.ones_like(w)
    return np.clip(w / p99, 0.0, 1.0)


def _abs_rel(pred: np.ndarray, gt: np.ndarray, weight: np.ndarray) -> float:
    valid = (gt > 1e-3) & np.isfinite(gt) & np.isfinite(pred) & (weight > 0)
    if valid.sum() < 16:
        return float("nan")
    err = np.abs(pred[valid] - gt[valid]) / np.maximum(gt[valid], 1e-3)
    w = weight[valid]
    return float((err * w).sum() / max(w.sum(), 1e-6))


def _delta(pred: np.ndarray, gt: np.ndarray, weight: np.ndarray, thresh: float = 1.25) -> float:
    valid = (gt > 1e-3) & np.isfinite(gt) & np.isfinite(pred) & (pred > 1e-3) & (weight > 0)
    if valid.sum() < 16:
        return float("nan")
    ratio = np.maximum(pred[valid] / gt[valid], gt[valid] / pred[valid])
    w = weight[valid]
    ok = (ratio < thresh).astype(np.float32)
    return float((ok * w).sum() / max(w.sum(), 1e-6))


def eval_depth(pred_path: str, gt_path: str, **ctx) -> dict[str, float]:
    """pred_path / gt_path are directories of per-frame .npy.

    ctx may contain:
        depth_kind: 'habitat_gt' | 'dap_estimated'
        rgb_video_path: path to method-generated mp4 (used for sky-mask + gradient)
                        if absent we fall back to GT video at gt_path/../videos/...
        gt_video_path: optional explicit GT video to use for masks when pred is missing
    """
    out: dict[str, float] = {
        f"depth_abs_rel/{v}": float("nan") for v in ("raw", "sky_masked", "texture_weighted")
    }
    out.update({
        f"depth_delta1_25/{v}": float("nan") for v in ("raw", "sky_masked", "texture_weighted")
    })

    pred_dir = Path(pred_path)
    gt_dir = Path(gt_path)
    if not pred_dir.is_dir() or not gt_dir.is_dir():
        return out

    pred, gt = _load_pred_gt(pred_dir, gt_dir)
    if pred.size == 0:
        return out
    if pred.shape != gt.shape:
        n = min(pred.shape[0], gt.shape[0])
        pred, gt = pred[:n], gt[:n]
    gt = _normalize_gt(gt, ctx.get("depth_kind", "dap_estimated"))

    raw_w = np.ones_like(gt, dtype=np.float32)

    rgb_path = ctx.get("rgb_video_path") or ctx.get("gt_video_path")
    sky = None
    grad_w = None
    if rgb_path and Path(rgb_path).is_file():
        try:
            import imageio.v3 as iio

            video = iio.imread(rgb_path)  # (T, H, W, 3)
            if video.ndim == 4 and video.shape[0] > 0:
                T = min(video.shape[0], pred.shape[0])
                video = video[:T]
                pred_ = pred[:T]
                gt_ = gt[:T]
                # resize masks to depth shape if needed
                Hd, Wd = pred_.shape[-2], pred_.shape[-1]
                if video.shape[1] != Hd or video.shape[2] != Wd:
                    try:
                        from PIL import Image

                        resized = np.stack([
                            np.asarray(Image.fromarray(f).resize((Wd, Hd), Image.BILINEAR))
                            for f in video
                        ])
                        video = resized
                    except Exception:
                        video = video[:, :Hd, :Wd]
                sky_mask = _sky_mask_color(video)
                grad = _gradient_weight(video)
                pred, gt, raw_w = pred_, gt_, np.ones_like(gt_, dtype=np.float32)
                sky = sky_mask
                grad_w = grad
        except Exception:
            pass

    out["depth_abs_rel/raw"] = _abs_rel(pred, gt, raw_w)
    out["depth_delta1_25/raw"] = _delta(pred, gt, raw_w)

    if sky is not None:
        sky_w = (~sky).astype(np.float32)
        out["depth_abs_rel/sky_masked"] = _abs_rel(pred, gt, sky_w)
        out["depth_delta1_25/sky_masked"] = _delta(pred, gt, sky_w)
    if grad_w is not None:
        out["depth_abs_rel/texture_weighted"] = _abs_rel(pred, gt, grad_w)
        out["depth_delta1_25/texture_weighted"] = _delta(pred, gt, grad_w)

    for k, v in list(out.items()):
        if v != v or math.isinf(v):  # NaN / inf guard
            out[k] = float("nan")
    return out
