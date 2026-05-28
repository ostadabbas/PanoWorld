"""Shared helpers for the metrics layer.

There are two helpers here:

1. ``align_to_eval_grid`` — single-video temporal resample (used by
   FVD/FAED/FID etc., where pred and gt are encoded independently).
2. ``align_pair_to_eval_grid`` — pred+gt joint align: temporal resample
   PLUS spatial resize to a common (H, W). Used for pixel-correspondence
   metrics (PSNR/SSIM/LPIPS/CLIP-T).

Temporal axis unifies heterogeneous fps / length sources:

  * PanoWorld / Cosmos baselines write @ 16 fps × 93 frames  (5.81 s)
  * 360DVD writes                       @  8 fps × 16 frames  (2.00 s)
  * Imagine360 writes                   @  8 fps × 64 frames  (8.00 s)
  * FYC writes                          @  8 fps × 64 frames  (8.00 s)
  * Argus writes                        @  8 fps × 25 frames  (3.13 s)
  * OmniRoam (Wan-2.1)                  @ 16 fps × 81 frames  (5.06 s)*
  * GT self_iid                         @ 25 fps × 130 frames (5.20 s)
  * GT argus_ood                        @  5 fps ×  25 frames (5.00 s)
  * GT habitat_ood                      @ 25 fps × 125 frames (5.00 s)

  *OmniRoam mp4 container reports fps=30 because the writer hardcodes that;
   the semantic fps of the backbone is 16 (Self-Forcing wan_shared_cfg.sample_fps).
   Pass 16, NOT the container's r_frame_rate, to align_to_eval_grid.

We resample everything to a fixed temporal grid:

  target_fps    = 16
  target_secs   =  5.0
  target_frames = 80

via nearest-neighbor index along T (no synthetic frames invented).

If a video is shorter than ``target_secs`` the indexer just clamps to the
last available frame (the resulting tail of repeats is recorded but does
not break any metric — repeated frames simply produce zero motion at the
boundary, which is a faithful representation of the underlying low-rate
source like Argus 5 fps).

Spatial axis (align_pair_to_eval_grid only): pred and gt are bilinearly
resized to a common (H, W). Default target is gt's spatial size, so any
baseline whose output already matches GT (e.g. dvd_360 / argus / FYC /
panoworld at 1024x512) is bit-identical to before. OmniRoam Preview
(960x480) is now resized up to 1024x512 instead of triggering a min-crop
that would silently drop GT pole/seam regions and bias OmniRoam scores.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


# Defaults matching eval_config.yaml -> eval_grid.
DEFAULT_TARGET_FPS    = 16
DEFAULT_TARGET_SECS   = 5.0
DEFAULT_TARGET_FRAMES = 80   # = DEFAULT_TARGET_FPS * DEFAULT_TARGET_SECS


def read_video_uint8(path: str | Path) -> np.ndarray | None:
    """Read an mp4 to (T, H, W, 3) uint8. Returns None on failure."""
    try:
        import imageio.v3 as iio
        v = iio.imread(str(path))
        if v.ndim != 4:
            return None
        return v.astype(np.uint8) if v.dtype != np.uint8 else v
    except Exception:
        return None


def resize_thwc(
    video_thwc: np.ndarray,
    target_h: int,
    target_w: int,
    *,
    interpolation: str = "bilinear",
) -> np.ndarray:
    """Spatially resize a (T,H,W,C) uint8/float array to (T, target_h, target_w, C).

    Used to bring methods with heterogeneous output resolutions onto a common
    spatial grid before pixel-level metrics (PSNR/SSIM/LPIPS/CLIP-T). This is
    important for fair evaluation when, e.g., OmniRoam Preview emits 960x480
    while the rest of the methods + GT are 1024x512: cropping to the min would
    silently discard the equator/pole boundary regions of GT, biasing the
    omniroam scores. Resizing pred up to GT keeps the full ERP intact.

    Notes
    -----
    - For ERP videos, bilinear resize is preferred over nearest because it
      avoids aliasing the equirectangular distortion at the poles.
    - cv2 is used when available (faster), otherwise PIL fallback.
    - dtype is preserved for uint8 inputs (saturating cast back).
    """
    if video_thwc is None:
        return None  # type: ignore[return-value]
    if video_thwc.ndim != 4:
        return video_thwc
    T, H, W, C = video_thwc.shape
    if H == target_h and W == target_w:
        return video_thwc
    src_dtype = video_thwc.dtype

    try:
        import cv2  # type: ignore
        interp_map = {
            "bilinear": cv2.INTER_LINEAR,
            "bicubic":  cv2.INTER_CUBIC,
            "nearest":  cv2.INTER_NEAREST,
            "area":     cv2.INTER_AREA,
        }
        flag = interp_map.get(interpolation, cv2.INTER_LINEAR)
        out = np.empty((T, target_h, target_w, C), dtype=src_dtype)
        for t in range(T):
            out[t] = cv2.resize(video_thwc[t], (target_w, target_h), interpolation=flag)
        return out
    except ImportError:
        from PIL import Image  # type: ignore
        pil_map = {
            "bilinear": Image.BILINEAR,
            "bicubic":  Image.BICUBIC,
            "nearest":  Image.NEAREST,
        }
        resample = pil_map.get(interpolation, Image.BILINEAR)
        out = np.empty((T, target_h, target_w, C), dtype=src_dtype)
        for t in range(T):
            frame = video_thwc[t]
            if frame.dtype != np.uint8:
                # PIL only supports common dtypes; round-trip via uint8 is
                # acceptable for visual metrics where inputs are already 0-255.
                frame_u8 = np.clip(frame, 0, 255).astype(np.uint8)
                pil = Image.fromarray(frame_u8)
                pil = pil.resize((target_w, target_h), resample)
                out[t] = np.asarray(pil).astype(src_dtype)
            else:
                pil = Image.fromarray(frame)
                pil = pil.resize((target_w, target_h), resample)
                out[t] = np.asarray(pil)
        return out


def align_to_eval_grid(
    video_thwc: np.ndarray,
    src_fps: float,
    *,
    target_fps: int = DEFAULT_TARGET_FPS,
    target_secs: float = DEFAULT_TARGET_SECS,
    target_frames: int | None = None,
) -> np.ndarray:
    """Resample ``video_thwc`` (T,H,W,C) to the common evaluation grid.

    Parameters
    ----------
    video_thwc   : ndarray with leading temporal axis. Any dtype/shape.
    src_fps      : frames-per-second of the input. Used together with
                   target_fps to map source indices into the target grid.
    target_fps   : desired fps after resampling.  Default 16.
    target_secs  : desired duration in seconds.   Default 5.0.
    target_frames: explicit override; if given, takes precedence over
                   target_fps * target_secs.

    Returns
    -------
    ndarray of shape (target_frames, H, W, C) with the *same* dtype as
    the input (uint8 stays uint8, float stays float).

    Notes
    -----
    Temporal indexing is nearest-neighbor in *time*:
        t_target_i = i / target_fps                   (seconds)
        t_source   = clamp(t_target_i, 0, T_src/src_fps)
        idx        = round(t_source * src_fps), clamped to [0, T_src-1]
    This means upsampling (e.g. argus 5→16) repeats source frames, and
    downsampling (e.g. habitat 25→16) drops every ~5th frame. Both are
    intentional choices: we do *not* synthesise intermediate frames.
    """
    if video_thwc is None:
        return None  # type: ignore[return-value]
    if video_thwc.ndim < 1:
        return video_thwc
    T_src = int(video_thwc.shape[0])
    if T_src == 0:
        return video_thwc

    if target_frames is None:
        target_frames = int(round(target_fps * target_secs))
    target_frames = int(target_frames)

    # Build mapping from target index -> source index.
    # Time stamp of target frame i (seconds): i / target_fps
    t_target = np.arange(target_frames, dtype=np.float64) / float(target_fps)
    src_idx_f = t_target * float(src_fps)
    src_idx   = np.round(src_idx_f).astype(np.int64)
    src_idx   = np.clip(src_idx, 0, T_src - 1)
    return video_thwc[src_idx]


def align_pair_to_eval_grid(
    pred_thwc: np.ndarray,
    gt_thwc:   np.ndarray,
    pred_fps:  float,
    gt_fps:    float,
    *,
    target_fps: int = DEFAULT_TARGET_FPS,
    target_secs: float = DEFAULT_TARGET_SECS,
    target_frames: int | None = None,
    target_hw: tuple[int, int] | None = None,
    spatial_interp: str = "bilinear",
) -> tuple[np.ndarray, np.ndarray]:
    """Align both pred and gt onto the same temporal + spatial eval grid.

    Temporal: nearest-neighbor resample to (target_fps, target_secs) - same as
    align_to_eval_grid.

    Spatial: when pred and gt resolutions differ (e.g. OmniRoam Preview emits
    960x480 vs everyone else's 1024x512), both are bilinearly resized to a
    common (H, W). The default target is GT's spatial size, so for the typical
    case where pred matches GT this is a no-op and bit-identical to before.

    Parameters
    ----------
    target_hw : optional (H, W) override. When None, uses gt_thwc's H, W so
                pred is brought to the GT grid (preserving full ERP including
                pole/equator regions instead of cropping them away).
    spatial_interp : 'bilinear' (default), 'bicubic', 'nearest', 'area'.

    Notes
    -----
    Previous behavior cropped both to ``min(H), min(W)``. That silently dropped
    GT pole/seam pixels for any baseline whose output was smaller than GT,
    biasing pixel metrics in that baseline's favor. The fix resizes pred up to
    GT instead, so all methods are scored against the *full* GT ERP.
    """
    pred_a = align_to_eval_grid(
        pred_thwc, pred_fps,
        target_fps=target_fps, target_secs=target_secs,
        target_frames=target_frames,
    )
    gt_a = align_to_eval_grid(
        gt_thwc, gt_fps,
        target_fps=target_fps, target_secs=target_secs,
        target_frames=target_frames,
    )
    if pred_a is None or gt_a is None:
        return pred_a, gt_a  # type: ignore[return-value]

    if target_hw is None:
        target_h = int(gt_a.shape[1])
        target_w = int(gt_a.shape[2])
    else:
        target_h, target_w = int(target_hw[0]), int(target_hw[1])

    pred_a = resize_thwc(pred_a, target_h, target_w, interpolation=spatial_interp)
    gt_a   = resize_thwc(gt_a,   target_h, target_w, interpolation=spatial_interp)
    return pred_a, gt_a


def fps_from_master_row(row: dict, default: float = 16.0) -> float:
    """Read fps from a master.csv row (string column 'fps'). Falls back to
    the model's training fps if missing/invalid (safer than guessing 25)."""
    try:
        return float(row.get("fps") or default)
    except (TypeError, ValueError):
        return float(default)


__all__ = [
    "DEFAULT_TARGET_FPS",
    "DEFAULT_TARGET_SECS",
    "DEFAULT_TARGET_FRAMES",
    "read_video_uint8",
    "resize_thwc",
    "align_to_eval_grid",
    "align_pair_to_eval_grid",
    "fps_from_master_row",
]
