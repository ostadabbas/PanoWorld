"""Visual quality metrics: PSNR, SSIM, LPIPS, FAED, FID, FVD, CLIP-T.

PSNR / SSIM are paired (need GT video).
LPIPS is paired.
FID / FVD / FAED are distributional (need a *set* of pred videos & a *set* of
GT videos — these are computed once at the end of run_eval over the union of
all clips, not per-clip; for per-clip we report per-frame LPIPS/PSNR/SSIM).
CLIP-T is video↔text alignment, recorded per clip (mean over frames).

All heavy models (LPIPS net, CLIP, FAED autoencoder, FVD I3D) are cached as
module-level singletons; the first call loads them, subsequent calls reuse.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

try:
    # package-relative import when run as `metrics.visual`
    from ._common import (
        DEFAULT_TARGET_FPS, DEFAULT_TARGET_SECS, DEFAULT_TARGET_FRAMES,
        align_pair_to_eval_grid, read_video_uint8,
    )
except ImportError:
    # script-style fallback (run_eval.py prepends the eval/ dir to sys.path)
    from _common import (  # type: ignore
        DEFAULT_TARGET_FPS, DEFAULT_TARGET_SECS, DEFAULT_TARGET_FRAMES,
        align_pair_to_eval_grid, read_video_uint8,
    )

_CACHE: dict[str, object] = {}


def _read_video(path: str) -> np.ndarray | None:
    return read_video_uint8(path)


def _psnr_per_frame(pred: np.ndarray, gt: np.ndarray) -> float:
    if pred.shape[0] == 0:
        return float("nan")
    diff = pred.astype(np.float64) - gt.astype(np.float64)
    mse = (diff * diff).reshape(diff.shape[0], -1).mean(-1)
    psnr = 10.0 * np.log10((255.0**2) / np.maximum(mse, 1e-12))
    return float(psnr.mean())


def _ssim_per_frame(pred: np.ndarray, gt: np.ndarray) -> float:
    """GPU-batched SSIM (T frames at once). Falls back to CPU skimage when
    torch / torchmetrics are unavailable, but at 1024x512 RGB the CPU path is
    ~0.17s/frame which makes a full 150-clip × 15-method sweep take hours."""
    if pred.shape[0] == 0:
        return float("nan")
    try:
        import torch
        from torchmetrics.functional.image import (
            structural_similarity_index_measure as _tm_ssim,
        )
    except Exception:
        torch = None
    if torch is not None:
        try:
            with torch.no_grad():
                p = torch.from_numpy(pred).float().permute(0, 3, 1, 2) / 255.0
                g = torch.from_numpy(gt).float().permute(0, 3, 1, 2) / 255.0
                if torch.cuda.is_available():
                    p = p.cuda(non_blocking=True)
                    g = g.cuda(non_blocking=True)
                val = _tm_ssim(p, g, data_range=1.0)
                return float(val.item())
        except Exception:
            pass
    try:
        from skimage.metrics import structural_similarity as ssim
    except Exception:
        return float("nan")
    vals = []
    for p, g in zip(pred, gt):
        try:
            vals.append(ssim(g, p, channel_axis=-1, data_range=255))
        except Exception:
            continue
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _get_lpips():
    if "lpips" in _CACHE:
        return _CACHE["lpips"]
    try:
        import torch
        import lpips

        net = lpips.LPIPS(net="alex").eval()
        if torch.cuda.is_available():
            net = net.cuda()
        _CACHE["lpips"] = (net, torch)
        return _CACHE["lpips"]
    except Exception as e:
        _CACHE["lpips"] = None
        return None


def _lpips_per_frame(pred: np.ndarray, gt: np.ndarray) -> float:
    bundle = _get_lpips()
    if bundle is None:
        return float("nan")
    net, torch = bundle
    with torch.no_grad():
        p = torch.from_numpy(pred).float().permute(0, 3, 1, 2).contiguous()
        g = torch.from_numpy(gt).float().permute(0, 3, 1, 2).contiguous()
        p = p / 127.5 - 1.0
        g = g / 127.5 - 1.0
        if torch.cuda.is_available():
            p = p.cuda()
            g = g.cuda()
        d = net(p, g).flatten()
        return float(d.mean().item())


def _get_clip():
    if "clip" in _CACHE:
        return _CACHE["clip"]
    try:
        import torch
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        model = model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        _CACHE["clip"] = (model, preprocess, tokenizer, torch)
        return _CACHE["clip"]
    except Exception:
        _CACHE["clip"] = None
        return None


def _clip_t_per_frame(pred_video: np.ndarray, caption: str) -> float:
    bundle = _get_clip()
    if bundle is None or not caption:
        return float("nan")
    model, preprocess, tokenizer, torch = bundle
    from PIL import Image

    with torch.no_grad():
        text = tokenizer([caption])
        if torch.cuda.is_available():
            text = text.cuda()
        text_feat = model.encode_text(text)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        sims = []
        step = max(1, pred_video.shape[0] // 8)
        for i in range(0, pred_video.shape[0], step):
            img = Image.fromarray(pred_video[i])
            inp = preprocess(img).unsqueeze(0)
            if torch.cuda.is_available():
                inp = inp.cuda()
            f = model.encode_image(inp)
            f = f / f.norm(dim=-1, keepdim=True)
            sims.append(float((f @ text_feat.T).item()))
        if not sims:
            return float("nan")
        return float(np.mean(sims))


def eval_visual(pred_path: str, gt_path: str, **ctx) -> dict[str, float]:
    """
    pred_path: method-generated mp4
    gt_path  : GT mp4

    ctx:
      caption        : text string for CLIP-T (optional)
      pred_fps       : fps of pred mp4 (default 16, the Cosmos training rate)
      gt_fps         : fps of gt mp4 (from master.csv 'fps' column)
      eval_target_fps      : override DEFAULT_TARGET_FPS
      eval_target_secs     : override DEFAULT_TARGET_SECS
      eval_target_frames   : override DEFAULT_TARGET_FRAMES (rare)

    Both pred and GT are temporally resampled to the eval grid (default
    80f @ 16 fps = 5.0 s) before any per-frame metric. PSNR / SSIM / LPIPS
    therefore compare frame-aligned 16-fps streams; CLIP-T is the per-frame
    mean over the grid.
    """
    out = {
        "vq_psnr": float("nan"),
        "vq_ssim": float("nan"),
        "vq_lpips": float("nan"),
        "vq_clip_t": float("nan"),
    }

    pred = _read_video(pred_path) if pred_path else None
    gt   = _read_video(gt_path) if gt_path else None
    if pred is None or gt is None:
        return out

    pred_fps = float(ctx.get("pred_fps") or DEFAULT_TARGET_FPS)
    gt_fps   = float(ctx.get("gt_fps")   or DEFAULT_TARGET_FPS)
    target_fps    = int(ctx.get("eval_target_fps")    or DEFAULT_TARGET_FPS)
    target_secs   = float(ctx.get("eval_target_secs") or DEFAULT_TARGET_SECS)
    target_frames = ctx.get("eval_target_frames")
    target_frames = int(target_frames) if target_frames else None

    pred_, gt_ = align_pair_to_eval_grid(
        pred, gt, pred_fps, gt_fps,
        target_fps=target_fps, target_secs=target_secs,
        target_frames=target_frames,
    )

    out["vq_psnr"]  = _psnr_per_frame(pred_, gt_)
    out["vq_ssim"]  = _ssim_per_frame(pred_, gt_)
    out["vq_lpips"] = _lpips_per_frame(pred_, gt_)

    cap = ctx.get("caption", "")
    if cap:
        out["vq_clip_t"] = _clip_t_per_frame(pred_, cap)

    for k, v in list(out.items()):
        if v != v or (isinstance(v, float) and math.isinf(v)):
            out[k] = float("nan")
    return out
