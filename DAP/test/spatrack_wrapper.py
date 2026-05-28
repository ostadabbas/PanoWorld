#!/usr/bin/env python3
"""
SpaTracker wrapper for the DAP panoramic-trajectory pipeline.

Called as a subprocess in the spatrack conda env.  Accepts frames + DAP depth
maps, runs SpaTracker in RGBD mode, and writes the standardised outputs:
    <out_dir>/tracks_2d.npy   (N, T, 2)   x,y pixel coords
    <out_dir>/visibility.npy  (N, T)       float [0,1]
    <out_dir>/meta.json
"""

import argparse
import json
import os
import sys
import time

# ── Fix cupy CUDA library loading order before importing torch/cupy ───────────
# SpaTracker uses cupy JIT compilation.  nvidia-* Python packages installed by
# PyTorch must take precedence over cupy-bundled CUDA libs to avoid symbol
# version mismatches (e.g. cublasSetEnvironmentMode removed in newer cublas).
def _prepend_nvidia_libs():
    import site
    for sp in site.getsitepackages():
        nvidia = os.path.join(sp, "nvidia")
        if not os.path.isdir(nvidia):
            continue
        lib_dirs = []
        for sub in ["cublas", "cusolver", "cusparse", "cudnn"]:
            d = os.path.join(nvidia, sub, "lib")
            if os.path.isdir(d):
                lib_dirs.append(d)
        if lib_dirs:
            existing = os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs + ([existing] if existing else []))
            break

_prepend_nvidia_libs()

# Make sure CONDA_PREFIX/CUDA_HOME is set so cupy.cuda.get_cuda_path() works
if not os.environ.get("CONDA_PREFIX"):
    import site
    for sp in site.getsitepackages():
        # sp is typically .../envs/<name>/lib/pythonX.Y/site-packages
        candidate = os.path.dirname(os.path.dirname(os.path.dirname(sp)))
        if os.path.isdir(os.path.join(candidate, "bin")):
            os.environ.setdefault("CONDA_PREFIX", candidate)
            os.environ.setdefault("CUDA_HOME", candidate)
            break

import cv2
import numpy as np
import torch

# ── locate third_party/SpaTracker relative to this file ──────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_SPA_ROOT = os.path.join(_REPO_ROOT, "third_party", "SpaTracker")
if _SPA_ROOT not in sys.path:
    sys.path.insert(0, _SPA_ROOT)

from models.spatracker.predictor import SpaTrackerPredictor  # noqa: E402

# ── constants ─────────────────────────────────────────────────────────────────
DAP_DEPTH_SCALE = 100.0   # DAP .npy ∈ [0,1] → metres; then ×100 for SpaTracker
_DEFAULT_CKPT = os.path.join(_SPA_ROOT, "checkpoints", "spaT_final.pth")


def sorted_files(d, ext):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(ext))


def load_frames(frames_dir, max_side=None):
    """Return float RGB tensor (T, H, W, 3) and original (H, W)."""
    paths = sorted_files(frames_dir, (".png", ".jpg", ".jpeg"))
    if not paths:
        raise FileNotFoundError(f"No image files in {frames_dir}")
    frames = []
    for p in paths:
        bgr = cv2.imread(os.path.join(frames_dir, p))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
    if not frames:
        raise RuntimeError("Could not read any frames")
    orig_H, orig_W = frames[0].shape[:2]
    if max_side and max(orig_H, orig_W) > max_side:
        scale = max_side / max(orig_H, orig_W)
        new_H = int(orig_H * scale) & ~1  # even
        new_W = int(orig_W * scale) & ~1
        frames = [cv2.resize(f, (new_W, new_H)) for f in frames]
    frames = np.stack(frames, axis=0)  # (T, H, W, 3)
    return frames, orig_H, orig_W


def load_depths(depth_dir, T, H, W):
    """Load DAP depth .npy files → (T, H, W) float32 in metres."""
    paths = sorted_files(depth_dir, ".npy")
    if len(paths) != T:
        # tolerate extra files by truncating / padding with last frame
        paths = paths[:T] if len(paths) >= T else paths + [paths[-1]] * (T - len(paths))
    depths = []
    for p in paths:
        d = np.load(os.path.join(depth_dir, p)).astype(np.float32)
        if d.max() <= 1.5:          # normalised [0,1] → metres
            d = d * DAP_DEPTH_SCALE
        if d.shape != (H, W):
            d = cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR)
        depths.append(d)
    return np.stack(depths, axis=0)  # (T, H, W)


def run_spatracker(frames_arr, depths_arr, grid_size, ckpt, seq_length=12, gpu=0):
    """
    frames_arr : (T, H, W, 3) uint8
    depths_arr : (T, H, W) float32 metres
    Returns: tracks_2d (N, T, 2), visibility (N, T)
    """
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    T, H, W, _ = frames_arr.shape

    # ── build video tensor (1, T, 3, H, W) ──────────────────────────────────
    video = torch.from_numpy(frames_arr).float()   # (T, H, W, 3)
    video = video.permute(0, 3, 1, 2)[None]        # (1, T, 3, H, W)
    video = video.to(device)

    # ── build depth tensor (T, 1, H, W) ─────────────────────────────────────
    depths = torch.from_numpy(depths_arr).float()  # (T, H, W)
    depths = depths[:, None]                        # (T, 1, H, W)
    depths = depths.to(device)

    # ── build model ──────────────────────────────────────────────────────────
    model = SpaTrackerPredictor(
        checkpoint=ckpt,
        interp_shape=(384, 512),
        seq_length=seq_length,
    ).to(device)

    # ── segmentation mask: full image ────────────────────────────────────────
    segm_mask = torch.ones((1, 1, H, W), device=device, dtype=torch.float32)

    print(f"   Running SpaTracker  T={T}  H={H}  W={W}  grid={grid_size}  "
          f"seq_len={seq_length}  device={device}")

    with torch.no_grad():
        pred_tracks, pred_visibility, T_Firsts = model(
            video,
            video_depth=depths,
            grid_size=grid_size,
            backward_tracking=False,
            depth_predictor=None,
            grid_query_frame=0,
            segm_mask=segm_mask,
            wind_length=seq_length,
        )

    # pred_tracks   : (1, T, N, 3)  last dim = x, y, depth
    # pred_visibility: (1, T, N)
    tracks_2d = pred_tracks[0, :, :, :2].cpu().numpy()   # (T, N, 2)
    vis = pred_visibility[0].float().cpu().numpy()        # (T, N)

    tracks_2d = tracks_2d.transpose(1, 0, 2)  # (N, T, 2)
    vis = vis.transpose(1, 0)                  # (N, T)

    print(f"   SpaTracker done: {tracks_2d.shape[0]} tracks × {T} frames")
    return tracks_2d, vis


def main():
    parser = argparse.ArgumentParser(
        description="SpaTracker RGBD wrapper for DAP pipeline"
    )
    parser.add_argument("--frames_dir", required=True, help="Directory of PNG frames")
    parser.add_argument("--depth_dir",  required=True, help="Directory of DAP .npy depth maps")
    parser.add_argument("--out_dir",    required=True, help="Output directory")
    parser.add_argument("--grid_size",  type=int, default=40)
    parser.add_argument("--seq_length", type=int, default=12,
                        help="SpaTracker window length [8, 12, 16]")
    parser.add_argument("--ckpt",       default=_DEFAULT_CKPT)
    parser.add_argument("--gpu",        type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Subsample to at most this many frames")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()

    print(f"[spatrack_wrapper] frames_dir = {args.frames_dir}")
    print(f"[spatrack_wrapper] depth_dir  = {args.depth_dir}")
    print(f"[spatrack_wrapper] out_dir    = {args.out_dir}")

    # ── load frames ──────────────────────────────────────────────────────────
    frames_arr, orig_H, orig_W = load_frames(args.frames_dir)
    T, H, W, _ = frames_arr.shape
    print(f"   Loaded {T} frames  ({H}×{W}, orig {orig_H}×{orig_W})")

    if args.max_frames and T > args.max_frames:
        idx = np.linspace(0, T - 1, args.max_frames, dtype=int)
        frames_arr = frames_arr[idx]
        T = len(idx)
        print(f"   Sub-sampled to {T} frames")

    # ── load depths ──────────────────────────────────────────────────────────
    depths_arr = load_depths(args.depth_dir, T, H, W)
    print(f"   Depths range: {depths_arr.min():.2f} – {depths_arr.max():.2f} m")

    # ── run tracker ──────────────────────────────────────────────────────────
    tracks_2d, vis = run_spatracker(
        frames_arr, depths_arr,
        grid_size=args.grid_size,
        ckpt=args.ckpt,
        seq_length=args.seq_length,
        gpu=args.gpu,
    )

    # ── if frames were sub-sampled or resized, scale tracks back to orig res ─
    if H != orig_H or W != orig_W:
        tracks_2d[..., 0] *= orig_W / W
        tracks_2d[..., 1] *= orig_H / H

    # ── save outputs ─────────────────────────────────────────────────────────
    np.save(os.path.join(args.out_dir, "tracks_2d.npy"), tracks_2d.astype(np.float32))
    np.save(os.path.join(args.out_dir, "visibility.npy"), vis.astype(np.float32))

    meta = {
        "tracker":    "spatrack",
        "num_tracks": int(tracks_2d.shape[0]),
        "num_frames": int(tracks_2d.shape[1]),
        "frame_H":    orig_H,
        "frame_W":    orig_W,
        "grid_size":  args.grid_size,
        "seq_length": args.seq_length,
        "elapsed_s":  round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[spatrack_wrapper] Done in {meta['elapsed_s']} s")
    print(f"   tracks_2d  → {os.path.join(args.out_dir, 'tracks_2d.npy')}")
    print(f"   visibility → {os.path.join(args.out_dir, 'visibility.npy')}")


if __name__ == "__main__":
    main()
