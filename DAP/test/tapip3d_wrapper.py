#!/usr/bin/env python3
"""
TAPIP3D wrapper for the DAP panoramic-trajectory pipeline.

Called as a subprocess in the tapip3d conda env.  Accepts frames + DAP depth
maps, builds the .npz package TAPIP3D expects, runs inference, and writes:
    <out_dir>/tracks_2d.npy   (N, T, 2)   x,y pixel coords
    <out_dir>/visibility.npy  (N, T)       float [0,1]
    <out_dir>/meta.json

We deliberately do NOT import inference.py (which has a top-level MegaSAM
import) and instead call the lower-level utils directly.
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

# ── locate third_party/TAPIP3D relative to this file ─────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_TAP_ROOT = os.path.join(_REPO_ROOT, "third_party", "TAPIP3D")
if _TAP_ROOT not in sys.path:
    sys.path.insert(0, _TAP_ROOT)

# ── now safe to import TAPIP3D utilities (avoids top-level MegaSAM import) ───
from utils.inference_utils import (           # noqa: E402
    load_model,
    get_grid_queries,
    inference,
)

# ── constants ─────────────────────────────────────────────────────────────────
DAP_DEPTH_SCALE = 100.0   # DAP .npy ∈ [0,1] → metres
_DEFAULT_CKPT = os.path.join(_TAP_ROOT, "checkpoints", "tapip3d_final.pth")

# Standard equirectangular → perspective approximation:
#   fx = fy = W / (2π),   cx = W/2,  cy = H/2


def erp_intrinsics(H: int, W: int) -> np.ndarray:
    """Per-frame perspective-approximation intrinsics for an ERP frame."""
    f = W / (2.0 * np.pi)
    K = np.array([[f, 0.0, W / 2.0],
                  [0.0, f, H / 2.0],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    return K


def sorted_files(d, ext):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(ext))


def load_frames(frames_dir, max_side=None):
    paths = sorted_files(frames_dir, (".png", ".jpg", ".jpeg"))
    if not paths:
        raise FileNotFoundError(f"No image files in {frames_dir}")
    frames = []
    for p in paths:
        bgr = cv2.imread(os.path.join(frames_dir, p))
        if bgr is None:
            continue
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    orig_H, orig_W = frames[0].shape[:2]
    if max_side and max(orig_H, orig_W) > max_side:
        scale = max_side / max(orig_H, orig_W)
        new_H = int(orig_H * scale) & ~1
        new_W = int(orig_W * scale) & ~1
        frames = [cv2.resize(f, (new_W, new_H)) for f in frames]
    return np.stack(frames, axis=0), orig_H, orig_W  # (T, H, W, 3) uint8


def load_depths(depth_dir, T, H, W):
    paths = sorted_files(depth_dir, ".npy")
    if len(paths) != T:
        paths = paths[:T] if len(paths) >= T else paths + [paths[-1]] * (T - len(paths))
    depths = []
    for p in paths:
        d = np.load(os.path.join(depth_dir, p)).astype(np.float32)
        if d.max() <= 1.5:
            d = d * DAP_DEPTH_SCALE
        if d.shape != (H, W):
            d = cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR)
        depths.append(d)
    return np.stack(depths, axis=0)   # (T, H, W)


def project_world_to_pixel(coords, intrinsics):
    """
    Project 3D world coords back to 2D pixel coords.
    coords       : (T, N, 3)  world XYZ  (identity extrinsics → camera frame)
    intrinsics   : (T, 3, 3)  per-frame intrinsics
    Returns      : (T, N, 2)  pixel coords (u, v)
    """
    T, N, _ = coords.shape
    pixels = np.zeros((T, N, 2), dtype=np.float32)
    for t in range(T):
        K = intrinsics[t]             # (3, 3)
        xyz = coords[t]               # (N, 3)
        Z = xyz[:, 2]
        valid = Z > 1e-4
        # perspective division
        u = np.where(valid, K[0, 0] * xyz[:, 0] / np.where(valid, Z, 1.0) + K[0, 2], 0.0)
        v = np.where(valid, K[1, 1] * xyz[:, 1] / np.where(valid, Z, 1.0) + K[1, 2], 0.0)
        pixels[t, :, 0] = u
        pixels[t, :, 1] = v
    return pixels


def run_tapip3d(frames_arr, depths_arr, grid_size, ckpt, device_str="cuda:0",
                num_iters=6, support_grid_size=16, resolution_factor=2,
                vis_threshold=0.9):
    """
    frames_arr : (T, H, W, 3) uint8
    depths_arr : (T, H, W) float32 metres
    Returns: tracks_2d (N, T, 2), visibility (N, T)
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    T, H, W, _ = frames_arr.shape

    # ── load model ────────────────────────────────────────────────────────────
    model = load_model(ckpt)
    model.to(device)
    inference_res = (
        int(model.image_size[0] * np.sqrt(resolution_factor)),
        int(model.image_size[1] * np.sqrt(resolution_factor)),
    )
    model.set_image_size(inference_res)
    inf_H, inf_W = inference_res
    print(f"   TAPIP3D inference resolution: {inf_H}×{inf_W}")

    # ── build intrinsics (T, 3, 3) for original frames ────────────────────────
    K_orig = erp_intrinsics(H, W)
    intrinsics_orig = np.tile(K_orig[None], (T, 1, 1))   # (T, 3, 3)

    # ── extrinsics: identity for all frames ───────────────────────────────────
    extrinsics = np.tile(np.eye(4, dtype=np.float32)[None], (T, 1, 1))  # (T, 4, 4)

    # ── resize video + depths to inference resolution + scale intrinsics ──────
    frames_inf = np.stack([cv2.resize(f, (inf_W, inf_H)) for f in frames_arr])
    depths_inf = np.stack([cv2.resize(d, (inf_W, inf_H), interpolation=cv2.INTER_LINEAR)
                           for d in depths_arr])
    intrinsics_inf = intrinsics_orig.copy()
    intrinsics_inf[:, 0, :] *= (inf_W - 1) / (W - 1)
    intrinsics_inf[:, 1, :] *= (inf_H - 1) / (H - 1)

    # ── convert to tensors ────────────────────────────────────────────────────
    vid_t    = (torch.from_numpy(frames_inf).permute(0, 3, 1, 2).float() / 255.0).to(device)
    dep_t    = torch.from_numpy(depths_inf).float().to(device)
    intr_t   = torch.from_numpy(intrinsics_inf).float().to(device)
    extr_t   = torch.from_numpy(extrinsics).float().to(device)

    # ── build grid queries (3D world coords at frame 0) ───────────────────────
    query_pt = get_grid_queries(
        grid_size=grid_size,
        depths=dep_t,
        intrinsics=intr_t,
        extrinsics=extr_t,
    )   # (N, 4)  [frame_idx, X, Y, Z]
    print(f"   TAPIP3D query points: {query_pt.shape[0]}  grid_size={grid_size}")

    # ── run inference ─────────────────────────────────────────────────────────
    print(f"   Running TAPIP3D  T={T}  grid={grid_size}  "
          f"num_iters={num_iters}  device={device}")

    with torch.autocast("cuda", dtype=torch.bfloat16):
        coords, visibs = inference(
            model=model,
            video=vid_t,
            depths=dep_t,
            intrinsics=intr_t,
            extrinsics=extr_t,
            query_point=query_pt,
            num_iters=num_iters,
            grid_size=support_grid_size,
            vis_threshold=vis_threshold,
        )
    # coords   : (T, N, 3)  world XYZ
    # visibs   : (T, N)     bool

    coords_np = coords.float().cpu().numpy()    # (T, N, 3)
    visibs_np = visibs.float().cpu().numpy()    # (T, N)

    print(f"   TAPIP3D done: {coords_np.shape[1]} tracks × {T} frames")

    # ── project 3D → 2D in INFERENCE resolution ───────────────────────────────
    pixels_inf = project_world_to_pixel(coords_np, intrinsics_inf)   # (T, N, 2)

    # ── scale back to original resolution ─────────────────────────────────────
    pixels_inf[:, :, 0] *= W / inf_W
    pixels_inf[:, :, 1] *= H / inf_H
    pixels_orig = pixels_inf   # (T, N, 2) in orig coords

    tracks_2d = pixels_orig.transpose(1, 0, 2)   # (N, T, 2)
    vis_out   = visibs_np.transpose(1, 0)         # (N, T)

    return tracks_2d, vis_out


def main():
    parser = argparse.ArgumentParser(
        description="TAPIP3D wrapper for DAP pipeline"
    )
    parser.add_argument("--frames_dir",       required=True)
    parser.add_argument("--depth_dir",        required=True)
    parser.add_argument("--out_dir",          required=True)
    parser.add_argument("--grid_size",        type=int,   default=32)
    parser.add_argument("--num_iters",        type=int,   default=6)
    parser.add_argument("--support_grid",     type=int,   default=16)
    parser.add_argument("--resolution_factor",type=int,   default=2)
    parser.add_argument("--vis_threshold",    type=float, default=0.9)
    parser.add_argument("--ckpt",             default=_DEFAULT_CKPT)
    parser.add_argument("--gpu",              type=int,   default=0)
    parser.add_argument("--max_frames",       type=int,   default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    print(f"[tapip3d_wrapper] frames_dir = {args.frames_dir}")
    print(f"[tapip3d_wrapper] depth_dir  = {args.depth_dir}")
    print(f"[tapip3d_wrapper] out_dir    = {args.out_dir}")

    frames_arr, orig_H, orig_W = load_frames(args.frames_dir)
    T = len(frames_arr)
    H, W = frames_arr.shape[1:3]
    print(f"   Loaded {T} frames  ({H}×{W})")

    if args.max_frames and T > args.max_frames:
        idx = np.linspace(0, T - 1, args.max_frames, dtype=int)
        frames_arr = frames_arr[idx]
        T = len(idx)
        print(f"   Sub-sampled to {T} frames")

    depths_arr = load_depths(args.depth_dir, T, H, W)
    print(f"   Depths range: {depths_arr.min():.2f} – {depths_arr.max():.2f} m")

    tracks_2d, vis = run_tapip3d(
        frames_arr, depths_arr,
        grid_size=args.grid_size,
        ckpt=args.ckpt,
        device_str=f"cuda:{args.gpu}",
        num_iters=args.num_iters,
        support_grid_size=args.support_grid,
        resolution_factor=args.resolution_factor,
        vis_threshold=args.vis_threshold,
    )

    # Scale back if load_frames resized
    if H != orig_H or W != orig_W:
        tracks_2d[..., 0] *= orig_W / W
        tracks_2d[..., 1] *= orig_H / H

    np.save(os.path.join(args.out_dir, "tracks_2d.npy"),   tracks_2d.astype(np.float32))
    np.save(os.path.join(args.out_dir, "visibility.npy"),  vis.astype(np.float32))

    meta = {
        "tracker":           "tapip3d",
        "num_tracks":        int(tracks_2d.shape[0]),
        "num_frames":        int(tracks_2d.shape[1]),
        "frame_H":           orig_H,
        "frame_W":           orig_W,
        "grid_size":         args.grid_size,
        "num_iters":         args.num_iters,
        "resolution_factor": args.resolution_factor,
        "elapsed_s":         round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[tapip3d_wrapper] Done in {meta['elapsed_s']} s")
    print(f"   tracks_2d  → {os.path.join(args.out_dir, 'tracks_2d.npy')}")
    print(f"   visibility → {os.path.join(args.out_dir, 'visibility.npy')}")


if __name__ == "__main__":
    main()
