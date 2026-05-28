#!/usr/bin/env python3
"""
Geometry Annotation Generator for Panoramic Video Training
===========================================================
Generates per-video depth maps (DAP) and trajectory annotations (CoTracker3)
for geometry-aware training losses (L_depth, L_track).

Works with any dataset that follows the Cosmos pano training layout:
  <data_dir>/videos/*.mp4
  <data_dir>/train.csv  (or val.csv)

Usage:
  # Default dataset (single GPU, ~4h for 2300 videos)
  python generate_annotations.py --gpu 0 --resume

  # Custom dataset
  python generate_annotations.py \\
      --data_dir /path/to/my_dataset \\
      --output_dir /path/to/my_annotations \\
      --gpu 0 --resume

  # Multi-GPU parallel (4 GPUs, ~1h)
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python generate_annotations.py \\
        --gpu 0 --total_shards 4 --shard_id $i --resume &
  done

Output per video (~880 KB):
  <output_dir>/<video_name>/
    depth/0000.npy ... 0019.npy   # (H, W) float16 depth maps
    tracks_2d.npy                  # (N, T, 2) float16 pixel coords
    tracks_3d.npy                  # (N, T, 3) float16 3D positions (stabilized)
    visibility.npy                 # (N, T) float16
    meta.json                      # resolution, camera motion, metrics
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import cv2
import torch
from pathlib import Path
from tqdm import tqdm

DAP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DAP")
sys.path.insert(0, DAP_ROOT)
sys.path.insert(0, os.path.join(DAP_ROOT, "test"))

_PANO_DATA_ROOT = os.environ.get(
    "PANO_DATA_ROOT", os.path.join(os.path.expanduser("~"), "pano_video_data")
)
TRAIN_DATA_DIR = os.environ.get(
    "PANO_TRAIN_DATA_DIR", os.path.join(_PANO_DATA_ROOT, "cosmos_pano_train")
)
DEFAULT_OUTPUT_DIR = os.environ.get(
    "PANO_ANNOTATIONS_DIR", os.path.join(_PANO_DATA_ROOT, "annotations")
)


def load_dap_model(config_path, gpu="0"):
    """Load DAP depth model once, reuse for all videos."""
    import yaml
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # DAP uses relative paths internally, must run from DAP root
    prev_cwd = os.getcwd()
    os.chdir(DAP_ROOT)
    try:
        from infer import load_model
        model, device = load_model(config)
    finally:
        os.chdir(prev_cwd)
    return model, device


def load_cotracker_model(device="cuda"):
    """Load CoTracker3 offline model once."""
    from cotracker.predictor import CoTrackerPredictor
    ckpt = os.path.expanduser("~/.cache/torch/hub/checkpoints/scaled_offline.pth")
    model = CoTrackerPredictor(checkpoint=ckpt, window_len=60, v2=False)
    model = model.to(device)
    model.eval()
    return model


def read_video_frames(video_path, max_frames=20, target_w=1024, target_h=512):
    """Read frames from a video, subsampling to max_frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        total_frames = len(frames)
        cap.release()
        if total_frames == 0:
            return []
        step = max(1, total_frames // max_frames)
        selected = frames[::step][:max_frames]
    else:
        step = max(1, total_frames // max_frames)
        selected = []
        for i in range(0, total_frames, step):
            if len(selected) >= max_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                selected.append(frame)
        cap.release()

    result = []
    for frame in selected:
        h, w = frame.shape[:2]
        if w != target_w or h != target_h:
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        result.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    return result


def run_dap_on_frames(dap_model, device, frames_rgb):
    """Run DAP depth on a list of RGB frames. Returns list of float32 depth maps."""
    from infer import infer_raw
    depths = []
    for frame in frames_rgb:
        pred = infer_raw(dap_model, device, frame)
        depths.append(pred)
    return depths


def run_cotracker_on_frames(ct_model, frames_rgb, grid_size=30, pad_frac=0.25, device="cuda"):
    """Run CoTracker3 with circular padding on frames."""
    from pano_trajectory_pipeline import circular_pad_video, unwrap_tracks

    H, W = frames_rgb[0].shape[:2]
    padded, pad_px = circular_pad_video(frames_rgb, pad_frac)

    vid_np = np.stack(padded, axis=0)
    vid_t = torch.from_numpy(vid_np).permute(0, 3, 1, 2)
    vid_t = vid_t.unsqueeze(0).float().to(device)

    with torch.no_grad():
        pred_tracks, pred_vis = ct_model(vid_t, grid_size=grid_size)

    tracks_np = pred_tracks[0].cpu().numpy().transpose(1, 0, 2)
    vis_np = pred_vis[0].cpu().numpy().T

    tracks_2d, vis = unwrap_tracks(tracks_np, vis_np, W, pad_px)
    return tracks_2d, vis


def process_single_video(video_path, out_dir, dap_model, dap_device,
                         ct_model, grid_size=30, max_frames=20,
                         max_depth=100.0):
    """Process a single video: depth + tracking + 3D lifting + camera motion."""
    from pano_trajectory_pipeline import (
        merge_wraparound_tracks, lift_tracks_to_3d,
        detect_and_compensate_camera_motion,
        compute_reprojection_error, compute_3d_smoothness,
        compute_rigidity_score, load_depth, MAX_DEPTH_SCALE,
        erp_pixel_to_lonlat, lonlat_to_xyz,
    )

    os.makedirs(out_dir, exist_ok=True)

    frames_rgb = read_video_frames(video_path, max_frames=max_frames)
    if len(frames_rgb) < 3:
        return None

    T = len(frames_rgb)
    H, W = frames_rgb[0].shape[:2]

    # DAP depth
    depths = run_dap_on_frames(dap_model, dap_device, frames_rgb)

    # Auto max_depth from scene P99
    all_depths = np.concatenate([d.ravel() * MAX_DEPTH_SCALE for d in depths[:5]])
    auto_max_depth = float(np.percentile(all_depths, 99))
    auto_max_depth = min(auto_max_depth, max_depth)

    # Save depth maps as float16
    depth_dir = os.path.join(out_dir, "depth")
    os.makedirs(depth_dir, exist_ok=True)
    depth_paths = []
    for i, d in enumerate(depths):
        p = os.path.join(depth_dir, f"{i:04d}.npy")
        np.save(p, d.astype(np.float16))
        depth_paths.append(p)

    # CoTracker3
    tracks_2d, vis = run_cotracker_on_frames(
        ct_model, frames_rgb, grid_size=grid_size, device=dap_device
    )
    tracks_2d, vis = merge_wraparound_tracks(tracks_2d, vis, W)

    # 3D lifting (need to load depth as float32 for computation)
    N = tracks_2d.shape[0]
    tracks_3d = np.zeros((N, T, 3), dtype=np.float64)
    sampled_d = np.zeros((N, T), dtype=np.float64)

    for t in range(T):
        depth_m = depths[t].astype(np.float32)
        if depth_m.max() <= 1.5:
            depth_m = depth_m * MAX_DEPTH_SCALE
        dH, dW = depth_m.shape

        for i in range(N):
            if vis[i, t] < 0.5:
                tracks_3d[i, t] = [np.nan, np.nan, np.nan]
                sampled_d[i, t] = np.nan
                continue

            px, py = tracks_2d[i, t]
            dx, dy = px * (dW / W), py * (dH / H)
            ix, iy = int(np.clip(int(dx), 0, dW - 2)), int(np.clip(int(dy), 0, dH - 2))
            fx, fy = dx - ix, dy - iy

            d = (depth_m[iy, ix] * (1 - fx) * (1 - fy) +
                 depth_m[iy, ix + 1] * fx * (1 - fy) +
                 depth_m[iy + 1, ix] * (1 - fx) * fy +
                 depth_m[iy + 1, ix + 1] * fx * fy)
            d = np.clip(d, 0.01, auto_max_depth)
            sampled_d[i, t] = d

            lon, lat = erp_pixel_to_lonlat(px, py, W, H)
            X, Y, Z = lonlat_to_xyz(lon, lat, d)
            tracks_3d[i, t] = [X, Y, Z]

    # Camera motion detection & compensation
    tracks_3d_stab, camera_info = detect_and_compensate_camera_motion(
        tracks_2d, tracks_3d, vis, W, H, method="auto"
    )

    # Consistency metrics on stabilized tracks
    reproj_err = compute_reprojection_error(tracks_2d, tracks_3d, vis, W, H)
    smoothness = compute_3d_smoothness(tracks_3d_stab, vis)
    rigidity = compute_rigidity_score(tracks_3d_stab, vis)

    # Save annotations
    np.save(os.path.join(out_dir, "tracks_2d.npy"), tracks_2d.astype(np.float16))
    np.save(os.path.join(out_dir, "tracks_3d.npy"), tracks_3d_stab.astype(np.float16))
    np.save(os.path.join(out_dir, "visibility.npy"), vis.astype(np.float16))

    if not camera_info["is_static"]:
        np.save(os.path.join(out_dir, "tracks_3d_raw.npy"), tracks_3d.astype(np.float16))
        if "poses" in camera_info:
            poses_arr = np.array([
                np.hstack([p['R'].ravel(), p['t']])
                for p in camera_info["poses"]
            ], dtype=np.float32)
            np.save(os.path.join(out_dir, "camera_poses.npy"), poses_arr)

    valid_re = reproj_err[~np.isnan(reproj_err)]
    valid_sm = smoothness[~np.isnan(smoothness)]
    valid_rg = rigidity[~np.isnan(rigidity)]

    meta = {
        "video_resolution": [W, H],
        "n_frames": T,
        "n_tracks": int(N),
        "grid_size": grid_size,
        "max_depth_m": float(auto_max_depth),
        "camera_motion": {
            "method": str(camera_info["method"]),
            "is_static": bool(camera_info["is_static"]),
            "inlier_ratio": float(camera_info["inlier_ratio"]) if camera_info.get("inlier_ratio") is not None else None,
        },
        "reproj_error_median_px": float(np.median(valid_re)) if len(valid_re) else None,
        "smoothness_median_m": float(np.median(valid_sm)) if len(valid_sm) else None,
        "rigidity_median_m": float(np.median(valid_rg)) if len(valid_rg) else None,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return meta


def main():
    parser = argparse.ArgumentParser(
        description="Generate geometry annotations (depth + tracks) for panoramic video datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io = parser.add_argument_group("I/O")
    io.add_argument("--data_dir", type=str, default=TRAIN_DATA_DIR,
                    help="Dataset root (must contain videos/ and optionally train.csv)")
    io.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                    help="Where to save annotations (one subdir per video)")
    io.add_argument("--split", type=str, default="train", choices=["train", "val", "all"],
                    help="Which split CSV to use for video list")

    model = parser.add_argument_group("Model / quality")
    model.add_argument("--gpu", type=str, default="0",
                       help="GPU ID for DAP + CoTracker inference")
    model.add_argument("--max_frames", type=int, default=20,
                       help="Frames to sample per video (uniform)")
    model.add_argument("--grid_size", type=int, default=30,
                       help="CoTracker grid size (N=grid_size^2 tracks)")
    model.add_argument("--max_depth", type=float, default=100.0,
                       help="Max depth clamp in meters")

    dist = parser.add_argument_group("Multi-GPU sharding")
    dist.add_argument("--total_shards", type=int, default=1,
                      help="Total shards for multi-GPU (1=single GPU)")
    dist.add_argument("--shard_id", type=int, default=0,
                      help="This shard index (0-based)")
    dist.add_argument("--resume", action="store_true",
                      help="Skip videos that already have meta.json")
    args = parser.parse_args()

    # Load video list — supports CSV split or direct directory scan
    video_dir = os.path.join(args.data_dir, "videos")
    if not os.path.isdir(video_dir):
        # Fallback: data_dir itself contains .mp4 files (flat layout)
        video_dir = args.data_dir

    if args.split == "all":
        video_names = sorted([f.replace(".mp4", "") for f in os.listdir(video_dir) if f.endswith(".mp4")])
    else:
        csv_path = os.path.join(args.data_dir, f"{args.split}.csv")
        if os.path.exists(csv_path):
            with open(csv_path) as f:
                lines = f.readlines()[1:]  # skip header
                video_names = [l.strip() for l in lines if l.strip()]
        else:
            print(f"  CSV not found: {csv_path}, falling back to --split all")
            video_names = sorted([f.replace(".mp4", "") for f in os.listdir(video_dir) if f.endswith(".mp4")])

    # Shard
    video_names = sorted(video_names)
    if args.total_shards > 1:
        video_names = [v for i, v in enumerate(video_names) if i % args.total_shards == args.shard_id]

    print(f"{'='*60}")
    print(f"  Batch Annotation Generator")
    print(f"{'='*60}")
    print(f"  GPU: {args.gpu}")
    print(f"  Videos: {len(video_names)} ({args.split})")
    print(f"  Shard: {args.shard_id}/{args.total_shards}")
    print(f"  Max frames/clip: {args.max_frames}")
    print(f"  Grid size: {args.grid_size}")
    print(f"  Output: {args.output_dir}")
    print(f"  Resume: {args.resume}")
    print(f"{'='*60}\n")

    # Load models once
    print("Loading DAP model...")
    dap_config = os.path.join(DAP_ROOT, "config", "infer.yaml")
    dap_model, dap_device = load_dap_model(dap_config, gpu=args.gpu)

    print("Loading CoTracker3 model...")
    ct_model = load_cotracker_model(device=dap_device)

    # Process
    os.makedirs(args.output_dir, exist_ok=True)
    results = {"processed": 0, "skipped": 0, "failed": 0, "static": 0, "dynamic": 0}
    failed_videos = []
    t_start = time.time()

    for idx, vname in enumerate(tqdm(video_names, desc="Processing videos")):
        video_path = os.path.join(video_dir, vname + ".mp4")
        out_dir = os.path.join(args.output_dir, vname)

        # Resume check
        if args.resume and os.path.exists(os.path.join(out_dir, "meta.json")):
            results["skipped"] += 1
            continue

        if not os.path.exists(video_path):
            # Try resolving symlink
            real_path = os.path.realpath(video_path)
            if not os.path.exists(real_path):
                tqdm.write(f"  SKIP {vname}: file not found")
                results["failed"] += 1
                failed_videos.append(vname)
                continue
            video_path = real_path

        try:
            meta = process_single_video(
                video_path, out_dir, dap_model, dap_device, ct_model,
                grid_size=args.grid_size, max_frames=args.max_frames,
                max_depth=args.max_depth,
            )
            if meta is None:
                results["failed"] += 1
                failed_videos.append(vname)
                continue

            results["processed"] += 1
            if meta["camera_motion"]["is_static"]:
                results["static"] += 1
            else:
                results["dynamic"] += 1

            if (results["processed"] % 50) == 0:
                elapsed = time.time() - t_start
                rate = results["processed"] / elapsed
                remaining = (len(video_names) - idx - 1) / rate if rate > 0 else 0
                tqdm.write(
                    f"  [{results['processed']}/{len(video_names)}] "
                    f"{rate:.1f} vid/s, ETA {remaining/3600:.1f}h"
                )

        except Exception as e:
            tqdm.write(f"  FAIL {vname}: {e}")
            results["failed"] += 1
            failed_videos.append(vname)

    # Summary
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Batch Annotation Complete!")
    print(f"{'='*60}")
    print(f"  Processed: {results['processed']}")
    print(f"  Skipped:   {results['skipped']}")
    print(f"  Failed:    {results['failed']}")
    print(f"  Static:    {results['static']}  Dynamic: {results['dynamic']}")
    print(f"  Time:      {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    if results["processed"] > 0:
        print(f"  Rate:      {results['processed']/elapsed:.1f} vid/s")
        print(f"  Avg:       {elapsed/results['processed']:.1f}s/vid")
    print(f"{'='*60}")

    # Save summary
    summary = {**results, "elapsed_s": elapsed, "failed_videos": failed_videos}
    summary_path = os.path.join(args.output_dir, f"summary_shard{args.shard_id}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary → {summary_path}")

    # Disk usage
    os.system(f"du -sh {args.output_dir}")


if __name__ == "__main__":
    main()
