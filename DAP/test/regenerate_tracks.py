#!/usr/bin/env python3
"""
Batch-regenerate track annotations with higher density (grid_size=100).

Reuses existing depth maps — only re-runs:
  1. CoTracker (grid_size=100)
  2. 3D lifting
  3. Camera motion detection
  4. Static track classification

Usage:
  conda activate cosmos
  cd $REPO_ROOT/DAP
  python test/regenerate_tracks.py \
      --ann_root $PANO_DATA_ROOT/annotations \
      --video_root $PANO_DATA_ROOT/cosmos_pano_train/videos \
      --grid_size 100 --skip_viz
"""

import os, sys, json, time, argparse, glob, subprocess, shutil
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pano_trajectory_pipeline import (
    merge_wraparound_tracks,
    lift_tracks_to_3d,
    detect_and_compensate_camera_motion,
    compute_reprojection_error,
    compute_3d_smoothness,
    compute_rigidity_score,
    save_annotations,
    load_rgb,
    sorted_files,
    extract_frames,
    circular_pad_video,
    unwrap_tracks,
    MAX_DEPTH_SCALE,
)

_COTRACKER_MODEL = None


def get_cotracker(device="cuda"):
    """Load CoTracker once and reuse across all videos."""
    global _COTRACKER_MODEL
    if _COTRACKER_MODEL is not None:
        return _COTRACKER_MODEL
    from cotracker.predictor import CoTrackerPredictor
    ckpt = os.path.expanduser("~/.cache/torch/hub/checkpoints/scaled_offline.pth")
    print(f"   Loading CoTracker3 from {ckpt}")
    _COTRACKER_MODEL = CoTrackerPredictor(checkpoint=ckpt, window_len=60, v2=False)
    _COTRACKER_MODEL = _COTRACKER_MODEL.to(device)
    _COTRACKER_MODEL.eval()
    return _COTRACKER_MODEL


def run_cotracker_cached(frames_rgb, grid_size, pad_frac=0.25, device="cuda"):
    """Run CoTracker with cached model — no reload per video."""
    model = get_cotracker(device)
    H, W = frames_rgb[0].shape[:2]
    padded, pad_px = circular_pad_video(frames_rgb, pad_frac)

    vid_np = np.stack(padded, axis=0)
    vid_t = torch.from_numpy(vid_np).permute(0, 3, 1, 2)
    vid_t = vid_t.unsqueeze(0).float().to(device)

    with torch.no_grad():
        pred_tracks, pred_vis = model(vid_t, grid_size=grid_size)

    tracks_np = pred_tracks[0].cpu().numpy().transpose(1, 0, 2)
    vis_np = pred_vis[0].cpu().numpy().T

    tracks_2d, vis = unwrap_tracks(tracks_np, vis_np, W, pad_px)
    return tracks_2d, vis


def process_one(ann_dir, video_path, grid_size, device, skip_viz, tmp_frames_root):
    """Regenerate track annotations for a single video."""
    name = os.path.basename(ann_dir)
    depth_dir = os.path.join(ann_dir, "depth")

    depth_files_list = sorted_files(depth_dir, ".npy")
    if len(depth_files_list) == 0:
        print(f"  [SKIP] {name}: no depth files")
        return False

    T = len(depth_files_list)
    depth_paths = [os.path.join(depth_dir, f) for f in depth_files_list]

    frames_dir = os.path.join(tmp_frames_root, name)
    os.makedirs(frames_dir, exist_ok=True)

    existing_frames = sorted_files(frames_dir, (".png", ".jpg", ".jpeg"))
    if len(existing_frames) < T:
        if not os.path.exists(video_path):
            print(f"  [SKIP] {name}: video not found at {video_path}")
            return False
        # All videos are 5s, depth has 20 frames → fps=4
        extract_fps = 4
        # Clear any partial/wrong-count frames so extract_frames doesn't skip
        for f in sorted_files(frames_dir, (".png", ".jpg", ".jpeg")):
            os.remove(os.path.join(frames_dir, f))
        extract_frames(video_path, frames_dir, fps=extract_fps, width=1024, height=512)
        existing_frames = sorted_files(frames_dir, (".png", ".jpg", ".jpeg"))
        if len(existing_frames) < T:
            print(f"  [WARN] {name}: extracted {len(existing_frames)} frames but depth has {T}")
            T = min(T, len(existing_frames))
            depth_paths = depth_paths[:T]

    frame_files = existing_frames[:T]
    frame_paths = [os.path.join(frames_dir, f) for f in frame_files]
    frames_rgb = [load_rgb(p) for p in frame_paths]
    H, W = frames_rgb[0].shape[:2]

    # Auto max_depth
    sample = np.concatenate([
        np.load(dp).ravel() * MAX_DEPTH_SCALE
        for dp in depth_paths[:5]
    ])
    max_depth = float(np.percentile(sample, 99))

    # 1. CoTracker (cached model, no reload)
    tracks_2d, vis = run_cotracker_cached(frames_rgb, grid_size=grid_size, device=device)
    tracks_2d, vis = merge_wraparound_tracks(tracks_2d, vis, W)
    N = tracks_2d.shape[0]
    print(f"  {name}: {N} tracks × {T} frames")

    # 2. 3D lifting
    tracks_3d, sampled_depth = lift_tracks_to_3d(
        tracks_2d, vis, depth_paths, W, H, max_depth=max_depth
    )

    # 3. Camera motion
    tracks_3d_stab, camera_info = detect_and_compensate_camera_motion(
        tracks_2d, tracks_3d, vis, W, H, method="auto"
    )

    # 4. Metrics
    reproj_err = compute_reprojection_error(tracks_2d, tracks_3d, vis, W, H)
    smoothness = compute_3d_smoothness(tracks_3d_stab, vis)
    rigidity = compute_rigidity_score(tracks_3d_stab, vis)

    # 5. Static classification
    inlier_ratio_per_track = np.zeros(N, dtype=np.float32)
    if "inlier_masks" in camera_info:
        inlier_masks_cam = camera_info["inlier_masks"]
        vis_count = (vis > 0.5).sum(axis=1).clip(min=1)
        inlier_ratio_per_track = inlier_masks_cam.sum(axis=1).astype(np.float32) / vis_count
    elif camera_info.get("is_static", True):
        inlier_ratio_per_track[:] = 1.0

    rigidity_thresh = np.nanmedian(rigidity) * 2.0 if np.any(~np.isnan(rigidity)) else 0.1
    is_rigid = np.where(np.isnan(rigidity), True, rigidity < rigidity_thresh)
    is_inlier = inlier_ratio_per_track > 0.5
    track_is_static = (is_rigid & is_inlier).astype(np.float32)

    # 6. Build meta
    valid_re = reproj_err[~np.isnan(reproj_err)]
    valid_sm = smoothness[~np.isnan(smoothness)]
    valid_rg = rigidity[~np.isnan(rigidity)]

    meta = {
        "video_resolution": [W, H],
        "n_frames": T,
        "n_tracks": int(N),
        "grid_size": grid_size,
        "max_depth_m": max_depth,
        "camera_motion": {
            "method": str(camera_info["method"]),
            "is_static": bool(camera_info["is_static"]),
            "inlier_ratio": float(camera_info["inlier_ratio"]) if camera_info.get("inlier_ratio") is not None else None,
        },
        "reproj_error_median_px": float(np.median(valid_re)) if len(valid_re) else None,
        "smoothness_median_m": float(np.median(valid_sm)) if len(valid_sm) else None,
        "rigidity_median_m": float(np.median(valid_rg)) if len(valid_rg) else None,
        "n_static_tracks": int(track_is_static.sum()),
    }

    # 7. Save (overwrites existing)
    save_annotations(
        ann_dir, tracks_2d, tracks_3d_stab, vis,
        sampled_depth, reproj_err, smoothness, rigidity, meta
    )
    np.save(os.path.join(ann_dir, "track_is_static.npy"),
            track_is_static.astype(np.float32))

    if not camera_info["is_static"]:
        np.save(os.path.join(ann_dir, "tracks_3d_raw.npy"),
                tracks_3d.astype(np.float32))
        if "poses" in camera_info:
            poses_arr = np.array([
                np.hstack([p['R'].ravel(), p['t']])
                for p in camera_info["poses"]
            ], dtype=np.float32)
            np.save(os.path.join(ann_dir, "camera_poses.npy"), poses_arr)

    # Remove old semantic_static (will be regenerated separately by GroundingSAM)
    sem_path = os.path.join(ann_dir, "semantic_static.npy")
    if os.path.exists(sem_path):
        os.remove(sem_path)

    # Cleanup temp frames to save disk space
    shutil.rmtree(frames_dir, ignore_errors=True)

    n_static = int(track_is_static.sum())
    print(f"  -> {N} tracks, {n_static} static ({100*n_static/max(N,1):.0f}%), max_depth={max_depth:.1f}m")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Batch-regenerate track annotations with higher density"
    )
    _data_root = os.environ.get(
        "PANO_DATA_ROOT", os.path.join(os.path.expanduser("~"), "pano_video_data")
    )
    parser.add_argument("--ann_root", type=str,
                        default=os.path.join(_data_root, "annotations"))
    parser.add_argument("--video_root", type=str,
                        default=os.path.join(_data_root, "cosmos_pano_train", "videos"))
    parser.add_argument("--grid_size", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_viz", action="store_true", default=True)
    parser.add_argument("--tmp_frames", type=str,
                        default="/tmp/pano_frames_cache",
                        help="Temp dir for extracted frames")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume from this folder name (skip earlier ones)")
    args = parser.parse_args()

    ann_dirs = sorted(glob.glob(os.path.join(args.ann_root, "web360_*")))
    print(f"Found {len(ann_dirs)} annotation folders")
    print(f"Grid size: {args.grid_size} → {args.grid_size**2} points")

    if args.resume_from:
        skip = True
        filtered = []
        for d in ann_dirs:
            if os.path.basename(d) == args.resume_from:
                skip = False
            if not skip:
                filtered.append(d)
        ann_dirs = filtered
        print(f"Resuming from {args.resume_from}, {len(ann_dirs)} remaining")

    # Pre-load CoTracker model once
    print("\n── Pre-loading CoTracker3 ──")
    get_cotracker(args.device)
    print("── Ready ──\n")

    done = 0
    failed = 0
    t_start = time.time()

    for i, ann_dir in enumerate(ann_dirs):
        name = os.path.basename(ann_dir)
        video_path = os.path.join(args.video_root, f"{name}.mp4")

        elapsed = time.time() - t_start
        rate = done / elapsed if elapsed > 0 and done > 0 else 0
        eta = (len(ann_dirs) - i) / rate / 3600 if rate > 0 else 0
        print(f"\n[{i+1}/{len(ann_dirs)}] {name}  (done={done}, fail={failed}, "
              f"rate={rate:.2f}/s, ETA={eta:.1f}h)")

        try:
            ok = process_one(
                ann_dir, video_path, args.grid_size,
                args.device, args.skip_viz, args.tmp_frames
            )
            if ok:
                done += 1
            else:
                failed += 1
        except Exception as e:
            import traceback
            print(f"  [ERROR] {name}: {e}")
            traceback.print_exc()
            failed += 1

    elapsed_total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Done! {done} succeeded, {failed} failed in {elapsed_total/3600:.1f}h")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
