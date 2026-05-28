#!/usr/bin/env python3
"""
Full batch annotation regeneration: depth + tracks + static masks.

Regenerates ALL annotations from scratch for each video:
  1. Frame extraction (4fps, 1024×512 → 20 frames)
  2. DAP depth inference
  3. CoTracker tracking (grid_size=100 → 10,000 points)
  4. 3D lifting + camera motion detection
  5. Static track classification (geometric)
  6. Save all annotations

NOTE: semantic_static.npy (GroundingSAM) must be regenerated separately
      in the gsam conda environment after this script finishes.

Usage:
  conda activate cosmos
  cd $REPO_ROOT/DAP
  python test/regenerate_annotations.py \
      --video_root $PANO_DATA_ROOT/cosmos_pano_train/videos \
      --ann_root   $PANO_DATA_ROOT/annotations \
      --grid_size 100 --extract_fps 4

Resume from a specific video:
  python test/regenerate_annotations.py ... --resume_from web360_100500
"""

import os, sys, json, time, argparse, glob, subprocess, shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pano_trajectory_pipeline import (
    run_cotracker,
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
    MAX_DEPTH_SCALE,
)

_DAP_MODEL = None
_DAP_DEVICE = None


def _ensure_dap_model(config_path):
    """Load DAP model once and cache it."""
    global _DAP_MODEL, _DAP_DEVICE
    if _DAP_MODEL is not None:
        return _DAP_MODEL, _DAP_DEVICE

    import yaml, torch

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    infer_dir = os.path.dirname(os.path.abspath(__file__))
    if infer_dir not in sys.path:
        sys.path.insert(0, infer_dir)

    saved_cwd = os.getcwd()
    os.chdir(project_root)
    try:
        from infer import load_model
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        _DAP_MODEL, _DAP_DEVICE = load_model(config)
    finally:
        os.chdir(saved_cwd)

    return _DAP_MODEL, _DAP_DEVICE


def run_dap_on_frames(frames_dir, depth_out_dir, config_path):
    """Run DAP depth on all frames. Always overwrites."""
    import cv2
    from infer import infer_raw

    model, device = _ensure_dap_model(config_path)

    npy_dir = os.path.join(depth_out_dir, "depth")
    os.makedirs(npy_dir, exist_ok=True)

    frame_files = sorted_files(frames_dir, (".png", ".jpg", ".jpeg"))

    for idx, ff in enumerate(frame_files):
        img_bgr = cv2.imread(os.path.join(frames_dir, ff))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pred = infer_raw(model, device, img_rgb)

        stem = f"{idx:04d}"
        np.save(os.path.join(npy_dir, stem + ".npy"), pred)

    return npy_dir


_COTRACKER_MODEL = None


def _ensure_cotracker(device="cuda"):
    """Load CoTracker once and cache it."""
    global _COTRACKER_MODEL
    if _COTRACKER_MODEL is not None:
        return _COTRACKER_MODEL

    import torch
    from cotracker.predictor import CoTrackerPredictor

    ckpt = os.path.expanduser("~/.cache/torch/hub/checkpoints/scaled_offline.pth")
    _COTRACKER_MODEL = CoTrackerPredictor(checkpoint=ckpt, window_len=60, v2=False)
    _COTRACKER_MODEL = _COTRACKER_MODEL.to(device)
    _COTRACKER_MODEL.eval()
    print(f"   CoTracker3 loaded from {ckpt}")
    return _COTRACKER_MODEL


def run_cotracker_cached(frames_rgb, grid_size, pad_frac=0.25, device="cuda"):
    """Run CoTracker using cached model (no reload per video)."""
    import torch
    from pano_trajectory_pipeline import circular_pad_video, unwrap_tracks

    model = _ensure_cotracker(device)

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


def process_one(video_path, ann_dir, grid_size, extract_fps, config_path, device):
    """Full pipeline for a single video."""
    name = os.path.basename(ann_dir)
    frames_dir = os.path.join("/tmp/pano_regen_frames", name)

    # 1. Extract frames
    if os.path.isdir(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)
    extract_frames(video_path, frames_dir, fps=extract_fps, width=1024, height=512)

    frame_files = sorted_files(frames_dir, (".png", ".jpg", ".jpeg"))
    T = len(frame_files)
    if T < 2:
        print(f"  [SKIP] {name}: only {T} frames")
        return False

    frame_paths = [os.path.join(frames_dir, f) for f in frame_files]
    frames_rgb = [load_rgb(p) for p in frame_paths]
    H, W = frames_rgb[0].shape[:2]

    # 2. DAP depth (always regenerate)
    depth_npy_dir = os.path.join(ann_dir, "depth")
    if os.path.isdir(depth_npy_dir):
        shutil.rmtree(depth_npy_dir)
    run_dap_on_frames(frames_dir, ann_dir, config_path)

    depth_files_list = sorted_files(depth_npy_dir, ".npy")
    depth_paths = [os.path.join(depth_npy_dir, f) for f in depth_files_list[:T]]

    # Auto max_depth
    sample = np.concatenate([
        np.load(dp).ravel() * MAX_DEPTH_SCALE
        for dp in depth_paths[:5]
    ])
    max_depth = float(np.percentile(sample, 99))

    # 3. CoTracker (cached model)
    tracks_2d, vis = run_cotracker_cached(frames_rgb, grid_size=grid_size, device=device)
    tracks_2d, vis = merge_wraparound_tracks(tracks_2d, vis, W)
    N = tracks_2d.shape[0]

    # 4. 3D lifting
    tracks_3d, sampled_depth = lift_tracks_to_3d(
        tracks_2d, vis, depth_paths, W, H, max_depth=max_depth
    )

    # 5. Camera motion
    tracks_3d_stab, camera_info = detect_and_compensate_camera_motion(
        tracks_2d, tracks_3d, vis, W, H, method="auto"
    )

    # 6. Metrics
    reproj_err = compute_reprojection_error(tracks_2d, tracks_3d, vis, W, H)
    smoothness = compute_3d_smoothness(tracks_3d_stab, vis)
    rigidity = compute_rigidity_score(tracks_3d_stab, vis)

    # 7. Static classification
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
    n_static = int(track_is_static.sum())

    # 8. Meta
    valid_re = reproj_err[~np.isnan(reproj_err)]
    valid_sm = smoothness[~np.isnan(smoothness)]
    valid_rg = rigidity[~np.isnan(rigidity)]

    meta = {
        "video_resolution": [W, H],
        "n_frames": T,
        "n_tracks": int(N),
        "grid_size": grid_size,
        "max_depth_m": max_depth,
        "depth_scale": MAX_DEPTH_SCALE,
        "extract_fps": extract_fps,
        "camera_motion": {
            "method": str(camera_info["method"]),
            "is_static": bool(camera_info["is_static"]),
            "inlier_ratio": float(camera_info["inlier_ratio"]) if camera_info.get("inlier_ratio") is not None else None,
        },
        "reproj_error_median_px": float(np.median(valid_re)) if len(valid_re) else None,
        "smoothness_median_m": float(np.median(valid_sm)) if len(valid_sm) else None,
        "rigidity_median_m": float(np.median(valid_rg)) if len(valid_rg) else None,
        "n_static_tracks": n_static,
    }

    # 9. Save
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

    # Remove stale semantic_static (will be regenerated by GroundingSAM)
    sem_path = os.path.join(ann_dir, "semantic_static.npy")
    if os.path.exists(sem_path):
        os.remove(sem_path)

    # Cleanup temp frames
    shutil.rmtree(frames_dir, ignore_errors=True)

    print(f"  ✓ {name}: {N} tracks, {T} frames, {n_static} static, max_depth={max_depth:.1f}m")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Full batch annotation regeneration (depth + tracks + static)"
    )
    _data_root = os.environ.get(
        "PANO_DATA_ROOT", os.path.join(os.path.expanduser("~"), "pano_video_data")
    )
    parser.add_argument("--video_root", type=str,
                        default=os.path.join(_data_root, "cosmos_pano_train", "videos"))
    parser.add_argument("--ann_root", type=str,
                        default=os.path.join(_data_root, "annotations"))
    parser.add_argument("--grid_size", type=int, default=100,
                        help="CoTracker grid density (100 → 10000 pts)")
    parser.add_argument("--extract_fps", type=int, default=4,
                        help="Frame extraction fps (4fps × 5s = 20 frames)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume from this folder name (skip earlier ones)")
    args = parser.parse_args()

    dap_config = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "infer.yaml",
    )
    assert os.path.exists(dap_config), f"DAP config not found: {dap_config}"

    # Find all videos
    video_files = sorted(glob.glob(os.path.join(args.video_root, "web360_*.mp4")))
    print(f"Found {len(video_files)} videos")
    print(f"Grid size: {args.grid_size} → {args.grid_size**2} points")
    print(f"Extract fps: {args.extract_fps}")

    if args.resume_from:
        skip = True
        filtered = []
        for v in video_files:
            name = Path(v).stem
            if name == args.resume_from:
                skip = False
            if not skip:
                filtered.append(v)
        video_files = filtered
        print(f"Resuming from {args.resume_from}, {len(video_files)} remaining")

    # Pre-load models
    print("\n── Loading models ──")
    _ensure_dap_model(dap_config)
    _ensure_cotracker(args.device)
    print("── Models loaded ──\n")

    done = 0
    failed = 0
    t_start = time.time()

    for i, video_path in enumerate(video_files):
        name = Path(video_path).stem
        ann_dir = os.path.join(args.ann_root, name)
        os.makedirs(ann_dir, exist_ok=True)

        elapsed = time.time() - t_start
        rate = done / elapsed if elapsed > 0 and done > 0 else 0
        eta = (len(video_files) - i) / rate / 3600 if rate > 0 else 0
        print(f"\n[{i+1}/{len(video_files)}] {name}  "
              f"(done={done}, fail={failed}, {rate:.2f} vid/s, ETA {eta:.1f}h)")

        try:
            ok = process_one(
                video_path, ann_dir, args.grid_size,
                args.extract_fps, dap_config, args.device,
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
    print(f"All done! {done} succeeded, {failed} failed in {elapsed_total/3600:.1f}h")
    print(f"Average: {elapsed_total/max(done,1):.1f}s per video")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
