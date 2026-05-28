"""Generate track_is_static.npy for all existing annotation folders.

Lightweight post-processing — does NOT re-run DAP or CoTracker.
Uses existing tracks_3d.npy (stabilized) + tracks_2d.npy + visibility.npy
to compute rigidity scores and classify each track as static or dynamic.

For folders with camera_poses.npy (dynamic camera), also re-computes
camera RANSAC inliers as a second signal.

Usage:
    python DAP/test/generate_static_masks.py --ann_dir $PANO_DATA_ROOT/annotations
"""

import argparse
import json
import os
import sys

import numpy as np
from pathlib import Path


def compute_rigidity_score(tracks_3d, vis, k=8):
    """Identical to pano_trajectory_pipeline.compute_rigidity_score."""
    from scipy.spatial import cKDTree

    N, T, _ = tracks_3d.shape
    ref_pts = np.zeros((N, 3))
    for i in range(N):
        for t in range(T):
            if vis[i, t] > 0.5 and not np.any(np.isnan(tracks_3d[i, t])):
                ref_pts[i] = tracks_3d[i, t]
                break

    tree = cKDTree(ref_pts)
    _, nn_idx = tree.query(ref_pts, k=k + 1)
    nn_idx = nn_idx[:, 1:]

    rigidity = np.full(N, np.nan)
    for i in range(N):
        dists_over_time = []
        neighbours = nn_idx[i]
        for t in range(T):
            if vis[i, t] < 0.5 or np.any(np.isnan(tracks_3d[i, t])):
                continue
            all_ok = all(
                vis[j, t] > 0.5 and not np.any(np.isnan(tracks_3d[j, t]))
                for j in neighbours
            )
            if not all_ok:
                continue
            dists = [np.linalg.norm(tracks_3d[i, t] - tracks_3d[j, t])
                     for j in neighbours]
            dists_over_time.append(dists)

        if len(dists_over_time) >= 3:
            arr = np.array(dists_over_time)
            rigidity[i] = arr.std(axis=0).mean()

    return rigidity


def compute_3d_displacement(tracks_3d, vis):
    """Total 3D displacement of each track (first visible → last visible).
    Computed on stabilized tracks so camera motion is already removed."""
    N, T, _ = tracks_3d.shape
    disp = np.zeros(N, dtype=np.float32)

    for i in range(N):
        first_t, last_t = -1, -1
        for t in range(T):
            if vis[i, t] > 0.5 and not np.any(np.isnan(tracks_3d[i, t])):
                if first_t < 0:
                    first_t = t
                last_t = t
        if first_t >= 0 and last_t > first_t:
            disp[i] = np.linalg.norm(tracks_3d[i, last_t] - tracks_3d[i, first_t])

    return disp


def erp_pixel_to_bearing(u, v, W, H):
    """Convert ERP pixel coords to unit bearing vectors."""
    theta = (u / W) * 2 * np.pi
    phi = (v / H) * np.pi
    bx = np.sin(phi) * np.sin(theta)
    by = -np.cos(phi)
    bz = np.sin(phi) * np.cos(theta)
    return bx, by, bz


def compute_camera_inlier_mask(tracks_2d, vis, W, H, camera_poses):
    """Re-compute inlier mask using saved camera poses.
    Points whose 2D motion is consistent with the camera pose = background."""
    N, T, _ = tracks_2d.shape
    inlier_scores = np.zeros(N, dtype=np.float32)
    vis_count = np.zeros(N, dtype=np.float32)

    ref_frame = 0
    ref_bx, ref_by, ref_bz = erp_pixel_to_bearing(
        tracks_2d[:, ref_frame, 0], tracks_2d[:, ref_frame, 1], W, H
    )
    ref_bearings = np.stack([ref_bx, ref_by, ref_bz], axis=-1)

    thresh_rad = np.deg2rad(3.0)

    for t in range(T):
        if t == ref_frame:
            continue

        R = camera_poses[t, :9].reshape(3, 3)

        valid = (vis[:, ref_frame] > 0.5) & (vis[:, t] > 0.5)
        if valid.sum() < 2:
            continue

        cur_bx, cur_by, cur_bz = erp_pixel_to_bearing(
            tracks_2d[:, t, 0], tracks_2d[:, t, 1], W, H
        )
        cur_bearings = np.stack([cur_bx, cur_by, cur_bz], axis=-1)

        rotated = (R @ ref_bearings.T).T
        angular_err = np.arccos(np.clip(
            np.sum(rotated * cur_bearings, axis=-1), -1, 1
        ))

        is_inlier = (angular_err < thresh_rad) & valid
        inlier_scores += is_inlier.astype(np.float32)
        vis_count += valid.astype(np.float32)

    vis_count = np.clip(vis_count, 1, None)
    return inlier_scores / vis_count


def process_folder(ann_dir):
    """Generate track_is_static.npy for a single annotation folder."""
    meta_path = os.path.join(ann_dir, "meta.json")
    if not os.path.exists(meta_path):
        return None

    tracks_3d_path = os.path.join(ann_dir, "tracks_3d.npy")
    vis_path = os.path.join(ann_dir, "visibility.npy")
    tracks_2d_path = os.path.join(ann_dir, "tracks_2d.npy")

    if not all(os.path.exists(p) for p in [tracks_3d_path, vis_path, tracks_2d_path]):
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    tracks_3d = np.load(tracks_3d_path).astype(np.float32)
    vis = np.load(vis_path).astype(np.float32)
    tracks_2d = np.load(tracks_2d_path).astype(np.float32)
    N = tracks_3d.shape[0]

    W, H = meta["video_resolution"]

    # Signal 1: Rigidity (on stabilized tracks)
    rigidity = compute_rigidity_score(tracks_3d, vis)
    valid_rig = rigidity[~np.isnan(rigidity)]
    rig_thresh = np.median(valid_rig) * 2.0 if len(valid_rig) > 0 else 0.1
    is_rigid = np.where(np.isnan(rigidity), True, rigidity < rig_thresh)

    # Signal 2: 3D displacement (on stabilized tracks, camera already compensated)
    disp_3d = compute_3d_displacement(tracks_3d, vis)
    valid_disp = disp_3d[disp_3d > 0]
    disp_thresh = np.median(valid_disp) * 3.0 if len(valid_disp) > 5 else 0.5
    is_low_disp = disp_3d < disp_thresh

    # Signal 3: Camera RANSAC inlier (only for dynamic cameras)
    cam_info = meta.get("camera_motion", {})
    is_static_cam = cam_info.get("is_static", True)
    poses_path = os.path.join(ann_dir, "camera_poses.npy")

    if not is_static_cam and os.path.exists(poses_path):
        camera_poses = np.load(poses_path).astype(np.float32)
        inlier_ratio = compute_camera_inlier_mask(tracks_2d, vis, W, H, camera_poses)
        is_inlier = inlier_ratio > 0.5
    else:
        is_inlier = np.ones(N, dtype=bool)

    # Combine: static = rigid AND low 3D displacement AND camera inlier
    track_is_static = (is_rigid & is_low_disp & is_inlier).astype(np.float32)

    # Save
    np.save(os.path.join(ann_dir, "track_is_static.npy"), track_is_static)
    if not os.path.exists(os.path.join(ann_dir, "rigidity.npy")):
        np.save(os.path.join(ann_dir, "rigidity.npy"), rigidity.astype(np.float32))

    n_static = int(track_is_static.sum())
    return n_static, N


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_dir", type=str, required=True,
                        help="Root annotation directory containing per-video folders")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing track_is_static.npy")
    args = parser.parse_args()

    folders = sorted([
        d for d in os.listdir(args.ann_dir)
        if os.path.isdir(os.path.join(args.ann_dir, d)) and d != "summary_shard0.json"
    ])

    print(f"Processing {len(folders)} annotation folders...")

    total_static, total_tracks = 0, 0
    skipped, errors = 0, 0

    for i, folder in enumerate(folders):
        ann_path = os.path.join(args.ann_dir, folder)

        if not args.force and os.path.exists(os.path.join(ann_path, "track_is_static.npy")):
            skipped += 1
            continue

        try:
            result = process_folder(ann_path)
            if result is None:
                skipped += 1
                continue
            n_static, n_total = result
            total_static += n_static
            total_tracks += n_total
        except Exception as e:
            print(f"  ERROR {folder}: {e}")
            errors += 1
            continue

        if (i + 1) % 100 == 0:
            pct = 100 * total_static / max(total_tracks, 1)
            print(f"  [{i+1}/{len(folders)}] processed, "
                  f"static: {total_static}/{total_tracks} ({pct:.0f}%)")

    print(f"\nDone!")
    print(f"  Processed: {len(folders) - skipped - errors}")
    print(f"  Skipped:   {skipped}")
    print(f"  Errors:    {errors}")
    if total_tracks > 0:
        print(f"  Static:    {total_static}/{total_tracks} "
              f"({100*total_static/total_tracks:.1f}%)")


if __name__ == "__main__":
    main()
