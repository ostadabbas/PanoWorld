"""
Panoramic 3D Trajectory Annotation Pipeline
=============================================
End-to-end pipeline from a panoramic video file:
  Step 0  – Extract frames (ffmpeg, configurable fps & resolution)
  Step 0.5– DAP depth inference (per-frame .npy depth maps)
  Step 1  – 2D tracking  (CoTracker3 | SpaTracker | TAPIP3D)
  Step 2  – Wrap-around track merging
  Step 3  – Lift 2D → 3D with DAP depth
  Step 4  – Consistency metrics
  Step 5  – Save annotations
  Step 6  – Visualisations (2D track video, 3D trajectory video + static plot)

Quick start (single video, CoTracker):
  python test/pano_trajectory_pipeline.py --video video_samples/1000050119.mp4

Compare all three trackers:
  python test/pano_trajectory_pipeline.py \
      --video video_samples/1000050207.mp4 \
      --tracker all --grid_size 50 --max_depth 5.0

Advanced (pre-extracted frames + depths):
  python test/pano_trajectory_pipeline.py \
      --frames_dir datasets/video_frames \
      --depth_dir  output/video_1000050115/depth_npy \
      --out_dir    output/video_1000050115/annotations
"""

import os, json, argparse, time, subprocess
import numpy as np
import cv2
import torch
from tqdm import tqdm
from pathlib import Path

# ╭──────────────────────────────────────────────────────────────╮
# │  CONFIG                                                      │
# ╰──────────────────────────────────────────────────────────────╯
MAX_DEPTH_SCALE = 100.0        # DAP convention: npy ∈ [0,1] → metres
ERP_WIDTH_IS_360 = True        # full 360° equirectangular


# ╭──────────────────────────────────────────────────────────────╮
# │  UTILS                                                       │
# ╰──────────────────────────────────────────────────────────────╯

def sorted_files(d, ext):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(ext))


def load_rgb(path):
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_depth(path, scale=MAX_DEPTH_SCALE):
    """Load DAP .npy and convert to metres."""
    d = np.load(path).astype(np.float32)
    if d.max() <= 1.5:          # still in normalised [0,1] range
        d = d * scale
    return d


# ╭──────────────────────────────────────────────────────────────╮
# │  1. CIRCULAR PADDING  (handles left↔right wrap-around)       │
# ╰──────────────────────────────────────────────────────────────╯

def circular_pad_frame(img, pad_frac=0.25):
    """
    Pad an equirectangular image by copying the right strip to the left
    and the left strip to the right.  pad_frac = fraction of width per side.

    Returns: padded_img, pad_px (number of pixels added each side)
    """
    H, W = img.shape[:2]
    pad_px = int(W * pad_frac)
    right_strip = img[:, -pad_px:]        # goes to left
    left_strip  = img[:,  :pad_px]        # goes to right
    if img.ndim == 3:
        padded = np.concatenate([right_strip, img, left_strip], axis=1)
    else:
        padded = np.concatenate([right_strip, img, left_strip], axis=1)
    return padded, pad_px


def circular_pad_video(frames, pad_frac=0.25):
    """Pad a list of RGB arrays. Returns (padded_list, pad_px)."""
    out, pad_px = [], 0
    for f in frames:
        p, pad_px = circular_pad_frame(f, pad_frac)
        out.append(p)
    return out, pad_px


def unwrap_tracks(tracks_2d, vis, W_orig, pad_px):
    """
    Map padded-image coordinates back to original ERP pixel space.
    tracks_2d : (N, T, 2)  — x, y in padded image
    Returns   : (N, T, 2)  — x, y in original image with wrap-around mod W
    """
    tracks = tracks_2d.copy()
    tracks[..., 0] = (tracks[..., 0] - pad_px) % W_orig   # circular mod
    return tracks, vis


# ╭──────────────────────────────────────────────────────────────╮
# │  2. 2D TRACKING  (CoTracker v3 with wrap-around)             │
# ╰──────────────────────────────────────────────────────────────╯

_COTRACKER3_OFFLINE_CKPT = os.path.expanduser(
    "~/.cache/torch/hub/checkpoints/scaled_offline.pth"
)

# Conda environments for third-party trackers
_CONDA_PREFIX = os.path.expanduser("~/miniconda3")
_SPATRACK_PYTHON  = os.path.join(_CONDA_PREFIX, "envs", "spatrack",  "bin", "python")
_TAPIP3D_PYTHON   = os.path.join(_CONDA_PREFIX, "envs", "tapip3d",   "bin", "python")
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def run_cotracker(frames_rgb, grid_size=30, pad_frac=0.25, device="cuda",
                  checkpoint=_COTRACKER3_OFFLINE_CKPT):
    """
    Run CoTracker3 (offline) on circularly-padded panoramic frames.
    Loads directly from a local checkpoint — no internet required.
    Returns unwrapped (N, T, 2) tracks and (N, T) visibility in original coords.
    """
    try:
        from cotracker.predictor import CoTrackerPredictor
    except ImportError:
        raise ImportError(
            "CoTracker not found.  Install with:\n"
            "  pip install git+https://github.com/facebookresearch/co-tracker.git"
        )

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"CoTracker3 checkpoint not found at {checkpoint}\n"
            "Download with:\n"
            "  python -c \"from huggingface_hub import hf_hub_download; "
            "hf_hub_download('facebook/cotracker3', 'scaled_offline.pth', "
            "local_dir='~/.cache/torch/hub/checkpoints')\""
        )

    H, W = frames_rgb[0].shape[:2]
    padded, pad_px = circular_pad_video(frames_rgb, pad_frac)
    print(f"   Circular padding: {W} → {padded[0].shape[1]}  (pad={pad_px}px each side)")

    # Stack into tensor  (1, T, 3, H, W_padded)
    vid_np = np.stack(padded, axis=0)                      # (T, H, W', 3)
    vid_t  = torch.from_numpy(vid_np).permute(0, 3, 1, 2)  # (T, 3, H, W')
    vid_t  = vid_t.unsqueeze(0).float().to(device)          # (1, T, 3, H, W')

    # Load CoTracker3 offline directly from local checkpoint
    print(f"   Loading CoTracker3 offline from: {checkpoint}")
    model = CoTrackerPredictor(checkpoint=checkpoint, window_len=60, v2=False)
    model = model.to(device)
    model.eval()

    print(f"   Running CoTracker3 (grid_size={grid_size}) on {len(frames_rgb)} frames ...")
    with torch.no_grad():
        pred_tracks, pred_vis = model(vid_t, grid_size=grid_size)
        # pred_tracks : (1, T, N, 2)  — x, y
        # pred_vis    : (1, T, N)

    tracks_np = pred_tracks[0].cpu().numpy().transpose(1, 0, 2)  # (N, T, 2)
    vis_np    = pred_vis[0].cpu().numpy().T                       # (N, T)

    # Unwrap from padded → original coordinates
    tracks_2d, vis = unwrap_tracks(tracks_np, vis_np, W, pad_px)
    return tracks_2d, vis


# ╭──────────────────────────────────────────────────────────────╮
# │  2b. FALLBACK: Dense Optical Flow tracker (no CoTracker)     │
# ╰──────────────────────────────────────────────────────────────╯

def run_optflow_tracker(frames_rgb, grid_step=16, pad_frac=0.15):
    """
    Simple Farneback optical-flow tracker with circular padding.
    Useful when CoTracker is not available.
    Returns (N, T, 2) tracks and (N, T) visibility.
    """
    H, W = frames_rgb[0].shape[:2]
    padded, pad_px = circular_pad_video(frames_rgb, pad_frac)
    Wp = padded[0].shape[1]

    # Initialise grid in the *original* region of the padded image
    ys = np.arange(grid_step // 2, H, grid_step)
    xs = np.arange(pad_px + grid_step // 2, pad_px + W, grid_step)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(np.float32)  # (N, 2)
    N = pts.shape[0]
    T = len(frames_rgb)

    all_tracks = np.zeros((N, T, 2), dtype=np.float32)
    all_vis    = np.ones((N, T), dtype=np.float32)
    all_tracks[:, 0] = pts

    prev_gray = cv2.cvtColor(padded[0], cv2.COLOR_RGB2GRAY)

    for t in tqdm(range(1, T), desc="   OptFlow tracking"):
        curr_gray = cv2.cvtColor(padded[t], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=5, winsize=21,
            iterations=5, poly_n=7, poly_sigma=1.5, flags=0
        )
        # Advect each point using bilinear-sampled flow
        for i in range(N):
            x, y = all_tracks[i, t - 1]
            ix, iy = int(round(x)), int(round(y))
            if 0 <= iy < H and 0 <= ix < Wp:
                dx, dy = flow[iy, ix]
                nx, ny = x + dx, y + dy
                # Circular wrap in padded space
                if nx < 0:       nx += Wp
                elif nx >= Wp:   nx -= Wp
                ny = np.clip(ny, 0, H - 1)
                all_tracks[i, t] = [nx, ny]
            else:
                all_tracks[i, t] = all_tracks[i, t - 1]
                all_vis[i, t] = 0.0
        prev_gray = curr_gray

    # Unwrap
    tracks_2d, vis = unwrap_tracks(all_tracks, all_vis, W, pad_px)
    return tracks_2d, vis


# ╭──────────────────────────────────────────────────────────────╮
# │  2b. SPATRACK & TAPIP3D  (subprocess runners)               │
# ╰──────────────────────────────────────────────────────────────╯

def run_spatrack_subprocess(frames_dir, depth_dir, out_dir, grid_size=40,
                             seq_length=12, gpu=0):
    """Run SpaTracker in a subprocess using the spatrack conda env."""
    wrapper = os.path.join(_THIS_DIR, "spatrack_wrapper.py")
    cmd = [
        _SPATRACK_PYTHON, wrapper,
        "--frames_dir", frames_dir,
        "--depth_dir",  depth_dir,
        "--out_dir",    out_dir,
        "--grid_size",  str(grid_size),
        "--seq_length", str(seq_length),
        "--gpu",        str(gpu),
    ]
    print(f"   CMD: {' '.join(cmd)}")
    # Pass CONDA_PREFIX so cupy can find the CUDA path via conda env variables
    env = os.environ.copy()
    env["CONDA_PREFIX"] = os.path.join(_CONDA_PREFIX, "envs", "spatrack")
    env.setdefault("CUDA_HOME", env["CONDA_PREFIX"])
    subprocess.run(cmd, check=True, env=env)
    tracks_2d  = np.load(os.path.join(out_dir, "tracks_2d.npy"))
    visibility = np.load(os.path.join(out_dir, "visibility.npy"))
    return tracks_2d, visibility


def run_tapip3d_subprocess(frames_dir, depth_dir, out_dir, grid_size=32,
                            num_iters=6, gpu=0):
    """Run TAPIP3D in a subprocess using the tapip3d conda env."""
    wrapper = os.path.join(_THIS_DIR, "tapip3d_wrapper.py")
    cmd = [
        _TAPIP3D_PYTHON, wrapper,
        "--frames_dir", frames_dir,
        "--depth_dir",  depth_dir,
        "--out_dir",    out_dir,
        "--grid_size",  str(grid_size),
        "--num_iters",  str(num_iters),
        "--gpu",        str(gpu),
    ]
    print(f"   CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    tracks_2d  = np.load(os.path.join(out_dir, "tracks_2d.npy"))
    visibility = np.load(os.path.join(out_dir, "visibility.npy"))
    return tracks_2d, visibility


# ╭──────────────────────────────────────────────────────────────╮
# │  3. ERP ↔ 3D PROJECTION                                     │
# ╰──────────────────────────────────────────────────────────────╯

def erp_pixel_to_lonlat(x, y, W, H):
    """Pixel (x, y) → (longitude, latitude) in radians.
       Convention: lon ∈ [-π, +π],  lat ∈ [-π/2, +π/2]
       y=0 → top of image → lat = +π/2 (north pole)
    """
    lon = (x / W) * 2 * np.pi - np.pi            # left=-π  right=+π
    lat = np.pi / 2 - (y / H) * np.pi            # top=+π/2  bot=-π/2
    return lon, lat


def lonlat_to_xyz(lon, lat, depth):
    """Spherical → Cartesian.  Y-up convention.
       Forward = +Z, Right = +X, Up = +Y.
    """
    X = depth * np.cos(lat) * np.sin(lon)
    Y = depth * np.sin(lat)
    Z = depth * np.cos(lat) * np.cos(lon)
    return X, Y, Z


def xyz_to_erp_pixel(X, Y, Z, W, H):
    """Cartesian → ERP pixel. Inverse of the above."""
    depth = np.sqrt(X**2 + Y**2 + Z**2)
    lat   = np.arcsin(np.clip(Y / (depth + 1e-8), -1, 1))
    lon   = np.arctan2(X, Z)
    x = ((lon + np.pi) / (2 * np.pi)) * W
    y = ((np.pi / 2 - lat) / np.pi) * H
    return x, y, depth


def lift_tracks_to_3d(tracks_2d, vis, depth_paths, W, H, max_depth=100.0):
    """
    Unproject 2D trajectories to 3D using per-frame depth maps.

    tracks_2d   : (N, T, 2)  x, y pixel coords
    vis         : (N, T)     visibility
    depth_paths : list of T .npy paths
    Returns     : tracks_3d (N, T, 3), sampled_depth (N, T)
    """
    N, T, _ = tracks_2d.shape
    tracks_3d = np.zeros((N, T, 3), dtype=np.float64)
    sampled_d = np.zeros((N, T), dtype=np.float64)

    for t in tqdm(range(T), desc="   Lifting to 3D"):
        depth_m = load_depth(depth_paths[t])
        dH, dW  = depth_m.shape

        for i in range(N):
            if vis[i, t] < 0.5:
                tracks_3d[i, t] = [np.nan, np.nan, np.nan]
                sampled_d[i, t] = np.nan
                continue

            px, py = tracks_2d[i, t]

            # Map track pixel to depth-map pixel (may differ in resolution)
            dx = px * (dW / W)
            dy = py * (dH / H)

            # Bilinear sample depth
            ix, iy = int(dx), int(dy)
            ix = np.clip(ix, 0, dW - 2)
            iy = np.clip(iy, 0, dH - 2)
            fx, fy = dx - ix, dy - iy

            d00 = depth_m[iy,     ix]
            d01 = depth_m[iy,     ix + 1]
            d10 = depth_m[iy + 1, ix]
            d11 = depth_m[iy + 1, ix + 1]
            d = (d00 * (1 - fx) * (1 - fy) +
                 d01 * fx       * (1 - fy) +
                 d10 * (1 - fx) * fy       +
                 d11 * fx       * fy)

            d = np.clip(d, 0.01, max_depth)
            sampled_d[i, t] = d

            lon, lat = erp_pixel_to_lonlat(px, py, W, H)
            X, Y, Z  = lonlat_to_xyz(lon, lat, d)
            tracks_3d[i, t] = [X, Y, Z]

    return tracks_3d, sampled_d


# ╭──────────────────────────────────────────────────────────────╮
# │  3b. CAMERA MOTION DETECTION & COMPENSATION (PnP / SVD)      │
# ╰──────────────────────────────────────────────────────────────╯

def erp_pixel_to_bearing(x, y, W, H):
    """Convert ERP pixel(s) to unit bearing vector(s) on the sphere.
    x, y can be arrays. Returns (bx, by, bz) each same shape as x."""
    lon = (x / W) * 2 * np.pi - np.pi
    lat = np.pi / 2 - (y / H) * np.pi
    bx = np.cos(lat) * np.sin(lon)
    by = np.sin(lat)
    bz = np.cos(lat) * np.cos(lon)
    return bx, by, bz


def _estimate_rotation_svd(src, dst):
    """Estimate rotation R such that dst ≈ R @ src using SVD (Kabsch/Wahba).
    src, dst: (N, 3) arrays of unit vectors or 3D points (centred).
    Returns R (3, 3)."""
    H = src.T @ dst
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    S = np.diag([1.0, 1.0, d])
    R = Vt.T @ S @ U.T
    return R


def estimate_camera_rotation(tracks_2d, vis, W, H, ref_frame=0,
                              ransac_iters=200, inlier_thresh_deg=2.0):
    """Estimate per-frame camera rotation relative to a reference frame.

    Uses bearing-vector correspondences from 2D tracks on the ERP image.
    Static background points vote for the dominant rotation via RANSAC.

    Returns:
        rotations : list of T rotation matrices (3,3). rotations[ref_frame] = I.
        is_static : bool, True if camera is essentially static across all frames.
        inlier_masks : (N, T) bool array, True for points consistent with camera rotation.
    """
    N, T, _ = tracks_2d.shape
    rotations = [np.eye(3)] * T
    inlier_masks = np.zeros((N, T), dtype=bool)
    thresh_rad = np.deg2rad(inlier_thresh_deg)

    ref_bx, ref_by, ref_bz = erp_pixel_to_bearing(
        tracks_2d[:, ref_frame, 0], tracks_2d[:, ref_frame, 1], W, H
    )
    ref_bearings = np.stack([ref_bx, ref_by, ref_bz], axis=-1)  # (N, 3)

    max_rotation_angles = []

    for t in range(T):
        if t == ref_frame:
            inlier_masks[:, t] = vis[:, t] > 0.5
            max_rotation_angles.append(0.0)
            continue

        valid = (vis[:, ref_frame] > 0.5) & (vis[:, t] > 0.5)
        valid_idx = np.where(valid)[0]
        if len(valid_idx) < 4:
            max_rotation_angles.append(0.0)
            continue

        cur_bx, cur_by, cur_bz = erp_pixel_to_bearing(
            tracks_2d[valid_idx, t, 0], tracks_2d[valid_idx, t, 1], W, H
        )
        cur_bearings = np.stack([cur_bx, cur_by, cur_bz], axis=-1)  # (M, 3)
        ref_b = ref_bearings[valid_idx]

        best_R = np.eye(3)
        best_inliers = np.zeros(len(valid_idx), dtype=bool)
        best_count = 0

        rng = np.random.default_rng(42 + t)
        n_pts = len(valid_idx)

        for _ in range(ransac_iters):
            sample = rng.choice(n_pts, size=min(4, n_pts), replace=False)
            R_cand = _estimate_rotation_svd(ref_b[sample], cur_bearings[sample])

            rotated = (R_cand @ ref_b.T).T  # (M, 3)
            angular_err = np.arccos(np.clip(
                np.sum(rotated * cur_bearings, axis=-1), -1, 1
            ))
            inliers = angular_err < thresh_rad
            count = inliers.sum()

            if count > best_count:
                best_count = count
                best_inliers = inliers
                best_R = R_cand

        if best_count >= 4:
            best_R = _estimate_rotation_svd(
                ref_b[best_inliers], cur_bearings[best_inliers]
            )

        rotations[t] = best_R

        inlier_full = np.zeros(N, dtype=bool)
        inlier_full[valid_idx[best_inliers]] = True
        inlier_masks[:, t] = inlier_full

        angle = np.arccos(np.clip((np.trace(best_R) - 1) / 2, -1, 1))
        max_rotation_angles.append(np.degrees(angle))

    max_rot = max(max_rotation_angles) if max_rotation_angles else 0.0
    median_rot = float(np.median(max_rotation_angles))
    is_static = max_rot < 1.0

    print(f"   Camera rotation: median={median_rot:.2f}°  max={max_rot:.2f}°"
          f"  → {'STATIC' if is_static else 'DYNAMIC'}")

    return rotations, is_static, inlier_masks


def estimate_camera_pose_pnp(tracks_3d_ref, tracks_2d, vis, W, H,
                              ref_frame=0, ransac_iters=300,
                              inlier_thresh_deg=2.0):
    """Full 6-DoF camera pose estimation using 3D-to-bearing PnP.

    For each frame t, estimates (R_t, t_t) such that the 3D points from
    ref_frame, when transformed by (R_t, t_t), produce bearing vectors
    matching the 2D observations at frame t.

    Uses 3D-3D Umeyama alignment between reference 3D points and
    current-frame 3D points (from depth), with RANSAC for robustness.

    Returns:
        poses : list of T dicts with keys 'R' (3,3) and 't' (3,).
        is_static : bool
        inlier_masks : (N, T) bool
    """
    N, T, _ = tracks_3d_ref.shape
    poses = [{'R': np.eye(3), 't': np.zeros(3)} for _ in range(T)]
    inlier_masks = np.zeros((N, T), dtype=bool)
    thresh_rad = np.deg2rad(inlier_thresh_deg)

    ref_pts = tracks_3d_ref[:, ref_frame]  # (N, 3)
    max_angles = []

    for t in range(T):
        if t == ref_frame:
            inlier_masks[:, t] = vis[:, t] > 0.5
            max_angles.append(0.0)
            continue

        valid = (
            (vis[:, ref_frame] > 0.5) & (vis[:, t] > 0.5)
            & ~np.any(np.isnan(ref_pts), axis=-1)
            & ~np.any(np.isnan(tracks_3d_ref[:, t]), axis=-1)
        )
        valid_idx = np.where(valid)[0]
        if len(valid_idx) < 4:
            max_angles.append(0.0)
            continue

        src = ref_pts[valid_idx]              # 3D at ref frame
        dst = tracks_3d_ref[valid_idx, t]     # 3D at frame t

        cur_bx, cur_by, cur_bz = erp_pixel_to_bearing(
            tracks_2d[valid_idx, t, 0], tracks_2d[valid_idx, t, 1], W, H
        )
        cur_bearings = np.stack([cur_bx, cur_by, cur_bz], axis=-1)

        best_R, best_t = np.eye(3), np.zeros(3)
        best_count = 0
        best_inliers = np.zeros(len(valid_idx), dtype=bool)

        rng = np.random.default_rng(42 + t)
        n_pts = len(valid_idx)

        for _ in range(ransac_iters):
            sample = rng.choice(n_pts, size=min(4, n_pts), replace=False)
            src_s, dst_s = src[sample], dst[sample]

            c_src = src_s - src_s.mean(axis=0)
            c_dst = dst_s - dst_s.mean(axis=0)
            R_cand = _estimate_rotation_svd(c_src, c_dst)
            t_cand = dst_s.mean(axis=0) - R_cand @ src_s.mean(axis=0)

            transformed = (R_cand @ src.T).T + t_cand
            pred_bearing = transformed / (np.linalg.norm(transformed, axis=-1, keepdims=True) + 1e-8)
            angular_err = np.arccos(np.clip(
                np.sum(pred_bearing * cur_bearings, axis=-1), -1, 1
            ))
            inliers = angular_err < thresh_rad
            count = inliers.sum()

            if count > best_count:
                best_count = count
                best_inliers = inliers
                best_R, best_t = R_cand, t_cand

        if best_count >= 4:
            src_in, dst_in = src[best_inliers], dst[best_inliers]
            c_src = src_in - src_in.mean(axis=0)
            c_dst = dst_in - dst_in.mean(axis=0)
            best_R = _estimate_rotation_svd(c_src, c_dst)
            best_t = dst_in.mean(axis=0) - best_R @ src_in.mean(axis=0)

        poses[t] = {'R': best_R, 't': best_t}
        inlier_full = np.zeros(N, dtype=bool)
        inlier_full[valid_idx[best_inliers]] = True
        inlier_masks[:, t] = inlier_full

        angle = np.arccos(np.clip((np.trace(best_R) - 1) / 2, -1, 1))
        max_angles.append(np.degrees(angle))

    max_rot = max(max_angles) if max_angles else 0.0
    median_rot = float(np.median(max_angles))
    trans_norms = [np.linalg.norm(p['t']) for p in poses]
    max_trans = max(trans_norms)
    is_static = max_rot < 1.0 and max_trans < 0.05

    print(f"   Camera pose: rot median={median_rot:.2f}° max={max_rot:.2f}°"
          f"  trans max={max_trans:.4f}m  → {'STATIC' if is_static else 'DYNAMIC'}")

    return poses, is_static, inlier_masks


def compensate_camera_motion(tracks_3d, vis, poses):
    """Stabilize 3D tracks by removing estimated camera motion.

    For each frame t, applies the inverse of the estimated camera pose
    to transform 3D points back to the reference coordinate system.

    tracks_3d : (N, T, 3)
    poses     : list of T dicts with 'R' and 't'
    Returns   : tracks_3d_stabilized (N, T, 3)
    """
    N, T, _ = tracks_3d.shape
    stabilized = tracks_3d.copy()

    for t in range(T):
        R = poses[t]['R']
        t_vec = poses[t]['t']
        R_inv = R.T

        for i in range(N):
            if vis[i, t] < 0.5 or np.any(np.isnan(tracks_3d[i, t])):
                continue
            stabilized[i, t] = R_inv @ (tracks_3d[i, t] - t_vec)

    return stabilized


def detect_and_compensate_camera_motion(tracks_2d, tracks_3d, vis, W, H,
                                         method="auto"):
    """High-level API: detect if camera is moving and compensate if needed.

    method: "auto" | "rotation" | "pnp" | "none"
      - "auto": first try rotation-only; if translation is detected, upgrade to PnP
      - "rotation": rotation-only (faster, good for tripod/handheld rotation)
      - "pnp": full 6-DoF (needed when camera translates, e.g. driving/walking)
      - "none": skip compensation

    Returns:
        tracks_3d_out : (N, T, 3) — stabilized if dynamic, original if static
        camera_info   : dict with detection results and per-frame poses
    """
    N, T, _ = tracks_3d.shape

    if method == "none":
        return tracks_3d, {"method": "none", "is_static": True}

    print(f"\n── Camera Motion Analysis ──")

    rotations, is_static_rot, inlier_rot = estimate_camera_rotation(
        tracks_2d, vis, W, H
    )

    if is_static_rot and method == "auto":
        print(f"   Camera is static. No compensation needed.")
        return tracks_3d, {
            "method": "rotation",
            "is_static": True,
            "rotations": rotations,
            "inlier_ratio": float(inlier_rot.sum()) / max(1, (vis > 0.5).sum()),
            "inlier_masks": inlier_rot,
        }

    if method in ("auto", "pnp"):
        print(f"   Camera is dynamic. Running full 6-DoF PnP estimation...")
        poses, is_static_pnp, inlier_pnp = estimate_camera_pose_pnp(
            tracks_3d, tracks_2d, vis, W, H
        )
        used_method = "pnp"
    else:
        poses = [{'R': R, 't': np.zeros(3)} for R in rotations]
        is_static_pnp = is_static_rot
        inlier_pnp = inlier_rot
        used_method = "rotation"

    if not is_static_pnp:
        print(f"   Compensating camera motion ({used_method})...")
        tracks_3d_stabilized = compensate_camera_motion(tracks_3d, vis, poses)
        inlier_ratio = float(inlier_pnp.sum()) / max(1, (vis > 0.5).sum())
        print(f"   Inlier ratio: {inlier_ratio:.1%} (static background points)")
    else:
        tracks_3d_stabilized = tracks_3d
        inlier_ratio = 1.0

    return tracks_3d_stabilized, {
        "method": used_method,
        "is_static": is_static_pnp,
        "poses": poses,
        "inlier_ratio": inlier_ratio,
        "inlier_masks": inlier_pnp if method in ("auto", "pnp") else inlier_rot,
    }


# ╭──────────────────────────────────────────────────────────────╮
# │  4. CONSISTENCY METRICS                                      │
# ╰──────────────────────────────────────────────────────────────╯

def compute_reprojection_error(tracks_2d, tracks_3d, vis, W, H):
    """
    Reproject 3D back to 2D and compute pixel error.
    This checks if depth + projection are self-consistent.
    """
    N, T, _ = tracks_3d.shape
    errors = np.full((N, T), np.nan)

    for i in range(N):
        for t in range(T):
            if vis[i, t] < 0.5 or np.any(np.isnan(tracks_3d[i, t])):
                continue
            X, Y, Z = tracks_3d[i, t]
            rx, ry, _ = xyz_to_erp_pixel(X, Y, Z, W, H)
            ox, oy    = tracks_2d[i, t]

            # Handle wrap-around distance in x
            dx = abs(rx - ox)
            dx = min(dx, W - dx)         # circular distance
            dy = abs(ry - oy)
            errors[i, t] = np.sqrt(dx**2 + dy**2)

    return errors


def compute_3d_smoothness(tracks_3d, vis):
    """
    Acceleration magnitude in 3D (second-order finite diff).
    Low = smooth trajectory, High = jitter / inconsistency.
    Returns (N, T) array.
    """
    N, T, _ = tracks_3d.shape
    accel = np.full((N, T), np.nan)

    for i in range(N):
        for t in range(1, T - 1):
            if vis[i, t-1] < 0.5 or vis[i, t] < 0.5 or vis[i, t+1] < 0.5:
                continue
            p0, p1, p2 = tracks_3d[i, t-1], tracks_3d[i, t], tracks_3d[i, t+1]
            if np.any(np.isnan(p0)) or np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
                continue
            a = p0 - 2 * p1 + p2         # discrete second derivative
            accel[i, t] = np.linalg.norm(a)
    return accel


def compute_rigidity_score(tracks_3d, vis, k=8):
    """
    For each point, measure how stable the distances to its k nearest
    spatial neighbours remain over time.  A rigid body → std ≈ 0.
    Returns (N,) rigidity score (lower = more rigid).
    """
    N, T, _ = tracks_3d.shape

    # Use the first visible frame per point to find neighbours
    ref_pts = np.zeros((N, 3))
    ref_t   = np.zeros(N, dtype=int)
    for i in range(N):
        for t in range(T):
            if vis[i, t] > 0.5 and not np.any(np.isnan(tracks_3d[i, t])):
                ref_pts[i] = tracks_3d[i, t]
                ref_t[i] = t
                break

    from scipy.spatial import cKDTree
    tree = cKDTree(ref_pts)
    _, nn_idx = tree.query(ref_pts, k=k + 1)   # includes self
    nn_idx = nn_idx[:, 1:]                       # drop self → (N, k)

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
            arr = np.array(dists_over_time)          # (num_frames, k)
            rigidity[i] = arr.std(axis=0).mean()     # avg std across neighbours

    return rigidity


# ╭──────────────────────────────────────────────────────────────╮
# │  5. WRAP-AROUND TRACK MERGING                                │
# ╰──────────────────────────────────────────────────────────────╯

def merge_wraparound_tracks(tracks_2d, vis, W, merge_px=20, max_dy=10):
    """
    Detect tracks that disappear near one edge and appear near the other
    within a few frames, and stitch them into continuous trajectories.

    merge_px : how close to the edge (in pixels) to consider a candidate
    max_dy   : max vertical displacement for a merge
    """
    N, T, _ = tracks_2d.shape
    merged = 0

    # Find tracks that lose visibility
    for i in range(N):
        for t in range(T - 1):
            if vis[i, t] > 0.5 and vis[i, t + 1] < 0.5:
                # Track i dies at frame t
                x_die, y_die = tracks_2d[i, t]
                at_right = x_die > W - merge_px
                at_left  = x_die < merge_px
                if not (at_right or at_left):
                    continue

                # Look for a track starting near the opposite edge at frame t+1
                for j in range(N):
                    if j == i:
                        continue
                    # j should be invisible at t and visible at t+1
                    if vis[j, t] > 0.5 or vis[j, t + 1] < 0.5:
                        continue
                    x_born, y_born = tracks_2d[j, t + 1]
                    if at_right and x_born < merge_px:
                        if abs(y_born - y_die) < max_dy:
                            # Merge: copy j's future into i
                            for tt in range(t + 1, T):
                                tracks_2d[i, tt] = tracks_2d[j, tt]
                                vis[i, tt]       = vis[j, tt]
                                vis[j, tt]       = 0     # mark j as consumed
                            merged += 1
                            break
                    elif at_left and x_born > W - merge_px:
                        if abs(y_born - y_die) < max_dy:
                            for tt in range(t + 1, T):
                                tracks_2d[i, tt] = tracks_2d[j, tt]
                                vis[i, tt]       = vis[j, tt]
                                vis[j, tt]       = 0
                            merged += 1
                            break

    print(f"   Merged {merged} wrap-around track pairs")
    return tracks_2d, vis


# ╭──────────────────────────────────────────────────────────────╮
# │  6. VISUALISATION                                            │
# ╰──────────────────────────────────────────────────────────────╯

def _motion_color(speed_px, max_speed=8.0):
    """
    Map instantaneous speed (px/frame) to an RGB colour.
    blue (0) → cyan → green → yellow → red (max_speed+)
    Uses the 'jet' colormap so static points are cool-coloured and
    dynamic points (moving person etc.) stand out in warm colours.
    """
    import matplotlib
    cmap = matplotlib.colormaps["jet"]
    t = float(np.clip(speed_px / max_speed, 0.0, 1.0))
    r, g, b, _ = cmap(t)
    return (int(r * 255), int(g * 255), int(b * 255))   # RGB


def visualize_tracks_on_frames(frames_rgb, tracks_2d, vis, out_path,
                               fps=6, max_tracks=None, trail_frames=12,
                               motion_speed_max=8.0):
    """
    Draw 2D trajectories coloured by instantaneous motion magnitude.

    • Each dot is coloured blue→red based on its speed at that frame.
      Static background → cool blue/green; moving objects → warm yellow/red.
    • A fading trail shows the last `trail_frames` positions.
      Trail colour inherits the motion colour so motion history is visible.
    • For very fast points a velocity arrow is drawn.

    max_tracks   : None = all; int = random spatial sample
    trail_frames : number of past frames to draw as trail (None = full history)
    motion_speed_max : px/frame mapped to full-red
    """
    H, W = frames_rgb[0].shape[:2]
    N, T, _ = tracks_2d.shape

    if max_tracks is None or max_tracks >= N:
        draw_idx = np.arange(N)
    else:
        rng = np.random.default_rng(0)
        draw_idx = np.sort(rng.choice(N, size=max_tracks, replace=False))

    # Pre-compute per-track per-frame speed (px / frame)
    dx = np.diff(tracks_2d[:, :, 0], axis=1)   # (N, T-1)
    dy = np.diff(tracks_2d[:, :, 1], axis=1)
    speed = np.sqrt(dx**2 + dy**2)              # (N, T-1)
    # speed[:, t] = speed between frame t and t+1
    # For frame t we use speed[:, t-1] (arrived speed), defaulting to 0 at t=0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    for t in tqdm(range(T), desc="   Viz tracks"):
        # --- base frame (darkened slightly so dots stand out) ---
        canvas = (frames_rgb[t].astype(np.float32) * 0.65).astype(np.uint8)

        # --- trail overlay (separate layer, blended in) ---
        trail_layer = np.zeros_like(canvas, dtype=np.float32)
        trail_alpha = np.zeros(canvas.shape[:2], dtype=np.float32)

        t_start = 0 if trail_frames is None else max(0, t - trail_frames)

        for i in draw_idx:
            if vis[i, t] < 0.5:
                continue

            spd = float(speed[i, t - 1]) if t > 0 else 0.0
            dot_c = _motion_color(spd, motion_speed_max)

            # ── trail ──
            for dt in range(t_start, t):
                if vis[i, dt] < 0.5:
                    continue
                x0 = int(round(tracks_2d[i, dt,     0]))
                y0 = int(round(tracks_2d[i, dt,     1]))
                x1 = int(round(tracks_2d[i, dt + 1, 0]))
                y1 = int(round(tracks_2d[i, dt + 1, 1]))
                if abs(x1 - x0) >= W // 2:     # skip wrap-around jumps
                    continue
                seg_spd   = float(speed[i, dt]) if dt < T - 1 else spd
                seg_color = _motion_color(seg_spd, motion_speed_max)
                age_alpha = 0.25 + 0.75 * ((dt - t_start) / max(t - t_start, 1))
                thickness = 2 if seg_spd > motion_speed_max * 0.3 else 1
                cv2.line(trail_layer, (x0, y0), (x1, y1),
                         seg_color, thickness, cv2.LINE_AA)
                cv2.line(trail_alpha.reshape(H, W, 1) if False else
                         # write alpha proportional to age
                         trail_alpha, (x0, y0), (x1, y1),
                         float(age_alpha), thickness, cv2.LINE_AA)

            # ── current dot ──
            cx, cy = int(round(tracks_2d[i, t, 0])), int(round(tracks_2d[i, t, 1]))
            radius = 4 if spd > motion_speed_max * 0.4 else 3
            cv2.circle(canvas, (cx, cy), radius, dot_c, -1, cv2.LINE_AA)
            # white outline for high-motion dots
            if spd > motion_speed_max * 0.25:
                cv2.circle(canvas, (cx, cy), radius + 1, (255, 255, 255), 1, cv2.LINE_AA)

            # ── velocity arrow for fast points ──
            if t > 0 and spd > motion_speed_max * 0.3:
                vx = tracks_2d[i, t, 0] - tracks_2d[i, t - 1, 0]
                vy = tracks_2d[i, t, 1] - tracks_2d[i, t - 1, 1]
                scale = min(20.0 / max(spd, 1e-3), 6.0)
                ex = int(round(cx + vx * scale))
                ey = int(round(cy + vy * scale))
                cv2.arrowedLine(canvas, (cx, cy), (ex, ey),
                                (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.4)

        # ── blend trail layer onto darkened canvas ──
        # Where trail_alpha > 0 blend trail_layer in
        a3 = np.clip(trail_alpha, 0, 1)[..., None]
        canvas = np.clip(
            canvas.astype(np.float32) * (1 - a3 * 0.7) + trail_layer * a3 * 0.7,
            0, 255
        ).astype(np.uint8)

        # ── legend ──
        import matplotlib
        cmap = matplotlib.colormaps["jet"]
        for li, label in enumerate(["static", "moving"]):
            c = _motion_color(li * motion_speed_max, motion_speed_max)
            cv2.circle(canvas, (W - 90, 18 + li * 22), 5, c, -1)
            cv2.putText(canvas, label, (W - 80, 23 + li * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"   Saved: {out_path}")


def _sample_draw_indices(N, max_tracks, seed=0):
    """Return a spatially-spread sample of track indices (random, not first-N)."""
    if max_tracks is None or max_tracks >= N:
        return np.arange(N)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(N, size=max_tracks, replace=False))


def visualize_3d_trajectories(tracks_3d, vis, out_path, max_tracks=300):
    """
    Render full 3D trajectory history as a static Matplotlib figure.
    Uses random sampling so all spatial regions are represented.
    """
    import matplotlib.pyplot as plt

    N, T, _ = tracks_3d.shape
    draw_idx = _sample_draw_indices(N, max_tracks)
    n_draw   = len(draw_idx)
    cmap     = plt.get_cmap("hsv")

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("#0d0d0d")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0d0d0d")

    for k, i in enumerate(draw_idx):
        mask = (vis[i] > 0.5) & ~np.any(np.isnan(tracks_3d[i]), axis=-1)
        if mask.sum() < 2:
            continue
        xs = tracks_3d[i, mask, 0]
        ys = tracks_3d[i, mask, 1]
        zs = tracks_3d[i, mask, 2]
        c  = cmap(k / n_draw)
        # Full trajectory line (history)
        ax.plot(xs, zs, ys, color=c, lw=0.7, alpha=0.65)
        # Start dot (hollow) and end dot (filled)
        ax.scatter(xs[:1],  zs[:1],  ys[:1],  color=c, s=12, marker="o",
                   facecolors="none", edgecolors=c, linewidths=0.8)
        ax.scatter(xs[-1:], zs[-1:], ys[-1:], color=c, s=14, zorder=5)

    def _style(ax):
        ax.tick_params(colors="#555", labelsize=7)
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor("#2a2a2a")
        ax.grid(True, color="#1e1e1e", linewidth=0.5)
        ax.set_xlabel("X (m)", color="#999", fontsize=8, labelpad=4)
        ax.set_ylabel("Z (m)", color="#999", fontsize=8, labelpad=4)
        ax.set_zlabel("Y (m)", color="#999", fontsize=8, labelpad=4)

    _style(ax)
    ax.set_title("3D Point Trajectories  (○ start → • end)",
                 color="white", fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"   Saved: {out_path}")


def visualize_3d_trajectory_video(tracks_3d, vis, out_path,
                                   fps=6, max_tracks=400, trail_len=None):
    """
    Animated MP4 of 3D trajectories building up frame by frame.

    Each frame renders the accumulated trail up to time t so the viewer
    can watch trajectories grow through the scene.
    trail_len : None = full history; int = keep last N frames only.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N, T, _ = tracks_3d.shape
    draw_idx = _sample_draw_indices(N, max_tracks)
    n_draw   = len(draw_idx)
    cmap     = plt.get_cmap("hsv")
    colors   = [cmap(k / n_draw) for k in range(n_draw)]

    # Pre-compute axis limits from all valid points
    valid = tracks_3d[draw_idx][vis[draw_idx].astype(bool)]
    valid = valid[~np.isnan(valid).any(axis=-1)]
    pad   = 0.3
    xlim  = (valid[:, 0].min() - pad, valid[:, 0].max() + pad)
    ylim  = (valid[:, 2].min() - pad, valid[:, 2].max() + pad)  # Z axis
    zlim  = (valid[:, 1].min() - pad, valid[:, 1].max() + pad)  # Y axis

    # Render each frame to a PNG in memory, then assemble with cv2
    print(f"   Rendering 3D trajectory video ({T} frames, {n_draw} tracks) ...")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None  # initialised after first frame to get (H, W)

    for t in tqdm(range(T), desc="   3D traj video"):
        fig = plt.figure(figsize=(12, 7), dpi=100)
        fig.patch.set_facecolor("#0d0d0d")
        ax  = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("#0d0d0d")

        t0 = 0 if trail_len is None else max(0, t - trail_len + 1)

        for k, i in enumerate(draw_idx):
            # Frames t0 … t
            seg_t  = np.arange(t0, t + 1)
            v_mask = vis[i, seg_t] > 0.5
            nan_ok = ~np.isnan(tracks_3d[i, seg_t]).any(axis=-1)
            good   = v_mask & nan_ok

            if good.sum() < 1:
                continue

            seg = tracks_3d[i, seg_t[good]]
            c   = colors[k]

            # Fading trail: older segments are more transparent
            n_seg = len(seg)
            if n_seg >= 2:
                # Draw as segments with linearly increasing alpha
                for s in range(n_seg - 1):
                    alpha = 0.15 + 0.85 * (s / (n_seg - 1))
                    ax.plot(seg[s:s+2, 0], seg[s:s+2, 2], seg[s:s+2, 1],
                            color=c, lw=0.9, alpha=alpha)

            # Current position dot
            ax.scatter(seg[-1:, 0], seg[-1:, 2], seg[-1:, 1],
                       color=c, s=16, zorder=5, alpha=1.0)

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)

        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor("#2a2a2a")
        ax.grid(True, color="#1e1e1e", linewidth=0.4)
        ax.tick_params(colors="#555", labelsize=6)
        ax.set_xlabel("X (m)", color="#888", fontsize=7, labelpad=3)
        ax.set_ylabel("Z (m)", color="#888", fontsize=7, labelpad=3)
        ax.set_zlabel("Y (m)", color="#888", fontsize=7, labelpad=3)
        ax.set_title(f"3D Trajectory History   frame {t+1:02d}/{T}",
                     color="white", fontsize=10, pad=6)

        # Rotate viewpoint slowly (5° per frame)
        ax.view_init(elev=20, azim=(-60 + t * 5) % 360)

        plt.tight_layout()

        # Render to numpy array
        fig.canvas.draw()
        w_px, h_px = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(h_px, w_px, 4)[..., :3]   # RGBA → RGB
        plt.close(fig)

        if writer is None:
            h, w = buf.shape[:2]
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        writer.write(cv2.cvtColor(buf, cv2.COLOR_RGB2BGR))

    if writer:
        writer.release()
    print(f"   Saved: {out_path}")


# ╭──────────────────────────────────────────────────────────────╮
# │  7. SAVE / LOAD ANNOTATIONS                                 │
# ╰──────────────────────────────────────────────────────────────╯

def save_annotations(out_dir, tracks_2d, tracks_3d, vis, sampled_depth,
                     reproj_err, smoothness, rigidity, meta):
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "tracks_2d.npy"),      tracks_2d.astype(np.float32))
    np.save(os.path.join(out_dir, "tracks_3d.npy"),      tracks_3d.astype(np.float32))
    np.save(os.path.join(out_dir, "visibility.npy"),      vis.astype(np.float32))
    np.save(os.path.join(out_dir, "sampled_depth.npy"),   sampled_depth.astype(np.float32))
    np.save(os.path.join(out_dir, "reproj_error.npy"),    reproj_err.astype(np.float32))
    np.save(os.path.join(out_dir, "smoothness.npy"),      smoothness.astype(np.float32))
    np.save(os.path.join(out_dir, "rigidity.npy"),        rigidity.astype(np.float32))

    # Summary stats
    valid_re = reproj_err[~np.isnan(reproj_err)]
    valid_sm = smoothness[~np.isnan(smoothness)]
    valid_rg = rigidity[~np.isnan(rigidity)]

    meta.update({
        "n_tracks":           int(tracks_2d.shape[0]),
        "n_frames":           int(tracks_2d.shape[1]),
        "reproj_error_median_px": float(np.median(valid_re)) if len(valid_re) else None,
        "reproj_error_mean_px":   float(np.mean(valid_re))   if len(valid_re) else None,
        "smoothness_median_m":    float(np.median(valid_sm))  if len(valid_sm) else None,
        "rigidity_median_m":      float(np.median(valid_rg))  if len(valid_rg) else None,
    })
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n   Annotations saved to {out_dir}/")
    print(f"   ├── tracks_2d.npy       ({tracks_2d.shape})")
    print(f"   ├── tracks_3d.npy       ({tracks_3d.shape})")
    print(f"   ├── visibility.npy      ({vis.shape})")
    print(f"   ├── sampled_depth.npy   ({sampled_depth.shape})")
    print(f"   ├── reproj_error.npy    ({reproj_err.shape})")
    print(f"   ├── smoothness.npy      ({smoothness.shape})")
    print(f"   ├── rigidity.npy        ({rigidity.shape})")
    print(f"   └── meta.json")
    print(f"\n   Reprojection error: {meta.get('reproj_error_median_px', '?'):.2f} px (median)")
    print(f"   3D smoothness:     {meta.get('smoothness_median_m', '?'):.4f} m  (median accel)")
    print(f"   Rigidity score:    {meta.get('rigidity_median_m', '?'):.4f} m  (median)")


# ╭──────────────────────────────────────────────────────────────╮
# │  COMPARE TRACKERS                                            │
# ╰──────────────────────────────────────────────────────────────╯

def compare_trackers(tracker_results, frames_rgb, depth_paths, W, H,
                     max_depth, out_dir, fps=6):
    """
    Compare multiple trackers.

    tracker_results: dict {name: (tracks_2d, vis, ann_dir)}
    Writes:
        comparison/metrics.json
        comparison/tracks_comparison.mp4
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmp_dir = os.path.join(out_dir, "comparison")
    os.makedirs(cmp_dir, exist_ok=True)

    metrics = {}
    for name, (tracks_2d, vis, ann_dir) in tracker_results.items():
        tracks_3d, sampled_depth = lift_tracks_to_3d(
            tracks_2d, vis, depth_paths, W, H, max_depth=max_depth
        )
        reproj_err = compute_reprojection_error(tracks_2d, tracks_3d, vis, W, H)
        smoothness = compute_3d_smoothness(tracks_3d, vis)
        rigidity   = compute_rigidity_score(tracks_3d, vis)

        valid_re = reproj_err[~np.isnan(reproj_err)]
        valid_sm = smoothness[~np.isnan(smoothness)]
        valid_rg = rigidity[~np.isnan(rigidity)]
        n_vis = int(vis.sum())

        m = {
            "n_tracks":               int(tracks_2d.shape[0]),
            "n_visible":              n_vis,
            "reproj_error_median_px": float(np.median(valid_re)) if len(valid_re) else None,
            "reproj_error_mean_px":   float(np.mean(valid_re))   if len(valid_re) else None,
            "smoothness_median_m":    float(np.median(valid_sm))  if len(valid_sm) else None,
            "rigidity_median_m":      float(np.median(valid_rg))  if len(valid_rg) else None,
        }
        metrics[name] = m

        # Save per-tracker 3D data
        np.save(os.path.join(ann_dir, "tracks_3d.npy"),    tracks_3d.astype(np.float32))
        np.save(os.path.join(ann_dir, "sampled_depth.npy"), sampled_depth.astype(np.float32))
        np.save(os.path.join(ann_dir, "reproj_error.npy"),  reproj_err.astype(np.float32))
        np.save(os.path.join(ann_dir, "smoothness.npy"),    smoothness.astype(np.float32))
        np.save(os.path.join(ann_dir, "rigidity.npy"),      rigidity.astype(np.float32))

    with open(os.path.join(cmp_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Print table ──────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Tracker Comparison")
    print(f"{'='*65}")
    print(f"  {'Tracker':<14} {'Tracks':>7} {'Reproj(px)':>10} "
          f"{'Smooth(m)':>10} {'Rigid(m)':>10}")
    print(f"  {'-'*14} {'-'*7} {'-'*10} {'-'*10} {'-'*10}")
    for name, m in metrics.items():
        re  = f"{m['reproj_error_median_px']:.2f}" if m['reproj_error_median_px'] else "N/A"
        sm  = f"{m['smoothness_median_m']:.4f}"    if m['smoothness_median_m']    else "N/A"
        rg  = f"{m['rigidity_median_m']:.4f}"      if m['rigidity_median_m']      else "N/A"
        print(f"  {name:<14} {m['n_tracks']:>7} {re:>10} {sm:>10} {rg:>10}")
    print(f"{'='*65}\n")

    # ── Side-by-side comparison video ────────────────────────────────────
    names = list(tracker_results.keys())
    T = len(frames_rgb)
    writer = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.join(cmp_dir, "tracks_comparison.mp4")

    # Pre-render per-tracker track frames using motion-coloured dots
    rendered = {}
    for name, (tracks_2d, vis, _) in tracker_results.items():
        print(f"   Rendering comparison frames for {name} ...")
        rendered[name] = _render_track_frames(frames_rgb, tracks_2d, vis)

    for t in range(T):
        panels = []
        for name in names:
            frame = rendered[name][t].copy()
            cv2.putText(frame, name, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 80), 2, cv2.LINE_AA)
            cv2.putText(frame, f"f{t+1:02d}", (10, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
            # Add metric
            m = metrics[name]
            re_str = f"reproj={m['reproj_error_median_px']:.2f}px" if m['reproj_error_median_px'] else ""
            cv2.putText(frame, re_str, (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 80), 1, cv2.LINE_AA)
            panels.append(frame)

        composite = np.concatenate(panels, axis=1)  # side by side
        bgr = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)

        if writer is None:
            h, w = bgr.shape[:2]
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        writer.write(bgr)

    if writer:
        writer.release()

    print(f"   Comparison metrics → {os.path.join(cmp_dir, 'metrics.json')}")
    print(f"   Comparison video   → {out_path}")
    return metrics


def _render_track_frames(frames_rgb, tracks_2d, vis, max_tracks=300):
    """Return list of T BGR frames with motion-coloured tracks."""
    N, T, _ = tracks_2d.shape
    rng = np.random.default_rng(0)
    if N > max_tracks:
        idx = rng.choice(N, max_tracks, replace=False)
        tracks_2d = tracks_2d[idx]
        vis = vis[idx]

    out = []
    for t in range(T):
        frame = frames_rgb[t].copy().astype(np.uint8)
        dark = (frame.astype(np.float32) * 0.65).astype(np.uint8)
        overlay = dark.copy()
        H, W = frame.shape[:2]

        for i in range(tracks_2d.shape[0]):
            if vis[i, t] < 0.5:
                continue
            cur_x, cur_y = int(tracks_2d[i, t, 0]), int(tracks_2d[i, t, 1])

            # Draw trail (last 8 frames)
            t_start = max(0, t - 8)
            pts = []
            for tt in range(t_start, t + 1):
                if vis[i, tt] > 0.5:
                    pts.append((int(tracks_2d[i, tt, 0]), int(tracks_2d[i, tt, 1])))
            for k in range(1, len(pts)):
                alpha = 0.3 + 0.7 * (k / max(1, len(pts) - 1))
                color = (int(80 * alpha), int(200 * alpha), int(80 * alpha))
                cv2.line(overlay, pts[k - 1], pts[k], color, 1, cv2.LINE_AA)

            # Speed-based color for current dot
            if t > 0 and vis[i, t - 1] > 0.5:
                dx = tracks_2d[i, t, 0] - tracks_2d[i, t - 1, 0]
                dy = tracks_2d[i, t, 1] - tracks_2d[i, t - 1, 1]
                spd = np.sqrt(dx**2 + dy**2)
            else:
                spd = 0.0
            r = int(min(255, spd * 15))
            b = max(0, 255 - r)
            color = (r, 60, b)
            cv2.circle(overlay, (cur_x, cur_y), 3, color, -1, cv2.LINE_AA)

        # Blend
        blended = cv2.addWeighted(dark, 0.3, overlay, 0.7, 0)
        out.append(blended)
    return out


# ╭──────────────────────────────────────────────────────────────╮
# │  STEP 0 – FRAME EXTRACTION                                   │
# ╰──────────────────────────────────────────────────────────────╯

def extract_frames(video_path, out_dir, fps=1, width=1024, height=512):
    """
    Extract frames from a video at `fps` frames/sec, resized to width×height.
    Skips extraction if frames already exist.
    Returns the output directory path.
    """
    os.makedirs(out_dir, exist_ok=True)
    existing = sorted_files(out_dir, (".png", ".jpg"))
    if existing:
        print(f"   Frames already extracted ({len(existing)} files in {out_dir}), skipping.")
        return out_dir

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps={fps},scale={width}:{height}",
        os.path.join(out_dir, "%04d.png"),
    ]
    print(f"   Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")

    n = len(sorted_files(out_dir, ".png"))
    print(f"   Extracted {n} frames → {out_dir}")
    return out_dir


# ╭──────────────────────────────────────────────────────────────╮
# │  STEP 0.5 – DAP DEPTH INFERENCE                              │
# ╰──────────────────────────────────────────────────────────────╯

def run_dap_inference(frames_dir, depth_dir, config_path, gpu="0",
                      vis_range="100m", cmap="Spectral"):
    """
    Run DAP depth inference on all frames in `frames_dir`.
    Writes per-frame .npy + colourised PNGs under `depth_dir`.
    Skips inference if .npy files already exist.
    Returns the path to the depth_npy subdirectory.
    """
    import sys, yaml, torch, torch.nn as nn

    npy_dir = os.path.join(depth_dir, "depth_npy")
    os.makedirs(npy_dir, exist_ok=True)
    existing_npy = sorted_files(npy_dir, ".npy")
    existing_frames = sorted_files(frames_dir, (".png", ".jpg", ".jpeg"))

    if len(existing_npy) >= len(existing_frames) > 0:
        print(f"   Depth maps already exist ({len(existing_npy)} files), skipping DAP.")
        return npy_dir

    # Resolve project root (one level above test/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Import from infer.py which lives in the same directory as this file
    infer_dir = os.path.dirname(os.path.abspath(__file__))
    if infer_dir not in sys.path:
        sys.path.insert(0, infer_dir)
    from infer import load_model, infer_raw, pred_to_vis, ensure_dir_for_file
    import yaml as _yaml

    with open(config_path) as f:
        config = _yaml.load(f, Loader=_yaml.FullLoader)

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu)

    model, device = load_model(config)

    frame_files = sorted_files(frames_dir, (".png", ".jpg", ".jpeg"))
    print(f"   Running DAP on {len(frame_files)} frames → {npy_dir}")

    for idx, ff in enumerate(tqdm(frame_files, desc="   DAP inference"), start=1):
        img_bgr = cv2.imread(os.path.join(frames_dir, ff))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        pred = infer_raw(model, device, img_rgb)

        stem = f"{idx:06d}"
        npy_path   = os.path.join(npy_dir, stem + ".npy")
        gray_path  = os.path.join(depth_dir, f"depth_vis_gray_{vis_range}", stem + ".png")
        color_path = os.path.join(depth_dir, f"depth_vis_color_{vis_range}", stem + ".png")
        ensure_dir_for_file(npy_path)
        ensure_dir_for_file(gray_path)
        ensure_dir_for_file(color_path)

        np.save(npy_path, pred)
        depth_gray, depth_color_rgb = pred_to_vis(pred, vis_range=vis_range, cmap=cmap)
        cv2.imwrite(gray_path,  depth_gray)
        cv2.imwrite(color_path, cv2.cvtColor(depth_color_rgb, cv2.COLOR_RGB2BGR))

    print(f"   DAP done. depth_npy → {npy_dir}")
    return npy_dir


# ╭──────────────────────────────────────────────────────────────╮
# │  MAIN                                                        │
# ╰──────────────────────────────────────────────────────────────╯

def main():
    parser = argparse.ArgumentParser(
        description="Panoramic 3D Trajectory Annotation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── video shortcut ──
    parser.add_argument("--video", type=str, default=None,
                        help="Path to input video. When set, frames are extracted "
                             "and DAP depth is run automatically. All output paths "
                             "are derived from the video stem unless overridden.")

    # ── DAP / extraction settings (used when --video is set) ──
    dap = parser.add_argument_group("DAP / extraction  (used with --video)")
    dap.add_argument("--config",      type=str, default="config/infer.yaml",
                     help="DAP config yaml")
    dap.add_argument("--extract_fps", type=int, default=1,
                     help="Frames per second to extract from the video")
    dap.add_argument("--frame_width", type=int, default=1024,
                     help="Width to resize extracted frames")
    dap.add_argument("--frame_height",type=int, default=512,
                     help="Height to resize extracted frames")
    dap.add_argument("--gpu",         type=str, default="0",
                     help="GPU index for DAP inference")
    dap.add_argument("--vis_range",   type=str, default="100m",
                     choices=["100m", "10m"],
                     help="DAP depth visualisation range")

    # ── manual paths (used when --video is NOT set) ──
    manual = parser.add_argument_group("Manual paths  (used without --video)")
    manual.add_argument("--frames_dir", type=str,
                        default="datasets/video_frames",
                        help="Directory of panoramic video frames")
    manual.add_argument("--depth_dir",  type=str,
                        default="output/video_1000050115/depth_npy",
                        help="Directory of DAP depth .npy files")
    manual.add_argument("--out_dir",    type=str,
                        default="output/video_1000050115/annotations",
                        help="Output directory for annotations")

    # ── trajectory / tracker settings ──
    trk = parser.add_argument_group("Tracker settings")
    trk.add_argument("--grid_size",  type=int, default=50,
                     help="CoTracker grid density (50 → 2500 pts)")
    trk.add_argument("--grid_step",  type=int, default=16,
                     help="Pixel step for OptFlow fallback grid")
    trk.add_argument("--max_depth",  type=float, default=None,
                     help="Max depth in metres. None = auto (P99 of scene)")
    trk.add_argument("--pad_frac",   type=float, default=0.25,
                     help="Circular padding fraction (0.25 = 25%% per side)")
    trk.add_argument("--tracker",    type=str, default="cotracker",
                     choices=["cotracker", "optflow", "spatrack", "tapip3d", "all"],
                     help="Tracker to use. 'all' runs cotracker+spatrack+tapip3d and compares.")
    trk.add_argument("--spatrack_grid",  type=int, default=40,
                     help="Grid size for SpaTracker")
    trk.add_argument("--tapip3d_grid",   type=int, default=32,
                     help="Grid size for TAPIP3D")
    trk.add_argument("--device",     type=str, default="cuda")
    trk.add_argument("--checkpoint", type=str,
                     default=_COTRACKER3_OFFLINE_CKPT,
                     help="CoTracker3 offline checkpoint path")
    trk.add_argument("--camera_motion", type=str, default="auto",
                     choices=["auto", "rotation", "pnp", "none"],
                     help="Camera motion compensation: auto detects and compensates, "
                          "rotation=rotation-only, pnp=full 6-DoF, none=skip")

    # ── output settings ──
    out = parser.add_argument_group("Output settings")
    out.add_argument("--fps",       type=int, default=6,
                     help="FPS for visualisation videos")
    out.add_argument("--skip_viz",  action="store_true",
                     help="Skip visualisation generation")

    args = parser.parse_args()

    # ── Resolve paths from --video if provided ──
    if args.video:
        video_path = os.path.abspath(args.video)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        stem = Path(video_path).stem          # e.g. "1000050119"
        base_out = os.path.join("output", stem)

        args.frames_dir = os.path.join("datasets", f"frames_{stem}")
        args.depth_dir  = base_out            # DAP writes subfolders here
        args.out_dir    = os.path.join(base_out, "annotations")

        print(f"\n{'='*60}")
        print(f" End-to-end pipeline for: {os.path.basename(video_path)}")
        print(f"{'='*60}")
        print(f"   frames_dir → {args.frames_dir}")
        print(f"   depth_dir  → {args.depth_dir}/depth_npy")
        print(f"   out_dir    → {args.out_dir}")

        # Step 0 – extract frames
        print(f"\n── Step 0: Frame Extraction  ({args.extract_fps} fps, "
              f"{args.frame_width}×{args.frame_height}) ──")
        extract_frames(video_path, args.frames_dir,
                       fps=args.extract_fps,
                       width=args.frame_width, height=args.frame_height)

        # Step 0.5 – DAP depth inference
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            args.config,
        )
        print(f"\n── Step 0.5: DAP Depth Inference ──")
        depth_npy_dir = run_dap_inference(
            args.frames_dir, args.depth_dir, config_path,
            gpu=args.gpu, vis_range=args.vis_range,
        )
        args.depth_dir = depth_npy_dir   # point to the npy subdirectory

        # Auto max_depth from scene P99 if not set
        if args.max_depth is None:
            npy_files = sorted_files(args.depth_dir, ".npy")
            sample = np.concatenate([
                np.load(os.path.join(args.depth_dir, f)).ravel() * MAX_DEPTH_SCALE
                for f in npy_files[:min(5, len(npy_files))]
            ])
            args.max_depth = float(np.percentile(sample, 99))
            print(f"   Auto max_depth = {args.max_depth:.1f} m (P99 of first 5 frames)")
    else:
        if args.max_depth is None:
            args.max_depth = 100.0

    # ── Load frames ──
    frame_files = sorted_files(args.frames_dir, (".png", ".jpg", ".jpeg"))
    depth_files = sorted_files(args.depth_dir,  ".npy")
    assert len(frame_files) > 0,  f"No frames found in {args.frames_dir}"
    assert len(depth_files) > 0,  f"No depth maps found in {args.depth_dir}"
    T = min(len(frame_files), len(depth_files))
    frame_files = frame_files[:T]
    depth_files = depth_files[:T]
    print(f"\n{'='*60}")
    print(f" Panoramic 3D Trajectory Pipeline")
    print(f"{'='*60}")
    print(f"   Frames : {T}  ({args.frames_dir})")
    print(f"   Depths : {T}  ({args.depth_dir})")

    frame_paths = [os.path.join(args.frames_dir, f) for f in frame_files]
    depth_paths = [os.path.join(args.depth_dir,  f) for f in depth_files]

    frames_rgb = [load_rgb(p) for p in tqdm(frame_paths, desc="Loading frames")]
    H, W = frames_rgb[0].shape[:2]
    print(f"   Resolution: {W} × {H}")

    # ── Determine which trackers to run ──────────────────────────────────────
    gpu_idx = int(args.device.replace("cuda:", "").replace("cuda", "0")) if "cuda" in args.device else 0

    if args.tracker == "all":
        # Run all three trackers and compare
        print(f"\n── Step 1: Running ALL trackers (cotracker + spatrack + tapip3d) ──")
        tracker_results = {}

        # CoTracker
        ct_dir = os.path.join(args.out_dir, "annotations_cotracker")
        os.makedirs(ct_dir, exist_ok=True)
        print(f"\n   [1/3] CoTracker3 ...")
        t0 = time.time()
        ct_2d, ct_vis = run_cotracker(
            frames_rgb, grid_size=args.grid_size,
            pad_frac=args.pad_frac, device=args.device,
            checkpoint=args.checkpoint,
        )
        ct_2d, ct_vis = merge_wraparound_tracks(ct_2d, ct_vis, W)
        np.save(os.path.join(ct_dir, "tracks_2d.npy"),   ct_2d.astype(np.float32))
        np.save(os.path.join(ct_dir, "visibility.npy"),   ct_vis.astype(np.float32))
        tracker_results["cotracker"] = (ct_2d, ct_vis, ct_dir)
        print(f"   CoTracker done: {ct_2d.shape[0]} tracks  ({time.time()-t0:.1f}s)")

        # SpaTracker
        spa_dir = os.path.join(args.out_dir, "annotations_spatrack")
        os.makedirs(spa_dir, exist_ok=True)
        print(f"\n   [2/3] SpaTracker ...")
        t0 = time.time()
        try:
            spa_2d, spa_vis = run_spatrack_subprocess(
                args.frames_dir, args.depth_dir, spa_dir,
                grid_size=args.spatrack_grid, gpu=gpu_idx,
            )
            spa_2d, spa_vis = merge_wraparound_tracks(spa_2d, spa_vis, W)
            np.save(os.path.join(spa_dir, "tracks_2d.npy"),   spa_2d.astype(np.float32))
            np.save(os.path.join(spa_dir, "visibility.npy"),   spa_vis.astype(np.float32))
            tracker_results["spatrack"] = (spa_2d, spa_vis, spa_dir)
            print(f"   SpaTracker done: {spa_2d.shape[0]} tracks  ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"   SpaTracker FAILED: {e}")

        # TAPIP3D
        tap_dir = os.path.join(args.out_dir, "annotations_tapip3d")
        os.makedirs(tap_dir, exist_ok=True)
        print(f"\n   [3/3] TAPIP3D ...")
        t0 = time.time()
        try:
            tap_2d, tap_vis = run_tapip3d_subprocess(
                args.frames_dir, args.depth_dir, tap_dir,
                grid_size=args.tapip3d_grid, gpu=gpu_idx,
            )
            tap_2d, tap_vis = merge_wraparound_tracks(tap_2d, tap_vis, W)
            np.save(os.path.join(tap_dir, "tracks_2d.npy"),   tap_2d.astype(np.float32))
            np.save(os.path.join(tap_dir, "visibility.npy"),   tap_vis.astype(np.float32))
            tracker_results["tapip3d"] = (tap_2d, tap_vis, tap_dir)
            print(f"   TAPIP3D done: {tap_2d.shape[0]} tracks  ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"   TAPIP3D FAILED: {e}")

        # Compare
        print(f"\n── Step 2-5: Lifting 3D + Metrics + Comparison ──")
        compare_trackers(
            tracker_results, frames_rgb, depth_paths, W, H,
            max_depth=args.max_depth,
            out_dir=args.out_dir,
            fps=args.fps,
        )

        # Visualise each tracker
        if not args.skip_viz:
            for name, (tr2d, trvis, ann_dir) in tracker_results.items():
                viz_dir = os.path.join(ann_dir, "viz")
                os.makedirs(viz_dir, exist_ok=True)
                print(f"\n── Visualisation: {name} ──")
                visualize_tracks_on_frames(
                    frames_rgb, tr2d, trvis,
                    os.path.join(viz_dir, "tracks_2d.mp4"), fps=args.fps
                )

        print(f"\n{'='*60}")
        print(f" Done!  Comparison → {args.out_dir}/comparison/")
        print(f"{'='*60}\n")

    else:
        # ── Single tracker mode ──────────────────────────────────────────
        print(f"\n── Step 1: 2D Point Tracking ({args.tracker}) ──")
        t0 = time.time()

        if args.tracker == "cotracker":
            tracks_2d, vis = run_cotracker(
                frames_rgb, grid_size=args.grid_size,
                pad_frac=args.pad_frac, device=args.device,
                checkpoint=args.checkpoint,
            )
        elif args.tracker == "spatrack":
            tracks_2d, vis = run_spatrack_subprocess(
                args.frames_dir, args.depth_dir,
                os.path.join(args.out_dir, "_spa_tmp"),
                grid_size=args.spatrack_grid, gpu=gpu_idx,
            )
        elif args.tracker == "tapip3d":
            tracks_2d, vis = run_tapip3d_subprocess(
                args.frames_dir, args.depth_dir,
                os.path.join(args.out_dir, "_tap_tmp"),
                grid_size=args.tapip3d_grid, gpu=gpu_idx,
            )
        else:
            tracks_2d, vis = run_optflow_tracker(
                frames_rgb, grid_step=args.grid_step, pad_frac=args.pad_frac
            )

        print(f"   {tracks_2d.shape[0]} tracks × {T} frames  ({time.time()-t0:.1f}s)")

        # ── Wrap-around merging ──
        print(f"\n── Step 2: Wrap-around Track Merging ──")
        tracks_2d, vis = merge_wraparound_tracks(tracks_2d, vis, W)

        # ── 3D Lifting ──
        print(f"\n── Step 3: Lifting 2D → 3D (DAP depth) ──")
        tracks_3d, sampled_depth = lift_tracks_to_3d(
            tracks_2d, vis, depth_paths, W, H, max_depth=args.max_depth
        )

        # ── Camera Motion Detection & Compensation ──
        print(f"\n── Step 3b: Camera Motion Detection & Compensation ──")
        tracks_3d_stabilized, camera_info = detect_and_compensate_camera_motion(
            tracks_2d, tracks_3d, vis, W, H, method=args.camera_motion
        )

        # ── Consistency Metrics (on stabilized tracks) ──
        print(f"\n── Step 4: Consistency Metrics ──")
        print("   Computing reprojection error ...")
        reproj_err = compute_reprojection_error(tracks_2d, tracks_3d, vis, W, H)

        print("   Computing 3D smoothness (stabilized) ...")
        smoothness = compute_3d_smoothness(tracks_3d_stabilized, vis)

        print("   Computing rigidity score (stabilized) ...")
        rigidity = compute_rigidity_score(tracks_3d_stabilized, vis)

        # ── Save ──
        print(f"\n── Step 5: Saving Annotations ──")
        meta = {
            "video_resolution":  [W, H],
            "n_frames":          T,
            "tracker":           args.tracker,
            "grid_size":         args.grid_size if args.tracker == "cotracker" else args.grid_step,
            "pad_frac":          args.pad_frac,
            "max_depth_m":       args.max_depth,
            "depth_scale":       MAX_DEPTH_SCALE,
            "camera_motion":     {
                "method": str(camera_info["method"]),
                "is_static": bool(camera_info["is_static"]),
                "inlier_ratio": float(camera_info["inlier_ratio"]) if camera_info.get("inlier_ratio") is not None else None,
            },
            "frame_files":       frame_files,
            "depth_files":       depth_files,
        }

        # ── Per-track static/dynamic classification ──
        # Combine two signals:
        #   1. Camera RANSAC inliers → consistent with global camera motion (= background)
        #   2. Rigidity score → low score means rigid (= static scene element)
        inlier_ratio_per_track = np.zeros(N, dtype=np.float32)
        if "inlier_masks" in camera_info:
            inlier_masks_cam = camera_info["inlier_masks"]  # (N, T)
            vis_count = (vis > 0.5).sum(axis=1).clip(min=1)
            inlier_ratio_per_track = inlier_masks_cam.sum(axis=1).astype(np.float32) / vis_count
        elif camera_info.get("is_static", True):
            inlier_ratio_per_track[:] = 1.0

        rigidity_thresh = np.nanmedian(rigidity) * 2.0 if np.any(~np.isnan(rigidity)) else 0.1
        is_rigid = np.where(np.isnan(rigidity), True, rigidity < rigidity_thresh)
        is_inlier = inlier_ratio_per_track > 0.5

        track_is_static = (is_rigid & is_inlier).astype(np.float32)  # (N,)
        n_static = int(track_is_static.sum())
        print(f"   Static tracks: {n_static}/{N} ({100*n_static/max(N,1):.0f}%)")

        # Save both raw and stabilized 3D tracks
        save_annotations(
            args.out_dir, tracks_2d, tracks_3d_stabilized, vis,
            sampled_depth, reproj_err, smoothness, rigidity, meta
        )
        np.save(os.path.join(args.out_dir, "track_is_static.npy"),
                track_is_static.astype(np.float32))
        print(f"   ├── track_is_static.npy ({track_is_static.shape})")
        if not camera_info["is_static"]:
            np.save(os.path.join(args.out_dir, "tracks_3d_raw.npy"),
                    tracks_3d.astype(np.float32))
            # Save per-frame camera poses
            if "poses" in camera_info:
                poses_arr = np.array([
                    np.hstack([p['R'].ravel(), p['t']])
                    for p in camera_info["poses"]
                ], dtype=np.float32)  # (T, 12)
                np.save(os.path.join(args.out_dir, "camera_poses.npy"), poses_arr)
                print(f"   ├── tracks_3d_raw.npy   (unstabilized)")
                print(f"   ├── camera_poses.npy    (T, 12: R_flat + t)")

        # ── Visualisation ──
        if not args.skip_viz:
            import matplotlib.pyplot as plt

            viz_dir = os.path.join(args.out_dir, "viz")
            os.makedirs(viz_dir, exist_ok=True)

            print(f"\n── Step 6: Visualisation ──")
            visualize_tracks_on_frames(
                frames_rgb, tracks_2d, vis,
                os.path.join(viz_dir, "tracks_2d.mp4"), fps=args.fps
            )
            visualize_3d_trajectories(
                tracks_3d, vis,
                os.path.join(viz_dir, "trajectories_3d.png")
            )
            visualize_3d_trajectory_video(
                tracks_3d, vis,
                os.path.join(viz_dir, "trajectories_3d.mp4"),
                fps=args.fps,
            )

        print(f"\n{'='*60}")
        print(f" Done!  All outputs → {args.out_dir}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()