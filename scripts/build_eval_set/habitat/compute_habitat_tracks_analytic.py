"""Generate analytical GT tracks for habitat clips from depth + camera_poses.

Habitat conventions used here (must match render_habitat_erp.py):
  - Camera frame: y-up, camera looks toward -Z.
  - EquirectangularSensor pixel (u, v) -> spherical angles in camera frame:
        lon (phi)   = (u / W - 0.5) * 2*pi      (lon=0  at center column)
        lat (theta) = (0.5 - v / H) * pi        (lat=0  at horizon)
    -> unit direction in camera frame:
        d_cam = (cos(theta)*sin(phi), sin(theta), -cos(theta)*cos(phi))
  - Depth is euclidean distance (||P_cam|| in camera frame), NOT z-distance.
    P_cam = d_cam * depth.
  - World point = c2w @ [P_cam; 1].

Output (per clip), matching argus / self_iid format:
  tracks_2d.npy   (N, T, 2)  float16   pixel coords (col, row), col wrapped to [0, W).
  tracks_3d.npy   (N, T, 3)  float16   world coords; rigid scene so constant over t.
  visibility.npy  (N, T)     float16   0/1, depth-consistency occlusion check.

Defaults: GRID=30 -> N=900 tracks per clip, matching argus / self_iid meta.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


# --------------------------------------------------------------- ERP utils ----

def pixel_to_dir_cam(u: np.ndarray, v: np.ndarray, H: int, W: int) -> np.ndarray:
    """ERP pixel -> unit direction in habitat camera frame (-Z forward, +Y up)."""
    phi = (u / W - 0.5) * 2.0 * np.pi
    theta = (0.5 - v / H) * np.pi
    cos_t = np.cos(theta)
    return np.stack(
        [np.sin(phi) * cos_t, np.sin(theta), -np.cos(phi) * cos_t],
        axis=-1,
    )


def dir_cam_to_pixel(d: np.ndarray, H: int, W: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Camera-frame point -> ERP pixel coordinates and euclidean distance."""
    norm = np.linalg.norm(d, axis=-1).clip(min=1e-12)
    dn = d / norm[..., None]
    phi = np.arctan2(dn[..., 0], -dn[..., 2])         # in [-pi, pi]
    theta = np.arcsin(np.clip(dn[..., 1], -1.0, 1.0))  # in [-pi/2, pi/2]
    u = (phi / (2.0 * np.pi) + 0.5) * W
    v = (0.5 - theta / np.pi) * H
    return u, v, norm


def sample_pixel_depth(depth: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinearly sample depth at fractional ERP pixel; horizontal wrap, vertical clip."""
    H_, W_ = depth.shape
    u_w = np.mod(u, W_)
    u0 = np.floor(u_w).astype(np.int32)
    u1 = (u0 + 1) % W_
    fu = u_w - u0
    v_c = np.clip(v, 0.0, H_ - 1.0)
    v0 = np.floor(v_c).astype(np.int32)
    v1 = np.clip(v0 + 1, 0, H_ - 1)
    fv = v_c - v0
    d00 = depth[v0, u0]
    d01 = depth[v0, u1]
    d10 = depth[v1, u0]
    d11 = depth[v1, u1]
    return (1 - fv) * ((1 - fu) * d00 + fu * d01) + fv * ((1 - fu) * d10 + fu * d11)


# ---------------------------------------------------------- per-clip logic ----

def process_clip(
    clip_dir: Path,
    grid: int = 30,
    occlusion_rel: float = 0.05,
    occlusion_abs: float = 0.05,
) -> tuple[int, int, int]:
    poses = np.load(clip_dir / "camera_poses.npy")  # (T, 4, 4) float32
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"{clip_dir.name}: unexpected camera_poses shape {poses.shape}")
    T = poses.shape[0]

    # Load all depth frames (float32, ~250 MB / clip; keeps batch ops simple)
    depth_dir = clip_dir / "depth"
    depth_files = sorted(depth_dir.glob("*.npy"))
    if len(depth_files) < T:
        raise ValueError(
            f"{clip_dir.name}: expected >= {T} depth frames, found {len(depth_files)}"
        )
    depths = np.stack([np.load(p) for p in depth_files[:T]], axis=0)  # (T, H, W)
    _, H, W = depths.shape

    # Sample N=grid*grid points uniformly across pixel space (cell centers)
    half_du = W / grid / 2.0
    half_dv = H / grid / 2.0
    us = np.linspace(half_du, W - half_du, grid)
    vs = np.linspace(half_dv, H - half_dv, grid)
    grid_u, grid_v = np.meshgrid(us, vs, indexing="xy")
    u0 = grid_u.reshape(-1).astype(np.float32)
    v0 = grid_v.reshape(-1).astype(np.float32)
    N = u0.size

    # Frame-0 depth -> world point per track
    d0 = sample_pixel_depth(depths[0], u0, v0)
    valid_mask = (d0 > 0.05) & np.isfinite(d0)
    dir_cam0 = pixel_to_dir_cam(u0, v0, H, W)
    p_cam0 = dir_cam0 * d0[:, None]
    p_h = np.concatenate([p_cam0, np.ones((N, 1), dtype=p_cam0.dtype)], axis=-1)
    p_world = (poses[0].astype(np.float32) @ p_h.T).T[:, :3]  # (N, 3)

    tracks_2d = np.zeros((N, T, 2), dtype=np.float32)
    tracks_3d = np.zeros((N, T, 3), dtype=np.float32)
    visibility = np.zeros((N, T), dtype=np.float32)

    # Frame 0 is the input itself
    tracks_2d[:, 0, 0] = u0
    tracks_2d[:, 0, 1] = v0
    tracks_3d[:, 0, :] = p_world
    visibility[:, 0] = valid_mask.astype(np.float32)

    # Re-project to every other frame via inv(c2w[t])
    for t in range(1, T):
        c2w = poses[t].astype(np.float32)
        R = c2w[:3, :3].T              # w2c rotation
        tvec = -R @ c2w[:3, 3]         # w2c translation
        p_cam_t = (R @ p_world.T + tvec[:, None]).T  # (N, 3)
        u_t, v_t, dist_t = dir_cam_to_pixel(p_cam_t, H, W)
        d_at_proj = sample_pixel_depth(depths[t], u_t, v_t)
        tol = np.maximum(occlusion_rel * dist_t, occlusion_abs)
        vis_t = (
            valid_mask
            & (d_at_proj > 0.05)
            & np.isfinite(d_at_proj)
            & (np.abs(dist_t - d_at_proj) < tol)
        )
        tracks_2d[:, t, 0] = np.mod(u_t, W)
        tracks_2d[:, t, 1] = np.clip(v_t, 0.0, H - 1.0)
        tracks_3d[:, t, :] = p_world  # rigid scene
        visibility[:, t] = vis_t.astype(np.float32)

    np.save(clip_dir / "tracks_2d.npy", tracks_2d.astype(np.float16))
    np.save(clip_dir / "tracks_3d.npy", tracks_3d.astype(np.float16))
    np.save(clip_dir / "visibility.npy", visibility.astype(np.float16))

    # Patch meta.json with track stats so it matches argus / self_iid
    meta_path = clip_dir / "meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}
        meta["n_tracks"] = int(N)
        meta["grid_size"] = int(grid)
        meta["track_method"] = "analytic_depth_pose"
        meta_path.write_text(json.dumps(meta, indent=2))

    return T, N, int(valid_mask.sum())


# ----------------------------------------------------------------- driver ----

def main():
    import os as _os
    _data_root = _os.environ.get(
        "PANO_DATA_ROOT", _os.path.join(_os.path.expanduser("~"), "pano_video_data")
    )
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--annotations_root",
        type=Path,
        default=Path(_data_root) / "test" / "habitat_ood" / "annotations",
    )
    ap.add_argument("--grid", type=int, default=30)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--clip", type=str, default=None,
                    help="Process only this single clip (e.g. hab_0000).")
    args = ap.parse_args()

    clip_dirs: Iterable[Path] = sorted(
        d for d in args.annotations_root.iterdir()
        if d.is_dir() and d.name.startswith("hab_")
    )
    if args.clip is not None:
        clip_dirs = [args.annotations_root / args.clip]

    n = len(list(clip_dirs))
    print(f"Found {n} habitat clip(s) under {args.annotations_root}")

    skipped, processed, failed = 0, 0, 0
    for i, cd in enumerate(sorted(clip_dirs)):
        tag = f"[{i + 1}/{n}] {cd.name}"
        if not (cd / "camera_poses.npy").is_file():
            print(f"  {tag}: SKIP (no camera_poses)")
            skipped += 1
            continue
        if not args.overwrite and (cd / "tracks_2d.npy").is_file():
            print(f"  {tag}: SKIP (tracks already present; use --overwrite to redo)")
            skipped += 1
            continue
        try:
            T, N, n_valid = process_clip(cd, grid=args.grid)
            print(f"  {tag}: T={T} N={N} valid_at_frame0={n_valid}")
            processed += 1
        except Exception as exc:
            print(f"  {tag}: FAILED ({exc})")
            failed += 1
    print(f"\nDone. processed={processed}  skipped={skipped}  failed={failed}")


if __name__ == "__main__":
    main()
