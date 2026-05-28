"""
Render 360-degree ERP video clips with perfect ground-truth depth & camera pose
from Habitat-Sim using Replica or HM3D scenes.

Output for each clip <id>:
  videos/<id>.mp4              1024x512, 25fps, 125 frames, 5 seconds  ← matches PanoWorld training (93 frames @ ~16 fps ≈ 5 s)
  depth/<id>/0000.npy ... 0124.npy   per-frame metric depth (H,W) float32
  camera_poses/<id>.npy        (T, 4, 4) camera-to-world (Habitat convention)
  meta/<id>.json               scene id, camera kind, traj kind, intrinsics, etc.

The trajectories are deliberately diverse (all distances/angles below describe
the entire 5-second clip, so peak motion stays at realistic indoor speeds):
  - 50% smooth forward walk (~0.15-0.30 m/s, total ~0.7-1.5 m over 5 s)
  - 25% slow rotation in place (total yaw ±45 deg over the clip)
  - 15% combined translate + rotate (~0.10-0.20 m/s, ±30 deg total)
  - 10% completely static (pure environment scan via "render only")

Tracks are NOT rendered here — they are computed analytically from depth+pose
when integrated into the test set (see consolidate_test_set.py back on the
training machine).

Usage on A100:
    conda activate habitat
    python render_habitat_erp.py \
        --scenes_root /path/to/Replica  \
        --out_root    /path/to/out      \
        --num_clips   50                 \
        --width       1024 --height 512  \
        --fps         25 --frames        25
"""

from __future__ import annotations
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np


def ensure_imports():
    """Late-import heavy deps so --help works without habitat installed."""
    global habitat_sim, magnum, mn
    import habitat_sim  # type: ignore
    import magnum as mn  # type: ignore


# --------------------------------------------------------------- scenes ----

REPLICA_SCENE_IDS = [
    # Replica official (FAIR) — 18 scenes; pick those with reasonable navigable area
    "apartment_0", "apartment_1", "apartment_2",
    "frl_apartment_0", "frl_apartment_1", "frl_apartment_2",
    "frl_apartment_3", "frl_apartment_4", "frl_apartment_5",
    "office_0", "office_1", "office_2", "office_3", "office_4",
    "room_0", "room_1", "room_2",
    "hotel_0",
]


def find_scene_path(scenes_root: Path, scene_id: str) -> Path | None:
    """Search common Replica/HM3D layouts."""
    candidates = [
        scenes_root / scene_id / "habitat" / "mesh_semantic.ply",
        scenes_root / scene_id / "mesh.ply",
        scenes_root / scene_id / f"{scene_id}.glb",
        scenes_root / scene_id / "habitat" / "info_semantic.json",  # full habitat layout
        # HM3D structure
        scenes_root / scene_id / f"{scene_id}.basis.glb",
    ]
    for c in candidates:
        if c.exists():
            return c
    # fall back: any glb/ply in scene dir
    sd = scenes_root / scene_id
    if sd.is_dir():
        for ext in ("*.glb", "*.basis.glb", "*.ply"):
            found = list(sd.glob(ext)) + list(sd.glob(f"**/{ext}"))
            if found:
                return found[0]
    return None


# ----------------------------------------------------------- simulator ----

def make_sim(scene_path: Path, width: int, height: int):
    """Build a Habitat simulator with ONE EquirectangularSensor (color + depth)."""
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = str(scene_path)
    backend_cfg.enable_physics = False
    backend_cfg.create_renderer = True

    agent_cfg = habitat_sim.AgentConfiguration()

    color_spec = habitat_sim.EquirectangularSensorSpec()
    color_spec.uuid = "color_erp"
    color_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_spec.resolution = [height, width]
    color_spec.position = [0.0, 0.0, 0.0]

    depth_spec = habitat_sim.EquirectangularSensorSpec()
    depth_spec.uuid = "depth_erp"
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_spec.resolution = [height, width]
    depth_spec.position = [0.0, 0.0, 0.0]

    agent_cfg.sensor_specifications = [color_spec, depth_spec]

    sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    return habitat_sim.Simulator(sim_cfg)


# ---------------------------------------------------- trajectory sampling ----

def sample_trajectory(sim, n_frames: int, kind: str, rng: random.Random):
    """Generate (n_frames, 4, 4) camera-to-world poses (Habitat convention).

    Returns None if no valid start point can be found.
    """
    nav = sim.pathfinder
    if not nav.is_loaded:
        return None

    # Random start
    for _ in range(50):
        p0 = nav.get_random_navigable_point()
        if np.isfinite(p0).all():
            break
    else:
        return None

    # Random initial yaw (radians), pitch=0 (eye level)
    yaw0 = rng.uniform(0, 2 * np.pi)

    poses = []
    pos = np.array(p0, dtype=np.float64)
    yaw = yaw0
    pitch = 0.0

    if kind == "static":
        for _ in range(n_frames):
            poses.append(make_pose(pos, yaw, pitch))
    elif kind == "rotate":
        # ~30 deg over the clip
        delta_yaw = np.deg2rad(rng.uniform(-45, 45)) / max(n_frames - 1, 1)
        for i in range(n_frames):
            poses.append(make_pose(pos, yaw + delta_yaw * i, pitch))
    elif kind == "walk":
        # Walk forward at indoor pace; total distance ~0.7-1.5 m over a 5 s clip.
        speed = rng.uniform(0.15, 0.30)  # m/s
        dt = 1.0 / 25.0  # assume 25 fps
        for i in range(n_frames):
            poses.append(make_pose(pos, yaw, pitch))
            forward = np.array([np.sin(yaw), 0.0, -np.cos(yaw)]) * speed * dt
            new_pos = pos + forward
            # Snap to navmesh
            snapped = nav.snap_point(new_pos)
            if np.isfinite(snapped).all() and nav.is_navigable(snapped):
                pos = np.array(snapped, dtype=np.float64)
            else:
                # blocked — stop walking, hold pose
                pass
    elif kind == "walk_rotate":
        speed = rng.uniform(0.10, 0.20)
        delta_yaw = np.deg2rad(rng.uniform(-30, 30)) / max(n_frames - 1, 1)
        dt = 1.0 / 25.0
        for i in range(n_frames):
            poses.append(make_pose(pos, yaw + delta_yaw * i, pitch))
            forward = np.array([np.sin(yaw + delta_yaw * i), 0.0,
                                -np.cos(yaw + delta_yaw * i)]) * speed * dt
            new_pos = pos + forward
            snapped = nav.snap_point(new_pos)
            if np.isfinite(snapped).all() and nav.is_navigable(snapped):
                pos = np.array(snapped, dtype=np.float64)
    else:
        raise ValueError(kind)

    return np.stack(poses, axis=0)  # (T, 4, 4)


def make_pose(pos: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    """Habitat convention: y-up; camera looks toward -Z."""
    Ry = np.array([
        [np.cos(yaw),  0.0, np.sin(yaw), 0.0],
        [0.0,          1.0, 0.0,         0.0],
        [-np.sin(yaw), 0.0, np.cos(yaw), 0.0],
        [0.0,          0.0, 0.0,         1.0],
    ])
    Rx = np.array([
        [1.0, 0.0,             0.0,            0.0],
        [0.0, np.cos(pitch),  -np.sin(pitch), 0.0],
        [0.0, np.sin(pitch),   np.cos(pitch), 0.0],
        [0.0, 0.0,             0.0,            1.0],
    ])
    R = Ry @ Rx
    R[:3, 3] = pos
    return R


# -------------------------------------------------------------- render ----

def set_agent_state(sim, pose: np.ndarray):
    """Apply a 4x4 c2w pose to the agent."""
    agent = sim.get_agent(0)
    state = agent.get_state()
    state.position = pose[:3, 3].astype(np.float32)
    R = pose[:3, :3]
    # quaternion from rotation matrix
    state.rotation = quat_from_matrix(R)
    state.sensor_states = {}
    agent.set_state(state, infer_sensor_states=True)


def quat_from_matrix(R: np.ndarray):
    """Habitat uses np.quaternion (w,x,y,z) via the magnum bridge."""
    import quaternion  # bundled with habitat-sim
    return quaternion.from_rotation_matrix(R)


def render_clip(sim, poses: np.ndarray):
    """Run the simulator at each pose; return color (T,H,W,3) uint8 and depth (T,H,W) float32."""
    colors, depths = [], []
    for pose in poses:
        set_agent_state(sim, pose)
        obs = sim.get_sensor_observations()
        color = obs["color_erp"]
        depth = obs["depth_erp"]
        if color.shape[-1] == 4:
            color = color[..., :3]
        colors.append(color.astype(np.uint8))
        depths.append(depth.astype(np.float32))
    return np.stack(colors, 0), np.stack(depths, 0)


def write_video(frames: np.ndarray, path: Path, fps: int):
    """Encode (T,H,W,3) uint8 into mp4 via ffmpeg pipe."""
    import subprocess
    T, H, W, _ = frames.shape
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    proc.stdin.write(frames.tobytes())
    proc.stdin.close()
    proc.wait()


# ---------------------------------------------------------------- main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes_root", type=Path, required=True,
                    help="Root directory containing Replica scenes (one subdir per scene)")
    ap.add_argument("--out_root", type=Path, required=True)
    ap.add_argument("--num_clips", type=int, default=50)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--frames", type=int, default=125,
                    help="Frames per clip (default 125 for 5-second clips at 25fps, "
                         "matching PanoWorld training distribution)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scenes", nargs="*", default=None,
                    help="Override scene id list; default = builtin Replica list")
    args = ap.parse_args()

    ensure_imports()

    rng = random.Random(args.seed)
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "videos").mkdir(exist_ok=True)
    (args.out_root / "depth").mkdir(exist_ok=True)
    (args.out_root / "camera_poses").mkdir(exist_ok=True)
    (args.out_root / "meta").mkdir(exist_ok=True)

    scene_ids = args.scenes or REPLICA_SCENE_IDS
    available = []
    for s in scene_ids:
        p = find_scene_path(args.scenes_root, s)
        if p is not None:
            available.append((s, p))
        else:
            print(f"  [warn] scene not found: {s}")
    if not available:
        raise SystemExit(f"no scenes found under {args.scenes_root}; "
                         f"expected subdirs like apartment_0/habitat/mesh_semantic.ply")
    print(f"available scenes: {len(available)}")

    kinds = ["walk"] * 25 + ["rotate"] * 12 + ["walk_rotate"] * 8 + ["static"] * 5
    rng.shuffle(kinds)
    kinds = kinds[:args.num_clips]

    manifest = []
    clip_idx = 0
    cur_scene = None
    sim = None
    try:
        for k in kinds:
            scene_id, scene_path = rng.choice(available)
            if scene_id != cur_scene:
                if sim is not None:
                    sim.close()
                print(f"[{clip_idx+1}/{args.num_clips}] loading scene {scene_id}")
                sim = make_sim(scene_path, args.width, args.height)
                cur_scene = scene_id

            # Try up to 5 trajectories until one is fully navigable
            poses = None
            for _ in range(5):
                poses = sample_trajectory(sim, args.frames, k, rng)
                if poses is not None and len(poses) == args.frames:
                    break
            if poses is None:
                print(f"  [skip] could not sample {k} trajectory in {scene_id}")
                continue

            colors, depths = render_clip(sim, poses)

            cid = f"hab_{clip_idx:04d}"
            write_video(colors, args.out_root / "videos" / f"{cid}.mp4", args.fps)
            depth_dir = args.out_root / "depth" / cid
            depth_dir.mkdir(exist_ok=True, parents=True)
            for ti, d in enumerate(depths):
                np.save(depth_dir / f"{ti:04d}.npy", d)
            np.save(args.out_root / "camera_poses" / f"{cid}.npy", poses.astype(np.float32))
            with open(args.out_root / "meta" / f"{cid}.json", "w") as f:
                json.dump({
                    "scene_id": scene_id,
                    "scene_path": str(scene_path),
                    "trajectory_kind": k,
                    "n_frames": int(args.frames),
                    "fps": int(args.fps),
                    "resolution": [args.height, args.width],
                    "depth_units": "meters",
                    "world_up_axis": "y",
                    "camera_convention": "habitat (looks toward -Z)",
                    "depth_min": float(depths.min()),
                    "depth_max": float(depths.max()),
                }, f, indent=2)
            manifest.append({"clip": cid, "scene": scene_id, "traj": k})
            print(f"  [{clip_idx+1}/{args.num_clips}] OK {cid} scene={scene_id} traj={k} "
                  f"depth=[{depths.min():.2f}, {depths.max():.2f}]m")
            clip_idx += 1
    finally:
        if sim is not None:
            sim.close()

    with open(args.out_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDONE: {len(manifest)} clips written to {args.out_root}")


if __name__ == "__main__":
    main()
