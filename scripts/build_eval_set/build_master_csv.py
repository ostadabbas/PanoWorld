"""Build $PANO_DATA_ROOT/test/master.csv from the three splits.

Columns:
    clip_id            unique within the master csv (split prefix)
    split              one of {self_iid, argus_ood, habitat_ood}
    video_path         absolute path to the .mp4 (1024x512 or smaller)
    annotation_dir     absolute path to per-clip GT directory
    caption_path       absolute path to {clip}.json gpt5mini caption
    n_frames           int
    fps                int (best-effort from ffprobe, default 25)
    resolution         "1024x512"
    is_static          1 if camera is static, else 0
    has_depth          1 if depth/ subdir or depth.npy present
    has_tracks         1 if tracks_2d.npy + tracks_3d.npy present
    has_camera_poses   1 if camera_poses.npy present
    depth_kind         "dap_estimated" | "habitat_gt"
    pose_kind          "pnp" | "habitat_gt" | "" (none)
    trajectory_kind    "static"|"walk"|"rotate"|"walk_rotate"|"" (only habitat)
    scene_id           only habitat (Replica scene)

150 rows: 50 self_iid + 50 argus_ood + 50 habitat_ood.
"""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path

_PANO_DATA_ROOT = os.environ.get(
    "PANO_DATA_ROOT", os.path.join(os.path.expanduser("~"), "pano_video_data")
)
ROOT = Path(os.environ.get("PANO_TEST_ROOT", os.path.join(_PANO_DATA_ROOT, "test")))
OUT  = ROOT / "master.csv"

FIELDS = [
    "clip_id", "split", "video_path", "annotation_dir", "caption_path",
    "n_frames", "fps", "resolution",
    "is_static", "has_depth", "has_tracks", "has_camera_poses",
    "depth_kind", "pose_kind", "trajectory_kind", "scene_id",
]


def has_depth(ann_dir: Path) -> bool:
    return (ann_dir / "depth").is_dir() or (ann_dir / "depth.npy").exists()


def has_tracks(ann_dir: Path) -> bool:
    return (ann_dir / "tracks_2d.npy").exists() and (ann_dir / "tracks_3d.npy").exists()


def has_poses(ann_dir: Path) -> bool:
    return (ann_dir / "camera_poses.npy").exists()


def build_self_iid() -> list[dict]:
    split_dir = ROOT / "self_iid"
    rows = []
    # videos may be symlinks; just enumerate annotations
    for ann in sorted((split_dir / "annotations").iterdir()):
        if not ann.is_dir():
            continue
        cid = ann.name
        meta_p = ann / "meta.json"
        if meta_p.exists():
            meta = json.load(open(meta_p))
        else:
            meta = {}
        n_frames = int(meta.get("n_frames", 20))
        is_static = bool(meta.get("camera_motion", {}).get("is_static", True))
        # video lives in self_iid/videos/<cid>.mp4 (symlink)
        video = split_dir / "videos" / f"{cid}.mp4"
        cap = split_dir / "captions" / f"{cid}.json"
        rows.append({
            "clip_id": f"self_iid::{cid}",
            "split":   "self_iid",
            "video_path":     str(video),
            "annotation_dir": str(ann),
            "caption_path":   str(cap) if cap.exists() else "",
            "n_frames":  n_frames,
            "fps":       25,
            "resolution":"1024x512",
            "is_static": int(is_static),
            "has_depth": int(has_depth(ann)),
            "has_tracks":int(has_tracks(ann)),
            "has_camera_poses": int(has_poses(ann)),
            "depth_kind":"dap_estimated",
            "pose_kind": "pnp" if has_poses(ann) else "",
            "trajectory_kind": "static" if is_static else "",
            "scene_id": "",
        })
    return rows


def build_argus_ood() -> list[dict]:
    split_dir = ROOT / "argus_ood"
    rows = []
    ann_root = split_dir / "annotations"
    for ann in sorted(ann_root.iterdir()):
        if not ann.is_dir():
            continue
        cid = ann.name  # e.g. "0000"
        meta_p = ann / "meta.json"
        meta = json.load(open(meta_p)) if meta_p.exists() else {}
        n_frames = int(meta.get("n_frames", 20))
        is_static = bool(meta.get("camera_motion", {}).get("is_static", True))
        video = split_dir / "videos_normalized" / f"{cid}.mp4"
        cap = split_dir / "captions" / f"{cid}.json"
        rows.append({
            "clip_id": f"argus_ood::{cid}",
            "split":   "argus_ood",
            "video_path":     str(video),
            "annotation_dir": str(ann),
            "caption_path":   str(cap) if cap.exists() else "",
            "n_frames":  n_frames,
            "fps":       25,
            "resolution":"1024x512",
            "is_static": int(is_static),
            "has_depth": int(has_depth(ann)),
            "has_tracks":int(has_tracks(ann)),
            "has_camera_poses": int(has_poses(ann)),
            "depth_kind":"dap_estimated",
            "pose_kind": "pnp" if has_poses(ann) else "",
            "trajectory_kind": "static" if is_static else "moving",
            "scene_id": "",
        })
    return rows


def build_habitat_ood() -> list[dict]:
    split_dir = ROOT / "habitat_ood"
    rows = []
    ann_root = split_dir / "annotations"
    for ann in sorted(ann_root.iterdir()):
        if not ann.is_dir():
            continue
        cid = ann.name  # e.g. "hab_0000"
        meta_p = ann / "meta.json"
        meta = json.load(open(meta_p)) if meta_p.exists() else {}
        n_frames = int(meta.get("n_frames", 25))
        traj_kind = meta.get("trajectory_kind", "")
        is_static = (traj_kind == "static")
        scene_id = meta.get("scene_id", "")
        video = split_dir / "videos" / f"{cid}.mp4"
        cap = split_dir / "captions" / f"{cid}.json"
        rows.append({
            "clip_id": f"habitat_ood::{cid}",
            "split":   "habitat_ood",
            "video_path":     str(video),
            "annotation_dir": str(ann),
            "caption_path":   str(cap) if cap.exists() else "",
            "n_frames":  n_frames,
            "fps":       25,
            "resolution":"1024x512",
            "is_static": int(is_static),
            # tracks not pre-computed for habitat; depth+pose are GT
            "has_depth": int(has_depth(ann)),
            "has_tracks":0,
            "has_camera_poses": int(has_poses(ann)),
            "depth_kind":"habitat_gt",
            "pose_kind": "habitat_gt" if has_poses(ann) else "",
            "trajectory_kind": traj_kind,
            "scene_id": scene_id,
        })
    return rows


def main():
    rows = []
    rows.extend(build_self_iid())
    rows.extend(build_argus_ood())
    rows.extend(build_habitat_ood())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {OUT}: {len(rows)} rows")
    by_split = {}
    for r in rows:
        by_split.setdefault(r["split"], []).append(r)
    for s, lst in by_split.items():
        n_static  = sum(r["is_static"]  for r in lst)
        n_depth   = sum(r["has_depth"]  for r in lst)
        n_tracks  = sum(r["has_tracks"] for r in lst)
        n_poses   = sum(r["has_camera_poses"] for r in lst)
        n_caps    = sum(1 for r in lst if r["caption_path"])
        print(f"  {s:13s} n={len(lst):3d}  static={n_static:3d}  "
              f"depth={n_depth:3d}  tracks={n_tracks:3d}  poses={n_poses:3d}  caps={n_caps:3d}")


if __name__ == "__main__":
    main()
