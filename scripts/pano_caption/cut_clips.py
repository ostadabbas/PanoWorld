#!/usr/bin/env python3
"""Cut self-collected long panoramic videos into fixed-length clips.

Splits each long video (typically 2-8 min, 3840x1920 @ 25-30 fps) into
non-overlapping 5-second clips via ffmpeg stream copy (fast, no quality loss).

Output layout (flat per-scene prefix):
    output_dir/
        <scene>__<video_stem>__clip000.mp4
        <scene>__<video_stem>__clip001.mp4
        ...
        manifest.json

Example:
    python -m scripts.pano_caption.cut_clips \
        --source_root $PANO_DATA_ROOT \
        --scenes Le_home_foodcourt Le_school street xiangyu shayda Le_prudential_apartment \
        --output_dir $PANO_DATA_ROOT/self_collected_clips \
        --clip_seconds 5
"""

import argparse
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def get_duration(video_path: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True, check=True,
    )
    return float(r.stdout.strip())


def cut_one_video(args: tuple) -> dict:
    scene, video_path, out_dir, clip_seconds, min_keep_seconds = args
    stem = Path(video_path).stem
    prefix = f"{scene}__{stem}"

    try:
        duration = get_duration(video_path)
    except Exception as e:
        return {"scene": scene, "video": video_path, "error": str(e), "clips": []}

    n_clips = int(duration // clip_seconds)
    # Optionally keep last partial clip if >= min_keep_seconds
    remainder = duration - n_clips * clip_seconds
    if remainder >= min_keep_seconds:
        n_clips += 1

    clips = []
    for i in range(n_clips):
        start = i * clip_seconds
        out_path = os.path.join(out_dir, f"{prefix}__clip{i:03d}.mp4")
        if os.path.exists(out_path):
            clips.append({"path": out_path, "start": start, "skipped": True})
            continue

        # Stream-copy for speed; -avoid_negative_ts to align PTS; -map 0:v:0 to drop audio
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start}", "-i", video_path,
            "-t", f"{clip_seconds}",
            "-map", "0:v:0",
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            out_path,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) < 1024:
            # Fallback: transcode (some files have keyframe misalignment)
            if os.path.exists(out_path):
                os.remove(out_path)
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{start}", "-i", video_path,
                "-t", f"{clip_seconds}",
                "-map", "0:v:0",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                out_path,
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                clips.append({"path": out_path, "start": start, "error": r.stderr[:400]})
                continue
        clips.append({"path": out_path, "start": start})

    return {"scene": scene, "video": video_path, "duration": duration, "clips": clips}


def main():
    ap = argparse.ArgumentParser(description="Cut self-collected pano videos into fixed-length clips")
    _default_source = os.environ.get(
        "PANO_DATA_ROOT", os.path.join(os.path.expanduser("~"), "pano_video_data")
    )
    ap.add_argument("--source_root", default=_default_source)
    ap.add_argument("--scenes", nargs="+", required=True,
                    help="Subfolder names under source_root to process")
    ap.add_argument("--output_dir", required=True,
                    help="Flat output directory for all clips")
    ap.add_argument("--clip_seconds", type=float, default=5.0)
    ap.add_argument("--min_keep_seconds", type=float, default=4.5,
                    help="If remainder after full clips is >= this, keep as partial clip. "
                         "Set to >= clip_seconds to strictly discard any tail (recommended for "
                         "consistent 5s clip length).")
    ap.add_argument("--recursive", action="store_true",
                    help="Recurse into scene dir to find mp4/mov/mkv. Required for datasets "
                         "with nested layouts (e.g. 360x_dataset/360x_dataset/panoramic/*.mp4).")
    ap.add_argument("--scene_label", default=None,
                    help="Override scene label used as output-filename prefix. Useful when the "
                         "scene folder has a long/nested name (e.g. --scenes 360x_dataset "
                         "--scene_label x360). Applies to ALL --scenes if set.")
    ap.add_argument("--workers", type=int, default=6, help="Parallel ffmpeg workers")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    jobs = []
    VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".MP4", ".MOV", ".MKV")
    for scene in args.scenes:
        scene_dir = os.path.join(args.source_root, scene)
        if not os.path.isdir(scene_dir):
            print(f"  [SKIP] {scene_dir} not found")
            continue
        label = args.scene_label or scene
        if args.recursive:
            # rglob flattens nested layouts; sort for deterministic ordering
            vpaths = sorted(str(p) for p in Path(scene_dir).rglob("*")
                            if p.is_file() and p.suffix in VIDEO_EXTS)
        else:
            vpaths = [os.path.join(scene_dir, vf) for vf in sorted(os.listdir(scene_dir))
                      if vf.endswith(VIDEO_EXTS)]
        for vp in vpaths:
            jobs.append((label, vp, args.output_dir, args.clip_seconds, args.min_keep_seconds))

    print(f"Found {len(jobs)} source videos across {len(args.scenes)} scenes.")
    print(f"Output: {args.output_dir}  |  clip_seconds={args.clip_seconds}  workers={args.workers}")

    manifest = {"clip_seconds": args.clip_seconds, "videos": []}
    n_clips_total = 0
    n_failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(cut_one_video, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs)):
            res = fut.result()
            manifest["videos"].append(res)
            if "error" in res:
                n_failed += 1
                print(f"  [{i+1}/{len(jobs)}] FAIL {res['video']}: {res['error']}")
            else:
                n_ok = sum(1 for c in res["clips"] if "error" not in c)
                n_bad = sum(1 for c in res["clips"] if "error" in c)
                n_clips_total += n_ok
                print(f"  [{i+1}/{len(jobs)}] {Path(res['video']).name} "
                      f"dur={res['duration']:.1f}s -> {n_ok} clips"
                      + (f" ({n_bad} failed)" if n_bad else ""))

    mpath = os.path.join(args.output_dir, "manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"Total clips: {n_clips_total}")
    print(f"Failed source videos: {n_failed}")
    print(f"Manifest: {mpath}")


if __name__ == "__main__":
    main()
