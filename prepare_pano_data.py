#!/usr/bin/env python3
"""Prepare panoramic video datasets.

Organizes WEB360 and x360dataset into the standard directory layout:
    output_dir/
        videos/          -> symlinks or copies of .mp4 files
        metas/           -> {video_basename}.txt  (one caption per file)

Data sources:
    WEB360:      CSV captions + pre-cut 512x1024 videos (2114 clips, 100 frames each)
    x360dataset: panoramic videos + index.json with category/weather metadata (231 long videos)

Usage:
    # Process both datasets (default)
    python prepare_pano_data.py

    # Process only WEB360
    python prepare_pano_data.py --datasets web360

    # Process only x360
    python prepare_pano_data.py --datasets x360

    # Custom paths
    python prepare_pano_data.py \
        --web360_dir /path/to/WEB360/WEB360 \
        --x360_dir /path/to/360x_dataset/360x_dataset \
        --output_dir /path/to/output

    # Generate train/val split CSV
    python prepare_pano_data.py --val_ratio 0.05
"""

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path


# ── x360 category → descriptive caption templates ────────────────────────
CATEGORY_TEMPLATES = {
    "Indoor Residential Spaces": "A panoramic view inside a residential living space with furniture and household items.",
    "Indoor Entertainment Venues": "A panoramic view of an indoor entertainment venue with ambient lighting.",
    "Transportation Interiors": "A panoramic view from inside a vehicle or public transport.",
    "Kitchen": "A panoramic view of a kitchen interior with counters, cabinets, and appliances.",
    "Indoor Shops & Retail& Commercial": "A panoramic view inside a retail shop or commercial store.",
    "Dining & Food Outlets": "A panoramic view of a restaurant or dining area with tables and chairs.",
    "Bars & Nightlife": "A panoramic view of a bar or nightlife venue with dim lighting and drinks.",
    "Outdoor Residences & Living": "A panoramic outdoor view of a residential area with buildings and vegetation.",
    "Agriculture & Rural": "A panoramic view of a rural agricultural landscape with fields and vegetation.",
    "Urban Constructions & street": "A panoramic view of an urban street with buildings, roads, and city infrastructure.",
    "Open Public Spaces": "A panoramic view of an open public space with people and structures.",
    "Outdoor Commercial & Markets": "A panoramic view of an outdoor market or commercial area.",
    "Nature": "A panoramic view of a natural landscape with trees, hills, or water.",
    "Parks & Recreational Areas": "A panoramic view of a park or recreational area with greenery.",
    "Outdoor Transportation": "A panoramic view of an outdoor transportation scene with roads or vehicles.",
    "Historic & Religious Sites": "A panoramic view of a historic or religious site with architectural details.",
    "Waterfronts & Water Bodies": "A panoramic view of a waterfront area with a lake, river, or ocean.",
    "Campus": "A panoramic view of a campus with buildings and walkways.",
    "Indoor Educational Spaces": "A panoramic view inside an educational facility such as a classroom or library.",
    "Artistic Spaces": "A panoramic view of an art gallery or museum with exhibits and displays.",
    "Hotel & Temporary Stay": "A panoramic view of a hotel lobby or guest room interior.",
    "Public Gathering & Conference Spaces": "A panoramic view of a conference hall or public gathering space.",
    "Scientific interior space": "A panoramic view inside a scientific or laboratory space.",
    "Storage & Utility": "A panoramic view of a storage or utility room.",
    "Indoor sports venues": "A panoramic view of an indoor sports facility.",
    "Outdoor Sports & Athletic Fields": "A panoramic view of an outdoor sports field or athletic area.",
    "Workspaces": "A panoramic view of a workspace or office environment.",
    "Elevators & Escalators&Stairs": "A panoramic view near elevators, escalators, or a staircase.",
}

WEATHER_SUFFIX = {
    "indoor": "",
    "sunny": " The scene is brightly lit by natural sunlight under clear skies.",
    "cloudy": " The sky is overcast with soft diffused lighting.",
    "rainy": " Rain is falling, creating wet surfaces and a gloomy atmosphere.",
    "haze": " The air is hazy, reducing visibility in the distance.",
    "clear": " The weather is clear with good visibility.",
    "haze+rainy": " The conditions are hazy and rainy with reduced visibility.",
}


def generate_x360_caption(category: str, weather: str) -> str:
    """Generate a descriptive caption from x360 metadata fields."""
    base = CATEGORY_TEMPLATES.get(category, f"A panoramic video of {category}.")
    suffix = WEATHER_SUFFIX.get(weather, "")
    return base + suffix


def process_web360(web360_dir: str, output_dir: str, stats: dict) -> list[str]:
    """Process WEB360 dataset: create video symlinks and caption txt files.

    Returns list of video basenames processed.
    """
    video_src = os.path.join(web360_dir, "videos_512x1024x100")
    csv_path = os.path.join(web360_dir, "WEB360_360TF_train.csv")

    if not os.path.exists(video_src):
        print(f"  [SKIP] WEB360 video dir not found: {video_src}")
        return []
    if not os.path.exists(csv_path):
        print(f"  [SKIP] WEB360 CSV not found: {csv_path}")
        return []

    captions = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row["videoid"].strip()
            name = row["name"].strip().strip('"')
            captions[vid] = name

    print(f"  Loaded {len(captions)} captions from CSV")

    videos_dir = os.path.join(output_dir, "videos")
    metas_dir = os.path.join(output_dir, "metas")

    processed = []
    skipped = 0
    for vid, caption in captions.items():
        src_video = os.path.join(video_src, f"{vid}.mp4")
        if not os.path.exists(src_video):
            skipped += 1
            continue

        basename = f"web360_{vid}"
        dst_video = os.path.join(videos_dir, f"{basename}.mp4")
        dst_caption = os.path.join(metas_dir, f"{basename}.txt")

        if not os.path.exists(dst_video):
            os.symlink(os.path.abspath(src_video), dst_video)

        with open(dst_caption, "w", encoding="utf-8") as f:
            f.write(caption)

        processed.append(basename)

    stats["web360_processed"] = len(processed)
    stats["web360_skipped"] = skipped
    print(f"  Processed {len(processed)} WEB360 videos ({skipped} skipped - no video file)")
    return processed


def process_x360(x360_dir: str, output_dir: str, stats: dict) -> list[str]:
    """Process x360dataset: create video symlinks and generate captions from metadata.

    Returns list of video basenames processed.
    """
    pano_dir = os.path.join(x360_dir, "panoramic")
    index_path = os.path.join(x360_dir, "index.json")

    if not os.path.exists(pano_dir):
        print(f"  [SKIP] x360 panoramic dir not found: {pano_dir}")
        return []
    if not os.path.exists(index_path):
        print(f"  [SKIP] x360 index.json not found: {index_path}")
        return []

    with open(index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    meta_by_uuid = {item["uuid"]: item for item in index_data}
    print(f"  Loaded {len(meta_by_uuid)} entries from index.json")

    videos_dir = os.path.join(output_dir, "videos")
    metas_dir = os.path.join(output_dir, "metas")

    video_files = [f for f in os.listdir(pano_dir) if f.endswith(".mp4")]
    print(f"  Found {len(video_files)} panoramic video files")

    processed = []
    no_meta = 0
    for vf in sorted(video_files):
        uuid = vf.replace(".mp4", "")
        src_video = os.path.join(pano_dir, vf)

        basename = f"x360_{uuid}"
        dst_video = os.path.join(videos_dir, f"{basename}.mp4")
        dst_caption = os.path.join(metas_dir, f"{basename}.txt")

        if not os.path.exists(dst_video):
            os.symlink(os.path.abspath(src_video), dst_video)

        meta = meta_by_uuid.get(uuid)
        if meta:
            caption = generate_x360_caption(
                meta.get("category", "unknown scene"),
                meta.get("weather", "indoor"),
            )
        else:
            caption = "A panoramic video of a scene."
            no_meta += 1

        with open(dst_caption, "w", encoding="utf-8") as f:
            f.write(caption)

        processed.append(basename)

    stats["x360_processed"] = len(processed)
    stats["x360_no_meta"] = no_meta
    print(f"  Processed {len(processed)} x360 videos ({no_meta} without metadata)")
    return processed


def process_self_collected(
    clips_dir: str,
    output_dir: str,
    stats: dict,
    basename_prefix: str = "self",
) -> list[str]:
    """Process self-collected pano clips (already cut + captioned by Gemini).

    Expects ``clips_dir`` to contain pairs of ``{basename}.mp4`` and either:
      * ``{basename}.json`` with multi-style captions -- preferred, produced by
        ``scripts.pano_caption.caption_with_gemini``. Shape is
        ``{"<model_key>": {"short": "...", "medium": "...", "long": "..."}}``
        and is consumed directly by
        ``cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video.VideoDataset._load_json_caption``.
      * ``{basename}.txt`` with a single-string caption -- legacy format,
        automatically promoted to JSON (``{"legacy": {"long": "..."}}``)
        so downstream loaders can still sample a single "long" style.

    Videos are symlinked into ``output_dir/videos/`` and captions are copied /
    normalized into ``output_dir/metas/`` as ``.json``. Clips without a valid
    caption are skipped so a partial caption run does not corrupt the split.

    The basename encoded by ``cut_clips.py`` is ``<scene>__<stem>__clip<N>``,
    which is both scene-aware (for split grouping) and globally unique.
    """
    if not os.path.isdir(clips_dir):
        print(f"  [SKIP] self_collected dir not found: {clips_dir}")
        return []

    videos_dir = os.path.join(output_dir, "videos")
    metas_dir = os.path.join(output_dir, "metas")

    processed = []
    no_caption = 0
    json_count = 0
    txt_count = 0
    per_scene = {}
    per_style_counts = {"short": 0, "medium": 0, "long": 0}

    for vf in sorted(os.listdir(clips_dir)):
        if not vf.endswith(".mp4"):
            continue
        stem = vf[:-4]
        src_video = os.path.join(clips_dir, vf)

        json_src = os.path.join(clips_dir, f"{stem}.json")
        txt_src = os.path.join(clips_dir, f"{stem}.txt")

        caption_obj = None  # final JSON payload to write

        if os.path.exists(json_src) and os.path.getsize(json_src) > 0:
            try:
                with open(json_src, "r", encoding="utf-8") as f:
                    caption_obj = json.load(f)
                # basic sanity: must be {model_key: {style: str}}
                if not (isinstance(caption_obj, dict) and caption_obj
                        and isinstance(next(iter(caption_obj.values())), dict)):
                    raise ValueError("unexpected JSON shape")
                model_key = next(iter(caption_obj.keys()))
                styles = caption_obj[model_key]
                # drop empty strings
                styles = {k: v.strip() for k, v in styles.items()
                          if isinstance(v, str) and v.strip()}
                if not styles:
                    raise ValueError("all styles empty")
                caption_obj = {model_key: styles}
                for k in styles:
                    if k in per_style_counts:
                        per_style_counts[k] += 1
                json_count += 1
            except Exception as e:
                print(f"    [WARN] bad JSON for {stem}: {e}")
                caption_obj = None

        if caption_obj is None and os.path.exists(txt_src) and os.path.getsize(txt_src) > 0:
            with open(txt_src, "r", encoding="utf-8") as f:
                caption = f.read().strip()
            if caption:
                caption_obj = {"legacy": {"long": caption}}
                per_style_counts["long"] += 1
                txt_count += 1

        if caption_obj is None:
            no_caption += 1
            continue

        basename = f"{basename_prefix}_{stem}"
        dst_video = os.path.join(videos_dir, f"{basename}.mp4")
        dst_caption = os.path.join(metas_dir, f"{basename}.json")

        if not os.path.exists(dst_video):
            os.symlink(os.path.abspath(src_video), dst_video)
        with open(dst_caption, "w", encoding="utf-8") as f:
            json.dump(caption_obj, f, ensure_ascii=False, indent=2)

        processed.append(basename)
        scene = stem.split("__", 1)[0] if "__" in stem else "unknown"
        per_scene[scene] = per_scene.get(scene, 0) + 1

    stats["self_processed"] = len(processed)
    stats["self_no_caption"] = no_caption
    stats["self_json_captions"] = json_count
    stats["self_txt_captions"] = txt_count
    stats["self_per_scene"] = per_scene
    stats["self_per_style"] = per_style_counts
    print(f"  Processed {len(processed)} self-collected clips "
          f"(json={json_count} txt={txt_count} missing={no_caption})")
    print(f"    style coverage: short={per_style_counts['short']} "
          f"medium={per_style_counts['medium']} long={per_style_counts['long']}")
    for scene, n in sorted(per_scene.items()):
        print(f"    - {scene}: {n}")
    return processed


def _group_key_from_basename(basename: str) -> str:
    """Group-key for scene-aware split.

    Ensures clips from the same source video don't leak across train/val.
    Example basename: "self_Le_home_foodcourt__VID_20260422_061031_00_121__clip007"
      -> group_key = "self_Le_home_foodcourt__VID_20260422_061031_00_121"
    For web360/x360/360x where basename doesn't have "__", the full basename is used.
    """
    if "__clip" in basename:
        return basename.rsplit("__clip", 1)[0]
    return basename


def write_split_csv(all_basenames: list[str], output_dir: str, val_ratio: float):
    """Write train.csv and val.csv listing video basenames.

    Grouped by source video: all clips from the same source video go to the
    same split (prevents train/val leakage for self-collected clips).
    """
    groups: dict[str, list[str]] = {}
    for b in all_basenames:
        groups.setdefault(_group_key_from_basename(b), []).append(b)

    group_keys = sorted(groups.keys())
    random.shuffle(group_keys)

    # Allocate groups to val until total val-clip count reaches the target
    target_val = int(len(all_basenames) * val_ratio)
    val_set, train_set = [], []
    for gk in group_keys:
        items = groups[gk]
        if len(val_set) < target_val:
            val_set.extend(items)
        else:
            train_set.extend(items)
    if not val_set:
        val_set = train_set[-1:]
        train_set = train_set[:-1]

    train_csv = os.path.join(output_dir, "train.csv")
    val_csv = os.path.join(output_dir, "val.csv")

    for path, items in [(train_csv, train_set), (val_csv, val_set)]:
        with open(path, "w", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["video_basename"])
            for item in sorted(items):
                writer.writerow([item])

    print(f"\n  Split: {len(train_set)} train / {len(val_set)} val")
    print(f"  Written: {train_csv}")
    print(f"  Written: {val_csv}")


def main():
    parser = argparse.ArgumentParser(description="Prepare pano datasets for Cosmos training")
    _data_root = os.environ.get(
        "PANO_DATA_ROOT", os.path.join(os.path.expanduser("~"), "pano_video_data")
    )
    parser.add_argument(
        "--web360_dir",
        default=os.path.join(_data_root, "WEB360", "WEB360"),
        help="Path to WEB360 dataset root (env: PANO_DATA_ROOT)",
    )
    parser.add_argument(
        "--x360_dir",
        default=os.path.join(_data_root, "360x_dataset", "360x_dataset"),
        help="Path to x360dataset root (env: PANO_DATA_ROOT)",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(_data_root, "cosmos_pano_train"),
        help="Output directory for unified dataset (env: PANO_DATA_ROOT)",
    )
    parser.add_argument(
        "--self_collected_dir",
        default=os.path.join(_data_root, "self_collected_clips"),
        help="Path to pre-cut + Gemini-captioned self-collected clips (mp4 + txt)",
    )
    parser.add_argument(
        "--datasets",
        default="all",
        choices=["all", "web360", "x360", "self", "web360+self", "x360+self"],
        help="Which datasets to process",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
        help="Fraction of data to hold out for validation (0 to skip split)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"=== Preparing Panoramic Training Data ===")
    print(f"Output: {args.output_dir}")
    print()

    os.makedirs(os.path.join(args.output_dir, "videos"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "metas"), exist_ok=True)

    all_basenames = []
    stats = {}

    do_web360 = args.datasets in ("all", "web360", "web360+self")
    do_x360 = args.datasets in ("all", "x360", "x360+self")
    do_self = args.datasets in ("all", "self", "web360+self", "x360+self")

    if do_web360:
        print("[WEB360] Processing WEB360...")
        all_basenames.extend(process_web360(args.web360_dir, args.output_dir, stats))
        print()
    if do_x360:
        print("[x360] Processing x360dataset...")
        all_basenames.extend(process_x360(args.x360_dir, args.output_dir, stats))
        print()
    if do_self:
        print("[self] Processing self-collected clips...")
        all_basenames.extend(process_self_collected(args.self_collected_dir, args.output_dir, stats))
        print()

    print(f"=== Summary ===")
    print(f"Total videos: {len(all_basenames)}")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    if args.val_ratio > 0 and len(all_basenames) > 0:
        write_split_csv(all_basenames, args.output_dir, args.val_ratio)

    # Write a manifest for quick inspection
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    manifest = {
        "total_videos": len(all_basenames),
        "stats": stats,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "sources": {
            "web360": args.web360_dir,
            "x360": args.x360_dir,
            "self_collected": args.self_collected_dir,
        },
        "split": "scene-aware (clips from same source video kept in same split)",
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Manifest: {manifest_path}")

    print(f"\n=== Done ===")
    print(f"Dataset ready at: {args.output_dir}")
    print(f"  videos/  -> {len(all_basenames)} symlinks")
    print(f"  metas/   -> {len(all_basenames)} caption .txt files")


if __name__ == "__main__":
    main()
