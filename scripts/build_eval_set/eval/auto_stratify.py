#!/usr/bin/env python3
"""Classify each clip in master.csv into orthogonal evaluation strata.

Outputs strata/strata.csv with columns:
    clip_id, scene_type, sky_pct, motion_kind, n_frames, split

scene_type ∈ {textured_indoor, outdoor_with_sky, mixed_low_texture}

Decision rule (in order of priority, first match wins):

1. habitat_ood  → always 'textured_indoor'  (Replica scenes are all interior)
2. caption keyword 'outdoor / outside / sky / street / park / forest / ocean /
   beach / mountain / road / field / sunset / sunrise / cityscape' → outdoor_with_sky
3. caption keyword 'indoor / room / bedroom / kitchen / bathroom / living /
   office / corridor / hallway / hotel / interior / studio / restaurant /
   shop / store / mall / classroom / lobby / cafe' → textured_indoor
4. else → mixed_low_texture

sky_pct is computed via a *quick* heuristic on the first frame of the GT video:
top 30% of equirectangular image → mean luminance & blueness ratio. Pixels
where (B > R) AND (luma > 150) are counted as sky-like. This is a coarse
estimate for stratification only; the per-frame sky mask used inside the
depth metric is computed separately at eval time using a real semantic
segmenter.

Usage:
    python auto_stratify.py \
        --master $PANO_DATA_ROOT/test/master.csv \
        --out strata/strata.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

OUTDOOR_KW = re.compile(
    r"\b(outdoor|outside|open[- ]?air|sky|skies|cloud|street|alley|park|forest|"
    r"woods|ocean|sea|beach|coast|shore|mountain|valley|hill|road|highway|"
    r"field|meadow|farm|sunset|sunrise|dawn|dusk|cityscape|skyline|"
    r"plaza|courtyard|garden|trail|harbor|river|lake|desert|canyon|"
    r"sidewalk|bridge|rooftop|balcony|sky line)\b",
    re.IGNORECASE,
)

INDOOR_KW = re.compile(
    r"\b(indoor|interior|inside|room|bedroom|kitchen|bathroom|living[- ]room|"
    r"dining|office|corridor|hallway|hotel|apartment|house|studio|"
    r"restaurant|cafe|coffee[- ]shop|shop|store|mall|classroom|lobby|"
    r"gym|library|museum|church|temple|theater|cinema|stairwell|stair|"
    r"elevator|garage|basement|attic|warehouse|factory|aisle|"
    r"bedroom|loft|laboratory|laboratory)\b",
    re.IGNORECASE,
)

SCENE_TYPES = ("textured_indoor", "outdoor_with_sky", "mixed_low_texture")


def load_caption_text(caption_path: str) -> str:
    """Read caption json (gpt-5-mini format) and concatenate short+medium+long."""
    if not caption_path or not Path(caption_path).is_file():
        return ""
    try:
        data = json.loads(Path(caption_path).read_text())
    except Exception:
        return ""
    chunks: list[str] = []

    def _walk(node):
        if isinstance(node, str):
            chunks.append(node)
        elif isinstance(node, list):
            for it in node:
                _walk(it)
        elif isinstance(node, dict):
            for v in node.values():
                _walk(v)

    _walk(data)
    return " ".join(chunks)[:4000]


def estimate_sky_pct(video_path: str) -> float:
    """Coarse first-frame sky percentage. Returns -1.0 on failure."""
    try:
        import imageio.v3 as iio
        import numpy as np

        frame = iio.imread(video_path, index=0)
    except Exception:
        return -1.0
    if frame.ndim != 3 or frame.shape[-1] < 3:
        return -1.0
    h, w = frame.shape[:2]
    top = frame[: max(1, h * 3 // 10)]
    r, g, b = top[..., 0].astype("int16"), top[..., 1].astype("int16"), top[..., 2].astype("int16")
    luma = (0.299 * r + 0.587 * g + 0.114 * b).astype("int16")
    sky_like = (b > r) & (b > g - 5) & (luma > 150)
    return float(sky_like.mean())


def classify_scene(split: str, caption_text: str, sky_pct: float) -> str:
    if split == "habitat_ood":
        return "textured_indoor"
    has_outdoor_kw = bool(OUTDOOR_KW.search(caption_text))
    has_indoor_kw = bool(INDOOR_KW.search(caption_text))

    if has_outdoor_kw and not has_indoor_kw:
        return "outdoor_with_sky"
    if has_indoor_kw and not has_outdoor_kw:
        return "textured_indoor"
    if sky_pct > 0.25:
        return "outdoor_with_sky"
    if sky_pct >= 0 and sky_pct < 0.05 and has_indoor_kw:
        return "textured_indoor"
    return "mixed_low_texture"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--skip-sky-pct",
        action="store_true",
        help="skip first-frame sky estimation (much faster, caption-only)",
    )
    args = ap.parse_args()

    master_path = Path(args.master)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows_in: list[dict] = []
    with master_path.open() as f:
        rows_in = list(csv.DictReader(f))

    print(f"[stratify] {len(rows_in)} clips loaded from {master_path}", flush=True)
    counts: dict[str, int] = {t: 0 for t in SCENE_TYPES}

    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "scene_type", "sky_pct", "motion_kind", "n_frames", "split"])
        for i, row in enumerate(rows_in):
            cid = row["clip_id"]
            split = row["split"]
            cap_text = load_caption_text(row.get("caption_path", ""))
            sky_pct = -1.0 if args.skip_sky_pct else estimate_sky_pct(row.get("video_path", ""))
            scene_type = classify_scene(split, cap_text, sky_pct)
            counts[scene_type] += 1
            motion = row.get("trajectory_kind") or ("static" if row.get("is_static") == "1" else "")
            w.writerow([
                cid,
                scene_type,
                f"{sky_pct:.3f}" if sky_pct >= 0 else "",
                motion,
                row.get("n_frames", ""),
                split,
            ])
            if (i + 1) % 25 == 0:
                print(f"[stratify]   {i+1}/{len(rows_in)}  scene={scene_type}", flush=True)

    print(f"[stratify] wrote {out_path}", flush=True)
    print("[stratify] scene_type breakdown:")
    for t, n in counts.items():
        print(f"  {t:>22s}: {n}")


if __name__ == "__main__":
    sys.exit(main())
