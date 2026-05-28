#!/usr/bin/env python3
"""A/B quality comparison between Gemini 2.5 Flash and OpenAI GPT-5-mini
as candidate "anchor" captioners for the hierarchical captioning pipeline
used on 360x / WEB360 / self-collected panoramic clips.

For each of ``--n`` diverse clips we:
  1) Extract the same 12 perspective views + identical prompt.
  2) Run Gemini 2.5 Flash and GPT-5-mini back-to-back.
  3) Write a side-by-side markdown report + per-clip JSON dumps.

Usage:
    # Set both keys via env vars (or via ~/.config/panoworld/, see docs/TOKENS.md)
    export GEMINI_API_KEY="AIzaSy..."
    export OPENAI_API_KEY="sk-..."
    python -m scripts.pano_caption.ab_compare_models \
        --clip_dir $DATA_ROOT/self_collected_clips \
        --out_md   $DATA_ROOT/logs/anchor_ab_report.md \
        --n 6
"""

import argparse
import base64
import io
import json
import os
import random
import sys
import time
from pathlib import Path

from PIL import Image


def _read_token_file(basename: str) -> str | None:
    """Resolve an API token file from $PANOWORLD_TOKEN_DIR, ~/.config/panoworld/, or cwd.

    See docs/TOKENS.md for the full resolution order.
    """
    candidates = []
    if (d := os.environ.get("PANOWORLD_TOKEN_DIR")):
        candidates.append(os.path.join(d, basename))
    candidates.append(os.path.expanduser(f"~/.config/panoworld/{basename}"))
    candidates.append(os.path.join(os.getcwd(), basename))
    for p in candidates:
        if os.path.isfile(p):
            try:
                return open(p).read().strip()
            except OSError:
                continue
    return None

# Reuse the view-extraction / prompt / parsing utilities so both backends see
# identical inputs and are judged on the same task.
from scripts.pano_caption.caption_with_gemini import (
    CAPTION_PROMPT,
    TIMESTAMP_LABELS,
    YAW_LABELS,
    extract_perspective_views,
    _parse_json_response,
    _validate_captions,
)


# ─────────────────────────── Gemini backend ───────────────────────────
def caption_gemini(client, model: str, images: list[Image.Image], n_timestamps: int = 3) -> dict:
    from google.genai import types

    contents = []
    ts_labels = TIMESTAMP_LABELS[:n_timestamps]
    for ti, t_label in enumerate(ts_labels):
        for yi, y_label in enumerate(YAW_LABELS):
            idx = ti * len(YAW_LABELS) + yi
            contents.append(f"Image [{t_label} / {y_label}]:")
            buf = io.BytesIO()
            images[idx].save(buf, format="JPEG", quality=88)
            contents.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"))
    contents.append(CAPTION_PROMPT)

    cfg = types.GenerateContentConfig(
        temperature=0.4,
        max_output_tokens=1536,
        response_mime_type="application/json",
        response_schema=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "short":  types.Schema(type=types.Type.STRING),
                "medium": types.Schema(type=types.Type.STRING),
                "long":   types.Schema(type=types.Type.STRING),
            },
            required=["short", "medium", "long"],
            property_ordering=["short", "medium", "long"],
        ),
        thinking_config=types.ThinkingConfig(thinking_budget=384),
    )
    resp = client.models.generate_content(model=model, contents=contents, config=cfg)
    parsed = _parse_json_response((resp.text or "").strip())
    for k in ("short", "medium", "long"):
        if k in parsed and isinstance(parsed[k], str):
            parsed[k] = parsed[k].strip().strip('"').strip()
    _validate_captions(parsed)
    return {k: parsed[k] for k in ("short", "medium", "long")}


# ─────────────────────────── OpenAI backend ───────────────────────────
def _image_to_data_url(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def caption_openai(client, model: str, images: list[Image.Image], n_timestamps: int = 3) -> dict:
    """Call OpenAI chat.completions with interleaved image + label content."""
    content = []
    ts_labels = TIMESTAMP_LABELS[:n_timestamps]
    for ti, t_label in enumerate(ts_labels):
        for yi, y_label in enumerate(YAW_LABELS):
            idx = ti * len(YAW_LABELS) + yi
            content.append({"type": "text", "text": f"Image [{t_label} / {y_label}]:"})
            content.append({
                "type": "image_url",
                "image_url": {"url": _image_to_data_url(images[idx]), "detail": "high"},
            })
    content.append({"type": "text", "text": CAPTION_PROMPT})

    # GPT-5 family uses `max_completion_tokens` and supports `reasoning_effort`.
    # JSON output ~350 tokens + modest reasoning budget keeps latency low.
    kwargs = dict(
        model=model,
        messages=[{"role": "user", "content": content}],
        response_format={"type": "json_object"},
        max_completion_tokens=2048,
    )
    if model.startswith("gpt-5") or model.startswith("o"):
        kwargs["reasoning_effort"] = "minimal"
    resp = client.chat.completions.create(**kwargs)
    text = resp.choices[0].message.content or ""
    parsed = _parse_json_response(text.strip())
    for k in ("short", "medium", "long"):
        if k in parsed and isinstance(parsed[k], str):
            parsed[k] = parsed[k].strip().strip('"').strip()
    _validate_captions(parsed)
    return {k: parsed[k] for k in ("short", "medium", "long")}


# ───────────────────────────── Runner ─────────────────────────────────
def pick_diverse_clips(clip_dir: Path, n: int, seed: int = 0) -> list[Path]:
    """One clip per scene prefix (up to n). Naming: {scene}__{src}__clip{NNN}.mp4."""
    clips = sorted(clip_dir.glob("*.mp4"))
    by_scene: dict[str, list[Path]] = {}
    for c in clips:
        scene = c.stem.split("__", 1)[0]
        by_scene.setdefault(scene, []).append(c)
    rng = random.Random(seed)
    picks: list[Path] = []
    for scene in sorted(by_scene):
        pool = by_scene[scene]
        picks.append(rng.choice(pool))
        if len(picks) >= n:
            break
    # If still fewer (not enough scenes), top up with random extras.
    if len(picks) < n:
        extras = [c for c in clips if c not in picks]
        rng.shuffle(extras)
        picks.extend(extras[: n - len(picks)])
    return picks[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip_dir", required=True)
    ap.add_argument("--out_md", required=True, help="Path to write side-by-side markdown report")
    ap.add_argument("--out_json_dir", default=None,
                    help="Optional dir to dump per-clip raw JSON outputs")
    ap.add_argument("--n", type=int, default=6, help="Number of clips to compare")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gemini_model", default="gemini-2.5-flash")
    ap.add_argument("--openai_model", default="gpt-5-mini")
    ap.add_argument("--n_timestamps", type=int, default=3)
    ap.add_argument("--yaws_deg", nargs="+", type=float, default=[0.0, 90.0, 180.0, 270.0])
    ap.add_argument("--fov_x", type=float, default=90.0)
    ap.add_argument("--out_h", type=int, default=480)
    ap.add_argument("--out_w", type=int, default=640)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # Keys
    gemini_key = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or _read_token_file("gemini_token.txt")
    )
    openai_key = (
        os.environ.get("OPENAI_API_KEY")
        or _read_token_file("openai_token.txt")
    )
    if not gemini_key or not openai_key:
        sys.exit(
            "ERROR: need both GEMINI_API_KEY and OPENAI_API_KEY. Set them as env "
            "vars or put the keys into ~/.config/panoworld/{gemini,openai}_token.txt. "
            "See docs/TOKENS.md."
        )

    from google import genai
    from openai import OpenAI
    gm_client = genai.Client(api_key=gemini_key)
    oai_client = OpenAI(api_key=openai_key)

    clips = pick_diverse_clips(Path(args.clip_dir), args.n, seed=args.seed)
    print(f"Selected {len(clips)} clips:")
    for c in clips:
        print(f"  - {c.name}")

    out_dir = Path(args.out_md).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    json_dir = Path(args.out_json_dir) if args.out_json_dir else None
    if json_dir:
        json_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        f"# Anchor-model A/B: {args.gemini_model} vs {args.openai_model}",
        "",
        f"Seed: `{args.seed}`  |  Clips: `{len(clips)}`  |  Prompt: same 3-style JSON (short/medium/long).",
        "",
    ]
    all_results = []
    for i, clip in enumerate(clips, 1):
        print(f"\n[{i}/{len(clips)}] {clip.name}")
        images = extract_perspective_views(
            str(clip), args.n_timestamps, tuple(args.yaws_deg),
            args.fov_x, args.out_h, args.out_w, args.device,
        )
        row = {"clip": clip.name}
        # Gemini
        try:
            t0 = time.time()
            caps_gm = caption_gemini(gm_client, args.gemini_model, images, args.n_timestamps)
            row["gemini"] = {"captions": caps_gm, "latency_s": round(time.time() - t0, 1)}
            print(f"  gemini {args.gemini_model}: ok ({row['gemini']['latency_s']}s)")
        except Exception as e:
            row["gemini"] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  gemini ERR: {row['gemini']['error']}")
        # OpenAI
        try:
            t0 = time.time()
            caps_oai = caption_openai(oai_client, args.openai_model, images, args.n_timestamps)
            row["openai"] = {"captions": caps_oai, "latency_s": round(time.time() - t0, 1)}
            print(f"  openai {args.openai_model}: ok ({row['openai']['latency_s']}s)")
        except Exception as e:
            row["openai"] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  openai ERR: {row['openai']['error']}")
        all_results.append(row)
        if json_dir:
            (json_dir / f"{clip.stem}_ab.json").write_text(json.dumps(row, indent=2))

        # Render to markdown
        lines += [f"## {i}. `{clip.name}`", ""]
        lines.append("| | Gemini | OpenAI |")
        lines.append("|---|---|---|")
        for style in ("short", "medium", "long"):
            gm_v = row.get("gemini", {}).get("captions", {}).get(style, row.get("gemini", {}).get("error", "—"))
            oai_v = row.get("openai", {}).get("captions", {}).get(style, row.get("openai", {}).get("error", "—"))
            # Escape pipes for markdown-table safety.
            gm_v = str(gm_v).replace("|", "\\|")
            oai_v = str(oai_v).replace("|", "\\|")
            lines.append(f"| **{style}** | {gm_v} | {oai_v} |")
        gm_lat = row.get("gemini", {}).get("latency_s", "—")
        oai_lat = row.get("openai", {}).get("latency_s", "—")
        lines.append(f"| _latency_ | {gm_lat}s | {oai_lat}s |")
        lines.append("")

    Path(args.out_md).write_text("\n".join(lines))
    print(f"\nReport written: {args.out_md}")
    if json_dir:
        print(f"Per-clip JSON dumps: {json_dir}")


if __name__ == "__main__":
    main()
