#!/usr/bin/env python3
"""Generate panoramic-aware captions for pano video clips via Gemini 2.5 Pro.

For each clip we extract perspective views (yaw in {0, 90, 180, 270}, pitch=0,
fov_x=90) at multiple evenly-spaced timestamps, then ask the VLM to produce
ONE unified scene description treating the views as a wrap-around 360 scene.

This sidesteps the two classic VLM failure modes on equirectangular input:
  (1) subjects split across the ERP seam
  (2) severe polar distortion

Output: for each {clip_basename}.mp4, write {clip_basename}.json next to it
with shape ``{"gemini2p5pro": {"short": "...", "medium": "...", "long": "..."}}``,
matching the JSON caption format already consumed by
``cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video.VideoDataset._load_json_caption``
so a training config can pick a style (or we can later add random sampling
that mirrors Cosmos' default ``caption_probs``).

Example:
    export GEMINI_API_KEY="AIzaSy..."   # or: echo "AIzaSy..." > ~/.config/panoworld/gemini_token.txt
    python -m scripts.pano_caption.caption_with_gemini \
        --clip_dir $DATA_ROOT/self_collected_clips \
        --out_dir  $DATA_ROOT/self_collected_clips \
        --model gemini-2.5-pro --workers 8

See docs/TOKENS.md for the full API-key resolution order.
"""

import argparse
import io
import json
import math
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _read_token_file(basename: str) -> str | None:
    """Locate an API token in the standard PanoWorld locations.

    Resolution order:
      1. ``$PANOWORLD_TOKEN_DIR/<basename>``  (if env var set)
      2. ``$HOME/.config/panoworld/<basename>``  (recommended)
      3. ``./<basename>``  (legacy, repo-root convenience)

    Returns the stripped contents of the first existing file, or None.
    See docs/TOKENS.md for full instructions.
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


CAPTION_PROMPT = """You are captioning a 360-degree panoramic video clip for training a panoramic \
video world model. You will see 12 perspective crops rendered from ONE short clip:
  - 3 timestamps: EARLY (near the start), MIDDLE (around the middle), LATE (near the end).
  - 4 compass directions at each timestamp: FRONT (yaw=0), RIGHT (yaw=90), BACK (yaw=180), LEFT (yaw=270), each 90-degree FOV.
  - The four directions at each timestamp together cover a continuous 360-degree scene: FRONT -> RIGHT -> BACK -> LEFT -> FRONT wraps around seamlessly. Objects can extend across adjacent views.
  - The three timestamps share the same camera location unless clearly shifted.

Produce THREE captions of different lengths describing the SAME clip, returned as JSON:
  - "short"  : a concise user-style prompt, 5 to 15 words. No sub-clauses. Think of how a real user would type a short prompt. Example shape: "A modern lobby with a man walking toward the entrance."
  - "medium" : a balanced description, 35 to 60 words. Include environment, 2-3 prominent objects with specific materials or colors, and the main activity if any. Written as 1-2 sentences.
  - "long"   : a cinematic-treatment paragraph, 90 to 130 words. Include environment + lighting, spatial layout across the four cardinal directions (phrase naturally as "In the front ... To the right ... Behind the camera ... On the left ..."), concrete materials/colors/textures, people and their specific activity if visible, camera behavior (static vs glide/rotation, inferred from the three timestamps), and a short closing clause on atmosphere/mood.

All three captions must be independently valid, grammatical, and describe the SAME clip. They are not a progressive rewrite of the same sentence; each is written from scratch at its own level of detail. They must be mutually consistent (e.g., all say "a man in a tan jacket" if a man is visible, never contradict each other).

Strict rules (apply to all three captions):
- Present tense, third person, natural prose.
- Do NOT start with "The video", "This panorama", "A panoramic view", "360-degree", "The scene shows", "The image shows", "This is", "Here is", "We see", or any meta/wrapper phrasing. Open directly with a noun phrase describing the environment (e.g., "A modern apartment lobby ...", "A bustling outdoor market ..."). The first word should be "A", "An", or a concrete noun.
- Do NOT mention the four perspective crops, the wrap-around construction, the timestamps, the AI, or the input format.
- Stay grounded in what is actually visible. Do not invent text, signage, or identities you cannot clearly read. If signage is unreadable, describe it generically ("neon signs in foreign lettering").
- NEVER use personal names, initials, or any identity markers for people, even if visible on clothing, ID cards, or signs. Refer to people only through descriptive phrases ("a man in a tan jacket", "a young woman carrying a shopping bag", "a figure in the distance"). If no person is clearly visible, omit people entirely.
- Panoramic captures can produce subtle warping near the top/bottom of each crop (polar distortion). Do NOT describe this distortion as reality ("mirrored reflections on the ceiling", "upside-down scenery"). Treat the underlying geometry as a normal room/scene.
- Camera behavior (long caption only): if the scene looks essentially identical across EARLY/MIDDLE/LATE, say "The camera remains static." If the viewpoint clearly shifts, describe the motion naturally ("the camera glides forward", "the camera slowly rotates clockwise").
- For the "long" caption, merge information across the four directions and three timestamps into one coherent paragraph, not a list of separate views.
- For the "short" caption, keep it strictly between 5 and 15 words. No newlines, no camera description.
- Use specific nouns over generic ones where possible: "wooden kitchen island" over "furniture", "red brick alley" over "a place".

Return ONLY valid JSON with exactly these three keys (no markdown fences, no prose before/after):
{"short": "...", "medium": "...", "long": "..."}"""


# Human-readable labels interleaved with images in the Gemini request.
# Order must match the output of ``extract_perspective_views``:
# [(t0, yaw0), (t0, yaw1), (t0, yaw2), (t0, yaw3), (t1, yaw0), ...]
TIMESTAMP_LABELS = ["EARLY", "MIDDLE", "LATE"]
YAW_LABELS = ["FRONT (yaw=0)", "RIGHT (yaw=90)", "BACK (yaw=180)", "LEFT (yaw=270)"]
def _model_key_for(model: str) -> str:
    """Derive a stable JSON top-level key from the VLM model id.

    Examples:
        gemini-2.5-pro   -> gemini2p5pro
        gemini-2.5-flash -> gemini2p5flash
        gemini-2.0-flash -> gemini2p0flash
        gpt-5-mini       -> gpt5mini
        gpt-5-mini-2025-08-07 -> gpt5mini20250807
    """
    return (model
            .replace("gemini-", "gemini")
            .replace(".", "p")
            .replace("-", ""))


# ── Perspective extraction (reuses cosmos equi2pers) ──────────────────
def _extract_views_one_frame(
    frame_hwc_uint8: np.ndarray,
    yaws_deg: tuple,
    fov_x: float,
    out_h: int,
    out_w: int,
    device: str = "cuda",
) -> list[Image.Image]:
    """Given one ERP frame (H,W,3), return list of PIL perspective images."""
    from cosmos_predict2._src.predict2.utils.pano_conditioning import equi2pers

    # (1, 3, H, W) float in [0, 1] — equi2pers works on any range, but grid_sample
    # needs floats.
    frame = torch.from_numpy(frame_hwc_uint8).to(device).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    images = []
    for yaw_deg in yaws_deg:
        yaw = torch.tensor([math.radians(yaw_deg)], device=device)
        roll = torch.zeros(1, device=device)
        pitch = torch.zeros(1, device=device)
        pers = equi2pers(frame, fov_x=fov_x, roll=roll, pitch=pitch, yaw=yaw,
                         out_h=out_h, out_w=out_w)  # (1, 3, out_h, out_w)
        arr = (pers[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        images.append(Image.fromarray(arr))
    return images


def extract_perspective_views(
    clip_path: str,
    n_timestamps: int,
    yaws_deg: tuple,
    fov_x: float,
    out_h: int,
    out_w: int,
    device: str,
) -> list[Image.Image]:
    """Return list of PIL images, len = n_timestamps * len(yaws_deg)."""
    # Use decord for fast frame reads
    from decord import VideoReader, cpu
    vr = VideoReader(clip_path, ctx=cpu(0))
    N = len(vr)
    # Evenly spaced timestamps, avoid exact first/last frame
    positions = [int((i + 0.5) / n_timestamps * (N - 1)) for i in range(n_timestamps)]
    images = []
    for p in positions:
        frame = vr[p].asnumpy()  # (H, W, 3)
        images.extend(_extract_views_one_frame(frame, yaws_deg, fov_x, out_h, out_w, device))
    return images


# ── Gemini API call ───────────────────────────────────────────────────
def _image_to_part(im: Image.Image):
    """Convert a PIL image into a Gemini Part (JPEG q=88 for quality/size tradeoff)."""
    from google.genai import types
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")


def _build_response_schema(types_mod):
    """Schema forcing {short, medium, long} string output."""
    return types_mod.Schema(
        type=types_mod.Type.OBJECT,
        properties={
            "short":  types_mod.Schema(type=types_mod.Type.STRING),
            "medium": types_mod.Schema(type=types_mod.Type.STRING),
            "long":   types_mod.Schema(type=types_mod.Type.STRING),
        },
        required=["short", "medium", "long"],
        property_ordering=["short", "medium", "long"],
    )


def _parse_json_response(text: str) -> dict:
    """Best-effort JSON parse; handle occasional ```json fences."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def _validate_captions(d: dict) -> None:
    """Sanity-check short/medium/long lengths and content."""
    for k in ("short", "medium", "long"):
        if k not in d or not isinstance(d[k], str) or not d[k].strip():
            raise ValueError(f"missing or empty '{k}' caption")
    n_short = len(d["short"].split())
    n_medium = len(d["medium"].split())
    n_long = len(d["long"].split())
    if not (3 <= n_short <= 25):
        raise ValueError(f"short length out of range: {n_short} words")
    if not (20 <= n_medium <= 90):
        raise ValueError(f"medium length out of range: {n_medium} words")
    if not (60 <= n_long <= 180):
        raise ValueError(f"long length out of range: {n_long} words")


ANCHOR_PREAMBLE = (
    "CONTEXT (from a different moment of the same source video): \"{anchor}\"\n"
    "Use this context ONLY to ground object identities, scene type, and spatial layout "
    "(e.g., indoor/outdoor, materials, landmarks). Describe THIS clip's moment faithfully "
    "based solely on the images below. Do not copy the context sentences.\n\n"
)


def _caption_one_gemini(
    client,
    model: str,
    images: list[Image.Image],
    n_timestamps: int = 3,
    max_retries: int = 7,
    anchor_text: str | None = None,
) -> dict:
    """Call Gemini with labeled 12 images + prompt; return dict with 3 styles.

    Images are expected in the order produced by ``extract_perspective_views``:
    [(t0,yaw0), (t0,yaw1), ..., (t2,yaw3)]. Labels are interleaved so Gemini
    grounds each crop to its direction/timestamp instead of guessing by order.

    If ``anchor_text`` is given, it is prepended as scene-grounding context
    (hierarchical captioning).

    Returns: ``{"short": "...", "medium": "...", "long": "..."}``
    """
    from google.genai import types

    n_yaws = len(YAW_LABELS)
    if len(images) != n_timestamps * n_yaws:
        raise ValueError(
            f"expected {n_timestamps * n_yaws} images, got {len(images)}"
        )

    contents = []
    ts_labels = (TIMESTAMP_LABELS[:n_timestamps] if n_timestamps <= len(TIMESTAMP_LABELS)
                 else [f"T{i}" for i in range(n_timestamps)])
    for ti, t_label in enumerate(ts_labels):
        for yi, y_label in enumerate(YAW_LABELS):
            idx = ti * n_yaws + yi
            contents.append(f"Image [{t_label} / {y_label}]:")
            contents.append(_image_to_part(images[idx]))
    prompt = CAPTION_PROMPT
    if anchor_text:
        prompt = ANCHOR_PREAMBLE.format(anchor=anchor_text.strip()) + prompt
    contents.append(prompt)

    # Only Gemini 2.5 / 3.x families support "thinking" tokens; 2.0 rejects
    # thinking_config, and "-lite" variants require budget in [512, 24576] (or 0).
    _mname = model.lower()
    supports_thinking = _mname.startswith(("gemini-2.5", "gemini-3"))
    # For non-lite models we use a small budget (384) to keep latency low; lite
    # variants minimum is 512, so bump to 512 there. Fast-path: 0 disables thinking.
    if "lite" in _mname:
        thinking_budget = 512
    else:
        thinking_budget = 384
    last_err = None
    for attempt in range(max_retries):
        base = min(30.0, 3.0 * (2 ** attempt))  # default backoff
        try:
            cfg_kwargs = dict(
                temperature=0.4,
                max_output_tokens=1536,
                response_mime_type="application/json",
                response_schema=_build_response_schema(types),
            )
            if supports_thinking:
                cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
            gen_cfg = types.GenerateContentConfig(**cfg_kwargs)
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=gen_cfg,
            )
            text = (resp.text or "").strip()
            if not text:
                last_err = "empty response"
            else:
                parsed = _parse_json_response(text)
                for k in ("short", "medium", "long"):
                    if k in parsed and isinstance(parsed[k], str):
                        parsed[k] = parsed[k].strip().strip('"').strip()
                _validate_captions(parsed)
                return {k: parsed[k] for k in ("short", "medium", "long")}
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            # Longer backoff for 503 overloaded / 429 quota (server-side capacity);
            # shorter for parse errors. Full jitter to avoid thundering herd.
            msg = str(e)
            if "503" in msg or "UNAVAILABLE" in msg or "overloaded" in msg.lower():
                base = min(60.0, 10.0 * (2 ** attempt))  # 10,20,40,60,60,60,60
            elif "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                base = min(120.0, 15.0 * (2 ** attempt))  # 15,30,60,120,...
            else:
                base = min(30.0, 3.0 * (2 ** attempt))
        time.sleep(base * (0.5 + random.random()))  # 50-150% jitter
    raise RuntimeError(f"caption_one failed after {max_retries} retries: {last_err}")


# ── OpenAI backend ─────────────────────────────────────────────────────
def _image_to_data_url(im: "Image.Image") -> str:
    import base64, io as _io
    buf = _io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _caption_one_openai(
    client,
    model: str,
    images: list,
    n_timestamps: int = 3,
    max_retries: int = 7,
    anchor_text: str | None = None,
) -> dict:
    """Call OpenAI chat.completions with interleaved labels + images.

    Uses json_object response format. For GPT-5 family we use
    ``max_completion_tokens`` and set ``reasoning_effort=minimal``.
    """
    n_yaws = len(YAW_LABELS)
    if len(images) != n_timestamps * n_yaws:
        raise ValueError(f"expected {n_timestamps * n_yaws} images, got {len(images)}")

    ts_labels = (TIMESTAMP_LABELS[:n_timestamps] if n_timestamps <= len(TIMESTAMP_LABELS)
                 else [f"T{i}" for i in range(n_timestamps)])
    content = []
    for ti, t_label in enumerate(ts_labels):
        for yi, y_label in enumerate(YAW_LABELS):
            idx = ti * n_yaws + yi
            content.append({"type": "text", "text": f"Image [{t_label} / {y_label}]:"})
            content.append({
                "type": "image_url",
                "image_url": {"url": _image_to_data_url(images[idx]), "detail": "high"},
            })
    prompt = CAPTION_PROMPT
    if anchor_text:
        prompt = ANCHOR_PREAMBLE.format(anchor=anchor_text.strip()) + prompt
    content.append({"type": "text", "text": prompt})

    last_err = None
    for attempt in range(max_retries):
        base = min(30.0, 3.0 * (2 ** attempt))
        try:
            kwargs = dict(
                model=model,
                messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"},
                max_completion_tokens=2048,
            )
            if model.startswith("gpt-5") or model.startswith("o"):
                kwargs["reasoning_effort"] = "minimal"
            resp = client.chat.completions.create(**kwargs)
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                last_err = "empty response"
            else:
                parsed = _parse_json_response(text)
                for k in ("short", "medium", "long"):
                    if k in parsed and isinstance(parsed[k], str):
                        parsed[k] = parsed[k].strip().strip('"').strip()
                _validate_captions(parsed)
                return {k: parsed[k] for k in ("short", "medium", "long")}
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            msg = str(e)
            # OpenAI errors: 429 = rate limit, 500/502/503/504 = server, 400 = client.
            if any(c in msg for c in ("500", "502", "503", "504", "overloaded")):
                base = min(60.0, 10.0 * (2 ** attempt))
            elif "429" in msg or "rate_limit" in msg.lower():
                base = min(120.0, 15.0 * (2 ** attempt))
            else:
                base = min(30.0, 3.0 * (2 ** attempt))
        time.sleep(base * (0.5 + random.random()))
    raise RuntimeError(f"caption_one failed after {max_retries} retries: {last_err}")


# ── Dispatcher ─────────────────────────────────────────────────────────
def caption_one(
    client,
    model: str,
    images: list,
    n_timestamps: int = 3,
    max_retries: int = 7,
    provider: str = "gemini",
    anchor_text: str | None = None,
) -> dict:
    if provider == "openai":
        return _caption_one_openai(client, model, images, n_timestamps,
                                   max_retries, anchor_text=anchor_text)
    return _caption_one_gemini(client, model, images, n_timestamps,
                               max_retries, anchor_text=anchor_text)


def _anchor_key_for_stem(stem: str) -> str | None:
    """Extract ``source_id`` from a clip stem.

    Our naming convention is ``{scene}__{source_id}__clipNNN``. The source_id
    itself may contain underscores, so we split off only the trailing ``__clipNNN``.
    Returns None if the stem doesn't match the pattern.
    """
    import re
    m = re.match(r"^(?P<key>.+?)__clip\d+$", stem)
    return m.group("key") if m else None


# ── Worker ─────────────────────────────────────────────────────────────
def process_clip(args_tuple):
    (clip_path, out_dir, client, model, n_timestamps, yaws_deg,
     fov_x, out_h, out_w, device, skip_existing, provider, anchor_map) = args_tuple
    stem = Path(clip_path).stem
    out_path = os.path.join(out_dir, f"{stem}.json")
    if skip_existing and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return {"clip": clip_path, "status": "skipped"}
    anchor_text = None
    if anchor_map:
        key = _anchor_key_for_stem(stem)
        if key is not None:
            anchor_text = anchor_map.get(key)
    try:
        images = extract_perspective_views(clip_path, n_timestamps, yaws_deg,
                                            fov_x, out_h, out_w, device)
        caps = caption_one(client, model, images, n_timestamps=n_timestamps,
                           provider=provider, anchor_text=anchor_text)
        # Store in the same {model_key: {style: caption}} format that
        # Cosmos' VideoDataset._load_json_caption already expects.
        out_obj = {_model_key_for(model): caps}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)
        return {"clip": clip_path, "status": "ok", "caption": caps}
    except Exception as e:
        return {"clip": clip_path, "status": "error", "error": str(e)}


def main():
    ap = argparse.ArgumentParser(description="Caption pano clips via Gemini or OpenAI VLMs")
    ap.add_argument("--clip_dir", required=True, help="Directory containing {clip}.mp4 files")
    ap.add_argument("--out_dir", default=None, help="Output dir for .json files (default: clip_dir)")
    ap.add_argument("--provider", default="gemini", choices=["gemini", "openai"],
                    help="VLM backend (default: gemini)")
    ap.add_argument("--model", default=None,
                    help="Model id. Defaults: gemini -> gemini-2.5-pro, openai -> gpt-5-mini")
    ap.add_argument("--n_timestamps", type=int, default=3,
                    help="Temporal samples per clip (each producing len(yaws) views)")
    ap.add_argument("--yaws_deg", nargs="+", type=float, default=[0.0, 90.0, 180.0, 270.0])
    ap.add_argument("--fov_x", type=float, default=90.0)
    ap.add_argument("--out_h", type=int, default=480)
    ap.add_argument("--out_w", type=int, default=640)
    ap.add_argument("--workers", type=int, default=8, help="Concurrent API calls")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit (process all)")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--skip_existing", action="store_true", default=True)
    ap.add_argument("--anchor_map", default=None,
                    help="Path to JSON mapping source_id -> anchor caption text "
                         "for hierarchical captioning. source_id is derived from "
                         "each clip stem (everything before '__clipNNN').")
    args = ap.parse_args()

    if args.model is None:
        args.model = "gpt-5-mini" if args.provider == "openai" else "gemini-2.5-pro"

    if args.provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            api_key = _read_token_file("gemini_token.txt")
        if not api_key:
            sys.exit(
                "ERROR: no Gemini key found. Set GEMINI_API_KEY env var, "
                "or write the key into ~/.config/panoworld/gemini_token.txt. "
                "See docs/TOKENS.md for details."
            )
        from google import genai
        client = genai.Client(api_key=api_key)
    else:  # openai
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            api_key = _read_token_file("openai_token.txt")
        if not api_key:
            sys.exit(
                "ERROR: no OpenAI key found. Set OPENAI_API_KEY env var, "
                "or write the key into ~/.config/panoworld/openai_token.txt. "
                "See docs/TOKENS.md for details."
            )
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

    out_dir = args.out_dir or args.clip_dir
    os.makedirs(out_dir, exist_ok=True)

    clip_paths = sorted(str(p) for p in Path(args.clip_dir).glob("*.mp4"))
    if args.shuffle:
        random.shuffle(clip_paths)
    if args.limit > 0:
        clip_paths = clip_paths[:args.limit]

    # Pre-filter already-captioned (now writes {stem}.json in multi-style format)
    if args.skip_existing:
        remaining = [p for p in clip_paths
                     if not (os.path.exists(os.path.join(out_dir, f"{Path(p).stem}.json"))
                             and os.path.getsize(os.path.join(out_dir, f"{Path(p).stem}.json")) > 0)]
    else:
        remaining = clip_paths
    print(f"Total clips: {len(clip_paths)}  |  to caption: {len(remaining)}  |  model: {args.model}")
    if not remaining:
        print("Nothing to do.")
        return

    anchor_map = None
    if args.anchor_map:
        with open(args.anchor_map) as f:
            anchor_map = json.load(f)
        n_covered = sum(1 for p in remaining
                        if _anchor_key_for_stem(Path(p).stem) in anchor_map)
        print(f"Anchor map: {len(anchor_map)} source IDs, "
              f"covers {n_covered}/{len(remaining)} remaining clips")

    yaws_tuple = tuple(args.yaws_deg)
    tasks = [(p, out_dir, client, args.model, args.n_timestamps, yaws_tuple,
              args.fov_x, args.out_h, args.out_w, args.device,
              args.skip_existing, args.provider, anchor_map)
             for p in remaining]

    n_ok, n_err, n_skip = 0, 0, 0
    t0 = time.time()
    # NOTE: equi2pers runs on cuda but is fast; the bottleneck is Gemini API,
    # so multi-threading (not multi-processing) is appropriate.
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_clip, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs)):
            res = fut.result()
            status = res["status"]
            if status == "ok":
                n_ok += 1
            elif status == "skipped":
                n_skip += 1
            else:
                n_err += 1
            if (i + 1) % 25 == 0 or status == "error":
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-3)
                eta = (len(remaining) - (i + 1)) / max(rate, 1e-3)
                extra = f" ERR={res.get('error', '')[:120]}" if status == "error" else ""
                print(f"  [{i+1}/{len(remaining)}] ok={n_ok} skip={n_skip} err={n_err} "
                      f"| {rate:.2f} clip/s | ETA {eta/60:.1f} min{extra}")

    print(f"\n=== Done ===  ok={n_ok}  skip={n_skip}  err={n_err}  elapsed={time.time()-t0:.1f}s")
    if n_err > 0:
        print("Re-run the same command to retry failed clips (skip_existing avoids re-work).")


if __name__ == "__main__":
    main()
