# Pano Captioning Pipeline

Generate training captions for self-collected 360° panoramic videos using
**Gemini 2.5 Pro** with 4-view perspective projections.

## Why perspective views?

VLMs are trained on perspective images; feeding them raw equirectangular
(ERP) frames fails in two common ways:

1. **Seam splitting** — objects/people straddling yaw=±180° get cut in half.
2. **Polar distortion** — ceilings/floors are stretched unrecognizably.

Our solution: for each clip, extract 4 perspective views at `yaw ∈ {0°, 90°,
180°, 270°}, pitch=0°, fov_x=90°` at 3 timestamps (12 images total), then
prompt the VLM to treat them as one wrap-around scene.

## Files

| File | Purpose |
|------|---------|
| `cut_clips.py` | ffmpeg stream-copy long videos → fixed-length (default 5s) clips |
| `caption_with_gemini.py` | 4-view perspective extraction + Gemini API call |
| `run_all.sh` | End-to-end orchestrator |
| `README.md` | This file |

Integration with the main dataset builder lives in
`../../prepare_pano_data.py :: process_self_collected()`.

## Setup

```bash
# 1. API key (pick one)
export GEMINI_API_KEY="AIzaSy..."
# OR (persistent, ignored by git)
mkdir -p ~/.config/panoworld
echo "AIzaSy..." > ~/.config/panoworld/gemini_token.txt
chmod 600        ~/.config/panoworld/gemini_token.txt

# 2. Python deps (already installed by install.sh)
conda activate cosmos
pip install google-genai
```

Get a key at https://aistudio.google.com/apikey. Gemini 2.5 Pro requires a
billing-enabled Google Cloud project (Flash is free-tier only). Full token
setup options (env var, ~/.config/panoworld, PANOWORLD_TOKEN_DIR) are in
[`../../docs/TOKENS.md`](../../docs/TOKENS.md).

## Usage

### End-to-end (recommended)

```bash
cd $REPO_ROOT  # path to your PanoWorld clone
bash scripts/pano_caption/run_all.sh
```

Uses defaults:
- `SOURCE_ROOT=$PANO_DATA_ROOT`
- `SCENES=Le_home_foodcourt Le_prudential_apartment Le_school shayda street xiangyu`
- `CLIPS_DIR=$SOURCE_ROOT/self_collected_clips`
- `CAPTION_MODEL=gemini-2.5-pro`

Override any via env vars, e.g.:
```bash
SCENES="xiangyu" CAPTION_MODEL=gemini-2.5-flash bash scripts/pano_caption/run_all.sh
```

### Stage-by-stage

```bash
# 1. cut only
python -m scripts.pano_caption.cut_clips \
    --source_root $PANO_DATA_ROOT \
    --scenes Le_school street \
    --output_dir $PANO_DATA_ROOT/self_collected_clips \
    --clip_seconds 5

# 2. caption only (can re-run to retry failed; skips already-captioned clips)
python -m scripts.pano_caption.caption_with_gemini \
    --clip_dir $PANO_DATA_ROOT/self_collected_clips \
    --model gemini-2.5-pro --workers 8

# Quick test on 20 clips:
python -m scripts.pano_caption.caption_with_gemini \
    --clip_dir $PANO_DATA_ROOT/self_collected_clips \
    --limit 20 --shuffle

# 3. register into cosmos_pano_train/
python prepare_pano_data.py \
    --self_collected_dir $PANO_DATA_ROOT/self_collected_clips \
    --output_dir $PANO_DATA_ROOT/cosmos_pano_train \
    --datasets self
```

## Costs (Gemini 2.5 Pro, April 2026 pricing)

| Item | Est. |
|------|------|
| 164 source videos, ~30k s total | → ~6000 × 5s clips |
| Per clip: 12 images × ~258 tok + prompt | ≈ 3300 in + 60 out tok |
| Per clip: `$2.00/M in + $8/M out` → | **~$0.005** |
| **Total** (6000 clips) | **~$30** |

For a quick check, `gemini-2.5-flash` costs ~$7 total and is often
indistinguishable for scene captions.

## Output format

Each clip produces `<basename>.mp4` + `<basename>.txt` (single caption line).
Basename format: `<scene>__<source_video_stem>__clip<NNN>`, e.g.
`Le_home_foodcourt__VID_20260422_061031_00_121__clip007`.

After `prepare_pano_data.py --datasets self`:

```
$PANO_DATA_ROOT/cosmos_pano_train/
├── videos/self_<basename>.mp4  (symlink to self_collected_clips/<basename>.mp4)
├── metas/self_<basename>.txt   (copied caption)
├── train.csv   (scene-aware split: clips from same source video stay together)
└── val.csv
```

## Tuning caption quality

The prompt is in `caption_with_gemini.py :: CAPTION_PROMPT`. Edit then
re-run with `--skip_existing=False` (or rm a specific .txt) to regenerate.

If captions come out too generic, try:
- Increasing `--n_timestamps` from 3 to 4 or 5 (better activity recognition)
- Upgrading to `gemini-2.5-pro` (default)
- Adding `--fov_x 105` to get slightly overlapping views (more context per image)
