#!/bin/bash
# ============================================================
# End-to-end pipeline: self-collected pano videos -> cosmos_pano_train/
#
#   1. Cut long videos into 5s clips (ffmpeg stream copy, fast)
#   2. Generate captions via Gemini 2.5 Pro (4-view perspective joint)
#   3. Integrate into cosmos_pano_train/ (symlinks + metas + train/val CSV)
#
# Requires:
#   - A Gemini API key. Resolution order (see docs/TOKENS.md):
#       1) $GEMINI_API_KEY env var
#       2) $PANOWORLD_TOKEN_DIR/gemini_token.txt
#       3) $HOME/.config/panoworld/gemini_token.txt
#       4) ./gemini_token.txt (relative to repo root, legacy)
#   - conda env "cosmos" with google-genai + decord + torch + ffmpeg
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---- config (override via env vars) ----
SOURCE_ROOT="${SOURCE_ROOT:-$HOME/pano_video_data}"
SCENES="${SCENES:?ERROR: set SCENES to a space-separated list of scene folder names (e.g. SCENES=\"scene1 scene2\")}"
CLIPS_DIR="${CLIPS_DIR:-$SOURCE_ROOT/self_collected_clips}"
CAPTION_OUT_DIR="${CAPTION_OUT_DIR:-$CLIPS_DIR}"
COSMOS_OUT="${COSMOS_OUT:-$SOURCE_ROOT/cosmos_pano_train}"
CLIP_SECONDS="${CLIP_SECONDS:-5}"
CUT_WORKERS="${CUT_WORKERS:-6}"
CAPTION_MODEL="${CAPTION_MODEL:-gemini-2.5-pro}"
CAPTION_WORKERS="${CAPTION_WORKERS:-8}"

# ---- activate env (assumes anaconda installed in the standard location) ----
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
if command -v conda >/dev/null 2>&1; then
    conda activate cosmos
fi
cd "$REPO_DIR"

# ---- token preflight ----
if [ -z "${GEMINI_API_KEY:-}" ]; then
    # Try standard PanoWorld locations
    for candidate in \
        "${PANOWORLD_TOKEN_DIR:-}/gemini_token.txt" \
        "$HOME/.config/panoworld/gemini_token.txt" \
        "$REPO_DIR/gemini_token.txt"; do
        if [ -n "$candidate" ] && [ -f "$candidate" ]; then
            export GEMINI_API_KEY="$(cat "$candidate" | tr -d '[:space:]')"
            break
        fi
    done
fi
if [ -z "${GEMINI_API_KEY:-}" ]; then
    echo "ERROR: GEMINI_API_KEY not found. Set the env var or put your key into" \
         "~/.config/panoworld/gemini_token.txt. See docs/TOKENS.md." >&2
    exit 1
fi

# ============================================================
# Stage 1: cut clips
# ============================================================
echo "============================================================"
echo "  Stage 1/3: cut long videos into ${CLIP_SECONDS}s clips"
echo "  source:    $SOURCE_ROOT"
echo "  scenes:    $SCENES"
echo "  clips to:  $CLIPS_DIR"
echo "============================================================"

python -m scripts.pano_caption.cut_clips \
    --source_root "$SOURCE_ROOT" \
    --scenes $SCENES \
    --output_dir "$CLIPS_DIR" \
    --clip_seconds "$CLIP_SECONDS" \
    --workers "$CUT_WORKERS"

N_CLIPS=$(ls "$CLIPS_DIR"/*.mp4 2>/dev/null | wc -l)
echo ""
echo "  -> $N_CLIPS clip files under $CLIPS_DIR"

# ============================================================
# Stage 2: Gemini captioning
# ============================================================
echo ""
echo "============================================================"
echo "  Stage 2/3: caption clips via $CAPTION_MODEL"
echo "  workers:   $CAPTION_WORKERS concurrent API calls"
echo "============================================================"

python -m scripts.pano_caption.caption_with_gemini \
    --clip_dir "$CLIPS_DIR" \
    --out_dir "$CAPTION_OUT_DIR" \
    --model "$CAPTION_MODEL" \
    --workers "$CAPTION_WORKERS"

N_CAPS=$(ls "$CAPTION_OUT_DIR"/*.txt 2>/dev/null | wc -l)
echo ""
echo "  -> $N_CAPS captions written"

# ============================================================
# Stage 3: integrate into cosmos_pano_train/
# ============================================================
echo ""
echo "============================================================"
echo "  Stage 3/3: integrate into $COSMOS_OUT"
echo "============================================================"

python prepare_pano_data.py \
    --self_collected_dir "$CLIPS_DIR" \
    --output_dir "$COSMOS_OUT" \
    --datasets self \
    --val_ratio 0.05

echo ""
echo "============================================================"
echo "  DONE"
echo "  Training data ready at: $COSMOS_OUT"
echo "  - videos/ + metas/ + train.csv + val.csv"
echo "============================================================"
