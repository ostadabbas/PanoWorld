#!/bin/bash
# ============================================================
# PanoWorld environment activation script
# ============================================================
# Usage:  source activate.sh
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate the cosmos conda env (looks for anaconda3 first, then miniconda3)
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
if command -v conda >/dev/null 2>&1; then
    conda activate cosmos
fi

# Move into the project root
cd "$SCRIPT_DIR"

# Environment variables
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PATH="$HOME/.local/bin:$PATH"

# Resolve HF token: env var > $PANOWORLD_TOKEN_DIR > ~/.config/panoworld > legacy repo-root
if [ -z "${HF_TOKEN:-}" ]; then
    for candidate in \
        "${PANOWORLD_TOKEN_DIR:-}/hf_token.txt" \
        "$HOME/.config/panoworld/hf_token.txt" \
        "$SCRIPT_DIR/hf_token.txt"; do
        if [ -n "$candidate" ] && [ -f "$candidate" ]; then
            export HF_TOKEN=$(cat "$candidate" | tr -d '[:space:]')
            break
        fi
    done
fi

# Optionally load other PanoWorld tokens too (Gemini, OpenAI) if present
for var in GEMINI_API_KEY OPENAI_API_KEY; do
    if [ -z "${!var:-}" ]; then
        case "$var" in
            GEMINI_API_KEY) base=gemini_token.txt ;;
            OPENAI_API_KEY) base=openai_token.txt ;;
        esac
        for candidate in \
            "${PANOWORLD_TOKEN_DIR:-}/$base" \
            "$HOME/.config/panoworld/$base"; do
            if [ -n "$candidate" ] && [ -f "$candidate" ]; then
                export "$var"="$(cat "$candidate" | tr -d '[:space:]')"
                break
            fi
        done
    fi
done

echo "PanoWorld environment activated."
echo "  project root: $SCRIPT_DIR"
echo "  HF_HOME:      $HF_HOME"
echo ""
echo "Try:"
echo "  python generate_pano.py --help"
