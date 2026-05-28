#!/bin/bash
# ============================================================
# PanoWorld install script
# ============================================================
# Usage:
#   bash install.sh
#
# Defaults (override via env vars before sourcing):
#   HF_HOME=$HOME/.cache/huggingface
#
# HuggingFace credentials are picked up from (in order):
#   1) $HF_TOKEN env var
#   2) $PANOWORLD_TOKEN_DIR/hf_token.txt
#   3) $HOME/.config/panoworld/hf_token.txt
#   4) ./hf_token.txt   (legacy, repo-root)
# See docs/TOKENS.md.
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  PanoWorld install script"
echo "============================================================"
echo ""

# ============================================================
# 1. Check and install system dependencies
# ============================================================
echo "[1/6] System dependencies..."

if ! command -v git-lfs &> /dev/null; then
    echo "   installing git-lfs..."
    sudo apt update && sudo apt install -y git-lfs
    git lfs install
else
    echo "   ok   git-lfs"
fi

for cmd in curl ffmpeg wget; do
    if ! command -v $cmd &> /dev/null; then
        echo "   installing $cmd..."
        sudo apt install -y $cmd
    else
        echo "   ok   $cmd"
    fi
done

# ============================================================
# 2. Create / activate the conda environment
# ============================================================
echo ""
echo "[2/6] Conda environment..."

# Initialize conda
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo "   ERROR: anaconda3 / miniconda3 not found in \$HOME; install conda first." >&2
    exit 1
fi

if conda env list | grep -q "^cosmos "; then
    echo "   ok   cosmos env already exists"
else
    echo "   creating cosmos env (Python 3.10 -- required by flash-attn)..."
    conda create -n cosmos python=3.10 -y
fi

conda activate cosmos
echo "   ok   activated cosmos (Python $(python --version | awk '{print $2}'))"

# ============================================================
# 3. Install the uv package manager
# ============================================================
echo ""
echo "[3/6] uv package manager..."

if ! command -v uv &> /dev/null && [ ! -f "$HOME/.local/bin/uv" ]; then
    echo "   installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
echo "   ok   uv ready"

# ============================================================
# 4. Install Python dependencies
# ============================================================
echo ""
echo "[4/6] Python dependencies..."

cd "$SCRIPT_DIR"
echo "   installing project deps into the active conda env..."
uv pip install -e ".[cu128]" --system

# ============================================================
# 4b. DAP pipeline extras (depth + trajectory annotation)
# ============================================================
echo ""
echo "[4b/6] DAP pipeline extras..."

echo "   installing einops, scipy, torchmetrics, opencv-headless..."
# Uninstall any opencv-python first (it pulls numpy>=2), then use the headless build
pip uninstall -y opencv-python 2>/dev/null || true
pip install -q "numpy>=1.24,<2"
pip install -q einops torchmetrics scipy "opencv-python-headless>=4.8,<4.11"

echo "   installing CoTracker3 (point tracking)..."
pip install -q git+https://github.com/facebookresearch/co-tracker.git

# Pre-download the CoTracker3 offline checkpoint
CT3_CKPT="$HOME/.cache/torch/hub/checkpoints/scaled_offline.pth"
if [ -f "$CT3_CKPT" ]; then
    echo "   ok   CoTracker3 checkpoint already cached"
else
    echo "   downloading CoTracker3 checkpoint (~98 MB)..."
    mkdir -p "$(dirname "$CT3_CKPT")"
    python -c "
import torch
torch.hub.load('facebookresearch/co-tracker', 'cotracker3_offline')
print('   ok   CoTracker3 checkpoint downloaded')
"
fi

# ============================================================
# 5. Configure the Hugging Face token
# ============================================================
echo ""
echo "[5/6] Hugging Face token..."

# Use a local-disk cache (avoid NFS write amplification)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_HOME"
echo "   cache: $HF_HOME"

# Resolve HF token: env var > $PANOWORLD_TOKEN_DIR > ~/.config/panoworld > legacy repo-root
TOKEN_FILE=""
for candidate in \
    "${PANOWORLD_TOKEN_DIR:-}/hf_token.txt" \
    "$HOME/.config/panoworld/hf_token.txt" \
    "$SCRIPT_DIR/hf_token.txt"; do
    if [ -n "$candidate" ] && [ -f "$candidate" ]; then
        TOKEN_FILE="$candidate"
        break
    fi
done

if [ -z "${HF_TOKEN:-}" ] && [ -n "$TOKEN_FILE" ]; then
    HF_TOKEN=$(cat "$TOKEN_FILE" | tr -d '[:space:]')
    export HF_TOKEN
fi

if [ -n "${HF_TOKEN:-}" ]; then
    echo "   logging in to Hugging Face..."
    huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || true
    echo "   ok   Hugging Face configured"
else
    echo "   WARN  no HF token found. Run 'huggingface-cli login' manually or see docs/TOKENS.md."
fi

# ============================================================
# 6. Install rclone (used to download the test set + checkpoints)
# ============================================================
echo ""
echo "[6/6] rclone..."

if command -v rclone &> /dev/null; then
    echo "   ok   rclone ($(rclone --version | head -1))"
else
    echo "   installing rclone..."
    curl -s https://rclone.org/install.sh | sudo bash 2>/dev/null || {
        echo "   WARN  rclone install failed; install manually from https://rclone.org/install/"
    }
fi

# Hint if no gdrive remote is configured
RCLONE_CONF="$HOME/.config/rclone/rclone.conf"
if [ -f "$RCLONE_CONF" ] && grep -q "\[gdrive\]" "$RCLONE_CONF" 2>/dev/null; then
    echo "   ok   rclone gdrive remote configured"
else
    echo "   WARN  no rclone gdrive remote. Run:"
    echo "         rclone config        # create a remote named 'gdrive'"
    echo "         docs: https://rclone.org/drive/"
fi

# ============================================================
# Done
# ============================================================
echo ""
echo "============================================================"
echo "  Install complete."
echo "============================================================"
echo ""
echo "Usage:"
echo ""
echo "  # 1. Activate the env"
echo "  cd $SCRIPT_DIR"
echo "  source activate.sh"
echo ""
echo "  # 2. Run panoramic inference"
echo "  python generate_pano.py \\"
echo "      --input_path  assets/panotest/lobby.png \\"
echo "      --prompt      \"\$(cat assets/panotest/lobby.txt)\" \\"
echo "      --pano_prompt \"<your 360-degree scene description>\" \\"
echo "      --output_dir  output/lobby_demo \\"
echo "      --finetune_checkpoint checkpoints/panoworld_main/model_ema_bf16.pt \\"
echo "      --resolution 512,1024 --num_frames 93 --guidance 7 --num_steps 35 --seed 42 \\"
echo "      --equirect_rope --use_clip --v3 --disable_guardrails --offload_diffusion_model"
echo ""
echo "Notes:"
echo "  - Model cache:        $HF_HOME (~60 GB, local disk, may not survive instance restart)"
echo "  - Download the pre-trained checkpoint: see docs/MODEL_ZOO.md"
echo "  - Download the evaluation test set:     see docs/DATASET.md"
echo "  - On a fresh instance, just re-run 'bash install.sh' -- it skips files that already exist."
echo "  - Full command reference: docs/INFERENCE.md and docs/PANOWORLD_INFERENCE_GUIDE.md"
echo "============================================================"
