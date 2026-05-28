#!/bin/bash
# ============================================================
# PanoWorld example-runner (NVIDIA-original Cosmos demos)
# ============================================================
# Usage:
#   source activate.sh
#   bash run_examples.sh [indoor|outdoor|all]
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Make sure an env is active (conda or venv)
if [ -z "$CONDA_DEFAULT_ENV" ] && [ -z "$VIRTUAL_ENV" ]; then
    echo "ERROR: activate an env first: source activate.sh"
    exit 1
fi

# Require the cosmos conda env if conda is in use
if [ -n "$CONDA_DEFAULT_ENV" ] && [ "$CONDA_DEFAULT_ENV" != "cosmos" ]; then
    echo "ERROR: activate the cosmos env first: source activate.sh"
    exit 1
fi

OUTPUT_DIR="$SCRIPT_DIR/output"
mkdir -p "$OUTPUT_DIR"

SCENE="${1:-all}"

echo "============================================================"
echo "  PanoWorld example inference"
echo "============================================================"
echo "  output_dir: $OUTPUT_DIR"
echo "  scene:      $SCENE"
echo "============================================================"
echo ""

run_indoor() {
    echo "running indoor_room..."
    python examples/inference.py \
        -i assets/custom/indoor_room.json \
        -o "$OUTPUT_DIR/indoor_room" \
        --inference-type=text2world \
        --model=2B/post-trained
    echo "indoor_room done -> $OUTPUT_DIR/indoor_room"
}

run_outdoor() {
    echo "running outdoor_street..."
    python examples/inference.py \
        -i assets/custom/outdoor_street.json \
        -o "$OUTPUT_DIR/outdoor_street" \
        --inference-type=text2world \
        --model=2B/post-trained
    echo "outdoor_street done -> $OUTPUT_DIR/outdoor_street"
}

case "$SCENE" in
    indoor)
        run_indoor
        ;;
    outdoor)
        run_outdoor
        ;;
    all)
        run_indoor
        echo ""
        run_outdoor
        ;;
    *)
        echo "Usage: bash run_examples.sh [indoor|outdoor|all]"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "  Inference complete."
echo "============================================================"
echo "Outputs: $OUTPUT_DIR"
echo ""
echo "Browse:"
echo "  ls -la $OUTPUT_DIR/"
echo "============================================================"
