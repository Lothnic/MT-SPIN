#!/bin/bash
# run_mtspin.sh — Full MT-SPIN self-play training loop
# Runs 3 iterations of: Generate → Train → Evaluate

set -e  # exit on first error

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPTS_DIR")"

echo "============================================"
echo "  MT-SPIN: Self-Play Fine-Tuning Pipeline"
echo "  Project: ${PROJECT_DIR}"
echo "============================================"

# ── Stage 0: Evaluate SFT baseline ──
echo ""
echo "=== Evaluating SFT Baseline ==="
uv run python scripts/eval.py

# ── Iterative Self-Play Loop ──
SFT_ADAPTER="models/nllb-200-kangri-lora/final_adapter"

for ITER in 0 1 2; do
    NEXT=$((ITER + 1))

    # Determine which adapter to use for generation
    if [ $ITER -eq 0 ]; then
        REF_ADAPTER="$SFT_ADAPTER"
    else
        REF_ADAPTER="models/spin_iter_${ITER}"
    fi

    # Determine curriculum strategy
    if [ $ITER -eq 0 ]; then
        CURRICULUM="easy"
    else
        CURRICULUM="hard"
    fi

    echo ""
    echo "============================================"
    echo "  MT-SPIN Iteration ${NEXT}"
    echo "  Reference adapter: ${REF_ADAPTER}"
    echo "  Curriculum: ${CURRICULUM}"
    echo "============================================"

    # Stage 1: Generate preference pairs
    echo "--- Stage 1: Generating candidates ---"
    uv run python spin/spin_generate.py \
        --adapter "$REF_ADAPTER" \
        --data "data/processed_dataset/train_combined.parquet" \
        --output "spin/spin_data/iteration_${ITER}" \
        --curriculum "$CURRICULUM" \
        --num_candidates 4 \
        --batch_size 8

    # Stage 2: DPO Training
    echo "--- Stage 2: DPO Training ---"
    uv run python spin/spin_train.py \
        --ref_adapter "$REF_ADAPTER" \
        --spin_data "spin/spin_data/iteration_${ITER}" \
        --output "models/spin_iter_${NEXT}" \
        --beta 0.1 \
        --lambda_reward 0.5 \
        --epochs 3

    # Stage 3: Evaluate the new adapter
    # TODO: Update eval.py to accept --adapter flag, or manually swap the adapter path
    echo "--- Stage 3: Evaluation ---"
    echo "Iteration ${NEXT} adapter saved to: models/spin_iter_${NEXT}"
    echo "Run eval.py manually with this adapter to record metrics."
    echo ""
done

echo ""
echo "============================================"
echo "  MT-SPIN Complete!"
echo "  Check models/spin_iter_{1,2,3} for adapters"
echo "============================================"
