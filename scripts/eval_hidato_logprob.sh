#!/bin/bash
# B-H1 post-SFT logprob threshold sweep.
#
# Reads B-H1 val parquet directly (so no env-side rollouts are needed —
# evaluate_rl.py's --eval-from-parquet path bypasses the env entirely).
# Outputs: P(solvable=True) and P(solvable=False) per sample, ROC AUC,
# and a threshold sweep showing precision/recall at varying τ.
#
# This is step 1 of the B-H1 post-SFT eval per scripts/run_hidato_sft.sh.
#
# Output: logs/eval_hidato_logprob.log + stdout report.

set -e
cd /root/autodl-tmp/world_model_termination_spa

SFT_PATH=${SFT_PATH:-outputs/sft_hidato_b_h1/final}
VAL_PARQUET=${VAL_PARQUET:-data/hidato_b_h1_combined/wm_val_no_post_bp.parquet}
N_PER_CLASS=${N_PER_CLASS:-200}

mkdir -p logs
LOG=logs/eval_hidato_logprob.log
: > "$LOG"

echo "============================================================" | tee -a "$LOG"
echo "  B-H1 logprob threshold sweep (Hidato SFT)" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
echo "  SFT checkpoint: $SFT_PATH" | tee -a "$LOG"
echo "  Val parquet:    $VAL_PARQUET" | tee -a "$LOG"
echo "  N per class:    $N_PER_CLASS" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

bash scripts/_run_with_env.sh /root/miniconda3/bin/python -u evaluate_rl.py \
    --env sudoku \
    --metric solvable-logprob \
    --sft-path "$SFT_PATH" \
    --eval-from-parquet "$VAL_PARQUET" \
    --n-per-class "$N_PER_CLASS" \
    --tag-name solvable \
    --skip-rl \
    2>&1 | tee -a "$LOG"

echo | tee -a "$LOG"
echo "=== B-H1 logprob eval complete ===" | tee -a "$LOG"
echo "Log: $LOG" | tee -a "$LOG"
