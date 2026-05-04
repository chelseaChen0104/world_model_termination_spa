#!/bin/bash
# B-H1 post-SFT Pass@1 sanity rollout.
#
# Uses rl_trainer_v6.py's quick_pass1() (same as the trainer's eval-during-training)
# to roll out greedy on Hidato puzzles and report Pass@1 + solvable_acc + bp_recall.
# Runs --n-total-steps 1 --eval-every 1, which only fires the initial eval and
# exits without modifying the model.
#
# Output: logs/eval_hidato_pass1.log

set -e
cd /root/autodl-tmp/world_model_termination_spa

SFT_PATH=${SFT_PATH:-outputs/sft_hidato_b_h1/final}
OUT_DIR=${OUT_DIR:-outputs/eval_hidato_pass1_tmp}

mkdir -p logs
LOG=logs/eval_hidato_pass1.log
: > "$LOG"

echo "============================================================" | tee -a "$LOG"
echo "  B-H1 Pass@1 sanity rollout (Hidato SFT)" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
echo "  SFT checkpoint: $SFT_PATH" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

bash scripts/_run_with_env.sh /root/miniconda3/bin/python -u src/training/rl_trainer_v6.py \
    --env hidato \
    --sft-checkpoint "$SFT_PATH" \
    --output-dir "$OUT_DIR" \
    --n-total-steps 1 --eval-every 1 \
    --reward-version v8 --viability-kl-coef 0.5 --seed 42 \
    --truncation-mode off \
    2>&1 | tee -a "$LOG"

# Conserve disk: keep jsonl + log, drop the saved model
if [ -d "$OUT_DIR" ]; then
    if [ -f "$OUT_DIR/rl_log.jsonl" ]; then
        cp "$OUT_DIR/rl_log.jsonl" logs/eval_hidato_pass1_rl_log.jsonl
    fi
    rm -rf "$OUT_DIR"
fi

echo | tee -a "$LOG"
echo "=== B-H1 Pass@1 eval complete ===" | tee -a "$LOG"
echo "Log: $LOG" | tee -a "$LOG"
