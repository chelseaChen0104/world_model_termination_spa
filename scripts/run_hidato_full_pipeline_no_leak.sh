#!/bin/bash
# Hidato full no-leak pipeline:
#   1. Strip doom-suffix leak from existing combined data
#   2. SFT on cleaned data (Qwen2.5-1.5B + standard hparams)
#   3. RL with v8 anchor + --action-quality-bonus 0.5
#   4. Eval (greedy Pass@1 + stochastic Pass@k)

set -e
cd /root/autodl-tmp/world_model_termination_spa

PY=/root/miniconda3/bin/python

INPUT_DIR=${INPUT_DIR:-data/hidato_b_h1_combined}
NO_LEAK_DIR=${NO_LEAK_DIR:-data/hidato_combined_no_leak}
SFT_OUT=${SFT_OUT:-outputs/sft_hidato_no_leak}
RL_OUT=${RL_OUT:-outputs/rl_hidato_no_leak_v8_aq}

N_RL_STEPS=${N_RL_STEPS:-200}
ACTION_QUALITY_BONUS=${ACTION_QUALITY_BONUS:-0.5}
LR=${LR:-1e-5}
KL_COEF=${KL_COEF:-0.05}
VIABILITY_KL_COEF=${VIABILITY_KL_COEF:-0.5}

mkdir -p logs
LOG_STRIP=logs/hidato_strip.log
LOG_SFT=logs/sft_hidato_no_leak.log
LOG_RL=logs/rl_hidato_no_leak_v8_aq.log
LOG_EVAL=logs/eval_hidato_no_leak_full.log

# Step 1
if [ ! -f "$NO_LEAK_DIR/wm_train_no_post_bp.parquet" ]; then
    echo "=== Step 1/4: strip doom-suffix leak (Hidato) ===" | tee "$LOG_STRIP"
    bash scripts/_run_with_env.sh "$PY" scripts/strip_doom_suffix.py \
        --input "$INPUT_DIR" --output "$NO_LEAK_DIR" 2>&1 | tee -a "$LOG_STRIP"
fi

# Step 2 — SFT
echo | tee "$LOG_SFT"
echo "============================================================" | tee -a "$LOG_SFT"
echo "  Step 2/4: Hidato SFT on no-leak data" | tee -a "$LOG_SFT"
echo "============================================================" | tee -a "$LOG_SFT"
bash scripts/_run_with_env.sh "$PY" -u src/training/simple_sft_trainer.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --train_file "$NO_LEAK_DIR/wm_train_no_post_bp.parquet" \
    --val_file "$NO_LEAK_DIR/wm_val_no_post_bp.parquet" \
    --output_dir "$SFT_OUT" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 --max_length 1024 \
    --eval_steps 25 --save_steps 100000 \
    2>&1 | tee -a "$LOG_SFT"

# Step 3 — RL
echo | tee "$LOG_RL"
echo "============================================================" | tee -a "$LOG_RL"
echo "  Step 3/4: Hidato RL with v8 anchor + action_quality_bonus=$ACTION_QUALITY_BONUS" | tee -a "$LOG_RL"
echo "============================================================" | tee -a "$LOG_RL"
bash scripts/_run_with_env.sh "$PY" -u src/training/rl_trainer_v6.py \
    --env hidato \
    --sft-checkpoint "$SFT_OUT/final" \
    --output-dir "$RL_OUT" \
    --n-total-steps "$N_RL_STEPS" \
    --n-puzzles-per-batch 4 --group-size 8 \
    --learning-rate "$LR" --kl-coef "$KL_COEF" \
    --eval-every 25 --seed 42 \
    --reward-version v8 \
    --viability-kl-coef "$VIABILITY_KL_COEF" \
    --action-quality-bonus "$ACTION_QUALITY_BONUS" \
    --truncation-mode off \
    --prepend-current-state --single-turn-eval \
    --max-response-tokens 512 \
    2>&1 | tee -a "$LOG_RL"

# Step 4 — Eval (greedy + stochastic)
echo | tee "$LOG_EVAL"
echo "============================================================" | tee -a "$LOG_EVAL"
echo "  Step 4/4: Eval (greedy Pass@1 + stochastic Pass@k)" | tee -a "$LOG_EVAL"
echo "============================================================" | tee -a "$LOG_EVAL"
bash scripts/_run_with_env.sh "$PY" scripts/sanity_check_checkpoint.py \
    --sft-path "$RL_OUT/final" --env hidato \
    --prepend-current-state --reset-history-per-step \
    --max-new-tokens 512 \
    --n-puzzles 30 --k 8 --temperature 0.7 \
    2>&1 | tee -a "$LOG_EVAL"

echo | tee -a "$LOG_EVAL"
echo "=== Hidato no-leak full pipeline complete ===" | tee -a "$LOG_EVAL"
echo "  SFT: $SFT_OUT/final" | tee -a "$LOG_EVAL"
echo "  RL:  $RL_OUT/final" | tee -a "$LOG_EVAL"
