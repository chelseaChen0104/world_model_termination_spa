#!/bin/bash
# SPA-full baseline: SFT with <observation> + <prediction> (state estimation +
# transition modeling, no termination tag), then PPO. SPA's full recipe before
# our <solvable> extension. The cleanest single-knob ablation of "what does
# our <solvable> tag add over base SPA?".
#
# Pipeline:
#   1. Build no-solvable dataset by stripping <solvable> tags from B-5.
#   2. SFT Qwen2.5-1.5B on the stripped dataset (~2 hr).
#   3. RL with reward v6 (no termination reward) (~3-4 hr).
#
# Output: outputs/baseline_spa_full_sft/    (SFT checkpoint)
#         outputs/baseline_spa_full_rl/     (RL checkpoint)
# Total wall: ~5-6 hr on A800.

set -e
cd /root/autodl-tmp/world_model_termination_spa

PY=/root/miniconda3/bin/python

# === Config ===
INPUT_DATA=${INPUT_DATA:-data/sudoku_4x4_llm_policy_minimal_spa_scale}
DATA_DIR=${DATA_DIR:-data/sudoku_4x4_no_solvable}
SFT_OUT=${SFT_OUT:-outputs/baseline_spa_full_sft}
RL_OUT=${RL_OUT:-outputs/baseline_spa_full_rl}

mkdir -p logs
LOG_DATA=logs/baseline_spa_full_data.log
LOG_SFT=logs/baseline_spa_full_sft.log
LOG_RL=logs/baseline_spa_full_rl.log

# === Step 1: build no-solvable dataset ===
if [ ! -f "$DATA_DIR/wm_train_no_post_bp.parquet" ]; then
    echo "=== Step 1/3: building no-solvable dataset ===" | tee "$LOG_DATA"
    bash scripts/_run_with_env.sh "$PY" scripts/strip_tags_from_parquet.py \
        --input "$INPUT_DATA" \
        --output "$DATA_DIR" \
        --variant se_pred 2>&1 | tee -a "$LOG_DATA"
else
    echo "=== Step 1/3: no-solvable dataset already exists, skipping ===" | tee "$LOG_DATA"
fi

# === Step 2: SFT on no-solvable data ===
echo | tee "$LOG_SFT"
echo "============================================================" | tee -a "$LOG_SFT"
echo "  Step 2/3: SFT on no-solvable data (SPA-full recipe)" | tee -a "$LOG_SFT"
echo "============================================================" | tee -a "$LOG_SFT"
bash scripts/_run_with_env.sh "$PY" -u src/training/simple_sft_trainer.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --train_file "$DATA_DIR/wm_train_no_post_bp.parquet" \
    --val_file "$DATA_DIR/wm_val_no_post_bp.parquet" \
    --output_dir "$SFT_OUT" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --max_length 1024 \
    --eval_steps 200 \
    --save_steps 100000 \
    2>&1 | tee -a "$LOG_SFT"

# === Step 3: RL on top of SPA-full SFT ===
echo | tee "$LOG_RL"
echo "============================================================" | tee -a "$LOG_RL"
echo "  Step 3/3: RL on SPA-full SFT (reward=v6, no termination reward)" | tee -a "$LOG_RL"
echo "============================================================" | tee -a "$LOG_RL"
bash scripts/_run_with_env.sh "$PY" -u src/training/rl_trainer_v6.py \
    --env sudoku \
    --grid-size 4 --difficulty easy \
    --sft-checkpoint "$SFT_OUT/final" \
    --output-dir "$RL_OUT" \
    --n-total-steps 200 \
    --n-puzzles-per-batch 4 --group-size 8 \
    --learning-rate 1e-5 --kl-coef 0.05 \
    --eval-every 25 --seed 42 \
    --reward-version v6 \
    --truncation-mode off \
    --skip-solvable-reward \
    2>&1 | tee -a "$LOG_RL"

echo | tee -a "$LOG_RL"
echo "=== SPA-full baseline complete ===" | tee -a "$LOG_RL"
echo "  SFT: $SFT_OUT/final" | tee -a "$LOG_RL"
echo "  RL:  $RL_OUT/final" | tee -a "$LOG_RL"
