#!/bin/bash
# Run B-9: 5×10 Pentomino SFT (10-piece variant {F,I,L,N,P,T,U,V,Y,Z}).
#
# Designed to fix the trajectory-length-distribution problem that caused B-7
# greedy collapse (5×4 board has only ~20 distinct tilings; 5×10 has ~4664).
# Greater puzzle variety should reduce memorization risk and produce richer
# multi-step trajectories.
#
# Same SPA-scale hparams as B-5 / B-8 / B-H1: lr=1e-4, ep=5, bs=16, max_length=1024.
#
# Data: data/pentomino_b9_llm_policy_minimal/wm_train_no_post_bp.parquet
#   - 4,652 train samples
#   - 52% solvable / 48% doom (much healthier balance than B-7's 81/19)
#   - Step dist: 2389 at step 0, 2028 at step 1, 233 at step 2-3
#
# Output: outputs/sft_pentomino_b9/
# Wall time on A800/H800: ~1.5-2 hr.

set -e
cd /root/autodl-tmp/world_model_termination_spa

DATA_DIR=${DATA_DIR:-data/pentomino_b9_llm_policy_minimal}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/sft_pentomino_b9}

mkdir -p logs
LOG=logs/sft_b9.log
: > "$LOG"

echo "============================================================" | tee -a "$LOG"
echo "  B-9 Pentomino 5×10 SFT" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
echo "  Train data: $DATA_DIR/wm_train_no_post_bp.parquet" | tee -a "$LOG"
echo "  Val data:   $DATA_DIR/wm_val_no_post_bp.parquet" | tee -a "$LOG"
echo "  Output:     $OUTPUT_DIR" | tee -a "$LOG"
echo "  Hparams:    lr=1e-4, ep=5, bs=4×4=16 effective, max_length=1024" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

bash scripts/_run_with_env.sh /root/miniconda3/bin/python -u src/training/simple_sft_trainer.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --train_file "$DATA_DIR/wm_train_no_post_bp.parquet" \
    --val_file "$DATA_DIR/wm_val_no_post_bp.parquet" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --max_length 1024 \
    --eval_steps 50 \
    --save_steps 100000 \
    2>&1 | tee -a "$LOG"

echo | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
echo "  B-9 SFT complete." | tee -a "$LOG"
echo "  Output: $OUTPUT_DIR/final/" | tee -a "$LOG"
echo "  Next steps:" | tee -a "$LOG"
echo "    1. Logprob threshold sweep (evaluate_rl.py --tag-name viability)" | tee -a "$LOG"
echo "    2. Pass@1 sanity rollout (greedy + stochastic on 30 puzzles)" | tee -a "$LOG"
echo "    3. RL with v8 anchor (run_rl_b9_v8.sh)" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
