#!/bin/bash
# 5×10 Pentomino SFT with solution-path augmentation. Mirror of
# run_pentomino_5x4_sft_augmented.sh (B-8) but for the 5×10 / 10-piece config.
#
# Pipeline:
#   1. Generate solution-path augmented samples (--max-tilings 466 → ~4660 samples)
#   2. Combine with LLM-policy data → ~50% augmented (similar ratio to B-8)
#   3. SFT Qwen2.5-1.5B with same hparams as B-8 / B-9 (lr=1e-4, ep=5, bs=16)
#
# Output: outputs/sft_pentomino_5x10_augmented/
# Wall time on A800: ~10 min augmenter + ~2 hr SFT = ~2h 10min.

set -e
cd /root/autodl-tmp/world_model_termination_spa

PY=/root/miniconda3/bin/python
AUG_DIR=${AUG_DIR:-data/pentomino_5x10_solution_paths}
COMBINED_DIR=${COMBINED_DIR:-data/pentomino_5x10_combined}
SFT_OUT=${SFT_OUT:-outputs/sft_pentomino_5x10_augmented}
MAX_TILINGS=${MAX_TILINGS:-466}

mkdir -p logs
LOG_AUG=logs/pentomino_5x10_augmenter.log
LOG_COMBINE=logs/pentomino_5x10_combine.log
LOG_SFT=logs/sft_pentomino_5x10_augmented.log

# Step 1: Augmenter (skip if already done)
if [ ! -f "$AUG_DIR/wm_train_solution_paths.parquet" ]; then
    echo "=== Step 1/3: solution-path augmenter (5×10, max $MAX_TILINGS tilings) ===" | tee "$LOG_AUG"
    bash scripts/_run_with_env.sh "$PY" -m src.data.solution_path_augmenter \
        --board-h 5 --board-w 10 \
        --piece-set F,I,L,N,P,T,U,V,Y,Z \
        --variant polyomino_minimal \
        --max-tilings "$MAX_TILINGS" \
        --output-dir "$AUG_DIR" 2>&1 | tee -a "$LOG_AUG"
else
    echo "=== Step 1/3: augmenter output already exists, skipping ===" | tee "$LOG_AUG"
fi

# Step 2: Combine
echo | tee "$LOG_COMBINE"
echo "=== Step 2/3: combine LLM-policy + augmented ===" | tee -a "$LOG_COMBINE"
bash scripts/_run_with_env.sh "$PY" scripts/combine_pentomino_5x10_with_augmented.py \
    --llm-dir data/pentomino_b9_llm_policy_minimal \
    --aug-dir "$AUG_DIR" \
    --output-dir "$COMBINED_DIR" 2>&1 | tee -a "$LOG_COMBINE"

# Step 3: SFT
echo | tee "$LOG_SFT"
echo "============================================================" | tee -a "$LOG_SFT"
echo "  Step 3/3: SFT on combined data" | tee -a "$LOG_SFT"
echo "============================================================" | tee -a "$LOG_SFT"
bash scripts/_run_with_env.sh "$PY" -u src/training/simple_sft_trainer.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --train_file "$COMBINED_DIR/wm_train_no_post_bp.parquet" \
    --val_file "$COMBINED_DIR/wm_val_no_post_bp.parquet" \
    --output_dir "$SFT_OUT" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --max_length 1024 \
    --eval_steps 50 \
    --save_steps 100000 \
    2>&1 | tee -a "$LOG_SFT"

echo | tee -a "$LOG_SFT"
echo "=== 5×10 Pentomino augmented SFT complete ===" | tee -a "$LOG_SFT"
echo "  Output: $SFT_OUT/final/" | tee -a "$LOG_SFT"
