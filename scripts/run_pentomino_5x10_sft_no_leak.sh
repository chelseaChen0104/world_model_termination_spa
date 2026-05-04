#!/bin/bash
# Pentomino 5×10 SFT — no-leak variant. Strips the env-render doom-reason
# suffix from the existing combined dataset (data/pentomino_5x10_combined),
# then trains SFT on the cleaned data. Same hparams as the leaked run for an
# apples-to-apples comparison.
#
# Hypothesis: the always-doom greedy collapse on 5×10 is largely caused by
# the model learning a text shortcut (`board now unsolvable` → viability=false).
# Stripping the suffix should force the model to predict viability from
# board content alone.

set -e
cd /root/autodl-tmp/world_model_termination_spa

PY=/root/miniconda3/bin/python
INPUT_DIR=${INPUT_DIR:-data/pentomino_5x10_combined}
NO_LEAK_DIR=${NO_LEAK_DIR:-data/pentomino_5x10_combined_no_leak}
SFT_OUT=${SFT_OUT:-outputs/sft_pentomino_5x10_no_leak}

mkdir -p logs
LOG_STRIP=logs/pentomino_5x10_strip.log
LOG_SFT=logs/sft_pentomino_5x10_no_leak.log

# Step 1: Strip
if [ ! -f "$NO_LEAK_DIR/wm_train_no_post_bp.parquet" ]; then
    echo "=== Step 1/2: strip doom-suffix leak ===" | tee "$LOG_STRIP"
    bash scripts/_run_with_env.sh "$PY" scripts/strip_doom_suffix.py \
        --input "$INPUT_DIR" \
        --output "$NO_LEAK_DIR" 2>&1 | tee -a "$LOG_STRIP"
else
    echo "=== Step 1/2: no-leak data already exists, skipping ===" | tee "$LOG_STRIP"
fi

# Step 2: SFT on no-leak data
echo | tee "$LOG_SFT"
echo "============================================================" | tee -a "$LOG_SFT"
echo "  Step 2/2: Pentomino 5×10 SFT on no-leak data" | tee -a "$LOG_SFT"
echo "============================================================" | tee -a "$LOG_SFT"
bash scripts/_run_with_env.sh "$PY" -u src/training/simple_sft_trainer.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --train_file "$NO_LEAK_DIR/wm_train_no_post_bp.parquet" \
    --val_file "$NO_LEAK_DIR/wm_val_no_post_bp.parquet" \
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
echo "=== Pentomino 5×10 no-leak SFT complete ===" | tee -a "$LOG_SFT"
echo "  Output: $SFT_OUT/final/" | tee -a "$LOG_SFT"
echo "  Next: bash scripts/eval_pentomino_5x10_no_leak_pass1.sh" | tee -a "$LOG_SFT"
