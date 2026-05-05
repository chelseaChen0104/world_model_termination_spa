#!/usr/bin/env bash
# Run a complete SAVE pipeline for a given Qwen size.
# Usage: run_size_pipeline.sh <model_id> <epochs>
# Example: run_size_pipeline.sh Qwen/Qwen2.5-0.5B-Instruct 3
#
# Steps: download model → SFT → Q2 eval → calibrate → generation transition test.
# Logs to logs/size_<short>.log; outputs to outputs/save_sudoku4_f_phi_<short>/.
set -e
MODEL=${1:-Qwen/Qwen2.5-0.5B-Instruct}
EPOCHS=${2:-3}
BATCH_SETS=${3:-8}
GRAD_CKPT=${4:-}     # set to "yes" to enable gradient checkpointing
OPTIM=${5:-adamw_torch}  # use 'paged_adamw_8bit' for 7B+
# Short identifier: strip HF user/ prefix, strip -Instruct suffix.
# Examples:
#   Qwen/Qwen2.5-7B-Instruct       -> Qwen2.5-7B
#   meta-llama/Llama-3.2-1B-Instruct -> Llama-3.2-1B
SHORT=$(echo "$MODEL" | sed 's|^[^/]*/||' | sed 's|-Instruct$||')
PROJ=/root/autodl-tmp/world_model_termination_spa
OUT="$PROJ/outputs/save_sudoku4_f_phi_${SHORT}"
LOG="$PROJ/logs/size_${SHORT}.log"
PY=/root/miniconda3/bin/python
mkdir -p "$(dirname "$LOG")"
echo "=== Size pipeline: $MODEL → $OUT ===" > "$LOG"
echo "started: $(date)" >> "$LOG"
cd "$PROJ"

# 1. Download model (offline mirror)
echo "" >> "$LOG"; echo "=== [1/4] download ===" >> "$LOG"
HF_ENDPOINT=https://hf-mirror.com $PY -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
print('Downloading $MODEL via mirror...')
t = AutoTokenizer.from_pretrained('$MODEL')
m = AutoModelForCausalLM.from_pretrained('$MODEL')
n = sum(p.numel() for p in m.parameters())
print(f'OK params={n/1e6:.1f}M')
" >> "$LOG" 2>&1 || { echo "DOWNLOAD_FAIL" >> "$LOG"; exit 1; }

# 2. SFT
echo "" >> "$LOG"; echo "=== [2/4] SFT ($EPOCHS epoch, batch_sets=$BATCH_SETS, grad_ckpt=$GRAD_CKPT, optim=$OPTIM) ===" >> "$LOG"
GC_FLAG=""
[ "$GRAD_CKPT" = "yes" ] && GC_FLAG="--gradient_checkpointing"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $PY -u scripts/sudoku_scripts/save_sft_train.py \
    --model_name_or_path "$MODEL" \
    --train data/sudoku4/sft/train_balanced.sft.jsonl \
    --val   data/sudoku4/sft/val_natural_calibration.sft.jsonl \
    --output_dir "$OUT" \
    --num_train_epochs "$EPOCHS" \
    --per_device_batch_sets "$BATCH_SETS" \
    --learning_rate 1e-5 \
    --optim "$OPTIM" \
    $GC_FLAG \
    >> "$LOG" 2>&1 || { echo "SFT_FAIL" >> "$LOG"; exit 1; }

# 3. Test eval (Q2 row)
echo "" >> "$LOG"; echo "=== [3/4] Q2 eval on test ===" >> "$LOG"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $PY -u scripts/sudoku_scripts/save_sft_eval.py \
    --checkpoint "$OUT/final" \
    --eval data/sudoku4/sft/test_natural_policy.sft.jsonl \
    --output_json "$OUT/eval_test.json" \
    >> "$LOG" 2>&1 || { echo "EVAL_FAIL" >> "$LOG"; }

# 4. Generation transition test (Q1 row)
echo "" >> "$LOG"; echo "=== [4/4] generation transition test ===" >> "$LOG"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $PY -u scripts/sudoku_scripts/save_sft_generation_test.py \
    --checkpoint "$OUT/final" \
    --eval data/sudoku4/sft/val_natural_calibration.sft.jsonl \
    --n_samples 50 \
    --output_json "$OUT/gen_test.json" \
    >> "$LOG" 2>&1 || { echo "GEN_FAIL" >> "$LOG"; }

echo "" >> "$LOG"
echo "=== DONE size pipeline $SHORT ===" >> "$LOG"
date >> "$LOG"
df -h /root/autodl-tmp >> "$LOG"
