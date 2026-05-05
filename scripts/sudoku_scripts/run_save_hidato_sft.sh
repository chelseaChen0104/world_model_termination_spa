#!/usr/bin/env bash
# Launch SAVE Hidato 5x4 f_phi SFT using the paper-aligned trainer
# (scripts/sudoku_scripts/save_sft_train.py) with the same hparams the
# Sudoku pilot run used (autodl2). Runs on autodl (Hidato machine).
#
# Pipeline:
#   1. Train: 4-component loss (L_trans + λ·L_viab + η·L_rank + μ·L_state)
#      with GroupedSampler keeping full sibling sets per batch.
#   2. Eval on test (writes eval_test.json).
#   3. Calibrate on val (temperature scaling + τ_keep + τ_fb).
#
# Run from project root on autodl:
#   tmux new -d -s save_fphi_hidato_paper \
#     'bash scripts/sudoku_scripts/run_save_hidato_sft.sh > logs/save_fphi_hidato_paper.log 2>&1'

set -euo pipefail

PROJ=/root/autodl-tmp/world_model_termination_spa
PY=/root/miniconda3/bin/python
DATE=$(date +%Y%m%d_%H%M%S)
OUT=$PROJ/outputs/save_hidato5x4_f_phi_paper
LOG=$PROJ/logs/save_fphi_hidato_paper_${DATE}.log

# Hidato machine has Qwen2.5-1.5B cached locally; force offline mode.
export HF_HOME=/root/autodl-tmp/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$(dirname "$LOG")"
cd "$PROJ"

TRAIN=$PROJ/data/hidato5x4/sft_pilot/train.sft.jsonl
VAL=$PROJ/data/hidato5x4/sft_pilot/val.sft.jsonl
TEST=$PROJ/data/hidato5x4/sft_pilot/test.sft.jsonl

echo "=== SAVE f_phi (paper-aligned) — Hidato 5×4 ==="
echo "  $(date)"
echo "  output: $OUT"
echo "  log:    $LOG"
echo "  train:  $TRAIN ($(wc -l < $TRAIN) samples)"
echo "  val:    $VAL ($(wc -l < $VAL) samples)"
echo "  test:   $TEST ($(wc -l < $TEST) samples)"
echo ""

# -- 1. SFT (4-component loss + grouped sampler) --
echo "=== [1/3] SFT ==="
$PY -u scripts/sudoku_scripts/save_sft_train.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --train "$TRAIN" \
    --val   "$VAL" \
    --output_dir "$OUT" \
    --num_train_epochs 3 \
    --learning_rate 1e-5 \
    --per_device_batch_sets 8 \
    --gradient_accumulation_steps 1 \
    --max_length 512 \
    --warmup_steps 100 \
    --logging_steps 10 \
    --eval_steps 100 \
    --save_steps 500 \
    --bf16 \
    2>&1 | tee "$LOG"

# -- 2. Test eval (Q2 row) --
echo ""
echo "=== [2/3] Q2 eval on test ==="
$PY -u scripts/sudoku_scripts/save_sft_eval.py \
    --checkpoint "$OUT/final" \
    --eval "$TEST" \
    --output_json "$OUT/eval_test.json" \
    2>&1 | tee -a "$LOG"

# -- 3. Calibration (temperature + τ_keep + τ_fb) --
echo ""
echo "=== [3/3] Calibration on val ==="
$PY -u scripts/sudoku_scripts/save_sft_calibrate.py \
    --checkpoint "$OUT/final" \
    --val "$VAL" \
    --output_json "$OUT/calibration.json" \
    2>&1 | tee -a "$LOG"

echo ""
echo "=== DONE — $(date) ==="
ls -lh "$OUT/final/" 2>/dev/null | head
echo ""
echo "Eval results:"
cat "$OUT/eval_test.json" 2>/dev/null
echo ""
echo "Calibration:"
cat "$OUT/calibration.json" 2>/dev/null
