#!/usr/bin/env bash
# Hidato Q1 cross-size sweep: train + eval f_phi at multiple model sizes
# on TOY data, mirroring autodl2's Sudoku Q1 matrix.
#
# Uses scripts/sudoku_scripts/{save_sft_train, save_sft_eval, save_sft_calibrate}.py
# (the paper-aligned trainer pulled from autodl2 — NOT autodl2-specific).
#
# Data paths:
#   train: data/hidato5x4/sft/train.sft.jsonl              (toy)
#   val:   data/hidato5x4/sft/val.sft.jsonl                (toy)
#   test:  data/hidato5x4/sft/test.sft.jsonl               (toy)
#
# Output: outputs/save_hidato5x4_f_phi_<short>/{eval_test,calibration,gen_test}.json
#
# Run on autodl:
#   tmux new -d -s hidato_q1_sweep \
#     'bash scripts/run_hidato_q1_size_sweep.sh > logs/hidato_q1_sweep.log 2>&1'

set -uo pipefail   # NOT -e: keep going on per-size failures

PROJ=/root/autodl-tmp/world_model_termination_spa
PY=/root/miniconda3/bin/python
cd "$PROJ"
mkdir -p logs outputs

# autodl can't reach huggingface.co directly; mirror works.
# HF_HUB_OFFLINE will be flipped per-step (download → train).
export HF_HOME=/root/autodl-tmp/cache/huggingface

TRAIN=$PROJ/data/hidato5x4/sft/train.sft.jsonl
VAL=$PROJ/data/hidato5x4/sft/val.sft.jsonl
TEST=$PROJ/data/hidato5x4/sft/test.sft.jsonl

# Ensure toy SFT files exist (materialize from raw if not).
if [ ! -f "$TRAIN" ]; then
    echo "[setup] materializing toy SFT files via save_sft_prepare.py"
    mkdir -p "$(dirname "$TRAIN")"
    $PY scripts/save_sft_prepare.py \
        --input  data/hidato5x4/train_balanced.jsonl \
        --output "$TRAIN" || true
    $PY scripts/save_sft_prepare.py \
        --input  data/hidato5x4/val_natural_calibration.jsonl \
        --output "$VAL" || true
    $PY scripts/save_sft_prepare.py \
        --input  data/hidato5x4/test_natural_policy.jsonl \
        --output "$TEST" || true
fi

echo "[init] toy SFT counts:"
wc -l "$TRAIN" "$VAL" "$TEST"
echo ""

# -----------------------------------------------------------------------------
# Per-size pipeline: download → train → eval → calibrate
# Args: $1=model_id, $2=epochs, $3=batch_sets, $4=grad_ckpt, $5=optim
# -----------------------------------------------------------------------------
run_size() {
    local MODEL="$1" EPOCHS="$2" BATCH_SETS="$3" GRAD_CKPT="${4:-no}" OPTIM="${5:-adamw_torch}"
    # Short identifier
    local SHORT
    SHORT=$(echo "$MODEL" | sed 's|^[^/]*/||' | sed 's|-Instruct$||')
    local OUT="$PROJ/outputs/save_hidato5x4_f_phi_${SHORT}"
    local LOG="$PROJ/logs/hidato_size_${SHORT}.log"
    echo "============================================================"
    echo "  [size $SHORT] $(date)"
    echo "  model: $MODEL  epochs: $EPOCHS  batch_sets: $BATCH_SETS"
    echo "  grad_ckpt: $GRAD_CKPT  optim: $OPTIM"
    echo "  output: $OUT"
    echo "============================================================"
    echo "" > "$LOG"
    echo "started: $(date)" >> "$LOG"

    if [ -f "$OUT/eval_test.json" ]; then
        echo "[skip] $OUT/eval_test.json already exists; skipping size $SHORT"
        echo "" | tee -a "$LOG"
        return 0
    fi

    # 1. download (via mirror)
    echo "" >> "$LOG"; echo "=== [1/4] download via hf-mirror ===" >> "$LOG"
    HF_ENDPOINT=https://hf-mirror.com $PY -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
print('Downloading $MODEL via mirror...')
t = AutoTokenizer.from_pretrained('$MODEL')
m = AutoModelForCausalLM.from_pretrained('$MODEL')
n = sum(p.numel() for p in m.parameters())
print(f'OK params={n/1e6:.1f}M')
" >> "$LOG" 2>&1 || { echo "DOWNLOAD_FAIL for $SHORT" | tee -a "$LOG"; return 1; }

    # 2. SFT
    echo "" >> "$LOG"; echo "=== [2/4] SFT ===" >> "$LOG"
    local GC_FLAG=""
    [ "$GRAD_CKPT" = "yes" ] && GC_FLAG="--gradient_checkpointing"
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $PY -u scripts/sudoku_scripts/save_sft_train.py \
        --model_name_or_path "$MODEL" \
        --train "$TRAIN" \
        --val   "$VAL" \
        --output_dir "$OUT" \
        --num_train_epochs "$EPOCHS" \
        --per_device_batch_sets "$BATCH_SETS" \
        --learning_rate 1e-5 \
        --optim "$OPTIM" \
        $GC_FLAG \
        >> "$LOG" 2>&1 || { echo "SFT_FAIL for $SHORT" | tee -a "$LOG"; return 1; }

    # 3. Eval (Q2 row + Q3 deceptive bench input)
    echo "" >> "$LOG"; echo "=== [3/4] eval on test ===" >> "$LOG"
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $PY -u scripts/sudoku_scripts/save_sft_eval.py \
        --checkpoint "$OUT/final" \
        --eval "$TEST" \
        --output_json "$OUT/eval_test.json" \
        >> "$LOG" 2>&1 || { echo "EVAL_FAIL for $SHORT" | tee -a "$LOG"; }

    # 4. Calibrate (temperature + tau_keep + tau_fb on val)
    echo "" >> "$LOG"; echo "=== [4/4] calibrate on val ===" >> "$LOG"
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $PY -u scripts/sudoku_scripts/save_sft_calibrate.py \
        --checkpoint "$OUT/final" \
        --val "$VAL" \
        --output_json "$OUT/calibration.json" \
        >> "$LOG" 2>&1 || { echo "CAL_FAIL for $SHORT" | tee -a "$LOG"; }

    # Free disk: delete intermediate optimizer state if present
    rm -rf "$OUT/checkpoint-"* 2>/dev/null

    echo "[done $SHORT] $(date)" | tee -a "$LOG"
    df -h /root/autodl-tmp | tail -1
    echo ""
}

# -----------------------------------------------------------------------------
# The sweep — ordered cheapest first to fail-fast cheaply.
# Sudoku autodl2 ran: 0.5B, 1.5B, 3B, 7B + LLaMA-1B, LLaMA-3B.
# We mirror that.
# -----------------------------------------------------------------------------
echo "[init] starting Q1 cross-size sweep at $(date)"

run_size "Qwen/Qwen2.5-0.5B-Instruct"      3 8 no  adamw_torch
run_size "Qwen/Qwen2.5-1.5B-Instruct"      3 8 no  adamw_torch
run_size "meta-llama/Llama-3.2-1B-Instruct" 3 8 no  adamw_torch
run_size "Qwen/Qwen2.5-3B-Instruct"        3 4 yes adamw_torch
run_size "meta-llama/Llama-3.2-3B-Instruct" 3 4 yes adamw_torch
# 7B last (~14GB DL + 14GB output → ~28GB; need disk headroom)
run_size "Qwen/Qwen2.5-7B-Instruct"        3 2 yes paged_adamw_8bit

echo ""
echo "=== ALL Q1 SIZES DONE — $(date) ==="
ls -la "$PROJ"/outputs/save_hidato5x4_f_phi_*/eval_test.json 2>&1
