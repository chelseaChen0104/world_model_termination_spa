#!/bin/bash
# Combined data + checkpoint sanity pipeline. Run before launching SFT (data check)
# and before launching RL (checkpoint check). Exits non-zero on any failure under
# --strict, so it can be wired into other launchers.
#
# Usage:
#   bash scripts/check_run_health.sh \
#       --env hidato \
#       --data-dir data/hidato_combined_no_leak \
#       --sft-path outputs/sft_hidato_no_leak/final \
#       [--strict]
#
# Exit codes: 0 if all checks pass, 1 otherwise (with --strict).

set -e
cd "$(dirname "$0")/.."

PY=${PY:-/root/miniconda3/bin/python}
[ -x "$PY" ] || PY=python   # local fallback
ENV_NAME=""
DATA_DIR=""
SFT_PATH=""
STRICT=""
SKIP_DATA=""
SKIP_CHECKPOINT=""
EXTRA_DATA_ARGS=""
EXTRA_CHECKPOINT_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env) ENV_NAME="$2"; shift 2 ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --sft-path) SFT_PATH="$2"; shift 2 ;;
        --strict) STRICT="--strict"; shift ;;
        --skip-data) SKIP_DATA=1; shift ;;
        --skip-checkpoint) SKIP_CHECKPOINT=1; shift ;;
        --grid-size|--difficulty|--board-h|--board-w|--piece-set|--max-new-tokens|--max-steps|--n-puzzles|--k|--temperature)
            EXTRA_CHECKPOINT_ARGS="$EXTRA_CHECKPOINT_ARGS $1 $2"; shift 2 ;;
        --prepend-current-state|--reset-history-per-step)
            EXTRA_CHECKPOINT_ARGS="$EXTRA_CHECKPOINT_ARGS $1"; shift ;;
        *) echo "[check_run_health] unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [ -z "$ENV_NAME" ]; then
    echo "[check_run_health] --env is required (sudoku|polyomino|hidato)" >&2
    exit 2
fi

EXIT=0

# ── Data sanity ──
if [ -z "$SKIP_DATA" ] && [ -n "$DATA_DIR" ]; then
    echo "============================================================"
    echo "  DATA SANITY: $DATA_DIR (env=$ENV_NAME)"
    echo "============================================================"
    "$PY" scripts/sanity_check_dataset.py \
        --input "$DATA_DIR" --env "$ENV_NAME" $STRICT || EXIT=$?
    echo
fi

# ── Checkpoint sanity ──
if [ -z "$SKIP_CHECKPOINT" ] && [ -n "$SFT_PATH" ]; then
    echo "============================================================"
    echo "  CHECKPOINT SANITY: $SFT_PATH (env=$ENV_NAME)"
    echo "============================================================"
    "$PY" scripts/sanity_check_checkpoint.py \
        --sft-path "$SFT_PATH" --env "$ENV_NAME" \
        --prepend-current-state --reset-history-per-step \
        --max-new-tokens 512 \
        $EXTRA_CHECKPOINT_ARGS $STRICT || EXIT=$?
fi

if [ "$EXIT" -ne 0 ]; then
    echo
    echo "=== ⚠️ check_run_health: FAILED (exit $EXIT) ==="
    exit "$EXIT"
fi
echo
echo "=== ✅ check_run_health: ALL CHECKS PASSED ==="
