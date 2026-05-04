#!/bin/bash
# Deployment-time min-step Pareto sweep on the gate-OFF-trained checkpoint.
#
# Goal: measure (Pass@1, savings) tradeoff at INFERENCE time, parametric in
# truncation_min_step. Closes the missing 4th cell of the deployment 2×2:
#   "OFF-trained checkpoint, eval w/ gate ON".
#
# Starting checkpoint: outputs/trunc_exp_b_off/final/  (cleanly RL'd Sudoku 4×4
# Pass@1 = 53.3% with gate off; this is our cleanest deployable model).
#
# Conditions:
#   - off       : gate disabled (sanity baseline; should reproduce 53.3%)
#   - ms0       : gate ON, τ=0.99, min_step=0  (most aggressive savings)
#   - ms2       : gate ON, τ=0.99, min_step=2
#   - ms3       : gate ON, τ=0.99, min_step=3
#   - ms4       : gate ON, τ=0.99, min_step=4  (gate barely fires)
#
# Each run is `--n-total-steps 1 --eval-every 1` so it only does the initial
# Pass@1 eval before exiting. ~2-3 min per condition × 5 = ~15 min total.
#
# Disk hygiene: deletes the saved checkpoint after each run, keeping only the
# rl_log.jsonl (which contains the eval result).

set -e
cd /root/autodl-tmp/world_model_termination_spa

SFT_PATH=outputs/trunc_exp_b_off/final
TAU=${TAU:-0.99}
SEED=${SEED:-42}

mkdir -p logs outputs/eval_deploy_jsonls
LOG=logs/eval_deploy_min_step_sweep.log
: > "$LOG"

echo "============================================================" | tee -a "$LOG"
echo "  Deployment-time min-step Pareto sweep" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
echo "  SFT checkpoint:  $SFT_PATH" | tee -a "$LOG"
echo "  τ:               $TAU" | tee -a "$LOG"
echo "  Conditions:      off, ms0, ms2, ms3, ms4" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

run_eval() {
    local label=$1
    local mode=$2
    local min_step=$3
    local outdir=outputs/eval_deploy_${label}
    local extra_flags=""
    if [ "$mode" = "off" ]; then
        extra_flags="--truncation-mode off"
    else
        extra_flags="--truncation-mode conservative --truncation-threshold $TAU --truncation-min-step $min_step"
    fi
    echo "" | tee -a "$LOG"
    echo "------ Running: $label (mode=$mode min_step=$min_step) ------" | tee -a "$LOG"
    rm -rf "$outdir"
    bash scripts/_run_with_env.sh python -u src/training/rl_trainer_v6.py \
        --env sudoku --grid-size 4 --difficulty easy \
        --sft-checkpoint "$SFT_PATH" \
        --output-dir "$outdir" \
        --n-total-steps 1 --eval-every 1 \
        --reward-version v8 --viability-kl-coef 0.5 --seed "$SEED" \
        $extra_flags 2>&1 | tee -a "$LOG" || true
    # Preserve jsonl, delete checkpoint to conserve disk
    if [ -f "$outdir/rl_log.jsonl" ]; then
        cp "$outdir/rl_log.jsonl" "outputs/eval_deploy_jsonls/${label}_rl_log.jsonl"
    fi
    rm -rf "$outdir"
}

run_eval "off"  off  0
run_eval "ms0"  on   0
run_eval "ms2"  on   2
run_eval "ms3"  on   3
run_eval "ms4"  on   4

echo "" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
echo "  Sweep complete — per-condition JSONLs:" | tee -a "$LOG"
ls -1 outputs/eval_deploy_jsonls/ | tee -a "$LOG"
echo "  Log: $LOG" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
