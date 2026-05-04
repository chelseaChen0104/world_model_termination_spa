#!/bin/bash
# Chained Sudoku-baseline runner for autodl2.
#
# Runs the three SPA-Table-5 baselines back-to-back so they don't need manual
# kicking. Each script handles its own data prep, SFT, and RL. Order:
#   1. vanilla RL (no SFT, ~3-4 hr)
#   2. SE-only baseline (SFT + RL, ~5-6 hr)
#   3. SPA-full baseline (SFT + RL, ~5-6 hr)
#
# Total wall: ~14-16 hr. Each step's success is independent — if one fails,
# the rest still run, and you can re-launch the failed one separately.

set -e
cd /root/autodl-tmp/world_model_termination_spa

mkdir -p logs
CHAIN_LOG=logs/sudoku_baselines_chain.log
: > "$CHAIN_LOG"

echo "============================================================" | tee -a "$CHAIN_LOG"
echo "  Sudoku baseline chain (vanilla → SE-only → SPA-full)" | tee -a "$CHAIN_LOG"
echo "  Started at $(date)" | tee -a "$CHAIN_LOG"
echo "============================================================" | tee -a "$CHAIN_LOG"

run_step() {
    local name=$1
    local script=$2
    echo | tee -a "$CHAIN_LOG"
    echo "------ $name | $(date) ------" | tee -a "$CHAIN_LOG"
    if bash "$script"; then
        echo "[OK] $name done at $(date)" | tee -a "$CHAIN_LOG"
    else
        echo "[FAIL] $name failed at $(date), continuing chain" | tee -a "$CHAIN_LOG"
    fi
}

run_step "vanilla RL"     scripts/run_baseline_vanilla_rl.sh
run_step "SE-only SFT+RL" scripts/run_baseline_se_only.sh
run_step "SPA-full SFT+RL" scripts/run_baseline_spa_full.sh

echo | tee -a "$CHAIN_LOG"
echo "============================================================" | tee -a "$CHAIN_LOG"
echo "  All baselines done at $(date)" | tee -a "$CHAIN_LOG"
echo "============================================================" | tee -a "$CHAIN_LOG"
