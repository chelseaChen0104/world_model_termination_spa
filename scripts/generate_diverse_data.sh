#!/bin/bash
# Generate diverse LLM-policy multi-difficulty training data.
#
# Runs 3 sequential data-gen jobs (easy / medium / hard) for 9x9 Sudoku
# using minimal data-gen prompt + Qwen2.5-1.5B-Instruct.
#
# Total ~4,500 trajectories → expected ~70k-90k single-step samples after
# reformat (vs current ~16k).
#
# Wall time on H800: ~4-5 hours.
#
# Usage:
#   bash scripts/generate_diverse_data.sh
#   (run from repo root, requires _run_with_env.sh to be on disk)

set -e

N_EASY=${N_EASY:-1500}
N_MEDIUM=${N_MEDIUM:-1500}
N_HARD=${N_HARD:-1500}

echo "============================================================"
echo "  Diverse LLM-Policy Multi-Difficulty Data Generation"
echo "============================================================"
echo "  Easy:   $N_EASY trajectories  → data/sudoku_llm_policy_easy/"
echo "  Medium: $N_MEDIUM trajectories → data/sudoku_llm_policy_medium/"
echo "  Hard:   $N_HARD trajectories  → data/sudoku_llm_policy_hard/"
echo "============================================================"

run_one() {
    local difficulty=$1
    local n=$2
    local out_dir=$3
    local seed=$4
    echo
    echo "============================================================"
    echo "  STARTING: difficulty=$difficulty n=$n out=$out_dir"
    echo "============================================================"
    python -u -m src.data.llm_trajectory_generator \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --num-trajectories "$n" \
        --max-steps 30 \
        --grid-size 9 \
        --difficulty "$difficulty" \
        --output-dir "$out_dir" \
        --seed "$seed" \
        --temperature 0.7 \
        --val-split 0.2 \
        --device auto \
        --multi-turn --max-context-turns 10
}

run_one easy   "$N_EASY"   data/sudoku_llm_policy_easy   42
run_one medium "$N_MEDIUM" data/sudoku_llm_policy_medium 142
run_one hard   "$N_HARD"   data/sudoku_llm_policy_hard   242

echo
echo "============================================================"
echo "  ALL DIFFICULTY RUNS COMPLETE"
echo "============================================================"
echo "Next: bash scripts/combine_diverse_to_minimal.sh"
