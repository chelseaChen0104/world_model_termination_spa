#!/bin/bash
# Option A: rollout-collection comparison for the Phase 2 truncation experiment.
# Runs the SAME training step sequence (same seed) twice on the v8 anchor checkpoint:
#   - 10 RL steps with truncation_mode=off
#   - 10 RL steps with truncation_mode=conservative (τ=0.99)
# Compare per-step rollout_time_s, n_truncated_early, total_response_tokens,
# mean_rollout_len from the resulting rl_log.jsonl files.
#
# τ=0.99 picked from threshold sweep (2026-05-01):
#   on Sudoku v8 anchor, P(false)|GT=False ≈ 1.0 (truncate correctly),
#   P(false)|GT=True mean 0.959 (mostly NOT truncate). See doc.
#
# Wall time on H800: ~10 min per condition × 2 = ~20 min total.

set -e
cd /root/autodl-tmp/world_model_termination_spa

SFT_PATH=${SFT_PATH:-outputs/rl_b5_phase3_v8_anchor/final}
N_STEPS=${N_STEPS:-10}
TAU=${TAU:-0.99}
SEED=${SEED:-42}

OFF_DIR=outputs/trunc_exp_off
ON_DIR=outputs/trunc_exp_on_tau${TAU}

mkdir -p logs

LOG_OFF=logs/trunc_exp_off.log
LOG_ON=logs/trunc_exp_on.log

echo "============================================================"
echo "  Phase 2 Truncation Experiment — Option A (rollout comparison)"
echo "============================================================"
echo "  Source checkpoint: $SFT_PATH"
echo "  Steps per condition: $N_STEPS"
echo "  τ_truncation: $TAU"
echo "  Seed: $SEED"
echo "  OFF output: $OFF_DIR"
echo "  ON  output: $ON_DIR"
echo "============================================================"

echo
echo "=== Run 1: truncation OFF ==="
> "$LOG_OFF"
bash scripts/_run_with_env.sh /root/miniconda3/bin/python -u src/training/rl_trainer_v6.py \
    --env sudoku --grid-size 4 --difficulty easy \
    --sft-checkpoint "$SFT_PATH" \
    --output-dir "$OFF_DIR" \
    --n-total-steps "$N_STEPS" \
    --n-puzzles-per-batch 4 --group-size 8 \
    --learning-rate 1e-5 --kl-coef 0.05 \
    --eval-every 9999 \
    --reward-version v8 --viability-kl-coef 0.5 \
    --seed "$SEED" \
    --truncation-mode off \
    2>&1 | tee "$LOG_OFF"

echo
echo "=== Run 2: truncation ON (τ=$TAU) ==="
> "$LOG_ON"
bash scripts/_run_with_env.sh /root/miniconda3/bin/python -u src/training/rl_trainer_v6.py \
    --env sudoku --grid-size 4 --difficulty easy \
    --sft-checkpoint "$SFT_PATH" \
    --output-dir "$ON_DIR" \
    --n-total-steps "$N_STEPS" \
    --n-puzzles-per-batch 4 --group-size 8 \
    --learning-rate 1e-5 --kl-coef 0.05 \
    --eval-every 9999 \
    --reward-version v8 --viability-kl-coef 0.5 \
    --seed "$SEED" \
    --truncation-mode conservative \
    --truncation-threshold "$TAU" \
    2>&1 | tee "$LOG_ON"

echo
echo "============================================================"
echo "  Both runs done. Compare metrics:"
echo "============================================================"
/root/miniconda3/bin/python -c "
import json
def summary(path, label):
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get('phase') in ('eval', 'init_eval'): continue
            if 'rollout_time_s' not in r: continue
            rows.append(r)
    n = len(rows)
    rt = sum(r['rollout_time_s'] for r in rows) / max(1, n)
    ttok = sum(r.get('total_response_tokens', 0) for r in rows) / max(1, n)
    mlen = sum(r.get('mean_rollout_len', 0) for r in rows) / max(1, n)
    ntr = sum(r.get('n_truncated_early', 0) for r in rows)
    nro = sum(r.get('n_rollouts', 0) for r in rows)
    print(f'{label:>14s}: n_steps={n}, mean_rollout_time={rt:.2f}s, '
          f'mean_tokens/step={ttok:.0f}, mean_len={mlen:.2f}, '
          f'truncated={ntr}/{nro} ({100*ntr/max(1,nro):.1f}%)')

summary('$OFF_DIR/rl_log.jsonl', 'truncation OFF')
summary('$ON_DIR/rl_log.jsonl',  'truncation ON')
"

echo "============================================================"
