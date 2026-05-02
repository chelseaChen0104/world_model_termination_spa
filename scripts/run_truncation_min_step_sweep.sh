#!/bin/bash
# Trajectory-position-aware truncation experiment.
#
# Tests whether delaying the truncation gate (only firing at step ≥ N) reduces
# the Pass@1 cost while keeping compute savings. Hypothesis: rollouts that
# the model says are doomed at step 1 are sometimes recoverable; truncating
# only at step ≥ 3 lets those rollouts run to natural success or doom.
#
# Sweeps min_step ∈ {0, 1, 2, 3, 4} at fixed τ=0.99, 10 rollout steps each.

set -e
cd /root/autodl-tmp/world_model_termination_spa

SFT_PATH=${SFT_PATH:-outputs/rl_b5_phase3_v8_anchor/final}
N_STEPS=${N_STEPS:-10}
TAU=${TAU:-0.99}
SEED=${SEED:-42}

mkdir -p logs

for min_step in 0 1 2 3 4; do
    OUT_DIR="outputs/min_step_sweep_step${min_step}"
    LOG="logs/min_step_sweep_${min_step}.log"
    rm -rf "$OUT_DIR"
    > "$LOG"
    echo "============================================================"
    echo "  τ=${TAU}, min_step=${min_step}  →  ${OUT_DIR}"
    echo "============================================================"
    bash scripts/_run_with_env.sh /root/miniconda3/bin/python -u src/training/rl_trainer_v6.py \
        --env sudoku --grid-size 4 --difficulty easy \
        --sft-checkpoint "$SFT_PATH" \
        --output-dir "$OUT_DIR" \
        --n-total-steps "$N_STEPS" \
        --n-puzzles-per-batch 4 --group-size 8 \
        --learning-rate 1e-5 --kl-coef 0.05 \
        --eval-every 9999 \
        --reward-version v8 --viability-kl-coef 0.5 \
        --seed "$SEED" \
        --truncation-mode conservative \
        --truncation-threshold "$TAU" \
        --truncation-min-step "$min_step" \
        2>&1 | tee "$LOG"
done

# Print comparison table
echo "============================================================"
echo "  MIN-STEP SWEEP COMPARISON (τ=$TAU)"
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
    if n == 0:
        print(f'{label:>22s}: NO DATA')
        return
    rt = sum(r['rollout_time_s'] for r in rows) / n
    ttok = sum(r.get('total_response_tokens', 0) for r in rows) / n
    mlen = sum(r.get('mean_rollout_len', 0) for r in rows) / n
    ntr = sum(r.get('n_truncated_early', 0) for r in rows)
    nro = sum(r.get('n_rollouts', 0) for r in rows)
    solv = sum(r.get('solved_rate', 0) for r in rows) / n
    print(f'{label:>22s}: rollout_time={rt:.2f}s, tokens/step={ttok:6.0f}, mean_len={mlen:.2f}, '
          f'truncated={ntr}/{nro} ({100*ntr/max(1,nro):4.1f}%), per-batch_solve={100*solv:.1f}%')

for ms in [0, 1, 2, 3, 4]:
    summary(f'outputs/min_step_sweep_step{ms}/rl_log.jsonl', f'min_step={ms} τ=$TAU')
"
echo "============================================================"
