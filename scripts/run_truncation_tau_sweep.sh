#!/bin/bash
# Truncation τ-sweep on v8 anchor checkpoint.
#
# For each τ ∈ {0.95, 0.99, 0.999, 0.9999}, run 10 rollout-collection steps
# from the same v8 anchor checkpoint (no training updates beyond what
# rl_trainer_v6.py does). Captures:
#   - % rollouts truncated
#   - Mean rollout length
#   - Total tokens / step
#   - Pass@1 (eval-time, with truncation)
#   - Pass@1 (eval-time, without truncation — clean re-eval)
#
# This characterizes the (compute savings, Pass@1 cost) Pareto frontier as a
# function of τ on the GATE behavior alone (training-time effects already
# measured in Option B).
#
# Wall time on H800: ~12 min per τ × 5 τ values = ~60 min total.

set -e
cd /root/autodl-tmp/world_model_termination_spa

SFT_PATH=${SFT_PATH:-outputs/rl_b5_phase3_v8_anchor/final}
N_STEPS=${N_STEPS:-10}
SEED=${SEED:-42}

mkdir -p logs

# Sweep: include OFF baseline + 4 τ values
declare -A TAU_RUNS=(
    [off]="off"
    [0.95]="0.95"
    [0.99]="0.99"
    [0.999]="0.999"
    [0.9999]="0.9999"
)

for tag in off 0.95 0.99 0.999 0.9999; do
    if [ "$tag" = "off" ]; then
        OUT_DIR="outputs/tau_sweep_off"
        TRUNC_MODE="off"
        TRUNC_THRESH="0.99"
    else
        OUT_DIR="outputs/tau_sweep_tau${tag}"
        TRUNC_MODE="conservative"
        TRUNC_THRESH="$tag"
    fi
    rm -rf "$OUT_DIR"
    LOG="logs/tau_sweep_${tag}.log"
    > "$LOG"

    echo "============================================================"
    echo "  τ=${tag}  →  ${OUT_DIR}"
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
        --truncation-mode "$TRUNC_MODE" \
        --truncation-threshold "$TRUNC_THRESH" \
        2>&1 | tee "$LOG"
done

# Print comparison table
echo "============================================================"
echo "  τ-SWEEP COMPARISON"
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
        print(f'{label:>14s}: NO DATA')
        return
    rt = sum(r['rollout_time_s'] for r in rows) / n
    ttok = sum(r.get('total_response_tokens', 0) for r in rows) / n
    mlen = sum(r.get('mean_rollout_len', 0) for r in rows) / n
    ntr = sum(r.get('n_truncated_early', 0) for r in rows)
    nro = sum(r.get('n_rollouts', 0) for r in rows)
    solv = sum(r.get('solved_rate', 0) for r in rows) / n
    print(f'{label:>14s}: rollout_time={rt:.2f}s, tokens/step={ttok:6.0f}, mean_len={mlen:.2f}, '
          f'truncated={ntr}/{nro} ({100*ntr/max(1,nro):4.1f}%), per-batch_solve={100*solv:.1f}%')

for tag in ['off', '0.95', '0.99', '0.999', '0.9999']:
    if tag == 'off':
        summary(f'outputs/tau_sweep_off/rl_log.jsonl', 'OFF')
    else:
        summary(f'outputs/tau_sweep_tau{tag}/rl_log.jsonl', f'τ={tag}')
"
echo "============================================================"
