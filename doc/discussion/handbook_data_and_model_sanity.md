# Data & Model Sanity Pipeline

A reference for `scripts/sanity_check_dataset.py`, `scripts/sanity_check_checkpoint.py`, and the orchestrator `scripts/check_run_health.sh`. Use this before any new SFT or RL run to catch the recurring bugs we've hit on this project.

## Why this exists

Over the course of the project we silently shipped multiple data and model bugs that wasted hours of compute:

| Bug | Where caught | Cost |
|---|---|---|
| Doom-suffix leak in Pentomino + Hidato data (`— board now unsolvable (REASON)`) | After Pentomino RL Pass@1=0% for two consecutive runs | ~6 hr GPU + days of misdiagnosis |
| Multi-turn-history collapse in `do_rollout` (model emits bare `<answer>` at turn 1+) | After B-H1 reported Pass@1=0% with AUC=1.000 | ~2 hr debug + 1 day of confusion |
| `max_response_tokens=256` truncating Hidato's long XML | After Hidato Pass@1 stayed at 0% even with format fix | ~1 hr eval re-runs |
| Hidato data has 98.5% duplicate rate (only ~183 unique samples) | Only at this pipeline's introduction | Found via duplicate threshold |
| `<viability>` tag↔label inconsistency from text-shortcut leak | Sanity check on actual model output | Days of "regime 1" misframing |

The pipeline below would have caught every one of these at launch time.

## Pipeline structure

```
   ┌─────────────────────────────────────────────────────────────┐
   │ scripts/check_run_health.sh  (orchestrator)                 │
   ├──────────────────────────────────────┬──────────────────────┤
   │ scripts/sanity_check_dataset.py      │ scripts/sanity_check_checkpoint.py │
   │   (data, parquet input)              │   (checkpoint, runs greedy + stochastic) │
   └──────────────────────────────────────┴──────────────────────┘
```

## When to run it

| Lifecycle stage | What to check | Command |
|---|---|---|
| **Before SFT** | data-only | `bash scripts/check_run_health.sh --env hidato --data-dir data/hidato_combined_no_leak --skip-checkpoint --strict` |
| **After SFT, before RL** | data + checkpoint | full command (see below) |
| **After RL** (sanity post-hoc) | checkpoint-only | `bash scripts/check_run_health.sh --env hidato --sft-path outputs/rl_hidato_v8_anchor/final --skip-data` |
| **After regenerating any data** | data-only | re-run dataset check |

Strict mode (`--strict`) makes the script exit non-zero on any failure — wire it into automation/CI.

## What gets checked

### Data (`sanity_check_dataset.py`)

| Check | Threshold | Catches |
|---|---|---|
| Schema | columns present, no nulls | corrupt parquet |
| Tag presence (`<observation>`, `<prediction>`/`<next_state>`, `<viability>`/`<solvable>`, `<answer>`) | 100% | broken formatter |
| Action format parses | ≥99% | wrong env action regex |
| `<viability>` ↔ `extra_info.is_solvable` consistency | ≥99% | label/tag drift |
| **Doom-suffix leak** (`— board now unsolvable`) | 0% | env-render leak (the Pentomino/Hidato bug) |
| **Duplicate rate** (exact prompt+response pairs) | ≤50% | low diversity (the Hidato 98.5% bug) |
| **Min unique samples** | ≥200 | too-small training set |
| **Smallest class fraction** | ≥10% | severe class imbalance |
| Class balance, step distribution, length stats | reported, no threshold | informational |

### Checkpoint (`sanity_check_checkpoint.py`)

Loads the SFT/RL checkpoint and actually runs rollouts. Reports:

| Section | What it does |
|---|---|
| **A. Verbose greedy rollout on puzzle 0** | Prints per-step prompt, response, parsed action, env outcome — see exactly what the model emits |
| **B. Greedy Pass@1 on N puzzles** | Aggregated greedy solve rate + full-XML rate + token-budget hit rate |
| **C. Stochastic Pass@k on N puzzles** | Stochastic per-batch + Pass@k — tells you whether RL has positive signal to bootstrap from |
| **D. Health checks** | Pass/fail summary against thresholds |

Health-check thresholds:

| Check | Threshold | Catches |
|---|---|---|
| Greedy emits FULL XML at every step | ≥95% of steps | multi-turn-history bare-answer collapse |
| Token budget not hit | <5% of steps truncated | `max_response_tokens` too low |
| Both viability classes appear | both `true` and `false` predicted at least once | bimodal collapse / always-doom regime |
| Stochastic per-batch solve > 0 | strictly positive | SFT has no positive signal — RL won't bootstrap |
| Greedy Pass@1 > 0 | strictly positive | greedy collapse despite positive stochastic signal — fixable via v8.2 / action_quality_bonus |

## Common failures and fixes

### `Greedy rollout produces FULL XML at every step` ❌

**Symptom**: at turn 1+ the model emits just `<answer>...</answer>` with no `<think>...`.

**Cause**: SFT was trained on per-step single-turn samples but the eval/rollout pipeline builds multi-turn chat history. The model never saw "previous-assistant-message + new-user-message" prompts.

**Fix**: pass `--prepend-current-state --reset-history-per-step` to the trainer/eval. Reset history per step matches single-turn SFT distribution. (See [eval_2026-05-01_truncation_full.md](../eval_2026-05-01_truncation_full.md) for the full debug trail that found this.)

### `Token budget not hit` ❌

**Symptom**: response length tightly clusters at `max_new_tokens-1`, often around the response cutoff. Often co-occurs with broken XML (response cut off mid-tag).

**Cause**: trainer's `cfg.max_response_tokens` (default 256) too low for verbose envs.

**Fix**: pass `--max-response-tokens 512` (or 1024 if responses are even longer).

### `Both viability classes appear` ❌

**Symptom**: greedy decode emits `<viability>=false` (or `=true`) on 100% of evaluations, regardless of board state.

**Cause**: bimodal calibration collapse. Model has two attractors (always-true, always-false); greedy picks whichever has marginally more logp mass.

**Fix attempts in order of cost**:
1. Strip data-side text shortcuts that prime the prediction (e.g., the doom-suffix leak)
2. Try v8.2 dual-token KL anchor during RL
3. Add `--action-quality-bonus 0.5` during RL for direct action-quality gradient
4. Larger model

### `Stochastic per-batch solve > 0` ❌

**Symptom**: Pass@k (k=4 or 8) is also 0% — model literally never solves any puzzle, even with sampling diversity.

**Cause**: SFT didn't learn the action policy at all. Could be too few unique training samples, too short SFT training, or the wrong reward shaping during a prior RL stage that destroyed the policy.

**Fix**: don't proceed to RL — fix SFT first (more data, more epochs, augmentation, stronger base model).

### `Doom-suffix leak == 0%` ❌

**Symptom**: training data contains `— board now unsolvable (REASON)` text in `<next_state>` of doom samples.

**Cause**: env's `render()` appends doom annotations on terminal states, captured into the SFT response.

**Fix**: run `scripts/strip_doom_suffix.py --input <leaked_dir> --output <clean_dir>`. Or patch `<env>.render()` at source.

### `Duplicate rate ≤ 50%` ❌

**Symptom**: dataset has many exact (prompt, response) duplicates.

**Cause**: small puzzle bank + repeated state recurrence + augmentation oversample. The Hidato dataset hit 98.5% from an 8-puzzle bank × 30× oversample.

**Fix**: expand puzzle bank diversity. For Hidato, generate more puzzles (vary grid sizes, given patterns). For Pentomino, vary piece subsets.

## Concrete invocation examples

### Sudoku — the clean baseline

```bash
bash scripts/check_run_health.sh \
    --env sudoku \
    --data-dir data/sudoku_4x4_llm_policy_minimal_spa_scale \
    --sft-path outputs/sft_sudoku_4x4/final \
    --grid-size 4 --difficulty easy
```

Expected: all data checks pass; checkpoint shows full-XML, both viability classes, Pass@1 in 30-50% range.

### Pentomino 5×10 (the no-leak run)

```bash
bash scripts/check_run_health.sh \
    --env polyomino \
    --data-dir data/pentomino_5x10_combined_no_leak \
    --sft-path outputs/sft_pentomino_5x10_no_leak/final \
    --board-h 5 --board-w 10 \
    --piece-set F,I,L,N,P,T,U,V,Y,Z
```

### Hidato

```bash
bash scripts/check_run_health.sh \
    --env hidato \
    --data-dir data/hidato_combined_no_leak \
    --sft-path outputs/sft_hidato_no_leak/final
```

### CI / automated mode

```bash
bash scripts/check_run_health.sh ... --strict || exit 1
```

Exits with code 1 on any failure. Wire into your launcher scripts before the long training step:

```bash
bash scripts/check_run_health.sh --env $ENV --data-dir $DATA --skip-checkpoint --strict
bash scripts/run_sft.sh                         # only runs if data check passed
bash scripts/check_run_health.sh --env $ENV --sft-path $SFT --skip-data --strict
bash scripts/run_rl.sh                          # only runs if checkpoint is healthy
```

## Maintenance

When you discover a new bug class, **add a check here** so we never debug the same thing twice. Keep thresholds conservative; adjust in code rather than via flags so they're version-controlled.

## File index

- [scripts/sanity_check_dataset.py](../../scripts/sanity_check_dataset.py)
- [scripts/sanity_check_checkpoint.py](../../scripts/sanity_check_checkpoint.py)
- [scripts/check_run_health.sh](../../scripts/check_run_health.sh)
- [scripts/strip_doom_suffix.py](../../scripts/strip_doom_suffix.py) — the actual fix for the env-render leak
- [scripts/debug_hidato_one_rollout.py](../../scripts/debug_hidato_one_rollout.py) — older single-rollout debug, superseded by `sanity_check_checkpoint.py` Section A
- [scripts/debug_polyomino_one_rollout.py](../../scripts/debug_polyomino_one_rollout.py) — same for Polyomino
