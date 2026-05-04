# Run Data Layout & Plotting

What's saved per run, where, and how to use it for plotting / paper figures.

## Per-run-type data layout

### RL runs (`rl_trainer_v6.py`)

Output directory `outputs/<run_name>/`:

| File | Contents | When written |
|---|---|---|
| `rl_log.jsonl` | One JSON per line: `{step, phase, ...}`. `phase="eval"` rows have `pass@1, solvable_acc, bp_recall, n_solved`. Other rows have per-step `reward, kl, pg_loss, clipfrac, via_kl, step_t`, etc. | Every training step + every eval |
| `final/` | Final saved policy (model.safetensors, tokenizer.json, etc.) | Once at end of training |
| `checkpoint-N/` | Saved every `--save-every` steps (default 100) | Periodic |

Plus the stdout/stderr in `logs/<run_name>.log`.

### SFT runs (`simple_sft_trainer.py` — uses HuggingFace `Trainer`)

Output directory `outputs/<run_name>/`:

| File / dir | Contents | When written |
|---|---|---|
| `runs/<timestamp>/events.out.tfevents...` | TensorBoard events (train_loss, eval_loss, learning_rate, grad_norm per step) | Every logging step |
| `final/` | Final saved policy | Once at end |
| `checkpoint-N/` | Periodic checkpoints (controlled by `--save_steps`) | Periodic |

Plus stdout in `logs/<run_name>.log`.

### Eval scripts

| Script | Output |
|---|---|
| `quick_pass1` (inside trainer) | Aggregate Pass@1 row in `rl_log.jsonl` |
| `sanity_check_checkpoint.py` | Stdout report (greedy Pass@1, stochastic Pass@k, full-XML rate, viability dist) — captured to `logs/*.log` |
| `evaluate_rl.py` | Stdout report — captured to `logs/*.log` |

## Unified extractor

[scripts/extract_run_data.py](../../scripts/extract_run_data.py) reads either `rl_log.jsonl` or TB events and writes a normalized `extracted_metrics.jsonl` to the same dir. Use this before plotting.

```bash
# Single run
python scripts/extract_run_data.py outputs/rl_b5_phase3_v8_anchor

# All runs
python scripts/extract_run_data.py outputs/*
```

Each `extracted_metrics.jsonl` row has `{step, ...metrics}`. Trivial to load with `pandas.read_json(..., lines=True)`.

## Standard plots

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_json("outputs/rl_b_h1_v8_anchor/rl_log.jsonl", lines=True)

# Pass@1 trajectory
eval_rows = df[df.phase == "eval"]
plt.plot(eval_rows.step, eval_rows["pass@1"], marker="o")
plt.xlabel("RL step"); plt.ylabel("Pass@1 greedy")

# Reward curve (training-time)
train_rows = df[df.phase != "eval"]
plt.plot(train_rows.step, train_rows.reward)

# KL drift
plt.plot(train_rows.step, train_rows.kl)
```

For SFT TB events:

```python
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator("outputs/sft_hidato_b_h1/runs/<ts>/events.out.tfevents...")
ea.Reload()
eval_loss = [(e.step, e.value) for e in ea.Scalars("eval/loss")]
```

## Gaps (not currently saved)

These could be added if needed:

| Missing | Why useful | Where to add |
|---|---|---|
| Per-puzzle eval results (puzzle_seed → solved/failed) | "Which puzzles does the model never solve?" analysis | `quick_pass1()` could write `eval_per_puzzle.jsonl` |
| Sample model responses at eval | Qualitative calibration analysis | Optional `--dump-eval-responses` flag |
| Token-level logprobs at viability positions | Bimodal-collapse quantification | Recomputable post-hoc via `evaluate_rl.py --metric solvable-logprob` |

Add these only when a specific plot/analysis needs them — they're recomputable from the final checkpoint.

## Cross-run comparison

To compare two runs side-by-side:

```python
import pandas as pd
import matplotlib.pyplot as plt

runs = {
    "v8 anchor": "outputs/rl_b5_phase3_v8_anchor/rl_log.jsonl",
    "v8.2 dual": "outputs/rl_b5_phase3_v8_2_dual_anchor/rl_log.jsonl",
}
fig, ax = plt.subplots()
for label, path in runs.items():
    df = pd.read_json(path, lines=True)
    e = df[df.phase == "eval"]
    ax.plot(e.step, e["pass@1"], label=label, marker="o")
ax.set_xlabel("RL step"); ax.set_ylabel("Pass@1 greedy"); ax.legend()
fig.savefig("doc/plots/v8_vs_v8_2_pass1.png", dpi=150)
```

[scripts/generate_paper_plots.py](../../scripts/generate_paper_plots.py) shows a more elaborate version with our project's styling.

## Existing data locations (current state, 2026-05-03)

Current preserved checkpoints / logs across the three clouds:

- **autodl1**: `outputs/sft_hidato_b_h1/`, `outputs/rl_b_h1_v8_anchor/`, `outputs/sft_pentomino_b8_augmented/`, `outputs/rl_b8_v8_anchor/`, etc.
- **autodl2**: `outputs/sft_sudoku_4x4_*`, `outputs/rl_b5_phase{1,2,3}_*`, `outputs/baseline_*` (Sudoku SPA-table comparisons), plus `_archived_jsonls/` for completed experiments where weights were deleted.
- **autodl3**: `outputs/sft_pentomino_5x10_no_leak/`, `outputs/sft_pentomino_5x4_no_leak/`, `outputs/rl_pentomino_5x4_no_leak_v8_aq/` (in flight).

To pull all the structured logs to local for plotting:

```bash
bash scripts/sync-down.sh                # default: data/, logs/, outputs/ (no model weights)
```

This already happens routinely. The metrics for every completed run are durable on local disk after sync-down.
