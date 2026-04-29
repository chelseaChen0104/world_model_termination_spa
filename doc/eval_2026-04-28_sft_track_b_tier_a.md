# Eval — SFT Track B, Tier A (termination metrics, single-turn) — 2026-04-28

First evaluation of `outputs/sft_sudoku_llm_policy/final` (Track B SFT — LLM-policy multi-turn data, filtered to ≤4000 tokens).

This is **Tier A** of the planned A/B/C eval cascade ([pipeline_design.md](pipeline_design.md) §5). Tier B + C are blocked pending the issue identified below.

## Setup

| | |
|---|---|
| Model | `outputs/sft_sudoku_llm_policy/final` (Qwen2.5-1.5B-Instruct + 3-epoch SFT, ~1.5h training) |
| Training data | `data/sudoku_llm_policy/wm_train_filtered.parquet` (6,221 samples, multi-turn) |
| Final train loss | 0.0172 |
| Final eval_loss | 0.01359 (no overfitting) |
| Eval set | 200 samples freshly generated from live `SudokuEnv` via random-play TrajectoryGenerator |
| Eval set distribution | 100 solvable, 100 unsolvable, **5 breaking points** |
| Decoding | greedy (`do_sample=False, temperature=0.1`), `max_new_tokens=512` |
| Prompt format | **single-turn**: `[system, user_state]` only |

## Results

### Format compliance (good)

| Tag | Present |
|---|---|
| Valid format (any tag parsed) | **100.0%** |
| `<solvable>` | 100.0% |
| `<breaking_point>` | 100.0% |
| `<answer>` | 69.0% |

The model produces valid XML structure for the termination tags every time. It fails to emit `<answer>` ~31% of the time — likely truncated by the 512-token generation budget after spending most of it on the two grid representations.

### Solvable prediction (degenerate)

| | Pred=True | Pred=False |
|---|---|---|
| GT=True (n=100) | 100 | 0 |
| GT=False (n=100) | 100 | 0 |

| Metric | Value |
|---|---|
| Accuracy | 50.0% (= class prior, random guessing) |
| Precision | 50.0% |
| Recall | 100.0% (vacuously — predicts True everywhere) |
| F1 | 66.7% |

**The model predicts `<solvable>true</solvable>` for all 200 samples**, regardless of input.

### Breaking-point prediction (degenerate)

| | Pred=True | Pred=False |
|---|---|---|
| GT=True (n=5) | 0 | 5 |
| GT=False (n=195) | 0 | 195 |

| Metric | Value |
|---|---|
| Accuracy | 97.5% (trivially — predicts False everywhere) |
| Precision | 0.0% (no true positives) |
| Recall | **0.0%** (0/5 BPs caught) |
| F1 | 0.0% |

Per-deadlock-type recall: 0% across all categories.

**The model predicts `<breaking_point>false</breaking_point>` for all 200 samples**, regardless of input.

## Diagnosis

The model converged in training (loss dropped 3× to 0.013, no overfitting), but at eval-time it produced **constant outputs that don't depend on the input grid**. This is a textbook distribution-shift collapse, not an under-training problem.

### Three contributing factors, in suspected order of severity:

1. **Single-turn eval prompt vs multi-turn training prompts.** `evaluate_model()` builds inputs as `[system, user_state]` only. SFT training data has mean ~17 messages per sample (10-turn sliding window). Step-0 samples (no priors) are a small minority of training data — the model may be effectively unconditioned on the kind of input the eval gives it.
2. **Eval state distribution: random-play vs LLM-policy training.** Eval samples are produced by `generate_balanced_eval_set()` which uses `TrajectoryGenerator` (random play). Training data is from `LLMTrajectoryGenerator` (Qwen plays the game). Random-play states have different visual characteristics (placements scattered uniformly) than LLM-policy states.
3. **Eval BP coverage: 5 of 200.** Even if the model worked, BP recall is being computed over only 5 samples — extremely high variance.

The fact that the model defaults to `solvable=true` is **inconsistent with class priors** (training data was 75% `solvable=false` after filtering). This rules out simple class-imbalance collapse and points at the model not having learned to condition on the input at all under the eval distribution.

### What this is NOT

- **Not a training failure.** Loss curves were healthy, eval_loss matched train_loss, format compliance is 100%.
- **Not a parser bug.** Format-compliance metric correctly extracted 100% of `<solvable>` / `<breaking_point>` tags.
- **Not a class-imbalance bias.** Class priors in training would predict the opposite default.

## Recommended next steps (before running Tier B/C)

Tier B and C will reproduce the same constant-output failure on this checkpoint — running them as-is wastes ~3.5 hours of GPU time. The fix to consider, in order of cost:

1. **Multi-turn eval prompt (5–10 min on GPU).** Modify `evaluate_model()` to build the prompt by replaying a short LLM-policy rollout from a generated puzzle and predicting on a state mid-rollout, matching training distribution. Re-run Tier A. If metrics jump materially, the diagnosis is confirmed and we move to Tier B + C with this eval.
2. **Eval set from LLM-policy distribution (~10 min on GPU).** Replace the random-play `generate_balanced_eval_set()` with one that uses `LLMTrajectoryGenerator` to produce eval states. This is what the trained model actually expects to see.
3. **Increase BP eval coverage.** Independent of the above, ensure ≥30 BP samples in any future eval set so the recall metric isn't ±20% from one sample flip.

If multi-turn eval *still* shows constant outputs, the issue is deeper (loss masking, training data label corruption, or boundary tokenization) and we'd run targeted probes before re-training.

## Notes

- Training time was 1h 30m, far below the 4h+ original estimate, thanks to the data-length filtering ([scripts/filter_long_samples.py](../scripts/filter_long_samples.py)) that dropped 60% of overly-long samples while preserving 94% of BP samples and 93% of solvable samples.
- Class balance after filtering improved markedly (24.6% solvable / 15.3% BP / 60.1% post-BP filler) compared to pre-filter (6.6% / 4.0% / 89.4%) — but the SFT model didn't translate that into useful classification under the current eval setup.
