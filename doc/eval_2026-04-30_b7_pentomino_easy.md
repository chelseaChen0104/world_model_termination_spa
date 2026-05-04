# Eval Report — Run B-7: Pentomino-Easy SFT (2026-04-30)

## Headline

**B-7 (5×4 Pentomino with `{L, P, W, Y}` pieces, B-5 hparams) hits ROC AUC = 1.000 — perfect `<viability>` discrimination.** This is the first cross-env transfer test of the recipe, and the discrimination signal is *stronger* than the 4×4 Sudoku baseline (B-5 AUC = 0.726).

The recipe transfers cleanly from Sudoku to a structurally different env (geometric piece placement vs cell-value puzzle), and the new tag set (`<observation>` + `<next_state>` + `<viability>` + `<answer>`) works end-to-end. See [doc/spec_pentomino.md](spec_pentomino.md) for the env spec.

## Setup

| | B-5 (4×4 Sudoku) | **B-7 (5×4 Pentomino)** |
|---|---|---|
| Task | 4×4 Sudoku, 6 empty cells | 5×4 board, 4 pentominoes `{L, P, W, Y}`, 20 distinct tilings |
| Tag set | `<solvable>` family | **`<viability>` family** (renamed) |
| Train data path | `data/sudoku_4x4_llm_policy_minimal_spa_scale/` | `data/pentomino_easy_llm_policy_minimal/` |
| **Train samples** (no_post_bp) | **6,571** | **2,964** (~45% of B-5) |
| Train class balance | 40% solvable / 60% BP | **18.8% solvable / 81.2% BP** |
| Val samples | 1,645 | 742 |
| Epochs / LR / batch | 5 / 1e-4 / 16 | 5 / 1e-4 / 16 (same) |
| max_length | 1,024 | 1,024 |
| eval_steps | 25 | 25 |
| Total updates | 2,050 | 925 |
| Final eval_loss | 0.0148 | 0.0361 (2.4× higher) |
| **`<solvable>` / `<viability>` ROC AUC** | **0.726** | **1.000** |

Why B-7 had less data: pentomino's harsher predictive gap kills most rollouts at step 1 (only ~6% of valid first moves are part of any tiling), yielding fewer (s, a, s′) samples per trajectory than Sudoku.

## Greedy classification eval (300 samples, 100 solvable / 100 unsolvable / 100 BP)

```
Format compliance:  100% valid + has <viability> + has <answer>

Greedy <viability> prediction (100 GT-True vs 100 GT-False):
  Accuracy:   49.5%
  Precision:  33.3%
  Recall:     1.0%
  F1:         1.9%

  Confusion matrix:
                Pred=True   Pred=False
    GT=True          1           99
    GT=False         2           98
```

**Greedy collapses to "always False"** — model predicts False on 99/100 solvable states. Same phenomenon as B-5: P(false) > P(true) at the argmax even on solvable states because the model is *uncalibrated* (heavy False prior from the 81/19 class imbalance in training).

Format compliance is 100% — the new tag set is emitted reliably.

## Logprob threshold-sweep eval — the headline

Teacher-forced single forward pass at the `<viability>` token, reading P(true) directly from logits.

### P(true) distributions

| Class | n | mean | median | std |
|---|---|---|---|---|
| GT=true | 100 | **0.548** | 0.641 | 0.271 |
| GT=false | 100 | **0.000** | 0.000 | 0.000 |

**Mean separation: +0.548** — vs B-5's +0.023 (24× larger). On unsolvable states the model is essentially 100% confident the state is doomed (P(true) ≈ 0); on solvable states it puts ~55% probability on True (median 64%).

### Threshold sweep

| τ | Acc | Prec(T) | Rec(T) | Spec | F1(T) |
|---|---|---|---|---|---|
| **0.10** | **97.0%** | **100%** | **94%** | **100%** | **96.9** |
| 0.20 | 91.0% | 100% | 82% | 100% | 90.1 |
| 0.30 | 90.5% | 100% | 81% | 100% | 89.5 |
| 0.50 | 78.5% | 100% | 57% | 100% | 72.6 |
| 0.70 | 68.5% | 100% | 37% | 100% | 54.0 |
| 0.90 | 51.0% | 100% | 2% | 100% | 3.9 |

At τ=0.10 we get **97% accuracy with 100% precision and 94% recall on the True class** — near-perfect classification at a low threshold. **Critical: precision on False is 94.3% at τ=0.10**, well above the 90% gate that would enable Phase 2 hard-truncation in the RL plan.

### ROC AUC

**1.000** — perfect ranking. Of the 100×100 = 10,000 (solvable, unsolvable) pairs, the model correctly ranks the solvable one higher in 100% of pairs.

## Why B-7 is so much stronger than B-5 on discrimination

Three plausible factors, all probably contributing:

1. **Pentomino's predictive gap is *visually local*.** A wrong piece placement leaves a literal isolated empty region the LLM can see directly in the rendered cells. Sudoku's "wrong number triggers a constraint cascade three steps later" requires propagation reasoning — much harder for the LLM to detect from raw text.

2. **Strong class-imbalance signal helps the BP-detection direction.** With 81% BP samples in training, the model develops a sharp "this looks doomed" prior. The 19% solvable samples are enough to learn the distinction since the BP signatures are so distinctive.

3. **Smaller state and action space.** 5×4 board with 4 piece letters has fewer patterns to memorize than 9×9 Sudoku with 9 digits and complex row/col/box constraints. The LLM's pattern-matching capacity is sufficient.

## Calibration is still an issue (same as B-5)

Although AUC is perfect, **greedy decoding doesn't surface the discrimination** — model still says "False" 99% of the time when greedy. The `<viability>` token's P(true) only reaches 0.55 on solvable states, never crossing 0.5. So the *ranking* is perfect but the *threshold* is off.

For deployed use (early-termination during RL rollouts), this isn't actually a problem — we'd use the threshold-based decision (P(true) < 0.10 → terminate), which is what NEAR-1.4 / Phase 2 of the RL plan envisions. With the 94.3% Prec(F) at τ=0.10, B-7 is *already* well past the Phase 2 truncation gate (≥90% Prec(F)). RL on B-7 could turn on hard truncation from step 1.

## Comparison vs SPA paper

The SPA paper doesn't have a `<viability>` analog (their tag set is only state-prediction, no termination tag), so there's no direct AUC comparison. SPA's headline numbers are Pass@1/Pass@8 on puzzle-solving, which we haven't measured for B-7 yet (next step).

What we *can* say: SPA demonstrates the world-modeling SFT recipe transfers within their puzzle family (Sokoban, FrozenLake, Sudoku). We extend that with one strict cross-env test (Sudoku → Pentomino) of our recipe with the added termination tag. Both succeed.

## What this means for the project

1. **Recipe transfers cross-env on the discrimination metric.** Sudoku → Pentomino with the same hparams + comparable data scale yields BETTER AUC. The recipe is not Sudoku-specific.

2. **Pentomino predictive gap is sharper but more learnable.** The 81/19 class imbalance + visual locality of the dead-ends combine to make `<viability>` a clean classification target. This may or may not transfer to harder pentomino setups (full 6×10) where deadlocks are more abstract.

3. **B-7 is RL-ready with truncation enabled.** Prec(F) at τ=0.10 = 94.3% clears the Phase 2 gate. RL can turn on conservative hard-truncation from step 1, saving compute.

4. **Greedy Pass@1 is an open question** for B-7. Like B-5, greedy collapses to a single class — would need calibration via RL or threshold-based decoding to lift solving ability. Not yet measured.

## Compute used

- Data gen: ~16 min on autodl1 (single cloud, 3,000 trajectories)
- B-7 SFT: ~30 min on autodl1 (925 train steps)
- B-7 eval: ~20 min on autodl1 (greedy 300 + logprob 200)
- **Total: ~66 min wall time end-to-end**

## Reproduction

```bash
# Data generation
ssh autodl 'cd /root/autodl-tmp/world_model_termination_spa && \
            N_TRAJ=3000 bash scripts/generate_pentomino_5x4.sh'

# B-7 SFT
ssh autodl 'cd /root/autodl-tmp/world_model_termination_spa && \
            bash scripts/run_pentomino_5x4_sft.sh'

# B-7 eval (greedy + viability-logprob)
ssh autodl 'cd /root/autodl-tmp/world_model_termination_spa && \
    bash scripts/_run_with_env.sh python -u evaluate_rl.py \
        --env sudoku --metric solvable-logprob --skip-rl \
        --grid-size 4 --difficulty easy \
        --sft-path outputs/sft_pentomino_easy_b7_spa_hparams/final \
        --eval-from-parquet data/pentomino_easy_llm_policy_minimal/wm_val.parquet \
        --n-per-class 100 --tag-name viability'
```

Logs: [logs/sft_b7.log](../logs/sft_b7.log), [logs/eval_b7.log](../logs/eval_b7.log).
Checkpoint: `outputs/sft_pentomino_easy_b7_spa_hparams/final/` (on autodl1).
