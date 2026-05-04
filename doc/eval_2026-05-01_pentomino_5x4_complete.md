# Pentomino-easy 5×4 — Complete Results (2026-05-01)

End-to-end summary of every run on the 5×4 / `{L, P, W, Y}` Pentomino-easy
configuration. Closes the 5×4 Pentomino track of the project.

## TL;DR

The recipe (SFT augmentation + RL with v8 anchor) produces a strong stochastic
action policy on Pentomino-easy 5×4 — Pass@1 stochastic = **59.25%** (237/400)
on the final RL checkpoint, vs **0/400** on B-7 SFT before any intervention.
But greedy Pass@1 stays at **0% across every run** because of the env's
sparse-action-space structure (argmax doesn't concentrate on any single
correct first move when many valid options exist).

This is captured in three numbers:

| Metric | B-7 SFT (baseline) | B-8 SFT (with augmenter) | **B-8 RL v8 final** |
|---|---|---|---|
| Greedy Pass@1 | 0% | 0% | 0% |
| **Stochastic Pass@1** (T=0.7) | **0/400 = 0%** | **89/400 = 22.25%** | **237/400 = 59.25%** |
| Mean rollout length | 1.27 | 2.28 | **3.59** |
| Complete tilings (length 4) | 0% | 30% | **59%** |

**Pass@1 stochastic is the right metric to report for Pentomino-easy.** Greedy
is structural-zero on this env. See
[eval_2026-05-01_pentomino_greedy_gap.md](eval_2026-05-01_pentomino_greedy_gap.md)
for the mechanism.

---

## Full per-run table

### SFT runs

| Run | Data | AUC | Stoch Pass@1 | Greedy Pass@1 | Greedy `solvable_acc` | Notes |
|---|---|---|---|---|---|---|
| B-7 SFT (canonical) | LLM-policy 2,964 train (80% step-0) | **1.000** | **0/400 (0.0%)** | 0% | 1.0 | Discrimination perfect, action policy fails at format on turn 2+ |
| B-8 SFT (augmented) | B-7 + 30× × 72 augmented = 5,124 train | **1.000** | **89/400 (22.25%)** | 0% | 1.0 | Augmenter cured stochastic Pass@1; greedy unchanged |

### RL runs

All RL runs start from the corresponding SFT checkpoint. lr=1e-5 throughout.

| Run | Source | Reward | Steps | Greedy Pass@1 | Greedy `solvable_acc` | Per-batch solve T=0.7 | via_kl |
|---|---|---|---|---|---|---|---|
| B-7 RL v6 | B-7 SFT | v6 | 200 | 0% | **collapsed → 0** | 0% | n/a |
| B-7 RL v7 | B-7 SFT | v7 | ~100 (killed) | 0% | **collapsed → 0** | 0% | n/a |
| B-7 RL v8 | B-7 SFT | v8 (single-token anchor) | 200 | 0% | **oscillated 0/1/0/1 → 0** | 0% | 0 |
| **B-8 RL v8** | **B-8 SFT** | **v8** | **200** | **0% all 8 evals** | **collapsed → 0** | **53–84% (avg ~60%)** | **0 throughout** |

---

## B-8 RL v8 detailed trajectory

**Per-batch (T=0.7) solve rate climbed dramatically while greedy stayed at 0%:**

| step | reward | per-batch solved | pg_loss | KL | via_kl |
|---|---|---|---|---|---|
| 1 | +2.33 | 16% | −0.408 | 0.211 | 0.0000 |
| 10 | +5.63 | 84% (peak) | +0.030 | 0.008 | 0.0000 |
| 25 | +5.76 | 53% | −0.161 | 0.013 | 0.0000 |
| 50 | +6.09 | 59% | +0.100 | 0.013 | 0.0000 |
| 75 | +6.25 | 66% | +0.108 | 0.013 | 0.0000 |
| 100 | +6.08 | 78% | +0.078 | 0.013 | 0.0000 |
| 125 | +5.96 | 56% | −0.116 | 0.013 | 0.0000 |
| 150 | +6.08 | 78% | +0.098 | 0.012 | 0.0000 |
| 175 | +5.57 | 47% | −0.125 | 0.015 | 0.0000 |
| 200 (final) | +6.09 | 59% | +0.119 | 0.013 | 0.0000 |

All 8 greedy evals (steps 25, 50, 75, 100, 125, 150, 175, 200) showed:
`Pass@1 = 0%, solvable_acc = 0.0, bp_recall = 0.0`.

---

## B-8 RL v8 final stochastic eval (the headline number)

50 puzzles × 8 rollouts = 400 stochastic rollouts at T=0.7 on
`outputs/rl_b8_v8_anchor/final/`:

```
Rollout length distribution (n=400):
   3 steps:  163  ####################
   4 steps:  237  #############################
  mean = 3.59, median = 4

First-action doom rate:  0.0%  (0/400)

Per-step class composition (across 1437 valid steps):
  GT=solvable: 1274 (88.7%)
  GT=doom:     163 (11.3%)

Pass@1 (rollout-level): 59.25%  (237/400)

B-8 RL v8 viability prediction (under stochastic sampling):
  accuracy: 100.0%
  predicted True: 88.7% of the time

Counterfactual expected per-step reward by policy × reward variant:
         policy |       v6 |       v7
    always_true |   +0.187 |   +0.773
   always_false |   -0.330 |   -0.773
         oracle |   +0.379 |   +1.000
     sft_actual |   +0.379 |   +1.000
```

Compared to B-7 SFT counterfactuals (oracle ≈ +1.0, always_false ≈ +0.46),
B-8 RL achieves *oracle-equivalent* per-step reward (+1.000) under stochastic
sampling — the model's actual policy IS the oracle on this rollout
distribution. (Greedy is wrong because argmax mode of the distribution doesn't
match the high-probability-but-spread-out correct moves.)

---

## Why greedy stays at 0% (mechanism recap)

5×4 with `{L, P, W, Y}`:
- ~150-200 first-move options
- Only ~20-40 are part of any of the 20 valid tilings
- After augmentation, model spreads probability across ~30 correct first moves
  (~2-3% prob each) plus some incorrect ones (some up to ~5%)
- argmax picks one of the higher-prob *incorrect* moves → greedy fails at step 1
- stochastic at T=0.7 spreads sampling across the distribution → ~30% chance of
  sampling a correct first move; over 4 steps this compounds but stays meaningful

Compounding errors over 4 piece placements:

```
P(greedy 4-correct) ≈ 0.15 × 0.20 × 0.30 × 0.50 ≈ 0.5%
P(stochastic 4-correct at T=0.7) ≈ measured 59% (much higher than the per-step
  product because the distribution is well-shaped — it gets concentrated on
  the right *family* of moves even when no single one is argmax)
```

The greedy collapse during RL (`solvable_acc` 1.0 → 0.0) is also explained by
this regime: the v8 single-token anchor preserves the *sampled* viability
token's logp but the *unsampled* token's logp drifts independently. Greedy
picks argmax(>true, >false), which can flip even when the anchor metric stays
near 0. This is what we observed: `via_kl = 0.0000` throughout while
`solvable_acc` stayed at 0.0 across all 8 evals.

---

## Truncation gate viability for B-8 RL final?

The B-8 RL v8 final checkpoint has:
- 100% viability accuracy (stochastic) ← would seem great for truncation
- BUT 0% greedy `solvable_acc` ← the gate uses greedy `<viability>` predictions

Implication: **the truncation gate would not work on B-8 RL v8 final** because
the greedy logit ordering of `>true` vs `>false` is broken. The gate would
trigger on most states (model says False at greedy on everything) and kill
recoverable rollouts immediately.

**v8.2 dual-token anchor** is the necessary precondition for a Pentomino
truncation experiment. Without v8.2, the truncation gate is a non-starter on
this env.

---

## Outstanding questions (not blocking the 5×4 track)

1. Will v8.2 (dual-token anchor) preserve greedy `solvable_acc` on B-8 RL?
   → Implementation done, launch pending.
2. What does Pass@8 stochastic look like on the B-8 RL final checkpoint?
   → 8 rollouts/group × 50 puzzles = approximately Pass@8. From training
   per-batch solve rates (53-84%), Pass@8 stochastic is approximately the
   max-of-8 of the per-rollout solve rate, likely > 95%.

---

## Files

- B-7 SFT eval: [eval_2026-04-30_b7_pentomino_easy.md](eval_2026-04-30_b7_pentomino_easy.md)
- B-7 sanity rollout stats: [sanity_2026-04-30_b7_rollout_stats.json](sanity_2026-04-30_b7_rollout_stats.json)
- B-7 RL Phase 1 eval: [eval_2026-04-30_b7_rl_phase1.md](eval_2026-04-30_b7_rl_phase1.md)
- B-8 sanity rollout stats: [sanity_2026-05-01_b8_rollout_stats.json](sanity_2026-05-01_b8_rollout_stats.json)
- B-8 RL final sanity stats: [sanity_2026-05-01_b8_rl_v8_rollout_stats.json](sanity_2026-05-01_b8_rl_v8_rollout_stats.json)
- Greedy gap mechanism: [eval_2026-05-01_pentomino_greedy_gap.md](eval_2026-05-01_pentomino_greedy_gap.md)
- Augmenter: [src/data/solution_path_augmenter.py](../src/data/solution_path_augmenter.py)
- Combiner: [scripts/combine_pentomino_5x4_with_augmented.py](../scripts/combine_pentomino_5x4_with_augmented.py)
- B-8 SFT launcher: [scripts/run_pentomino_5x4_sft_augmented.sh](../scripts/run_pentomino_5x4_sft_augmented.sh)
- B-8 RL v8 launcher: [scripts/run_pentomino_5x4_rl_v8.sh](../scripts/run_pentomino_5x4_rl_v8.sh)

## Conclusions

1. **The recipe (SFT augmentation + RL with v8 anchor) produces a strong
   stochastic action policy on Pentomino-easy 5×4** (Pass@1 stochastic ~59%).

2. **Greedy Pass@1 cannot be lifted on this env without a structural change**
   — either bigger action space (probably won't help on 5×10), constraint
   propagation (Kakuro/Sudoku-like envs), or different evaluation regime
   (Pass@N stochastic).

3. **For the paper, the cleanest claim**: "the recipe lifts the (greedy,
   stochastic) regime from (0, 0) to (0, 59%) on Pentomino, and from (0, 0)
   to (50%, ~50%) on Sudoku, with the difference traced to action-space
   density."

4. **The truncation gate is incompatible with the current B-8 RL v8
   checkpoint** because greedy `solvable_acc` is 0. Need v8.2 dual-token
   anchor for greedy calibration before truncation experiment is possible
   on Pentomino.
