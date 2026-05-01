# Eval Report — B-7 RL Phase 1 (2026-04-30)

## Headline

**B-7 RL Phase 1 (200 steps, lr=1e-5, v6 reward) regressed the model.** Pass@1 stayed at 0%, but the `<viability>` calibration *worsened*: greedy went from "always True on solvable" (correct) → "always False on everything" (collapsed). KL drifted to 1.7 (massive). The B-7 SFT checkpoint (AUC=1.000) remains the canonical pentomino model; **the B-7 RL checkpoint at `outputs/rl_b7_phase1/final` is strictly worse and should not be used downstream**.

This is a meaningful negative result. It shows that the v6 reward shape that worked for B-5 4×4 Sudoku (Pass@1 0% → 6.67%) actively destroys calibration on pentomino-easy because of the trajectory-length distribution mismatch.

## Setup

| field | value |
|---|---|
| SFT checkpoint (initial policy + ref) | `outputs/sft_pentomino_easy_b7_spa_hparams/final` (B-7, AUC=1.000) |
| Output dir | `outputs/rl_b7_phase1/final` (use only for diagnostic; do not deploy) |
| Reward | v6 (TP +1.0, FN −0.7, FP −0.5, TN +0.3, fmt +0.05/tag, success +3.0) — same as Phase 1 v6.1 |
| Optimizer | AdamW, lr=1e-5, β=0.05 KL coef |
| Rollout config | 4 puzzles × 8 group_size = 32 rollouts/step; T=0.7 |
| Total RL steps | 200 |
| Wall time | 103.7 min on H800 (~31s/step avg) |

## What happened (the trajectory)

| step | reward | solved% | KL | clipfrac | pg_loss |
|---|---|---|---|---|---|
| 1 | +0.889 | 0.0% | 0.186 | 0.021 | +0.066 |
| 26 | +1.141 | 0.0% | 1.379 | 0.011 | −0.058 |
| 51 | +1.434 | 0.0% | 1.790 | 0.007 | +0.042 |
| 76 | +1.447 | 0.0% | 1.749 | 0.009 | −0.001 |
| 101 | +1.500 | 0.0% | 1.645 | 0.010 | +0.000 |
| 151 | +1.500 | 0.0% | 1.725 | 0.010 | +0.000 |
| 200 | +1.500 | 0.0% | 1.677 | 0.010 | +0.000 |

**Reward plateaued at +1.50** from step ~50 onward, never lifted by success_bonus (since 0% solve rate). **pg_loss collapsed to 0.000** from step ~76 — the policy stopped learning anything.

| eval @ step | Pass@1 | `<viability>` greedy acc | BP recall |
|---|---|---|---|
| 0 (init / B-7 SFT) | 0% | **1.000** | 0.00 |
| 25 | 0% | 1.000 | 0.00 |
| 50 | 0% | **0.000** ← flipped | 0.00 |
| 100 | 0% | 0.000 | 0.00 |
| 200 (final) | 0% | 0.000 | 0.00 |

**Key observation**: at step 25 the model was still saying True greedily on all 30 eval puzzles (correctly, since they're empty boards = solvable). By step 50 it had flipped to saying False on all of them (wrong). Between steps 25 and 50, the policy crossed a tipping point — the reward landscape pulled it firmly toward all-False.

## Diagnosis

### Root cause: trajectory-length distribution

Pentomino-easy (5×4 with `{L, P, W, Y}`) has a *very harsh predictive gap*. Out of ~172 valid first moves, only ~10 are part of any of the 20 distinct tilings → **~94% of random/sampled first moves create immediately unsolvable states**. Trajectory length distribution is heavily skewed toward 1-step rollouts:

```
Trajectory length distribution (B-7 RL training):
  1 step:  ~90% (model picks a doom-causing first move, env terminates)
  2 steps: ~5%
  3 steps: ~3%
  4 steps: ~2% (rare; only valid tilings or near-tilings)
```

In RL training, this means almost every (s, a, s') sample has GT=False (doom). The reward landscape per sample:

| Outcome | Reward |
|---|---|
| Predict False (TP, GT=False) | +1.0 |
| Predict True (FN, GT=False) | −0.7 |
| Predict True (TN, GT=True, rare) | +0.3 |
| Predict False (FP, GT=True, rare) | −0.5 |

Asymmetry: **on doom states, predict-False (+1.0) > predict-True (−0.7) by +1.7 reward units**. With ~90% of samples being doom states, the gradient signal is strongly biased toward "always predict False." The "+0.3 for correctly predicting True on solvable states" is too rare and too small to survive.

### Why this DIDN'T happen on B-5

For B-5 (4×4 Sudoku), trajectories average 3-5 steps before BP, so the per-batch composition is more mixed:
- Pre-BP steps (GT=True): ~50% of samples
- BP step (GT=False): ~20%
- Post-BP steps: not seen (env terminates)

The TN/FP signal on solvable states is at parity with the TP/FN signal on doomed states. Both gradients flow. v6.1 (success_bonus 10→3) produced reward landscape where calibration *recovered* (60.9% → 62.0%) and Pass@1 *lifted* (0% → 6.67%).

### Why is success_bonus useless here?

In B-7 RL, **0% of rollouts reached success in any of 200 steps**. Success_bonus=3.0 never fired. The puzzle is hard enough (and the model bad enough at solving) that the trajectory-success reward signal contributes nothing. The only signal left is per-step `<viability>` correctness, which biases toward all-False.

## Implications

1. **The v6 reward shape doesn't generalize across envs as-is.** It worked on Sudoku because trajectories are long enough to balance TN and TP signals. On pentomino-easy, the trajectory distribution skewed the reward landscape too far toward "predict False."

2. **B-7 SFT alone (no RL) is the right pentomino model for downstream use.** AUC=1.000 SFT > AUC≈0.5 RL. For Phase 2 truncation, B-7 SFT is already past the gate (Prec(F)=94.3% at τ=0.10).

3. **The B-7 RL checkpoint should be marked deprecated.** Don't deploy it; don't continue from it.

## What would fix this

Three options, in increasing complexity:

### Option A: Bigger board (recommended — see analysis in main response)

Move from 5×4/4-piece to **5×5/5-piece** (or 5×6/6-piece). Longer rollouts (~2-3 steps avg) → more balanced TN/TP signal → calibration shouldn't collapse. ~1 day of work: regen P-0 if needed, regen data, retrain SFT (call it B-9?), redo RL.

### Option B: Filter trajectories during RL training

Use `do_rollouts_batched` but discard rollouts of length < 2. Forces the policy to see more pre-BP/TN signal during training. Doesn't fix the underlying mechanism but mitigates symptoms.

### Option C: Reward shape v7 — anchor on the baseline AUC

Add a regularizer that penalizes calibration regression. E.g., add a KL term against the SFT model's `<viability>` distribution specifically. Or add an auxiliary classification head and freeze its weights, using its predictions as an auxiliary reward signal. ~3 days of code.

**My recommendation: Option A.** Bigger boards address the root cause and have research value (does AUC=1.0 hold on harder predictive gaps?). Options B/C are bandages.

## Compute used

- Wall time: 1 hour 44 minutes on H800 (autodl1)
- Total rollouts: 200 steps × 32 = 6,400 rollouts, of which 0 succeeded
- Cost: a real loss — we destroyed the SFT calibration with no gain

## Reproduction

```bash
ssh autodl 'cd /root/autodl-tmp/world_model_termination_spa && \
            N_TOTAL_STEPS=200 LR=1e-5 \
            bash scripts/run_rl_b7_phase1.sh'
```

Logs: [logs/rl_b7_phase1.log](../logs/rl_b7_phase1.log).
Failed checkpoint: `outputs/rl_b7_phase1/final` (do not deploy).
