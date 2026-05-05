# Pentomino 5×6 π_θ — SFT + RL Results

**Date**: 2026-05-05
**Author**: this session (Pentomino + Hidato local machine)
**Models trained**: 4 SFT variants + 1 RL run on top of best SFT
**Hardware**: autodl3 (NVIDIA A800 PCIe, 80 GB)
**Backbone**: Qwen2.5-1.5B-Instruct (full-parameter SFT, bf16)
**Action format**: plain `place {piece} ori={K} at row {R} col {C}` — no XML tags

## TL;DR

After three SFT iterations and one RL attempt, **v3 SFT is the strongest policy: Pass@1 = 15/172 = 8.7%** on full-rollout episodes. RL on top of v3 did not improve over the SFT baseline (ended at 13/172 = 7.6%, within noise).

We adopt **v3 SFT as the production Pentomino π_θ** for downstream SAVE data generation. The π_θ is the "lt/ht candidate sampler" for f_φ training — it does not need to be a top solver, just diverse enough to produce viable+doomed candidate mixes per state.

## Per-stage results

### SFT progression

| Variant | Train data | Per-step local_valid | Episode Pass@1 | eval_loss (final) |
|---|---|---|---|---|
| v1 (3 ep, no aug) | 7K samples (172 subsets × ≤10 tilings × 6 steps) | 65% | **0 / 172 (0.0%)** | 0.169 |
| v2 (6 ep, +dihedral aug) | 28K samples (4× via 5×6 D₂ symmetries) | 84.5% | **0 / 172 (0.0%)** | 0.175 |
| **v3 (6 ep, +dihedral +DAgger-lite)** | **~70K samples** (+ off-canonical solver-recovery samples) | **90.0%** | **15 / 172 (8.7%)** | **0.142** |

Per-step local_valid rate at each step (v3 only, on the held-out v1 val set, n=200):

| step | local_valid |
|---|---|
| 0 | 100.0% |
| 1 | 97.1% |
| 2 | 78.8% (mid-game, hardest) |
| 3 | 75.8% |
| 4 | 90.9% |
| 5 | 97.0% |

### RL attempt (on top of v3 SFT)

| Hyperparameter | Value |
|---|---|
| Algorithm | GRPO (group-relative advantage) + token-level KL anchor to v3 SFT |
| Steps | 100 |
| Puzzles per batch (K) | 4 |
| Group size (G) | 4 |
| Effective rollouts per step | 16 |
| Learning rate | 1e-6 |
| KL coefficient | 0.05 |
| Sample temperature | 1.0, top-p 0.95 |
| Reward shaping | +1.0 success / +0.1 valid step / −0.1 invalid step |

**Result**: Pass@1 ended at **13 / 172 = 7.6%**, change −1.1 pp from v3 SFT. Within noise (172 binomial samples around p≈0.08 has 95% CI ±4 pp).

Training trajectory across 100 steps:
- `mean_reward` flat at 0.10–0.24 with high variance, no monotonic trend
- `n_complete` per group of 16 rollouts: typically 0, occasionally 1
- `KL(π || π_ref)` per token: ±0.001 to ±0.009 — policy barely moved
- `loss` hovering ±0.02 throughout — no descent trend

## Diagnosis: why RL did not help

Three compounding problems:

1. **Sparse-reward bottleneck.** With Pass@1 ~10%, only 1-2 of 16 group rollouts succeed on average. Most groups have **all-near-zero rewards → group std ≈ 0 → advantages ≈ 0 → no gradient signal**. GRPO needs positive variance within groups to attribute credit.

2. **Conservative LR + small advantage = tiny policy updates.** With LR=1e-6 and per-token advantage near zero, the policy barely moved from SFT initialization (KL stayed under 0.01 per token). Even if the gradient direction were correct, the magnitude was too small to make a difference in 100 steps.

3. **Reward shaping too weak.** +0.1 per valid step gives noisy ±0.4 spread between minimal and "almost-success" rollouts; the +1.0 success bonus almost never fires. The shaping doesn't strongly differentiate "made it to step 5 then died" from "died at step 1".

The fundamental issue is **the RL recipe was tuned for envs with denser positive signal** (Sudoku rl_b5 had ~30% Pass@1 starting point; here we start at ~9%). Pentomino 5×6 multi-subset is at the RL difficulty edge for greedy GRPO.

## What might have made RL work (not pursued in this run)

- **Larger group size (G=8 or 16)** so each group has higher chance of containing at least one success, giving non-zero advantages.
- **Higher LR (1e-5)** — match the v8 Sudoku recipe; willing to take more KL drift in exchange for actual learning.
- **Stronger shaped reward**: per-step reward proportional to "fraction of board tiled" (e.g., +0.2 × fraction_filled), so even unsuccessful rollouts get differentiated rewards.
- **Curriculum**: start episodes with more pieces pre-placed (effectively shorter trajectories) → higher initial success rate → denser RL signal → gradually unwrap to full 6-step rollouts.
- **DAgger-style relabeling during rollouts**: when the model fails at step k, query solver for the recovery action, add as a supervised correction.

These are reasonable retry directions. None are blocking the SAVE pipeline; we proceed with v3 SFT.

## What this means for the paper

The SAVE paper's central machinery (sibling-action contrastive learning for f_φ + CVCP at inference) is independent of how π_θ was trained. π_θ enters the pipeline as:

| Use of π_θ | What it needs |
|---|---|
| Sibling-set data gen (lt/ht candidate sampling) | Produce a mix of viable + doomed candidates per state. v3 SFT at 8.7% Pass@1 satisfies this. |
| f_φ training | π_θ does not appear in f_φ's loss. The recipe is solver-derived viability + token CE + (planned) L_rank — independent of how candidates were sourced. |
| CVCP at inference | Sample K candidates; π_θ logprobs used as tie-break. Recipe-agnostic. |

Implementation-detail asymmetry: Sudoku and Hidato use RL'd π_θ (rl_b5_phase3_v8_anchor and rl_b_h1_v8_anchor); Pentomino uses SFT-only π_θ (v3). Disclose in the paper's "Implementation details" subsection. Reviewers care about benchmark integrity, not π_θ training recipe parity across envs.

## Artifacts

| Item | Path |
|---|---|
| v3 SFT checkpoint (chosen π_θ) | `autodl3:/tmp/sft_pentomino5x6_pi_theta_v3/final/` (2.9 GB) |
| v3 SFT training log | `autodl3:/root/autodl-tmp/world_model_termination_spa/logs/pi_theta_pent5x6_v3.log` |
| v3 Pass@1 eval JSON | `autodl3:/tmp/pent5x6_v3_pass1_eval.json` |
| RL run final checkpoint (not used) | `autodl3:/tmp/rl_pentomino5x6_v6/final/` |
| RL training log | `autodl3:/root/autodl-tmp/world_model_termination_spa/logs/rl_pentomino5x6.log` |
| RL structured per-step log | `autodl3:/tmp/rl_pentomino5x6_v6/rl_log.jsonl` |
| SFT generator | [scripts/generate_pi_theta_sft_pentomino.py](../scripts/generate_pi_theta_sft_pentomino.py) (with `--augment`, `--dagger-deviations` flags) |
| SFT trainer | [scripts/train_pi_theta_sft.py](../scripts/train_pi_theta_sft.py) |
| RL trainer (additive, plain-action GRPO) | [scripts/rl_pentomino5x6_multisubset.py](../scripts/rl_pentomino5x6_multisubset.py) |
| Eval script (per-step + episode Pass@1) | [scripts/eval_pi_theta_pass1.py](../scripts/eval_pi_theta_pass1.py) |

## Decision and next step

**Adopted**: v3 SFT as Pentomino production π_θ.
**Next**: generate Pentomino SAVE sibling-set data using v3 SFT for lt/ht sampling, mirroring the Sudoku/Hidato pilot+paper-final pipeline. Then train Pentomino f_φ.

— end of report —
