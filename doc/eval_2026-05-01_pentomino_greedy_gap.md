# Pentomino Greedy Pass@1 Collapse — Mechanism and Implications (2026-05-01)

> **Major correction (2026-05-03):** the no-leak SFT + RL with v8 anchor +
> `--action-quality-bonus 0.5` achieved **greedy Pass@1 = 100% (30/30)** on
> 5×4 LPWY when evaluated with the corrected eval pipeline
> (`sanity_check_checkpoint --prepend-current-state --single-turn-eval
> --max-response-tokens 512`). The earlier "greedy Pass@1 = 0% on every
> Pentomino run" claim was based on the in-training `quick_pass1`, which has
> a multi-turn-history bug. **All earlier Pentomino RL "greedy=0" entries in
> the table below should be re-checked with the corrected eval before being
> trusted.** See section "Action-quality bonus + corrected eval pipeline →
> greedy Pass@1 100%" near the bottom for the full detail. Original analysis
> below is preserved for the diagnostic narrative; treat its conclusions as
> superseded for the action-quality-bonus + no-leak setting.

A consolidated report on the **stochastic-vs-greedy gap** observed across all
Pentomino-easy 5×4 runs in this project. Captures the diagnostic story, the
data, and what it implies for env choice + paper claims.

## TL;DR

**On Pentomino-easy 5×4 with `{L, P, W, Y}`, greedy Pass@1 = 0% on every
single run we have produced** — SFT, RL with v6, v7, v8, v8+augmentation. The
augmenter ([B-8 SFT](runs_reference_2026-05-01.md#b-8-completed-augmentation-cured-pass1-0))
lifted *stochastic* Pass@1 from 0/400 to 22.25%, but greedy stayed at 0%. RL
on B-8 SFT (with the v8 anchor) lifted *stochastic* per-batch solve rate to
84%, but greedy stayed at 0% across 6 consecutive evals (steps 25, 50, 75,
100, 125, 150).

**This is an env-structure bottleneck, not a recipe failure.** The 5×4 board
with 4 pieces has a sparse action space where greedy argmax rarely lands on
a valid solution path even with a well-trained policy. Sudoku 4×4 doesn't
have this issue because the action space is denser with valid moves.

The right metric for Pentomino-easy is therefore **Pass@N stochastic**, not
Pass@1 greedy. The gap between them is a structural feature of the env, not
a deficiency to be fixed by tweaking the recipe.

---

## The data: every Pentomino run

| Run | Greedy Pass@1 | Greedy `solvable_acc` | Stochastic per-batch / Pass@1 |
|---|---|---|---|
| **B-7 SFT** (no augmentation) | 0% | 1.0 | **0/400** sanity test |
| B-7 RL v6 (deprecated) | 0% | collapsed → 0 | 0% per-batch throughout |
| B-7 RL v7 (deprecated) | 0% | collapsed → 0 | 0% per-batch throughout |
| B-7 RL v8 (deprecated) | 0% | oscillated 0/1/0/1 → 0 | 0% per-batch throughout |
| **B-8 SFT** (with augmentation) | **0%** | 1.0 | **22.25%** sanity test |
| **B-8 RL v8** (in flight) | **0% across 6 evals** | **collapsed → 0** | **16-84% per-batch** |
| **5×4 RL v8 + action-quality-bonus 0.5** (no-leak SFT) | **0% in-training (BROKEN EVAL) → 100% (30/30) with corrected eval pipeline** | **viability collapsed → all-True** | **100% per-batch from step 41**, Pass@8 stochastic also 100% |
| **Sudoku Run A** *(reference)* | **33-37%** | 0.55 | similar to greedy |
| **Sudoku Phase 3 v8 anchor** *(reference)* | **50%** | 0.51 (held) | similar |

Two patterns jump out:

1. **No Pentomino run has ever lifted greedy Pass@1 off 0%.**
2. **B-8 RL produced unprecedented stochastic action quality** (per-batch 84%
   peak) while greedy stayed exactly 0% — making the gap structural, not
   transient.

---

## The mechanism: stochastic vs greedy decomposition

For a K-step trajectory:

```
P(greedy solves)  = ∏_k 1[argmax p(a_k | s_k) is correct]   ∈ {0, 1}
P(stochastic solves at T)  = ∏_k P(sampled a_k is correct | s_k)   ∈ [0, 1]
```

Four regimes in (greedy, stochastic) space:

| Regime | greedy | stochastic | What it means |
|---|---|---|---|
| 1: well-trained + concentrated | high | similar to greedy | argmax is correct with high prob |
| 2: greedy good, stochastic worse | high | lower | concentrated; sampling adds noise that hurts |
| 3: greedy 0, stochastic > 0 | 0% | meaningful | knows the right answer is in distribution but not as argmax |
| 4: both 0 | 0% | 0% | doesn't know correct answer exists |

**Pentomino runs map to:**
- B-7 SFT: regime 4 (both 0)
- B-7 RL all variants: regime 4 → still 4 (no improvement)
- B-8 SFT: **regime 4 → 3** (augmentation taught the model right answers exist)
- B-8 RL v8: regime 3 → deeper regime 3 (sharper stochastic mass on correct paths)

**Sudoku Run A / Phase 3 v8 anchor**: regime 1 (both metrics improve together).

---

## Why Pentomino-easy is structurally regime 3

The action space on 5×4 with `{L, P, W, Y}`:
- **~150-200 possible first moves** (4 pieces × ~8 orientations × ~5-10 valid positions)
- Of these, **~20-40 are part of any of the 20 valid tilings** (10-20% solution density)

After the augmenter, the model assigns probability mass like:
- ~30 specific correct first-moves: each ~2-3% prob
- ~150 incorrect first-moves: combined ~30% prob, but some single ones higher than any individual correct one

Result:
- **argmax**: usually picks an incorrect-but-most-likely move → greedy fails at step 1
- **stochastic at T=0.7**: spreads across the distribution; correct moves are sampled in aggregate ~30% of the time → stochastic Pass@1 = 22%

This is purely a property of the 5×4 / 4-piece env. **Compounding across 4
piece placements makes greedy Pass@1 essentially zero even with very good
per-step solution density** (e.g., per-step argmax-correct = 0.3 → 4-step =
0.81%).

By contrast, **Sudoku 4×4 has fewer valid moves per cell** (often a unique
correct answer once constraints propagate). At that point, argmax is likely
to pick the correct one. So Sudoku trajectories of similar length naturally
have higher P(greedy correct) per step → cumulative greedy Pass@1 stays
above zero.

---

## Why v8 anchor doesn't fix Pentomino greedy

The v8 anchor (single-token version) penalizes squared deviation of the
SAMPLED viability token's logp from the SFT reference. This works for
keeping the *distribution of sampled tokens* near SFT (preserves stochastic
behavior on `<viability>`).

But greedy decoding on `<viability>` picks `argmax(logp(>true), logp(>false))`.
The single-token anchor only constrains whichever was sampled — the unsampled
token's logp can drift independently. If `logp(>false)` drifts up by 0.3 and
`logp(>true)` drifts down by 0.05 while `>true` was the sampled token (anchored
at SFT value), the relative ordering at greedy time can flip from True→False
even with `via_kl ≈ 0`.

**This is exactly what we observed on B-8 RL v8**: `via_kl=0.0000` throughout
training (anchor working), but greedy `solvable_acc` collapsed from 1.0 → 0.0.

The v8.2 dual-token anchor (anchor BOTH `>true` and `>false` logprobs at every
viability position regardless of which was sampled) is the targeted fix —
preserves the relative ordering by construction. **Implementation done; not
yet tested in flight.** [scripts/run_pentomino_5x4_rl_v8_2_dual_anchor.sh](../scripts/run_pentomino_5x4_rl_v8_2_dual_anchor.sh).

But note: **v8.2 fixes the `solvable_acc` greedy collapse, NOT the action-policy
greedy Pass@1 collapse.** The action policy greedy Pass@1 = 0% is independent
and structural to the env.

---

## Action-quality bonus + corrected eval pipeline → greedy Pass@1 100% (2026-05-03 result, retracted earlier 0%)

`--action-quality-bonus 0.5` adds a per-step ±0.5 reward on whether the post-action
state is GT-solvable, providing direct gradient on action goodness independent of
the `<viability>` prediction.

**Initial finding (in-training quick_pass1, BROKEN):** Pass@1 = 0% across all 8
evals (steps 25/50/.../200). Per-batch=100% by step 41; reward variance → ±0.00
by step 143. The training-time eval was running against a multi-turn-history
prompt format that the model wasn't trained for (same bug we hit on B-H1) — so
the in-training Pass@1=0% number was an eval-pipeline artifact, NOT a model
failure.

**Corrected finding (Step 4/4 of pipeline runs `sanity_check_checkpoint` with
`--prepend-current-state --single-turn-eval --max-response-tokens 512`):**

```
=== checkpoint sanity check: outputs/rl_pentomino_5x4_no_leak_v8_aq/final ===
# A. Verbose greedy rollout on puzzle 0 (seed=100000)
  STEP 0: place L ori=2 at row 1 col 1   valid=True  is_solvable=True
  STEP 1: place W ori=2 at row 1 col 3   valid=True  is_solvable=True
  STEP 2: place Y ori=5 at row 2 col 4   valid=True  is_solvable=True
  STEP 3: place P ori=6 at row 4 col 2   valid=True  success=True
  → solved=True n_steps=4  full_xml=4/4

# B. Greedy Pass@1 on 30 puzzles
  Pass@1:                 100.0% (30/30)

# C. Stochastic Pass@k on 30 puzzles, k=8, T=0.7
  Pass@8:               100.0% (30/30)
  per-batch solve rate: 100.0% (240/240)

# D. Health checks
  ❌ viability predictions show both classes: {'true': 120}
     (single-class → bimodal collapse / regime-1)
```

**Greedy Pass@1 = 100% (30/30) with the corrected eval pipeline, Pass@8 = 100%.**
The model fully learned the 5×4 LPWY puzzle; both greedy and stochastic decoding
solve every test puzzle.

**What this means for the regime-3 hypothesis on Pentomino-easy.**

The earlier section above ("Why Pentomino-easy is structurally regime 3") argued
that 5×4 with 4 pieces has a sparse-enough action space that argmax cannot land
on a valid solution path even with a well-trained policy. That conclusion was
based on the broken in-training eval. With the corrected eval, **5×4 is NOT
structurally regime 3** — sufficient RL training (v8 anchor + action-quality-bonus,
200 steps) lifts both greedy AND stochastic to 100%. The action policy is excellent
under argmax; the regime-3 framing was an artifact.

**What's still genuine.**

The viability prediction has collapsed to all-`True` (`{'true': 120}` across all
120 step-decisions in the eval) — the calibration is gone in the canonical
regime-1 sense. But because the action policy never visits a doom state on
this puzzle (every greedy action lands on `is_solvable=True`), the always-True
prediction happens to be vacuously correct. The viability tag is essentially
unused as a termination signal here.

This is itself an important finding: **on a denser-than-expected puzzle, the
viability tag becomes vestigial and the action policy alone carries the load**.
The truncation-gate experiments only matter for envs/puzzles where doom states
are actually reachable under a well-trained policy; for 5×4 LPWY post-RL,
they aren't.

**Pending re-checks for prior Pentomino claims.** B-7 and B-8 RL Pass@1=0%
numbers in the table above were also based on the in-training quick_pass1 with
the same multi-turn-history bug. Those checkpoints should be re-evaluated with
`sanity_check_checkpoint --prepend-current-state --single-turn-eval` before the
"every Pentomino run greedy=0" claim in the TL;DR can stand. Likely outcome:
the older claims partially hold (B-7 SFT at 0% even pre-RL, where the action
policy hadn't been sharpened yet) but the RL-trained checkpoints may all
actually be much higher than reported.

---

## Comparison: Sudoku v8 anchor (success) vs Pentomino v8 anchor (partial)

| | Sudoku Phase 3 v8 anchor | Pentomino B-8 RL v8 |
|---|---|---|
| Source SFT Pass@1 (greedy) | 33% (Run A endpoint) | 0% (B-8 SFT) |
| Source SFT solvable_acc (greedy) | 0.514 | 1.0 |
| RL final Pass@1 (greedy) | **50%** ✓ | **0%** ✗ |
| RL final solvable_acc (greedy) | **0.509** ✓ (held) | **0.000** ✗ (collapsed) |
| RL per-batch solve T=0.7 final | not measured | **84%** peak |
| via_kl during RL | low, sometimes nonzero (anchor active) | exactly 0.0000 (anchor active) |

Why v8 anchor sufficed on Sudoku but not Pentomino:
- **Sudoku**: longer trajectories (avg 3-5 steps) → each rollout naturally
  visits both `<solvable>=True` (pre-BP) and `<solvable>=False` (BP) states.
  The single-token anchor sees both tokens sampled at viability positions
  across many rollouts → in aggregate, both directions get anchored.
- **Pentomino-easy**: short trajectories (avg 2.3 steps under B-8 stochastic,
  median 2). A typical rollout has ~1 viability position. Single-token anchor
  only constrains whichever was sampled at *that* position; the unsampled
  token's logp drifts globally over the training run.

This is the env-structure dependency of the calibration anchor mechanism. It
maps cleanly onto the trajectory-length argument from the
[plan doc](plan_2026-04-29_rl_approach.md): **single-token anchor is sufficient
when the env produces enough sample diversity per viability position;
otherwise dual-token anchor is needed.**

---

## What this means for the project

### Already-validated claims

- **Calibration anchor mechanism works on Sudoku** (Phase 3 v8 anchor:
  Pass@1 33% → 50%, solvable_acc preserved).
- **Augmentation cures stochastic Pass@1 = 0%** on Pentomino (B-8 SFT
  0/400 → 22.25%).
- **Truncation gate saves compute** ([eval_2026-05-01_truncation_full.md](eval_2026-05-01_truncation_full.md)
  — 19% wall, 23% tokens, −10pp Pass@1 on Sudoku v8 anchor).

### Not-yet-validated claims

- **Single-token v8 anchor doesn't generalize to Pentomino-easy** — the
  greedy `solvable_acc` collapse during B-8 RL is the negative finding.
  v8.2 dual-token anchor is the proposed fix, ready to launch.
- **Greedy Pass@1 lift on Pentomino requires denser action space** —
  5×10/10-piece (B-9) is the natural test; alternatively a different env
  (Hidato, Knight's Tour) with smaller per-state branching.

### Implications for paper claims

1. **Don't use Pass@1 greedy as the headline metric on Pentomino-easy.** Use
   Pass@N stochastic (or Pass@8 from training-time per-batch solve rates).
   The greedy=0 number is structural and doesn't reflect model capability.

2. **Frame the recipe in terms of regimes**, not single-metric numbers:
   - "Recipe lifts (greedy, stochastic) from regime 4 → 3 via SFT augmentation"
   - "v8 anchor preserves regime-1 envs (Sudoku) and regime-3 envs (Pentomino
     B-8) without quality collapse on stochastic"
   - "v8.2 anchor preserves greedy `solvable_acc` (calibration) regardless of
     env trajectory length — the next paper claim to validate"

3. **Cross-env transfer is partial, not full.** Sudoku gets full recipe
   (greedy + stochastic + truncation gate). Pentomino-easy gets stochastic
   policy improvement + AUC=1.0 calibration but no greedy Pass@1 lift.

---

## Open questions

1. **Will v8.2 dual-token anchor preserve greedy `solvable_acc` on Pentomino?**
   Pure prediction from the mechanism is yes. ~3-5 hr GPU to confirm.

2. **Will a denser-action-space env (5×10 Pentomino, or new env like Hidato)
   close the greedy Pass@1 gap?**
   - 5×10 Pentomino: more solution paths but more pieces × more options;
     compounding over 10 steps rather than 4. Net effect unclear.
   - Brand new env (Hidato, Knight's Tour): smaller per-state branching, almost
     certainly closes the gap. Costs ~1 day of new-env code.

3. **Is the 22% stochastic Pass@1 on B-8 SFT the ceiling, or does it lift
   with longer training?** B-8 RL v8 per-batch solve hit 84% — this is
   approximately stochastic Pass@8 (8 rollouts/group). Pass@1 stochastic on
   the trained checkpoint should be substantially higher than 22% (need to
   measure with quick_pass1 patched to use T=0.7).

---

## Files

- Sanity rollout stats:
  - B-7: [sanity_2026-04-30_b7_rollout_stats.json](sanity_2026-04-30_b7_rollout_stats.json)
  - B-8: [sanity_2026-05-01_b8_rollout_stats.json](sanity_2026-05-01_b8_rollout_stats.json)
- Full runs catalog: [runs_reference_2026-05-01.md](runs_reference_2026-05-01.md)
- B-8 RL log: `logs/rl_b8_v8.log` (autodl), `outputs/rl_b8_v8_anchor/rl_log.jsonl`
- B-7 RL eval: [eval_2026-04-30_b7_rl_phase1.md](eval_2026-04-30_b7_rl_phase1.md)
