# Phase 2 Truncation Experiment — Full Results (2026-05-01)

Combines Options A and B and the clean re-eval that disambiguates training-time
quality cost from eval-time gate side effects. Supersedes
[eval_2026-05-01_truncation_option_a.md](eval_2026-05-01_truncation_option_a.md)
as the canonical results doc.

## TL;DR

**On Sudoku 4×4 with the v8-anchor checkpoint and τ=0.99:**

| metric | OFF (truncation) | ON (truncation) | Δ |
|---|---|---|---|
| **Step time** (avg over 50 RL steps) | 64.81 s | 52.27 s | **−19.4%** |
| **Tokens / step** | 16,195 | 12,402 | −23.4% |
| Mean rollout length | 4.55 steps | 3.49 steps | −23.3% |
| Truncated rollouts | 0 / 1,600 | 881 / 1,600 (55.1%) | — |
| **Pass@1 greedy (clean re-eval)** | **53.3%** (16/30) | **43.3%** (13/30) | **−10.0pp** |
| solvable_acc (clean re-eval) | 0.522 | 0.475 | −0.05 |
| bp_recall (clean re-eval) | 1.000 | 1.000 | unchanged ✓ |

**Headline tradeoff: 19% wall-time savings + 23% token savings, at a cost of 10pp
Pass@1.** bp_recall preserved at 1.0 — the truncation gate never misses a true
doom state, exactly the property the v8 anchor was designed to ensure.

---

## Setup

| Field | Value |
|---|---|
| Source checkpoint | `outputs/rl_b5_phase3_v8_anchor/final` (Sudoku v8 anchor; baseline Pass@1 ~50%) |
| Env | Sudoku 4×4 easy |
| Reward | v8 (= v7 + viability-tag KL anchor at coef 0.5) |
| LR / KL coef | 1e-5 / 0.05 |
| Seed | 42 (same in both conditions for apples-to-apples) |
| Truncation threshold τ | 0.99 (truncate if `logp(false_token) > log(0.99)`) |

τ=0.99 was selected from a threshold sweep on the v8 checkpoint:
`P(false)|GT=False mean 0.997, median 1.000; P(false)|GT=True mean 0.959`.
ROC AUC = 0.949. τ=0.99 places the gate inside the gap between class means.

---

## Option A — rollout-only experiment (10 steps × 2 conditions)

| Metric | OFF | ON | Δ |
|---|---|---|---|
| Mean rollout time | 15.16 s | 14.10 s | −7.0% |
| Mean tokens / step | 16,094 | 12,562 | −21.9% |
| Mean rollout length | 4.50 | 3.52 | −21.8% |
| Truncated rollouts | 0 / 320 | 173 / 320 (54.1%) | — |
| Per-batch solve rate (mean) | ~28% | ~33% | preserved |

Option A only ran 10 RL steps and therefore measured *rollout-phase* savings
without amortizing PPO update savings. Rollout-phase wall savings were
**−7.0%** even though token savings were **−21.9%**, because batched-rollout
parallelism caps savings at the longest-surviving rollout per turn.

---

## Option B — full RL training (50 steps × 2 conditions)

50 RL steps from the v8 anchor checkpoint, eval_every=25 to track Pass@1
trajectory. PPO updates happen as in real RL training.

| Metric | OFF | ON | Δ |
|---|---|---|---|
| Mean **step time** | 64.81 s | **52.27 s** | **−19.4%** |
| Mean rollout time | 15.16 s | 14.41 s | −5.0% |
| **PPO update time (= step − rollout)** | 49.65 s | 37.86 s | **−23.7%** |
| Mean tokens / step | 16,195 | 12,402 | −23.4% |
| Mean rollout length | 4.55 | 3.49 | −23.3% |
| Total wall (50 steps) | 3,240 s (54 min) | 2,613 s (44 min) | **−10 min** |
| Truncated rollouts | 0 / 1,600 | 881 / 1,600 (55.1%) | — |

The big finding from Option B that Option A missed: **PPO update phase saves
even more than rollout phase** (−24% vs −5%) because shorter rollouts mean
fewer tokens to compute logprobs over during the update. Combined: −19% step
time. This is the right number for the paper's compute-savings claim.

### Pass@1 trajectory under each condition (with eval-side truncation effects)

| eval @ step | OFF Pass@1 | OFF solvable_acc | OFF bp_recall | ON Pass@1 | ON solvable_acc | ON bp_recall |
|---|---|---|---|---|---|---|
| 0 (init) | 50.0% | 0.509 | 1.000 | **30.0%** | **0.801** | 1.000 |
| 25 | 53.3% | 0.481 | 1.000 | 23.3% | 0.769 | 1.000 |
| 50 (final) | 53.3% | 0.522 | 1.000 | 20.0% | 0.798 | 1.000 |

**The "trajectory" looks like a Pass@1 collapse on ON, but it's misleading.**
Both runs start from the *same* v8 checkpoint, yet step-0 Pass@1 differs
(50% vs 30%). That's because `quick_pass1()` runs through the same
`do_rollout()` code that has the truncation gate wired in — so the eval
itself was being truncated. Under truncation_mode=conservative, eval-time
greedy rollouts get killed whenever the model says False with high
confidence, killing some recoverable rollouts and underestimating the
trained policy's true Pass@1. Note also: solvable_acc actually IMPROVED
under ON (0.5 → 0.8) because the eval state distribution shifted to states
the model is more confident about (truncated rollouts are removed from the
denominator).

The bias is consistent across ON's eval points — shows ~30 pp underestimate
of the trained policy's Pass@1.

### Clean re-eval (Pass@1 disambiguated)

To get the trained policy's *actual* Pass@1, we re-evaluated the ON-final
checkpoint with `truncation_mode=off` on the same 30-puzzle eval set:

| Metric | OFF-final (clean) | ON-final (clean) | Δ |
|---|---|---|---|
| Pass@1 | 53.3% | **43.3%** | **−10.0 pp** |
| solvable_acc | 0.522 | 0.475 | −0.05 |
| bp_recall | 1.000 | 1.000 | unchanged |

So the actual training cost of v8 + truncation training is **−10pp Pass@1**,
not the −33pp the eval-during-training table suggested. That's a meaningful
but recoverable cost.

Reproduce the re-eval:
```bash
ssh autodl2 'cd /root/autodl-tmp/world_model_termination_spa && \
    bash scripts/_run_with_env.sh /root/miniconda3/bin/python -u src/training/rl_trainer_v6.py \
        --env sudoku --grid-size 4 --difficulty easy \
        --sft-checkpoint outputs/trunc_exp_b_on_tau0.99/final \
        --output-dir outputs/trunc_exp_b_on_reeval_clean \
        --n-total-steps 1 --eval-every 1 \
        --reward-version v8 --viability-kl-coef 0.5 --seed 42 \
        --truncation-mode off'
```

---

## Methodology fix going forward (hygiene only — does NOT close the −10pp)

The eval-during-training conflation is a real bug we should fix in the trainer:
`quick_pass1()` should always run with `truncation_mode=off`, regardless of the
training mode. The training rollouts get the gate; the eval should always be a
clean measurement.

**Important**: this is hygiene. The headline −10pp Pass@1 number above already
comes from a clean re-eval (gate forced off), so fixing `quick_pass1()` will
*not* change the −10pp — it just makes future training-log Pass@1 numbers
trustworthy without an extra re-eval pass. The actual −10pp is a real
training-time degradation in the gate-on RL'd checkpoint vs the gate-off RL'd
checkpoint; the fix for *that* is recipe-level (see "Open questions" §3-§4).

Suggested patch (one-line guard inside `quick_pass1`):
```python
old_trunc = cfg.truncation_mode
cfg.truncation_mode = "off"  # eval never truncates
try:
    ...  # existing eval logic
finally:
    cfg.truncation_mode = old_trunc
```

---

## Paper figure

```
Table N — Compute savings via <solvable>-based truncation gate (Sudoku 4×4)

                          Truncation     Truncation
Metric                      OFF            ON (τ=0.99)       Δ
─────────────────────────────────────────────────────────────────
Per-step training time    64.8 s          52.3 s          −19.4%
  Rollout phase           15.2 s          14.4 s           −5.0%
  PPO update phase        49.6 s          37.9 s          −23.7%
Tokens / step             16,195          12,402          −23.4%
Mean rollout length       4.55 steps      3.49 steps      −23.3%
% rollouts truncated      0%              55.1%           +55pp

Pass@1 (greedy, clean)    53.3%           43.3%           −10.0 pp
solvable_acc              0.522           0.475           −0.047
bp_recall                 1.000           1.000           preserved
```

This is the table for the paper's Phase 2 truncation section.

---

## τ-sweep finding (2026-05-01) — bimodal confidence

After Option B, we ran a τ-sweep at {0.95, 0.99, 0.999, 0.9999} on the v8
anchor checkpoint to characterize the savings/quality Pareto frontier. Result:

| condition | rollout_time | tokens/step | mean_len | truncated | per-batch solve |
|---|---|---|---|---|---|
| OFF (baseline) | 15.51 s | 16,014 | 4.48 | 0 / 320 (0.0%) | 40.3% |
| τ=0.95 | 14.36 s | 12,575 | 3.52 | 173 / 320 (54.1%) | 33.1% |
| τ=0.99 | 14.15 s | 12,483 | 3.50 | 177 / 320 (55.3%) | 32.2% |
| τ=0.999 | 14.33 s | 12,529 | 3.51 | 176 / 320 (55.0%) | 32.2% |
| τ=0.9999 | 14.25 s | 12,416 | 3.48 | 178 / 320 (55.6%) | 32.8% |

**All τ values from 0.95 → 0.9999 give nearly identical results** — ~55%
truncation rate, ~22% token savings, ~7-8 pp solve cost across the board.

The reason: the v8 anchor checkpoint's `<solvable>=False` predictions are
**bimodal in confidence**. When the model says False, it's almost always
> 99.99% confident (logp(false) > log(0.9999)). There's no middle confidence
band where τ tuning would matter. This is a side effect of the v8
single-token anchor mechanism (preserves logp of sampled token, but the
unsampled token's logp drifts to extreme values).

**Implication**: τ tuning alone won't reduce the −10pp Pass@1 cost. The
remedies are either (a) a stronger calibration mechanism (v8.2 dual-token
anchor — implemented, pending RL run), or (b) a different gate condition
that doesn't depend solely on the bimodal confidence — see next section.

---

## Trajectory-position-aware truncation gate (`--truncation-min-step`)

Added 2026-05-01 as an alternative dimension to gate on (since τ alone
doesn't fine-tune): **only fire the truncation gate after a rollout has
accumulated some minimum number of steps**. Hypothesis: rollouts that look
doomed at step 1 are sometimes recoverable; truncating only at step ≥ 3
lets those rollouts run a bit longer before the gate kicks in.

### Implementation

- New config field [`RLConfig.truncation_min_step: int = 0`](../src/training/rl_trainer_v6.py)
  (default 0 = current behavior, fire on any step).
- Both rollout paths (`do_rollout` and `do_rollouts_batched`) check
  `len(steps) >= cfg.truncation_min_step` before firing the gate.
- New CLI flag `--truncation-min-step N` (composable with any reward version
  and with `--truncation-mode conservative`).

### Min-step sweep experiment

Sweeps `min_step ∈ {0, 1, 2, 3, 4}` at fixed τ=0.99, 10 rollout-collection
steps each, on the v8 anchor checkpoint.
Launcher: [scripts/run_truncation_min_step_sweep.sh](../scripts/run_truncation_min_step_sweep.sh).

### Results (2026-05-01)

Aggregate stats reported by the trainer at end of 10-step rollout collection
(`n=320` rollouts per setting, batch=4 puzzles × group=8):

| min_step | mean_len | tokens/step | truncated/320 | trunc % | per-batch solve | rollout time |
|---------:|---------:|------------:|--------------:|--------:|----------------:|-------------:|
| 0        | 3.48     | 12,415      | 176           | 55.0%   | 32.8%           | 13.98s       |
| 1        | 3.51     | 12,529      | 176           | 55.0%   | 32.5%           | 14.24s       |
| 2        | 3.86     | 13,766      | 116           | 36.2%   | 32.5%           | 14.71s       |
| 3        | 4.16     | 14,841      | 54            | 16.9%   | 31.2%           | 14.89s       |
| 4        | 4.28     | 15,268      | 39            | 12.2%   | 32.2%           | 14.66s       |

**Findings**:

1. **`min_step=0` and `min_step=1` are identical** (55.0% truncation, ~12.5k
   tokens/step). The gate never actually fires at step 0 — at the very first
   prediction the model never says ">false" with τ ≥ 0.99 confidence
   (consistent with a fresh board having no committed moves to flag as doom).

2. **Raising `min_step` monotonically reduces truncation rate**: 55% → 36% →
   17% → 12% as we delay the gate from step 1 → 2 → 3 → 4. Tokens/step rises
   correspondingly (+11% / +19% / +23%). The gate is firing predominantly
   at steps 1–2.

3. **Per-batch in-rollout solve rate is essentially flat (31–33%) across
   the entire sweep**. The truncation gate is saving compute without
   reducing the fraction of solvable puzzles that get solved during
   training. Disabling the gate at later steps just keeps already-doomed
   rollouts running longer — it does *not* recover puzzles that were
   prematurely killed.

4. **Min-step sweep proves a weaker thing than originally claimed.**
   - The sweep shows per-batch solve rate (1-of-8 stochastic) flat across
     `min_step ∈ {0..4}`. This means the gate's false-positive rate is
     *low enough that 8 stochastic samples per puzzle absorb it as
     noise* — not that the gate is 100% precise.
   - Crucially, **eval-time Pass@1 is 1-of-1 greedy**. No recovery margin.
     Even a 7–8% per-firing FP rate would look flat in per-batch (because
     1 of 8 stochastic rollouts always survives) but cost ~15pp on greedy
     Pass@1 — which is roughly what we see (the v8 anchor checkpoint
     shows 50% → 30% Pass@1, gate-OFF eval → gate-ON eval, [§ Pass@1
     trajectory under each condition](#pass1-trajectory-under-each-condition-with-eval-side-truncation-effects)
     line "0 (init)").
   - So the gate is NOT 100% accurate; the min-step sweep just doesn't
     have the resolution to detect its FP rate. The per-batch metric is
     forgiving by 8× sampling; Pass@1 is not.

   **The −10pp Pass@1 cost (clean re-eval) is still a real training-time
   degradation in the trained model itself**, not an eval artifact. It
   comes from comparing **two distinct RL-trained checkpoints**
   (`trunc_exp_b_off/final` vs `trunc_exp_b_on_tau0.99/final`), both
   evaluated cleanly with `truncation_mode=off`.
   - However, the −10pp Pass@1 number ([§ Clean re-eval](#clean-re-eval-pass1-disambiguated))
     comes from comparing the **two distinct RL-trained checkpoints**
     (`trunc_exp_b_off/final` vs `trunc_exp_b_on_tau0.99/final`), both
     evaluated cleanly with `truncation_mode=off`. So the gap is real
     degradation in the *trained model itself*, not an eval-measurement
     artifact. The min-step sweep narrows the search but does not pin
     down the actual mechanism.
   - **Remaining candidate mechanisms** (none disproven yet):
     1. *Truncated reward signal* — gate-on rollouts terminate before
        seeing their true return; bias against "keep going".
     2. *PPO advantage estimation on truncated trajectories* — partial
        returns underestimate value for would-have-solved cases.
     3. *Effective training-distribution shift* — gate kills rollouts
        before the late-trajectory states are visited; the policy gets
        less gradient signal on long-horizon recovery patterns.
     4. *Greedy/stochastic decoupling* — per-batch (1-of-8 stochastic)
        is forgiving; greedy Pass@1 (1-of-1) is what actually regresses.
        These mechanisms could degrade greedy without showing up in
        per-batch rollout solve rate.

5. **Optimal min-step setting (compute savings only): `min_step=0`** —
   55% truncation rate, ~22% token reduction. There is no quality reason
   to delay the gate within the min-step axis itself; that lever doesn't
   touch the −10pp Pass@1 cost.

Caveat: 4 of 5 sweep runs failed to checkpoint at the very end with
"No space left on device" — autodl2 was at the disk-full edge. The
in-rollout statistics are unaffected (computed before save), so the
table above is the real result; the sweep is functionally complete.

### Combined with v8.2 anchor

Independent axis: v8.2 (dual-token anchor) addresses the *bimodality*
of confidence directly by anchoring both `>true` and `>false` logprobs.
Min-step gates orthogonally on *trajectory position*. The min-step sweep
above shows there is no quality benefit to delaying the gate, so the
v8.2 + min-step combination is no longer the priority — v8.2 alone with
`min_step=0` is the natural next experiment.

### Status (2026-05-01)

- [x] Trainer code patched (`truncation_min_step` config + CLI flag)
- [x] Sweep launcher written (`run_truncation_min_step_sweep.sh`)
- [x] Sweep results collected (autodl2)

---

## Open questions for follow-up

The −10pp Pass@1 cost between gate-on and gate-off RL'd checkpoints is a
real training-time degradation in the trained model. Min-step ruled out one
mechanism (in-training kills of recoverable rollouts); the other candidate
mechanisms (truncated reward signal, PPO advantage bias on partial returns,
training-distribution shift, greedy/stochastic decoupling) are all
unresolved.

1. **τ-sweep** (DONE — bimodal confidence; tuning τ alone doesn't move the
   gate's behavior).
2. **min-step sweep** (DONE — `min_step=0` optimal; in-training kills
   are well-targeted, but this lever doesn't touch the −10pp).
3. **v8.2 dual-token anchor** (in flight — autodl2). Original motivation
   was eval-time bimodality; given the corrected interpretation, v8.2's
   relevance to the −10pp is now via *training-time* effects: better
   `<solvable>` calibration during training → fewer false-positive gate
   fires during rollout collection → less reward-signal contamination.
   Whether this fully closes the −10pp is the open empirical question.
4. **"Train without gate, deploy with gate" experiment** (NEW priority,
   not yet run). If the −10pp is purely a training-time artifact, then a
   model RL'd *without* the gate (gate-off arm checkpoint already exists:
   `outputs/trunc_exp_b_off/final/`) and *deployed* with the gate at
   inference should give the clean headline: "compute saved at deployment,
   no Pass@1 cost." Concretely: take the OFF-final checkpoint, run a clean
   eval with `truncation_mode=off` AND with `truncation_mode=conservative
   --truncation-threshold 0.99`, compare per-puzzle tokens/seconds and
   Pass@1. ~30 min GPU. This is the cheapest path to a defensible "the
   gate works at deployment" claim, independent of v8.2.
5. **Eval-during-training fix**: patch `quick_pass1()` to always run
   with `truncation_mode=off` — see [Methodology fix going forward](#methodology-fix-going-forward-hygiene-only--does-not-close-the-10pp)
   above. Hygiene-only, not a fix for the −10pp. ~10 min code.
6. **Larger N for τ-sweep / Pass@1 statistics**: 30-puzzle eval has
   noticeable variance (one puzzle = 3.3pp). For paper-quality numbers,
   eval at 100+ puzzles. ~30 min more GPU per condition.

---

## Files

- Eval doc (this file): [eval_2026-05-01_truncation_full.md](eval_2026-05-01_truncation_full.md)
- Plan doc: [plan_2026-05-01_truncation_experiment.md](plan_2026-05-01_truncation_experiment.md)
- Trainer code: [src/training/rl_trainer_v6.py](../src/training/rl_trainer_v6.py) — gate at do_rollouts_batched and do_rollout, CLI flags `--truncation-mode` / `--truncation-threshold`.
- Launchers:
  - Option A: [scripts/run_truncation_exp_option_a.sh](../scripts/run_truncation_exp_option_a.sh)
  - Option B: [scripts/run_truncation_exp_option_b.sh](../scripts/run_truncation_exp_option_b.sh)
- Checkpoints (autodl2): `outputs/trunc_exp_b_off/final/`, `outputs/trunc_exp_b_on_tau0.99/final/`
- JSONLs (synced local): `/tmp/trunc_b_off.jsonl`, `/tmp/trunc_b_on.jsonl`, `/tmp/trunc_b_on_reeval.jsonl`
