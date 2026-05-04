# RL Approach Plan (2026-04-29)

Revisits [future_steps.md](future_steps.md) NEAR-1.4 (RL truncation strategy) and NEAR-2 (RL training itself), updated for what we now know after [B-5](eval_2026-04-29_b5_4x4_spa_replication.md): the recipe works on 4×4, but the resulting model is uncalibrated and biased toward predicting `<solvable>=false`. That bias profile materially changes the reward shape and truncation gating we should use.

This doc supersedes future_steps.md NEAR-1.4 / NEAR-2 reward-shape language. The truncation phasing (Mechanisms 1/2/3) is unchanged in principle.

---

## 1. Where we are (post-B-5)

What we have:
- A working 4×4 SFT checkpoint at `outputs/sft_sudoku_4x4_minimal_b5_spa_hparams/final` (autodl2). ROC AUC = 0.726, format compliance 100%.
- An RL trainer skeleton in [`src/training/rl_trainer.py`](../src/training/rl_trainer.py) — `LiveEnvTerminationRLTrainer` + `compute_termination_reward_v2`. Currently v3-era: depends on `verl`, rewards `<breaking_point>` rather than `<solvable>`, and counts the dropped tags (`<steps_left>`, `<terminate_prob>`) in format compliance. Needs surgery before use.
- `vLLM` un-banned (SPEC v5) — the planned rollout accelerator.

What we don't have:
- A working 9×9 SFT checkpoint. B-4 confirmed 9×9 is too hard for Qwen-1.5B SFT alone. **9×9 RL is blocked** until B-6 lands (9×9 SPA-scale data → SFT on autodl1, currently ~9h out).
- A `<solvable>`-aware reward function (current is `<breaking_point>`-flavored).
- Pass@1 / Pass@k eval as a primary metric (we have it in `evaluate_rl.py` but haven't anchored against SPA).

So the RL plan below targets the **B-5 4×4 checkpoint** as the experimental subject. 9×9 will fork later if B-6 produces anything trainable.

### B-5's calibration profile (the constraint that shapes everything below)

From [eval_2026-04-29_b5_4x4_spa_replication.md](eval_2026-04-29_b5_4x4_spa_replication.md):

| Class | n | mean P(true) | median |
|---|---|---|---|
| GT=True | 100 | 0.045 | 0.032 |
| GT=False | 200 | 0.022 | 0.001 |

| Metric (greedy at τ=0.5) | Value |
|---|---|
| Greedy "True" frequency | 131/300 = 44% |
| Greedy precision on True | 24.4% |
| Greedy recall on True | 32.0% |
| Logprob precision on **False** at τ=0.10 | 68.5% |
| Logprob spec on False at τ=0.10 | 95.5% |

Three operationally important facts:
1. **The model strongly favors `false` in logprob terms** (mean P(true) ≈ 0.04 even on actually-solvable). So sampling-based rollouts will produce mostly-False outputs unless temperature is high.
2. **Precision on False is only ~68%** — when the model says "doomed," it's wrong about 1 in 3 times. This is far below the threshold (≥90%) at which hard truncation would be safe.
3. **Pass@1 = 0/50 (0%) on 4×4 puzzles** (baseline ran 2026-04-29 evening). The model has learned `<solvable>` discrimination but cannot actually solve the puzzles end-to-end. This is what motivates v6 (§3) — pure `<solvable>` correctness reward wouldn't lift Pass@1; we need a task-success component.

---

## 2. RL training plan — phased

### Phase 0: Surgery on `rl_trainer.py` (~1 day, no GPU)

Required changes before any RL run:
1. Replace `compute_termination_reward_v2`'s BP-targeted asymmetric rewards with `<solvable>`-targeted ones. Current code rewards `breaking_point_pred` vs `is_breaking_point_gt`; needs to reward `solvable_pred` vs `is_solvable_gt` with the asymmetric shape.
2. Update `required_tags` from `["<solvable>", "<breaking_point>", "<steps_left>", "<terminate_prob>", "<answer>"]` → `["<observation>", "<prediction>", "<solvable>", "<answer>"]` (matching v4 minimal format).
3. Decide: keep verl, migrate to TRL, or write a thin custom PPO loop. **Recommendation:** thin custom PPO loop with vLLM for rollout sampling — TRL adds dependency weight we don't need, verl's distributed assumptions don't fit our single-GPU setup. ~300 LOC of trainer code.
4. Add a config flag `truncation_mode ∈ {'off', 'conservative'}` and a `truncation_threshold` parameter. Phase 1 runs use `'off'`. Phase 2 flips to `'conservative'`.

### Phase 1: Pure reward shaping, no truncation (~6–12 GPU-hours on autodl2)

Run RL on B-5 4×4 with the v6 reward (§3) and **no rollout truncation**. Goal: lift Pass@1 (currently 0%) AND improve `<solvable>` calibration via gradient feedback.

Per-batch protocol:
- Sample initial puzzle states from a fixed seed pool (50–100 distinct 4×4 easy puzzles). Resampled fresh each epoch.
- Per puzzle, generate **K = 8 multi-step rollouts** at temperature T = 0.7 with vLLM. Each rollout runs from `s_0` until the puzzle is solved, deadlocked, or hits `max_steps=12`.
- Score each rollout with v6 reward (sum of per-step `<solvable>`+format rewards + trajectory-success bonus).
- GRPO advantage = group-relative trajectory reward (subtract group mean across the 8 rollouts of the same starting puzzle).
- KL penalty against B-5 reference (β = 0.05 to start). Tighten if drift; loosen if stuck.

Schedule:
- 200–400 RL steps. Each step = one batch of (n_puzzles × 8) rollouts.
- Eval every 50 steps: (a) `<solvable>` AUC + precision-on-False at τ ∈ {0.10, 0.30, 0.50, 0.70, 0.90} (the existing logprob threshold sweep) and (b) Pass@1 / Pass@8 on 50 fixed-seed eval puzzles. Log format compliance.

Phase 1 success criteria:
- **Pass@1 lifts from 0% to ≥10%** (primary signal — RL is doing what it's supposed to do).
- AUC stays ≥ 0.726 (calibration must not regress).
- **Precision-on-False at some threshold ≥ 0.90** — gate to Phase 2 (truncation).
- Format compliance stays ≥ 99%.

### Phase 2: Conservative hard truncation (Mechanism 2 of NEAR-1.4)

Triggered only when Phase 1 reaches Prec(F) ≥ 0.90 at some τ*. Set the runtime truncation threshold τ_runtime slightly stricter than τ* (e.g., τ_runtime = max(τ*, 0.95)) for safety margin.

Truncation rule (per rollout step t):
```
if model_emit("<solvable>=false") and P(false) > τ_runtime:
    truncate rollout — set done=True, mark as "early-terminated"
    reward = 0 for the truncated portion (no synthetic reward)
```

Continue training; recheck Prec(F) every 50 steps. If Prec(F) drifts below 0.85, fall back to Phase 1 (truncation off) until it recovers.

This sequence is essentially NEAR-1.4 steps 2–5 from future_steps.md, with the τ* gate concretely specified at 0.90.

### Phase 3 (optional): Action filtering (Mechanism 3)

Only after Phase 2 is validated as a separate ablation. When the model emits an action with `<solvable>=false`, **resample** instead of truncating. Doesn't shorten the trajectory; spends compute searching for non-doom-inducing actions. Tested as ablation against Mechanism 2 to see whether truncate-vs-resample produces materially different Pass@1 / GPU-hour curves.

---

## 3. Reward shape — revised to v6 after Pass@1 baseline (2026-04-29 evening)

**Important amendment (2026-04-29).** Pass@1 baseline on B-5 came back at **0/50 (0%)** — the SFT model can't actually solve 4×4 puzzles even with 8 samples at T=0.7. AUC=0.726 on `<solvable>` is real, but it's a *prediction-quality* metric, not a *solving-ability* metric, and the two have come apart.

**Implication:** the v5 reward (`<solvable>` correctness only) would make RL push calibration but not Pass@1. SPA's reward includes task success — that's why their Pass@1 lifts. To match that on our project's headline claim (NEAR-1.5: early-termination compute savings only matter if RL produces solvers worth running), we need a two-component reward.

**v6 supersedes v5.** v5 numbers below are kept for traceability; v6 is the operative spec.

### 3.0 v6 reward — multi-step rollouts with per-step + end-of-trajectory components

Rollouts are now **multi-step** (full puzzle attempts), not single-step decisions. Each rollout runs from `s_0` until the puzzle is solved, deadlocked, or hits `max_steps=12` (4×4 has 6 empty cells, 2× safety margin).

```
For each step t in the rollout, applied to that step's last response token:
  step_reward[t] =
      + format_reward[t]              # +0.05 per of 4 required tags (max +0.2)
      + solvable_reward[t]            # see table below

End-of-trajectory bonus, applied to the final response token:
  trajectory_bonus =
      +10.0  if env.is_solved (puzzle complete, all cells filled correctly)
       0.0   otherwise (deadlocked OR max_steps hit)
```

**Per-step `<solvable>` correctness reward** (reduced from v5 since rewards now accumulate across steps in a trajectory):

| Outcome | v6 reward |
|---|---|
| TP: pred False, GT False *(caught doom this step)* | **+1.0** |
| FN: pred True,  GT False *(missed doom this step)* | **−0.7** |
| FP: pred False, GT True  *(spurious doom alarm)* | **−0.5** |
| TN: pred True,  GT True  *(correct salvation prediction)* | **+0.3** |

### 3.1 v6 reward landscape on a 6-step rollout (4×4 has 6 empty cells)

| Rollout type | Per-step sum | Trajectory bonus | Total |
|---|---|---|---|
| Perfect play (solves puzzle, all TN, all format good) | 6×(+0.3 + 0.2) = **+3.0** | +10.0 | **+13.0** |
| Solves but with poor `<solvable>` calibration (all FP) | 6×(−0.5 + 0.2) = **−1.8** | +10.0 | **+8.2** |
| Fails on step 3 with perfect doom detection (TN, TN, TP, TP, TP, TP) | 2×0.3 + 4×1.0 + 6×0.2 = **+5.8** | 0 | **+5.8** |
| Fails immediately, all FN (always says "true" on doomed states) | 6×(−0.7 + 0.2) = **−3.0** | 0 | **−3.0** |
| Always-False collapse on solvable rollout (all FP, fails too) | 6×(−0.5 + 0.2) = **−1.8** | 0 | **−1.8** |

Reward ordering:
- **Perfect play** dominates everything (+13.0)
- **Solving with bad calibration** still beats failing-with-perfect-calibration (+8.2 > +5.8) — task success is the dominant signal, but per-step calibration matters
- Margin between "solving but ignoring `<solvable>`" and "perfect play" is +4.8 — large enough that RL will push to fix calibration once solving is learned
- Margin between "failing-with-perfect-doom-detection" and "always-False collapse" is +5.8 vs −1.8 — keeps the calibration gradient strong

### 3.2 Why these specific magnitudes

- **Trajectory bonus (+10.0)**: dominates the per-step sum (max ~+3.0 for a 6-step success), but doesn't crush it. Reviewers who say "you just optimized for solve-rate" can be answered with "calibration is also rewarded, with a +4.8 margin per puzzle for getting both right."
- **Asymmetry preserved**: TP +1.0 > TN +0.3, FN −0.7 < FP −0.5 — catching doom is more rewarded than salvation, missing doom is more penalized than spurious doom. Same shape as v5, scaled down.
- **Format weight** is unchanged from v5 (+0.05/tag, max +0.2/step) — small enough to not compete with correctness but enough to maintain XML compliance pressure.

### 3.3 Old v5 reward (kept for reference; superseded by §3.0 above)

The original v5 single-step reward, now superseded:

```
TP (pred False, GT False):     +3.0
FN (pred True,  GT False):     -2.0
FP (pred False, GT True):      -1.0
TN (pred True,  GT True):      +0.5
Format: +0.05 per tag (max +0.2)
```

v5's reward landscape on a balanced single-step batch (100 solvable + 100 unsolvable):
- Always False: +200; Always True: -150; Perfect: +350. Slack: 150 reward units.

v5 was correct for single-step (s_t, a_t) RL but doesn't drive Pass@1 because rollouts in v5 are single-decision — there's no notion of "puzzle solved" at all. v6 fixes this by switching to multi-step rollouts and adding the trajectory-level success bonus.

### 3.1 Problem with the v4 shape from NEAR-2

NEAR-2 currently specifies:
```
<solvable>=false, GT false (TP):    +3.0
<solvable>=true,  GT false (FN):    -2.0
<solvable>=false, GT true  (FP):    -0.5
<solvable>=true,  GT true  (TN):     0   (or small +0.5)
Format: +0.1 per tag (v4: 4 tags = max +0.4)
```

On a balanced batch (100 solvable, 100 unsolvable), the reward landscape is:

| Policy | TP-units | FN-units | FP-units | TN-units | Total reward |
|---|---|---|---|---|---|
| Always False | 100×3 | 0 | 100×(-0.5) | 0 | **+250** |
| Always True | 0 | 100×(-2) | 0 | 0 | -200 |
| Perfect | 100×3 | 0 | 0 | 100×0 | +300 |
| B-5's current behavior (~95% False) | ~95×3 | 0 | ~95×(-0.5) | 0 | **+237** |

Always-False is at +250, perfect is +300 — only **50 reward units of slack** between collapse and perfection. Worse, B-5 already sits at +237 (collapsed-toward-False), so RL only sees +63 reward room left to "fix." With KL regularization fighting to keep the policy near B-5, the gradient pull toward perfection is weak. **Risk: Phase 1 plateaus at always-False instead of learning discrimination.**

### 3.2 Proposed v5 shape

Two changes:
- **Add small positive TN reward (+0.5)** — so correctly predicting `<solvable>=true` on a solvable state has positive signal. Preserves asymmetry (TP +3 ≫ TN +0.5) but breaks the always-False local optimum.
- **Increase FP penalty (-0.5 → -1.0)** — make spurious doom predictions more costly. Still less harsh than FN (-2.0) so catching doom remains the dominant priority.

```
<solvable>=false, GT false (TP):    +3.0
<solvable>=true,  GT false (FN):    -2.0
<solvable>=false, GT true  (FP):    -1.0    [was -0.5]
<solvable>=true,  GT true  (TN):    +0.5    [was 0]
Format: +0.05 per tag (4 tags = max +0.2; reduced so reward ≪ correctness signal)
```

Same balanced-batch landscape under v5:

| Policy | Total reward |
|---|---|
| Always False | 100×3 + 100×(-1) = **+200** |
| Always True | 100×(-2) + 100×(0.5) = **-150** |
| Perfect | 100×3 + 100×(0.5) = **+350** |
| B-5 current (~95% False) | ~95×3 + ~95×(-1) ≈ **+190** |

Slack between collapse and perfection: **+150 reward units (3× wider).** B-5's starting point is at +190 with +160 room to gain. Gradient signal toward learning is much stronger.

The asymmetry property is preserved: catching doom (TP +3.0) is worth more than correct salvation (TN +0.5), but salvation is no longer free. Doom-misclassification (FN -2.0) remains the worst outcome.

### 3.3 What balanced sampling does for the gradient

The reward landscape above assumes 100 solvable / 100 unsolvable per batch. If we sampled in proportion to the trajectory-natural rate (~33/67 solvable/unsolvable), always-False's reward goes UP relative to perfect (more unsolvable states being correctly identified) and the slack shrinks. **`LiveTrajectorySampler` must be configured to enforce 50/50 per-batch.** This is already implemented (per CLAUDE.md "balanced sampling: ~50% solvable / ~50% unsolvable per batch from live env") — needs to be verified active in the rl config.

### 3.4 Format compliance reward — reduced

Current v2 gives +0.1 per tag for 5 tags = +0.5 max. After dropping 3 tags (now 4 required: `<observation>`, `<prediction>`, `<solvable>`, `<answer>`), if we kept +0.1 per tag we'd have +0.4 max for format — comparable to TN +0.5, which is too much weight on a near-trivial behavior. Reducing to +0.05 per tag (max +0.2) keeps format compliance pressure but doesn't compete with the prediction-correctness signal.

---

## 4. What to build / change concretely

### Code changes (Phase 0)

| File | Change | Approx LOC |
|---|---|---|
| `src/training/rl_trainer_v6.py` (new) | Fresh trainer: v6 reward (§3), multi-step rollouts (puzzle from `s_0` to terminal/timeout), GRPO with KL penalty against ref, vLLM-backed sampler. Leave existing verl-based `rl_trainer.py` untouched. | ~400 lines |
| `src/training/rl_trainer_v6.py` (above) | Includes a `compute_v6_reward(rollout, env)` helper that walks the rollout, applies per-step `<solvable>` reward by checking each `(s_t, a_t)` against env oracle, plus end-of-trajectory bonus on `env.is_solved`. | (folded into above) |
| `src/training/rl_trainer_v6.py` (above) | `truncation_mode ∈ {'off', 'conservative'}` + `truncation_threshold` parameter, gating logic in the rollout loop. Phase 1 uses `'off'`. Phase 2 flips to `'conservative'`. | (folded into above) |
| `src/training/config/rl_sudoku_4x4_b5.yaml` | New config: B-5 reference checkpoint, vLLM rollout settings, group_size=8, T=0.7, β=0.05, n_puzzles_per_batch=8 (8 puzzles × 8 rollouts = 64 trajectories per RL step). | ~80 lines |
| `evaluate_rl.py` | Add `--track-precision-recall` mode that emits Prec(F)/Rec(F) sweep at τ ∈ {0.1, 0.3, 0.5, 0.7, 0.9} as JSON. Used by Phase 1 monitoring loop. Existing Pass@1 mode reused as-is. | ~50 lines |
| `scripts/run_sudoku_4x4_rl_v6_phase1.sh` | New launcher for Phase 1 RL on autodl2. | ~30 lines |

### Infra checks (one-shot)

- vLLM compatibility with Qwen2.5-1.5B-Instruct + the B-5 SFT checkpoint (`final/`). Should work; verify before the run. Test on autodl2 (the autodl1 GPU is occupied with 9×9 gen until B-6 starts).
- Memory budget: B-5 model in vLLM (3 GB) + training weights (~6 GB with Adam optimizer in mixed precision) + KV cache for 8 parallel rollouts × 1024 max_length on H800 80 GB. Should fit comfortably.
- HF Trainer is **not** appropriate here — RL needs custom rollout/scoring/PPO loop, HF Trainer assumes static datasets.

---

## 5. Observability — what to log per RL step

Mandatory:
- `reward/total`, `reward/format`, `reward/solvable_correctness` (sub-decomposed by TP/FN/FP/TN)
- `kl_to_ref` (KL divergence vs B-5 SFT reference)
- `entropy` (policy entropy at the `<solvable>` token specifically, separately from the rest of the response)
- Greedy-decode prediction frequency: `predictions/true_rate`, `predictions/false_rate` (for collapse detection)
- Format compliance %: should stay at 100% across the run

Every 50 steps:
- Full balanced eval (300 samples, same as B-5's eval): AUC + Prec(F)/Rec(F)/Spec/Acc at τ ∈ {0.1, 0.3, 0.5, 0.7, 0.9}
- Pass@1 on a fixed-seed 100-puzzle set (Phase 1 anti-goal: this should not drop)

Every 200 steps:
- Save a checkpoint
- Append a row to `runs_ledger_2026-04-29.md` with the eval delta vs SFT

---

## 6. Anti-goal trip-wires

Halt the run and investigate if any of:

| Trip-wire | Threshold | Reason |
|---|---|---|
| AUC drops below SFT baseline | < 0.726 (5% margin) | RL is destroying the SFT signal |
| Pass@1 drops materially | 5%+ below B-5's SFT Pass@1 | RL is hurting agent performance — termination is over-fired |
| `predictions/false_rate` exceeds 0.95 | sustained for 100+ steps | Always-False collapse — reward shape is wrong or β too low |
| Format compliance drops | < 99% | Model is breaking the XML scaffold under reward pressure |
| KL to ref > 1.0 | sustained | Policy has drifted too far — the gradient is not correcting |

For each trip-wire, the recovery is: rollback to the last checkpoint, adjust the offending hparam (KL β, temperature, reward weights), restart.

---

## 7. Open questions still

- **`<solvable>` confidence extraction during rollouts.** When the model emits a sampled token at the `<solvable>` position, do we use:
  - (a) the per-step token logit (single-token probability at that exact decoding step), or
  - (b) a separate teacher-forced forward pass after the rollout to extract calibrated P(true)?
  - (a) is cheaper but biased by sampling temperature; (b) is what `evaluate_rl.py --metric solvable-logprob` already uses. **Recommendation: use (b) for Phase 2 truncation gating, (a) for in-rollout sampling.**
- **Group size**: 8 rollouts per state was the SPA paper's default. Smaller groups (4) save compute; larger (16) reduce variance. Stick with 8 unless we see high variance.
- **KL coefficient β schedule**: fixed 0.05 vs cosine-decay vs adaptive (target a fixed KL value)? Start fixed, switch to adaptive if it plateaus or runs away.
- **Pass@1 anchoring**: SPA's Sokoban Pass@1 baseline = 25.6%. Sudoku 4×4 number isn't reported in the paper main tables. We can't directly compare without running Pass@1 ourselves on B-5 (which is in the repo as `evaluate_rl.py --metric pass-k`, but never run yet) — should be done before Phase 1 starts so we have a baseline to compare against.

---

## 8. Concrete next actions

In order, with rough effort estimates:

1. **(now, no GPU)** Run Pass@1 on B-5 — anchors against SPA. Uses existing `evaluate_rl.py --metric pass-k`. ~30 min on autodl2.
2. **(today, no GPU)** Phase 0 code surgery: update reward to v5, gate truncation, write `rl_trainer_v5.py`. ~1 day.
3. **(today, no GPU)** Write Phase 1 config + launcher script. ~30 min.
4. **(when ready)** Smoke test: 50 RL steps on B-5 with the new trainer. Verify no infra blowup, log shape correct, KL behaves. ~30 min on autodl2.
5. **(after smoke test passes)** Phase 1 RL run: 200–400 steps, watching the trip-wires + 50-step eval cadence. ~6–12 GPU-hours on autodl2. **Decide Phase 2 entry based on Prec(F) gate.**

Phase 1 may overlap with B-6 (9×9 SFT after 9×9 SPA-scale gen) running on autodl1 — they're on different clouds with different data, so no contention. If B-6 produces a usable 9×9 checkpoint, fork the RL plan: same Phase 0/1/2 sequence on the 9×9 model, separate config.

---

## 9. What this leaves to NEAR-1.5

Once Phase 2 is running with truncation enabled, NEAR-1.5 (compute-budget experiment) is just a controlled comparison: run Phase 1 (no truncation) and Phase 2 (truncation at varying τ ∈ {0.5, 0.7, 0.9, 0.95}) for the same number of training samples and measure wall-clock to a target Pass@1. The infrastructure is the same; the experiment is just a sweep over `truncation_threshold`. Effort: 1–2 days of GPU time, 0 new code.
