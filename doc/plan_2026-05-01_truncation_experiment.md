# Plan — Phase 2 Truncation Experiment on Sudoku v8 Anchor

This is the experiment that quantifies the project's headline value claim:
**calibrated `<solvable>` predictions enable compute savings via early termination
of doomed rollouts during agentic RL.** Without a measured number, this is a
hypothesis; with it, it's a paper figure.

The v8 anchor checkpoint at `outputs/rl_b5_phase3_v8_anchor/final` is the first
asset we have where the prerequisites are met:
- `bp_recall = 1.000` — never miss a real doom state.
- `solvable_acc = 0.509`, but `Prec(False)` is the gate metric (not aggregate
  acc); needs to be measured at the chosen τ.
- `Pass@1 = 50%` — the baseline for "don't regress quality" comparisons.

This document plans the work end-to-end: code change, threshold selection,
two measurement designs (A and B), and what each gives the paper.

---

## 1. The code change (shared by A and B)

The trainer at [src/training/rl_trainer_v6.py:271-274](../src/training/rl_trainer_v6.py#L271)
already has a hook:

```python
if cfg.truncation_mode == "conservative" and pred is False:
    # NOTE: needs P(false) > truncation_threshold from a teacher-forced pass; not wired up yet.
    pass
```

The `pass` is a no-op. Wire it up:

```python
if cfg.truncation_mode == "conservative" and pred is False:
    # The model just emitted "<solvable>false" (or "<viability>false"). We have
    # `step.viability_token_positions` from the rollout-time anchor, and the
    # rollout-time logp at that position is in step.old_logps. If logp(>false)
    # is high enough, kill the rollout to save compute.
    via_pos = step.viability_token_positions
    if via_pos and step.old_logps:
        false_logp = step.old_logps[via_pos[0]]   # logp of the sampled "false" token
        if false_logp > math.log(cfg.truncation_threshold):
            r["alive"] = False
            r["truncated_early"] = True
            break  # rollout ends here
```

Two tracking fields to add for the experiment metric:

```python
# in StepRecord:
truncated_early: bool = False

# in Rollout: a counter of truncated rollouts in the batch is enough.
```

Plus a metric counter in the rollout loop output, so we report:
- `n_rollouts_total`
- `n_rollouts_truncated_early`
- `mean_steps_per_rollout` (with vs without truncation)
- `total_response_tokens` (with vs without)
- `wall_time_s` (with vs without)

**Effort**: ~1 hour code + smoke test.

---

## 2. Threshold selection (τ)

Run the existing `evaluate_rl.py --metric solvable-logprob` on the v8 checkpoint
to compute Prec(False) / Rec(False) at various τ. Pick the smallest τ where
Prec(False) ≥ 0.90 (the gate from
[doc/plan_2026-04-29_rl_approach.md](plan_2026-04-29_rl_approach.md) Phase 2).

Effort: ~10 min — script already exists.

**Sample command**:
```bash
python evaluate_rl.py \
    --rl-path outputs/rl_b5_phase3_v8_anchor/final \
    --metric solvable-logprob \
    --tag-name solvable \
    --grid-size 4 --difficulty easy \
    --n-solvable 200 --n-unsolvable 200 \
    --threshold-sweep 0.05 0.10 0.20 0.30 0.50 0.70 0.90
```

We pick whichever τ is smallest while still satisfying Prec(False) ≥ 0.90. That
becomes `cfg.truncation_threshold` for the experiment.

---

## 3. Option A — rollout-collection-only experiment (cheaper, faster)

**What**: Generate rollouts on the v8 checkpoint with truncation OFF and ON,
without doing PPO updates. Compare wall time, tokens, and Pass@N.

```bash
# Run 1 (truncation OFF):
python -m src.training.rl_trainer_v6 \
    --env sudoku --grid-size 4 --difficulty easy \
    --sft-checkpoint outputs/rl_b5_phase3_v8_anchor/final \
    --output-dir outputs/trunc_exp_off \
    --n-total-steps 25 \
    --reward-version v8 \
    --truncation-mode off \
    --eval-every 9999    # disable eval; we only want rollout timing

# Run 2 (truncation ON, τ from threshold sweep):
python -m src.training.rl_trainer_v6 \
    --env sudoku --grid-size 4 --difficulty easy \
    --sft-checkpoint outputs/rl_b5_phase3_v8_anchor/final \
    --output-dir outputs/trunc_exp_on \
    --n-total-steps 25 \
    --reward-version v8 \
    --truncation-mode conservative \
    --truncation-threshold 0.10  # or whatever τ produces Prec(F) >= 0.90
```

**Note**: 25 steps × 32 rollouts/step = 800 rollouts per condition. Same seeds
mean rollouts unfold identically until truncation kicks in.

**Pros**:
- Fast: ~30 min × 2 = 1 hour GPU.
- Isolates rollout-generation savings cleanly.
- Same seeds → directly comparable trajectories.

**Cons**:
- Doesn't measure PPO update savings (which also depends on total tokens).
- Not a full RL training run, so not exactly the agentic-RL setting.

**Output metric**: 
```
Compute savings (rollout-gen only) = (T_off − T_on) / T_off
Quality preserved = same per-rollout reward distribution within sampling noise
```

Effort: ~2 hr total (1 hr code + 0.5 hr setup + 1 hr GPU).

---

## 4. Option B — full RL training comparison (more direct)

**What**: Run actual short RL training (50 steps) with truncation OFF, then 50
steps with truncation ON, both starting from the v8 checkpoint. Compare end-to-end
wall time and final Pass@1.

```bash
# Run 1 (truncation OFF):
bash scripts/run_truncation_exp_option_b.sh    # OFF arm    # 50 steps, otherwise like Run A

# Run 2 (truncation ON):
bash scripts/run_truncation_exp_option_b.sh    # ON arm     # same but truncation_mode=conservative
```

**Pros**:
- This *is* the paper claim — agentic RL with calibrated truncation.
- Captures both rollout savings AND PPO update savings.
- Directly compares Pass@1 trajectories (does truncation hurt learning?).

**Cons**:
- Slower: ~3-4 hr GPU per condition ≈ 6-8 hr total.
- More variance from gradient noise; needs more steps to be conclusive.

**Output metrics**:
```
Per-step wall time (with vs without)
Per-step token count (with vs without)
Pass@1 trajectory (with vs without) — should NOT regress
Total wall time savings = (T_off_total − T_on_total) / T_off_total
```

Effort: ~8 hr total (1 hr code shared with A + 6-8 hr GPU + analysis).

---

## 5. Recommended sequencing

1. Implement the code change (~1 hr) — shared across A and B.
2. Threshold sweep on v8 checkpoint (~10 min) — picks τ.
3. **Option A first** (1 hr GPU). Fast, gives a real number on rollout savings.
4. If A's savings are meaningful (>20%) and Pass@N is preserved → run **Option B** (~6 hr GPU) for the apples-to-apples paper figure.
5. If A shows no savings or quality regression → debug before committing to B.

This gates the expensive option on the cheap one.

---

## 6. Data dependencies

- **v8 anchor checkpoint** (`outputs/rl_b5_phase3_v8_anchor/final`):
  on local + on autodl2 (synced 2026-05-01). Required for both A and B.
- **B-5 SFT checkpoint** (used as `ref_policy` in PPO under option B):
  on local + on autodl2. Already there.
- **Trainer code with truncation gate wired up**: pending implementation.

If autodl2 is turned off after sync, autodl can run the experiment by syncing
the v8 weights up from local first.

---

## 7. What numbers to report

For the paper:

```
Table N — Compute savings via <solvable>-based truncation (Sudoku 4×4)

                         Truncation  Truncation
Metric                     OFF        ON (τ=??)     Δ
─────────────────────────────────────────────────────
Wall time / step          35.0 s     22.0 s       −37%
Tokens / step             1,500     940           −37%
Mean rollout length       4.2 steps  2.8 steps    −33%
% rollouts truncated      0%         55%
Pass@1 (final eval)       50.0%      48.3%        −1.7pp (within noise)
Prec(F) at τ              n/a        0.92         (gate satisfied)
```

This is the headline figure — turn on truncation, save 37% wall time, lose
nothing on Pass@1.

---

## 8. Risks and how to mitigate

| Risk | Symptom | Mitigation |
|---|---|---|
| τ too aggressive → premature termination of recoverable rollouts | Pass@1 regresses | Increase τ (more conservative); rerun |
| τ too loose → no termination happens | Wall time barely changes | Decrease τ; might need to recalibrate the model or pick a different checkpoint |
| Savings larger but Pass@1 also drops | The model's calibration on borderline solvable states is poor (matches Run A's `Prec(F)=0.38`) | Confirms the gate condition wasn't really met; need a stronger calibration anchor or different checkpoint |
| Numerical instability in τ comparison (logp noise) | Some rollouts get truncated, others don't, on identical states | Use a margin: only truncate when logp(false) > log(τ) + ε |

---

## 9. Status (2026-05-01)

- [x] v8 anchor checkpoint trained, validated (Pass@1 50%, bp_recall 1.0)
- [x] Code hook present in trainer (no-op)
- [x] v8 anchor weights synced to local
- [ ] Truncation gate logic wired up (implementation TODO)
- [ ] Threshold sweep on v8 checkpoint
- [ ] Option A run
- [ ] Option B run (gated on A's results)
- [ ] Paper figure
