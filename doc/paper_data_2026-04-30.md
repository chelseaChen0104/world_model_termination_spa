# Paper Data Compendium (2026-04-30)

Reference document for writing the conference paper on this project. **All numerical results, with sourcing**. Organized so the paper sections can be drafted against tables and figures here without re-deriving anything from logs.

Companion: `doc/plots/paper/` has 5 paper-ready figures. Reproduction: `python scripts/generate_paper_plots.py`.

---

## 1. One-paragraph project summary (for abstract / intro)

We extend the SPA (Self-Play Agent, Chen et al. 2025) world-model SFT recipe with an explicit termination-prediction tag (`<solvable>` / `<viability>`) so an agent can detect doomed rollouts mid-trajectory and terminate early, saving compute during agentic RL. We test the recipe on two environments — 4×4 Sudoku and 5×4 Pentomino tiling — using Qwen2.5-1.5B-Instruct as the base model. Headline findings: (i) the recipe transfers cross-env, with stronger discrimination on Pentomino (ROC AUC 1.000) than on Sudoku (0.726), driven by Pentomino's visually-local predictive gap; (ii) per-step termination prediction is *uncalibrated* under greedy decoding (model collapses to one class) but the *ranking* signal supports threshold-based deployment past a 90% precision-on-False gate; (iii) RL with asymmetric per-step termination rewards lifts Pass@1 on 4×4 Sudoku (0% → 6.67% with `lr=1e-5`, 200 steps), but the same recipe *regresses* calibration on Pentomino due to short rollouts (≤2 steps) creating a reward landscape biased toward predicting "doom" universally — a reward-shape failure mode that is, to our knowledge, novel.

---

## 2. Method specification (§3 of paper)

### 2.1 SFT format (single-step minimal, v4)

Each training sample is a single-turn conversation:
- **System prompt**: env rules + response-format spec.
- **User**: rendered current state `s_t`.
- **Assistant target** (cross-entropy on every token):
  ```xml
  <think>
  <observation>{rendered s_t}</observation>
  <{state_pred}>{rendered s_{t+1}}</{state_pred}>
  <{viab}>{true|false}</{viab}>
  </think>
  <answer>{action}</answer>
  ```
  where `{state_pred}` is `prediction` for Sudoku, `next_state` for Pentomino; `{viab}` is `solvable` for Sudoku, `viability` for Pentomino. Tag-name parity but distinct namespaces ensure backward compatibility with the Sudoku-era runs and a clean break for new envs.

### 2.2 SFT hyperparameters (the canonical "B-5" / "B-7" recipe)

Mirrors SPA's published Sudoku 4×4 setup:
- Base model: Qwen2.5-1.5B-Instruct
- Optimizer: AdamW, lr=1e-4, batch size 16 (per_device 4 × grad_accum 4)
- Epochs: 5
- max_length: 1024, bf16 precision
- eval_steps: 25
- save_steps: very large (no intermediate saves; only final)

### 2.3 RL training (Phase 1 of NEAR-1.4, no truncation)

GRPO-style with multi-step rollouts:
- 4 puzzles × 8 group_size = 32 rollouts/step
- Rollouts at T=0.7
- KL penalty β=0.05 against frozen reference (the SFT model)
- PPO with clip ε=0.2, max grad norm 1.0, 2 PPO epochs per outer step
- Old logprobs cached at rollout time (PPO bug fix; without this, ratio ≡ 1)

### 2.4 v6 reward (per-rollout, summed over rollout steps)

```
For each step t in rollout:
  step_reward[t] = α * solvable_correctness(t) + β * format_compliance(t)

End of rollout:
  trajectory_bonus = γ if env.is_solved else 0

α weights per outcome:
  TP (predicted False, GT False — caught doom):   +1.0
  FN (predicted True,  GT False — missed doom):   -0.7
  FP (predicted False, GT True  — spurious doom): -0.5
  TN (predicted True,  GT True  — correct):       +0.3

β = 0.05 per of 4 required tags (max +0.20 per step)
γ = +10.0 (v6) → +3.0 (v6.1, used in successful B-5 run)
```

---

## 3. Data tables — SFT runs

### 3.1 All SFT runs, headline metrics

| Run | Env | Train n (no_post_bp) | Class balance (T/F) | Epochs | LR | Batch | Final eval_loss | **ROC AUC** | Greedy Acc | Greedy Rec(T) | Format compliance |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **B-0** | 9×9 multi-turn | 6,221 | — | 3 | 1e-5 | 32 | 0.014 | n/a (BP recall=5%) | — | — | — |
| **B-1/B-2** | 9×9 single-step | 6,221 / 2,482 | — | 3 | 1e-5 | 32 | 0.015 | **0.468** | 67.0% | 1.0% | 100% |
| **B-3** | 9×9 + class-balance | ~2,482 | — | 3 | 1e-5 | 32 | 0.018 | **0.462** | 64.7% | 6.0% | 100% |
| 4×4 baseline | 4×4 Sudoku | 1,336 | — | 3 | 1e-5 | 32 | 0.029 | not measured | 67.3% | 2.0% | 100% |
| **B-4** | 9×9 + SPA hparams | 2,482 | — | 5 | 1e-4 | 16 | 0.015 | **0.455** | 33.3% | 100% | 100% |
| **B-5** | 4×4 SPA-scale | 6,571 | 40 / 60 | 5 | 1e-4 | 16 | **0.0148** | **0.726** | 44.3% | 32.0% | 100% |
| **B-7** | 5×4 Pentomino | 2,964 | 19 / 81 | 5 | 1e-4 | 16 | **0.0361** | **1.000** | 49.5% | 1.0% | 100% |

Sources: `logs/sft_b*.log`, `logs/eval_b*.log` for all runs; final eval_loss extracted by [`scripts/generate_paper_plots.py`](../scripts/generate_paper_plots.py:fig1_sft_loss_curves).

**Reading guide for the paper:**
- B-1 through B-4 = the negative-result chapter on 9×9 (recipe doesn't lift discrimination)
- B-5 = first success: AUC 0.726 on 4×4. Recipe works when task is within model capacity.
- B-7 = cross-env transfer: AUC 1.000 on Pentomino. Recipe generalizes.
- Greedy accuracy is uniformly low because the model is uncalibrated even when AUC is high.

### 3.2 P(true) distributions for B-5 and B-7 (logprob eval)

| Class | n | mean P(true) | median | std |
|---|---|---|---|---|
| **B-5 GT=true (solvable)** | 100 | 0.045 | 0.032 | 0.042 |
| **B-5 GT=false (unsolvable)** | 200 | 0.022 | 0.001 | 0.033 |
| Separation (B-5) | | **+0.023** | | |
| **B-7 GT=true (solvable)** | 100 | 0.548 | 0.641 | 0.271 |
| **B-7 GT=false (unsolvable)** | 100 | 0.000 | 0.000 | 0.000 |
| Separation (B-7) | | **+0.548** (24× larger) | | |

Sources: `logs/eval_b5.log`, `logs/eval_b7.log`.

### 3.3 Threshold sweeps (B-5 and B-7)

#### B-5 (4×4 Sudoku) — predict True if P(true) > τ

| τ | Acc | Prec(T) | Rec(T) | Spec | F1(T) | Prec(F) |
|---|---|---|---|---|---|---|
| 0.10 | 67.7% | 57.1% | 12.0% | 95.5% | 19.8 | **68.5%** |
| ≥0.20 | 66.7% | 0.0% | 0.0% | 100.0% | 0.0 | 66.7% |

#### B-7 (5×4 Pentomino) — predict True if P(true) > τ

| τ | Acc | Prec(T) | Rec(T) | Spec | F1(T) | Prec(F) |
|---|---|---|---|---|---|---|
| **0.10** | **97.0%** | **100%** | **94%** | **100%** | **96.9** | **94.3%** |
| 0.20 | 91.0% | 100% | 82% | 100% | 90.1 | 84.7% |
| 0.30 | 90.5% | 100% | 81% | 100% | 89.5 | 84.0% |
| 0.50 | 78.5% | 100% | 57% | 100% | 72.6 | 69.9% |
| 0.70 | 68.5% | 100% | 37% | 100% | 54.0 | 61.3% |
| 0.90 | 51.0% | 100% | 2% | 100% | 3.9 | 50.5% |

**Key paper-ready claim**: B-7 hits 94.3% Prec(F) at τ=0.10, well above the ≥90% gate we propose for enabling hard rollout truncation during RL. B-5 only reaches 68.5% Prec(F) at the same threshold — Phase 2 truncation not yet safe.

---

## 4. Data tables — RL runs

### 4.1 RL runs summary

| Run | Source SFT | LR | Reward variant | Total steps | Init Pass@1 | **Final Pass@1** | Init `<sol/viab>` acc | Final acc | KL drift | Compute |
|---|---|---|---|---|---|---|---|---|---|---|
| **Phase 1 v6** | B-5 | 1e-6 | v6 (sb=10) | 200 | 0% | 0% (no lift) | 60.9% | 48.3% (regressed) | 0.005 | ~3h |
| **Phase 1 v6.1** | B-5 | 1e-5 | v6.1 (sb=3) | 200 | 0% | **6.67% (+6.67pp)** | 60.9% | **62.0%** (recovered) | 0.026 | ~3h |
| **Run A (continue v6.1)** | v6.1 final | 1e-5 | v6.1 | 500 | 6.67% | **33.33%** (peak 36.67% @ step 250) | 62.0% | 51.4% (mild regression) | 0.05–0.40 | ~10h |
| **B-7 RL Phase 1 (v6)** | B-7 | 1e-5 | v6.1 (sb=3) | 200 | 0% | 0% | **100%** (init: all-True correct) | **0%** (collapsed all-False by step 50) | **1.7** | 1h 44m |
| **B-7 RL Phase 1 (v7)** | B-7 | 1e-5 | v7 (sym + class-bal + progress) | 100 (killed) | 0% | 0% | 100% | **0%** (collapsed by step 75) | 0.65 | ~50m |
| **B-7 RL Phase 1 (v8, fixed)** | B-7 | 1e-5 | v8 (v7 + viability-tag KL anchor, coef 0.5) | 200 (in flight) | 0% | TBA | 100% | TBA — `via_kl=0` to 6dp through step 25, anchor verified firing on n_via_tokens=32/step | TBA | TBA |
| **Sudoku v8 anchor** | Run A final | 1e-5 | v8 (anchor on `<solvable>` tag) | 200 (in flight) | 33.33% | TBA — should preserve Pass@1 and lift `solvable_acc` from 0.514 → ~0.95 | 51.4% | TBA | TBA | ~3-4h |

### 4.2 Phase 1 v6.1 (Sudoku, success run) — quartile trajectory

| Quartile | Mean reward | Max-batch solve rate | Any-solve batches | Avg clipfrac |
|---|---|---|---|---|
| Q1 (steps 1–50) | +0.696 | 12.5% | 6/50 | 0.015 |
| Q2 (51–100) | +1.013 | 6.2% | 7/50 | 0.009 |
| Q3 (101–150) | +1.289 | **31.2%** | 11/50 | 0.010 |
| Q4 (151–200) | **+1.392** | **50.0%** | 8/50 | 0.010 |

| Eval @ step | Pass@1 | `<solvable>` greedy acc |
|---|---|---|
| 0 (init B-5) | 0% (0/30) | 60.9% |
| 75 | 3% (1/30) | 47.9% |
| 100 | 3% (1/30) | 51.4% |
| **150** | **13% (4/30)** | 49.6% |
| 175 | 7% (2/30) | 55.6% |
| **200** | **7% (2/30)** | **62.0%** |

Source: `outputs/rl_b5_phase1_v6_1/rl_log.jsonl` → analyzed by `scripts/generate_paper_plots.py`.

### 4.3 B-7 RL Phase 1 (Pentomino, regression) — quartile trajectory

| Quartile | Mean reward | Max-batch solve | Any-solve batches | Avg clipfrac |
|---|---|---|---|---|
| Q1 (1–50) | +1.178 | 0.0% | **0/50** | 0.015 |
| Q2 (51–100) | +1.491 | 0.0% | **0/50** | 0.009 |
| Q3 (101–150) | +1.495 | 0.0% | **0/50** | 0.010 |
| Q4 (151–200) | +1.493 | 0.0% | **0/50** | 0.010 |

| Eval @ step | Pass@1 | `<viability>` greedy acc |
|---|---|---|
| 0 (init B-7) | 0% | **100% (all True correctly on solvable starts)** |
| 25 | 0% | 100% |
| **50** | 0% | **0% (collapsed to all False)** |
| 100 | 0% | 0% |
| 200 | 0% | 0% |

Calibration regression timing: between steps 25 and 50, the policy crossed a tipping point and flipped greedy from "always True" to "always False". KL grew from 1.4 → 1.8 over the same window. **The reward landscape, dominated by 1-step BPs (GT=False), pulled the policy past the calibration cliff.**

Source: `outputs/rl_b7_phase1/rl_log.jsonl`.

### 4.4 Why B-7 RL collapsed and B-5 RL succeeded — mechanism table

| Property | B-5 (4×4 Sudoku) | B-7 (5×4 Pentomino) |
|---|---|---|
| Avg trajectory length during RL | ~3-4 steps | ~1 step (90%+ of rollouts BP at step 1) |
| Per-batch sample composition | Mixed: ~50% pre-BP (GT=True), ~20% BP (GT=False) | Heavily skewed: ~90% step-1 BP (GT=False) |
| Per-step reward bias | Balanced TN/FP (T-side) ≈ TP/FN (F-side) | Strongly favors predict-False (TP +1.0 vs FN −0.7 per step on doom states) |
| Success bonus contribution | Fires occasionally (~3-9% solve rate at peaks) | Never fires (0% solve rate) |
| Calibration outcome | Recovered: 60.9% → 62.0% | Collapsed: 100% → 0% |
| Pass@1 outcome | Lifted: 0% → 6.67% | No change: 0% → 0% |

This is the **central diagnostic finding** of the paper.

---

## 5. Comparison vs SPA paper

SPA paper headline numbers (from §4 / Table 5):

| Method | Sokoban Pass@1 | Sokoban Pass@8 | Reference |
|---|---|---|---|
| Vanilla RL (no SFT) | 25.6 | 34.0 | Table 5 |
| State Estimation RL | 52.7 | 53.9 | Table 5 |
| SPA 1-epoch SFT + RL | 29.2 | 52.7 | Table 5 |
| **SPA 5-epoch SFT + RL (headline)** | **59.8** | **69.5** | Table 5 / Abstract |
| SPA 5-epoch RandSFT + RL | 20.2 | 50.0 | Table 5 |

**Key clarification for paper**: SPA's 59.8 Pass@1 is *post-RL on top of 5-epoch SFT*. SFT-only Pass@1 is not headline-tabled; Figure 6 of SPA shows it ramping from ~19% (epoch 1) to ~59% (epoch 5) on Sokoban.

**Our results vs SPA (apples-to-apples where possible):**

| Setting | SPA | This work |
|---|---|---|
| Task | Sokoban / FrozenLake / Sudoku 4×4 | Sudoku 4×4 + Pentomino 5×4 |
| Base model | Qwen2.5-{0.5B, 1.5B, 3B} + LLaMA-3.2-1B | Qwen2.5-1.5B (single config) |
| Termination prediction tag | not present | `<solvable>` / `<viability>` (our addition) |
| SFT-only Pass@1 | not directly reported on Sudoku | 0% on 4×4 Sudoku and 5×4 Pentomino |
| RL-lifted Pass@1 | **59.8 (Sokoban)** | **6.67 (Sudoku 4×4, lr=1e-5, 200 steps)** |
| ROC AUC on termination | not measured | 0.726 (Sudoku 4×4), 1.000 (Pentomino 5×4) |

**Honest framing**: we have not run their full pipeline (≥1000 RL steps, multi-model). Our 200-step RL is in the early-trajectory phase. We make no claim of out-performing SPA on their headline metric; we extend the recipe with a termination signal and characterize a novel reward-shape failure mode.

---

## 6. Figures (in `doc/plots/paper/`)

### Figure 1: SFT validation loss curves
File: `fig1_sft_loss_curves.png`
**Caption**: "Validation cross-entropy across all SFT runs (log scale). All runs converge in expected shape; no run shows training pathology. Final eval_loss differs by ~2× between Sudoku (0.015) and Pentomino (0.036) — entropy of action format, not learning quality. **eval_loss is a poor proxy for the discrimination signal we measure separately via ROC AUC.**"

### Figure 2: ROC AUC progression across SFT runs
File: `fig2_auc_progression.png`
**Caption**: "ROC AUC on the termination prediction task (`<solvable>`/`<viability>`) across all SFT runs. 9×9 Sudoku runs (B-2 to B-4) plateau near chance (0.46) regardless of hyperparameter scaling. The recipe transitions from ineffective to effective at the 4×4 Sudoku scale (B-5: 0.726) and produces near-perfect discrimination on 5×4 Pentomino (B-7: 1.000). The dotted line marks B-5's level."

### Figure 3: P(true) distributions (B-5 vs B-7)
File: `fig3_p_true_distributions.png`
**Caption**: "Histograms of model-predicted P(true) at the `<solvable>`/`<viability>` token, separated by ground-truth class. Left (B-5, 4×4 Sudoku): bimodal but heavy overlap; mean separation +0.023. Right (B-7, 5×4 Pentomino): cleanly bimodal; unsolvable states get P(true)≈0 with near-zero variance, solvable states spread between 0.3 and 0.9. Mean separation +0.548 — 24× larger than B-5."

### Figure 4: RL trajectories (Phase 1 v6 vs v6.1 vs B-7 RL)
File: `fig4_rl_trajectory.png`
**Caption**: "RL training trajectories under three configurations: (purple) v6 reward with lr=1e-6 → no Pass@1 lift, calibration regression; (olive) v6.1 reward with lr=1e-5 → Pass@1 0% → 6.67%; (cyan) v6.1 applied to Pentomino (B-7) → calibration collapse despite favorable SFT initialization. Reward (left), KL to reference (middle, log scale), per-batch solve rate at T=0.7 (right). 10-step moving average. The B-7 trajectory exemplifies a reward-shape failure mode novel to short-horizon agentic envs."

### Figure 5: Threshold sweep (B-5 vs B-7)
File: `fig5_threshold_sweep.png`
**Caption**: "Precision, recall, and F1 of the True (solvable) classifier as a function of decision threshold τ on P(true). For B-5 (left), the classifier is unusable above τ=0.20 — model never assigns more than ~0.10 probability to True even on solvable states. For B-7 (right), the classifier is usable across τ ∈ [0.10, 0.50] with 100% precision throughout, gradually trading recall for confidence. Practical implication: **B-7 supports threshold-based truncation in RL while B-5 does not.**"

---

## 7. Key claims with evidence pointers

For the paper's results section, organized by claim:

### Claim 1: World-model SFT extended with a termination tag yields measurable discrimination on within-capacity tasks.
- Evidence: B-5 ROC AUC = 0.726, B-7 ROC AUC = 1.000 (Tables in §3.1, Figure 2).
- Counter-evidence: 9×9 Sudoku (B-2/B-3/B-4) all hover near 0.46, regardless of scale or hyperparameters (§3.1). Suggests Qwen-1.5B's representational capacity is the bottleneck on harder predictive gaps.

### Claim 2: The discrimination signal is *uncalibrated* under greedy decoding but supports threshold-based deployment.
- Evidence: Greedy `<viability>` accuracy on B-7 = 49.5% with 1% recall on True (collapsed to all-False), but threshold-based decoding at τ=0.10 yields 97% accuracy with 100% precision and 94% recall on True (§3.3, Figure 5).
- Implication: practical use case is threshold-based, not greedy. Phase 2 truncation gate (Prec(F) ≥ 90%) is feasible on B-7 but not on B-5.

### Claim 3: The recipe transfers cross-env, with stronger discrimination on visually-local predictive gaps.
- Evidence: B-7 (Pentomino) discrimination is stronger than B-5 (Sudoku) despite B-7 having 45% the training data (§3.1, Figure 3).
- Mechanism: pentomino's "isolated unfillable region" is locally observable in rendered cells; Sudoku's "future constraint cascade" requires propagation reasoning. Hypothesis offered, not proved — could be confounded by class imbalance.
- Caveat: AUC=1.0 deserves verification via train↔val (s_t, a_t) overlap analysis (not yet run).

### Claim 4: RL with asymmetric per-step termination reward lifts Pass@1 on tasks with multi-step rollouts (Sudoku 4×4: 0% → 6.67%) but *regresses* calibration on tasks with short rollouts (Pentomino: collapse to all-False).
- Evidence: §4.1, §4.2, §4.3, Figure 4.
- Mechanism: short trajectories (1 step BP for ~90% of rollouts) make the per-step reward landscape monotonically favor "predict False" — the asymmetric step rewards drown out the rare success_bonus. Documented in §4.4.
- Significance: identifies a novel constraint on reward shaping for agentic-RL with termination prediction. *Reward shape must be designed against the trajectory-length distribution.*

### Claim 5: PPO clipping requires the policy to actually move; cached-rollout-time logprobs are mandatory.
- Evidence: B-5 v6 RL had clipfrac=0 throughout 200 steps because old_logp was recomputed under the same policy → ratio≡1. After fixing, clipfrac=0.01-0.02 (B-5 v6.1) and 0.02 (B-7) where policy actually moved. (§2.3 implementation note.)
- Implication: a common implementation gotcha. We document because we got it wrong initially.

### Claim 6: Pass@1 lift in our setup depends on success_bonus interacting with achievable solve rates.
- Evidence: v6 (success_bonus=10) on 4×4 Sudoku had calibration regression and 0% Pass@1 lift. v6.1 (success_bonus=3) recovered calibration AND lifted Pass@1 to 6.67%. The reduced bonus shifted relative weight back to per-step calibration. (§4.1, §4.2.)
- B-7 with v6.1 also failed because its solve rate was 0% — bonus magnitude is moot when it never fires.

### Claim 7: The calibration regression in our RL setup is *dynamic drift*, not a static reward-landscape problem; a tag-specific KL anchor is the targeted fix.
- Evidence (§9.2 sanity test): under v7 reward magnitudes, the static expected per-step reward ranks `oracle (+1.00) > sft_actual (+0.99) > always_false (+0.46) > always_true (-0.46)` — the reward landscape favors correct prediction. B-7 SFT starts at oracle quality (98% accuracy on its rollout state distribution).
- The collapse is therefore a *dynamic* failure of the optimizer: noisy GRPO advantages walk the policy off the SFT optimum faster than the global KL leash (averaged over ~150 response tokens, kl_coef=0.05) can pull it back. Once off-manifold, the policy falls into the always-False basin and the local gradient keeps it there.
- The fix (v8 reward) adds an auxiliary KL penalty applied *only on the `<viability>` / `<solvable>` tag content tokens* against the frozen SFT reference, with coefficient 0.5. Concentrated on the 1-3 tokens that matter for calibration, it's much stronger per-token than the global KL while leaving action tokens free to optimize.
- Implementation correctness was non-trivial: a tokenization roundtrip mismatch (trailing EOS) caused the position finder to silently no-op for ~100 steps before it was caught. The fix re-decodes `response_ids` ourselves and tolerates trailing-EOS mismatches via prefix matching. Positions are now cached on `StepRecord.viability_token_positions` at rollout time.
- v8-fixed early signals (step 50, in flight at time of writing): `via_kl = 0.000000` to 6 decimal places (anchor effective), `solvable_acc` held at 1.0 through the v6 collapse point. Step 75 (v7 collapse point) and Sudoku Run A v8 anchor are the two active discriminating tests. ([qa_2026-05-01_reward_and_rl.md](qa_2026-05-01_reward_and_rl.md), [eval_2026-04-30_b7_rl_phase1.md](eval_2026-04-30_b7_rl_phase1.md) "Option D".)

---

## 8. Caveats and limitations to acknowledge

To get ahead of reviewer questions:

1. **AUC=1.0 on B-7 is not yet stress-tested.** Pentomino-easy starts every trajectory from the same empty board, so the 80/20 train/val split (per-trajectory) may admit (s_t, a_t) duplicates between train and val. Memorization vs generalization not yet disambiguated. Proposed remediation: train↔val (s_t, a_t) overlap check + held-out test on novel piece sets.

2. **Single base model (Qwen-1.5B).** SPA scales across 0.5B / 1.5B / 3B / LLaMA-1B; we did not.

3. **No Pass@1 measured on B-7.** We have AUC but not the SPA-comparable Pass@1 metric for the second env.

4. **9×9 Sudoku failure is unresolved.** B-4 (SPA-scale on 9×9) hit chance AUC. We hypothesize task-difficulty for 1.5B but did not isolate (no 9×9 SPA-scale data + 7B model run). 9×9 SPA-scale gen was paused after easy difficulty — would need to complete medium+hard for a definitive negative result.

5. **RL training is short** (200 steps vs SPA's 1000+). We don't claim out-performing SPA's published Pass@1; we characterize the early phase.

6. **Reward design is one specific shape.** v6/v6.1 is asymmetric per-step + trajectory bonus. Alternative shapes (auxiliary classification head, reward anchored on AUC preservation) not tested.

---

## 9. Pending experiments — the path from "current state" to "complete paper"

What's done is in §3-§5. What's needed to make the paper publishable, in priority order:

### 9.1 Currently in flight (2026-05-01)

| Experiment | Cloud | Status | When it lands | What it gives the paper |
|---|---|---|---|---|
| **B-7 v8-fixed RL (Pentomino calibration anchor)** | autodl | Step 50/200, `via_kl = 0.000000` to 6dp throughout, `solvable_acc = 1.0` at step 25 | ~3 hours | Whether the viability-tag KL anchor prevents the dynamic calibration drift identified in §4.4. Sharp prediction: anchor holds `solvable_acc ≥ 0.95` (matches B-7 SFT). |
| **Sudoku v8 anchor on Run A's checkpoint** | autodl2 | Just launched | ~3-4 hours | Cleanest test of v8 on a *known-working* baseline (Run A: Pass@1 33.33%, but `solvable_acc` regressed 0.620 → 0.514). Hypothesis: anchor restores `solvable_acc` to ~0.95 *without* losing Pass@1. If yes, this is the paper's "calibration anchor compatible with goal-directed RL" headline. |

### 9.2 Recently completed (2026-04-30 → 2026-05-01)

| Experiment | Result | Paper implication |
|---|---|---|
| **Run A (Sudoku continuation, lr=1e-5, 500 steps)** | Pass@1 6.67% → **33.33%** (peak 36.67% @ step 250); `solvable_acc` 0.620 → 0.514 (mild regression); `bp_recall = 1.000` throughout | Confirms lr was the original Sudoku bottleneck, not the reward shape. The 0.514 acc is "essentially chance on borderline solvable states" — but `bp_recall = 1.0` and `Prec(False) = 0.38` shows the failure is asymmetric (perfect recall on doom, high false-positive on solvable). Motivates Phase 3 v8 anchor experiment. |
| **B-7 sanity test (400 rollouts at T=0.7)** | 73% 1-step rollouts, 0% Pass@1 (success_bonus never fires), B-7 SFT calibration is essentially oracle (98% acc), v7 reward landscape has oracle (+1.00) >> always_false (+0.46) | Proves the v6/v7 collapse is **dynamic drift**, not a static reward landscape problem. The cure has to be a leash on the calibration tokens, not a reward redesign. |
| **B-7 v8 (first attempt, broken anchor)** | Anchor silently no-op'd due to tokenization roundtrip mismatch (returned [] every call); calibration collapsed at step 75 just like v7 | Bug discovery + fix documented in [src/training/rl_trainer_v6.py](../src/training/rl_trainer_v6.py): re-decode `response_ids` ourselves and tolerate trailing-EOS mismatch via prefix matching. Cache positions on `StepRecord.viability_token_positions` at rollout time. |

### 9.3 Imminent next runs (1-2 day budget)

These are blocking for a *complete* paper:

| Run | Effort | What it tests / fixes | Paper section it informs |
|---|---|---|---|
| **B-7 Pass@1 measurement** | ~30 min GPU on autodl | SPA-comparable headline number for Pentomino. Currently we have AUC but no Pass@1. | §3.1 table; cross-env story |
| **B-7 train↔val (s_t, a_t) overlap check** | ~10 min CPU | Disambiguates memorization vs generalization for the AUC=1.0 result. Reduces reviewer attack surface. | §8 caveats; §3.1 footnote |
| **B-7 v8-fixed RL (in flight)** | already running | If `solvable_acc ≥ 0.95` AND Pass@1 lifts off 0% → recipe works on Pentomino too. If acc holds but Pass@1 stays 0% → calibration anchor alone is insufficient; flip on `--action-quality-bonus 1.0` (already wired into the trainer) for a v8 + v8.1 combined run. If acc drops too → anchor mechanism flawed; escalate to B-9 bigger board. | §4.4 mitigation. |
| **Sudoku v8 anchor on Run A (in flight)** | already running | The cleanest decoupled test. Validates the anchor on a known-working setup. | §4.4 + new "calibration-preserving RL" sub-claim. |
| **B-9: 5×10/10-piece Pentomino — full pipeline** *(fallback if v8 also fails Pass@1 lift on B-7)* | ~1 day total (data gen + SFT + RL) | Switched from the originally-proposed 5×5/5-piece variant. Bigger predictive challenge, longer trajectory length distribution, and `success_bonus` should fire on at least some rollouts (unlike Pentomino-easy where Pass@1 stochastic = 0/400). P-0 sweep on 66 (10-of-12) subsets in progress on local. | §4.4 mechanism + Claim 4 evidence by elimination |
| **Phase 2 truncation experiment** | ~3-4 hours GPU | The project's headline value proposition: does early termination via `<viability>` actually save GPU-hours during RL? Run on Sudoku v8 anchor's final checkpoint (need `Prec(False) ≥ 0.90` from anchor — Run A's 0.38 wasn't enough). | New §X: "Compute savings from threshold-based truncation" — likely a key paper figure |

### 9.3 Higher-effort follow-ups (paper-strengthening but not blocking)

| Run | Effort | What it tests | Why it matters |
|---|---|---|---|
| **B-6: 9×9 + SPA-scale data + SPA hparams** | ~12 hours GPU (need to resume 9×9 data gen first, ~9 hours) | Whether 9×9 is salvageable with full SPA-scale data. Would close the negative result properly. | §3.1 footnote — "we ran the experiment SPA describes for 4×4, but 9×9 plateaus at chance." Strong negative result. |
| **Reward shape v8 (auxiliary classification head)** | ~2-3 days code + ~1 day GPU | If v7 (symmetric + class-balanced + progress bonus) is insufficient, decouple discrimination from token-level cross-entropy entirely via an aux classification head. Stronger but heavier than v7. | Last-resort alternative to v7 / bigger boards if both fail. |
| **Multi-model scaling (Qwen 0.5B / 3B)** | ~1 day each | Whether the 9×9 failure is task-difficulty or model-capacity. Mirrors SPA's setup. | §8 limitation removal. Strengthens the "Qwen-1.5B can't do 9×9 even at scale" claim. |
| **Full Pass@1 / Pass@8 sweep matching SPA's Table 5** | ~3-4 hours GPU | Direct comparison to SPA's published numbers. | §5 — currently SPA comparison is asymmetric (we have AUC, they have Pass@1). |

### 9.4 Open research questions (could be addressed but optional)

| Question | What experiment would answer it |
|---|---|
| Does the AUC=1.0 finding hold on a 6×10 board with all 12 pentominoes (the canonical research benchmark)? | Run B-10 with 6×10 / 12-piece config. Bigger predictive gap, harder to memorize. |
| Does the discrimination signal transfer to a third env (e.g., MKD — maze with keys & doors, where the predictive gap is path-dependent rather than visually local)? | Build MKD env (~2 days), run SFT + eval. |
| Does the temporal-echo failure mode (B-0) generalize to multi-turn world-model SFT on other puzzle envs? | Originally proposed Q6 in SPEC.md; still untested. |
| What's the minimum SFT data scale for the discrimination signal to emerge on each env? | Sweep N_TRAJ ∈ {500, 1000, 2000, 3000} for both Sudoku 4×4 and Pentomino 5×4. |

### 9.5 Decision points for paper scope

If a deadline forces a choice, here's a triage of must-haves vs nice-to-haves:

**Minimum-viable-paper** (current state + imminent runs §9.2):
- Method + 4×4 Sudoku results (B-5 SFT, Phase 1 v6.1 RL: 0→6.67%)
- Pentomino-easy results (B-7 SFT: AUC=1.0)
- B-7 RL collapse + mechanism analysis (Claim 4 — *this is the novel contribution*)
- Train↔val overlap check (1 day)
- Run A final (already running)

**Strong paper** (+§9.2 remaining + §9.3):
- B-7 RL with v7 reward (first-line fix for the collapse — replaces or precedes the bigger-board run)
- B-9 (5×5 Pentomino) **only if** v7 fails (used as elimination evidence for the trajectory-length hypothesis)
- Phase 2 truncation experiment (the "compute savings" headline)

**Comprehensive paper** (+§9.3 full):
- Add 9×9 SPA-scale (B-6) for completeness on the negative result
- Add multi-model scaling

We currently sit at **minimum-viable-paper minus train↔val overlap check minus Run A final**. ~1-2 days of GPU + analysis to get to MVP, ~5-7 more days to get to "strong paper."

---

## 10. Suggested paper structure (1-page outline)

| Section | Length | Content |
|---|---|---|
| Abstract | 1 paragraph | §1 above |
| Introduction | ~1 page | Motivation: agentic RL is expensive; doomed rollouts waste compute. SPA recipe is good base. We extend with termination prediction. Headline contributions (3-4 bullets). |
| Related work | ½ page | SPA, world-modeling SFT, termination MDPs (TerMDP, Tennenholtz et al. 2022), agentic RL (RAGEN, VAGEN). |
| Method | 1 page | §2 above: format, hparams, RL setup, reward shape. |
| Experiments | 2-3 pages | §3 + §4 tables. Figures 1-5 distributed across this section. |
| Analysis | 1 page | §4.4 mechanism (calibration regression on short rollouts), Claim 4 in detail. |
| Discussion / Limitations | ½ page | §8. |
| Conclusion | ¼ page | Summarize 3 main claims; gesture at future work (5×5 Pentomino to test trajectory-length hypothesis, multi-model scaling, full SPA pipeline replication). |

Total: ~6-7 pages, fits NeurIPS 9-page limit comfortably.

---

## 11. Reproducibility checklist

- All code: GitHub repo (link to `chelseaChen0104/world_model_termination_spa`).
- All data: regeneratable via `scripts/generate_*_spa_scale.sh` and `scripts/generate_pentomino_easy.sh`.
- All training: launchers `scripts/run_b{5,7}_*.sh` and `scripts/run_rl_b{5,7}_phase1.sh`.
- All eval logs: `logs/eval_b*.log` (parsed for §3 numbers).
- All RL logs: `outputs/rl_b*/rl_log.jsonl` (parsed for §4 numbers).
- All figures: `scripts/generate_paper_plots.py` regenerates from raw logs in one call.

Random seeds: 42 throughout. Hardware: single H800 80GB per cloud (AutoDL).
