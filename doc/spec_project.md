# Project Spec — World Model Termination Prediction

**Status:** living document. Update when scope changes.

- **v1** — first draft. Excluded Sokoban for "heuristic detector exists." Wrong reason.
- **v2** — re-included Sokoban + FrozenLake citing SPA paper alignment. Wrong reason.
- **v3** — 2026-04-28: excluded Sokoban + FrozenLake on the *right* reason — predictive-gap criterion (see §1).
- **v4** — 2026-04-28: switched to single-step SFT samples + minimal tag set + action-conditional `<solvable>` after Tier A revealed temporal-echo failure mode.
- **v5** (current, 2026-04-29) — **un-banned vLLM as a local inference accelerator.** v3 had grouped vLLM with Ray under "distributed training" out-of-scope, but vLLM can run single-process and provides ~5–10× speedup on autoregressive rollouts (matters for RL and Pass@k eval). Ray-based distributed coordination remains out of scope. Also: logprob-based eval revealed B-2/B-3 SFT have **AUC ≈ 0.46 — no learned discrimination at the `<solvable>` token**. Adding research questions on data-scale and task-difficulty (Q7, Q8 below) before committing to RL.
- **v5 amendment** (2026-04-29 evening) — **Q7/Q8 answered.** B-4 disproved "scale alone fixes 9×9" (AUC 0.455 with SPA hparams). B-5 (4×4 + SPA hparams + SPA-scale data) hit AUC 0.726 — first SFT run with real discrimination. **The recipe is sound; 9×9 is too hard for Qwen-1.5B SFT alone.** Strategic shift: pivoting next compute toward (a) one final 9×9 SFT test (B-6, with 9×9 SPA-scale data — gen running), and (b) RL on the B-5 4×4 checkpoint. See [eval_2026-04-29_b5_4x4_spa_replication.md](eval_2026-04-29_b5_4x4_spa_replication.md).

**Use:** before any new approach, environment, or experiment, evaluate it against §5 (decision rubric). After every milestone, evaluate the result against §3 (success criteria) — that's what "on track" means.

---

## 1. Mission

Demonstrate that an LLM agent equipped with an **SPA-style internalized world model** is the right substrate for **termination prediction** in environments that exhibit a **predictive gap** — environments where a state can be *visually solvable but already doomed*, with the formal failure arriving multiple steps later. Concretely: predict per-state solvability, the breaking-point step where an episode transitions from solvable to unsolvable, and steps-to-failure.

The thesis combines two prior threads:
- **SPA (Chen et al. 2025)** — internalized world models via self-play SFT (state estimation + transition modeling) outperform vanilla RL and online world modeling (VAGEN) on OOD environments (Sokoban / FrozenLake / Sudoku) by lifting *Pass@1 / Pass@k*.
- **TerMDP (Tennenholtz et al. 2022)** — termination is a learnable, history-dependent signal that, when modeled, improves agent behavior via a dynamic discount factor.

Our contribution sits at their intersection but on a **narrower experimental setting than SPA**: with SPA's grounding, can the model also internalize *when an episode is doomed* — in environments where "doomed" is non-trivial to detect? The narrowing is principled and load-bearing (see "Predictive gap" below).

The research question is not "can an LLM solve a puzzle." It is "given an LLM with grounded world-model SFT, can it know it's lost in the **interval between sealing its fate and formal failure** — and is that signal more accurate than every cheaper alternative?"

### Predictive gap (the load-bearing criterion)

For an environment to test our research question, it must satisfy:

> **There exists a state S at step K where (i) the state is locally legal — the agent can still take valid actions, (ii) no action sequence from S reaches the goal, (iii) formal termination arrives at step K+M for some M ≥ 1.** The interval [K, K+M] is the predictive gap. Our project asks: can the model classify S as doomed at step K, before the gap closes at step K+M?

Environments without this gap (where doomed-state ≡ formal-termination state, or where doomed-state is visually obvious) are **not testing the question we are asking**. They might be testing fast deadlock recognition, or task completion under OOD shift — useful problems, but different from ours.

---

## 2. Research questions (concrete, falsifiable)

1. **Q1 — Does SPA-style world-model SFT improve termination prediction?** Compared to SFT without `<observation>`/`<prediction>` world-modeling tags, does adding them lift breaking-point recall and solvability F1? Hypothesis: **yes**, by the same OOD-grounding mechanism that lifts Pass@1 in the SPA paper.
2. **Q2 — Does adding termination prediction to the SPA training objective hurt or help downstream RL?** Concretely: does training the model to also produce `<solvable>` / `<breaking_point>` / `<steps_left>` tags during SFT reduce the SPA paper's Pass@1 / Pass@k gains on the **same env** (Sudoku), or does it preserve / improve them?
3. **Q3 — Does asymmetric-reward RL (TP +3.0, FN −2.0, FP −0.5) on the termination objective lift BP recall above what SFT alone can deliver**, without breaking the format-compliance and answer-accuracy objectives RL is otherwise optimizing?
4. **Q4 — Does training on LLM-policy data outperform training on random-play data** on an LLM-policy-distribution eval set? — **DEFERRED 2026-04-28**: Track A SFT was dropped to focus compute on the LLM-policy headline. Q4 is testable later if a random-play SFT checkpoint is added; until then, the comparison can be made indirectly via the heuristic baseline.
5. **Q5 — Does multi-turn context (game history) improve BP detection over single-turn** in fully-observable domains? — **ANSWERED 2026-04-28: NO on Sudoku.** Multi-turn SFT with sliding-window priors created a *temporal-echo shortcut* that the model exploited: 84.7% of samples have priors that match the target label, so the model learned to copy prior assertions rather than condition on the grid. Multi-turn eval showed BP recall = 5%. We abandoned multi-turn SFT in favor of single-step samples (§7). Multi-turn RL rollouts are unaffected.
6. **Q6 — Does this temporal-echo failure mode generalize to other multi-turn world-model SFT setups?** — *added v4*. If yes (across Kakuro / Nonogram / our Sudoku), it's a finding that affects SPA-style training broadly; if no, it's a Sudoku-specific quirk.
7. **Q7 — Is the failure to learn `<solvable>` discrimination a data-scale problem, a task-difficulty problem, or a model-capacity problem?** — *added v5; mostly answered 2026-04-29*: **task difficulty is the dominant factor.** Run B-4 (9×9 + SPA hparams + 30× more grad signal) still gave AUC 0.455. Run B-5 (4×4 + SPA hparams + SPA-scale data, 6,571 samples) gave AUC 0.726. The recipe works on 4×4 but not on 9×9 at this capacity. 9×9 SPA-scale gen running to confirm with one more 9×9 SFT (B-6).
8. **Q8 — Does our SPA-comparable result hold on the SPA paper's own configuration?** — *added v5; answered 2026-04-29*: **yes, ROC AUC = 0.726 on 4×4 with SPA hparams + SPA-scale data.** First SFT run with real discrimination (vs AUC 0.46 ≈ chance on every prior 9×9 attempt). See [eval_2026-04-29_b5_4x4_spa_replication.md](eval_2026-04-29_b5_4x4_spa_replication.md). Calibration is poor (mean P(true) ≈ 0.045) but ranking is sound. Pass@k vs SPA's published 1.6 → 59.6 numbers is a follow-up to anchor scale.

Q6-as-cross-env-transfer (from v2) is **dropped** because Sokoban and FrozenLake fall outside the predictive-gap criterion (§1). The new Q6 is the temporal-echo generalization question.

Anything that doesn't help answer one of Q1–Q8 is suspect.

---

## 3. Success criteria — "are we on track?"

A milestone counts as on-track when **all** of:

- ✅ Metrics include the **SPA paper's headline Pass@1 and Pass@8 (or Pass@k)** on at least one of {Sokoban, FrozenLake, Sudoku} — this is the comparability constraint with SPA.
- ✅ Plus our termination-specific metrics: **BP recall, BP precision, BP F1, solvability F1, per-deadlock-type recall**. Never raw per-step accuracy on imbalanced data as a headline.
- ✅ Eval set is **in-distribution with deployment** — generated by an LLM playing the game, not by random play.
- ✅ Eval set is **balanced** for solvability metrics so we don't reward majority-class prediction.
- ✅ Improvement is reported relative to **at least three baselines**: (a) vanilla RL on the base model, (b) State-Estimation-only RL (SPA without transition modeling), (c) VAGEN-style online world modeling — these are SPA's own baseline set. Plus a heuristic baseline (e.g., `sudoku_baseline.py`) where one exists.
- ✅ The result is **reproducible on a single H800** (no distributed training, no model parallelism). SPA used the same constraint.

Numerical signal targets (not commitments):

| Stage | Pass@1 (SPA env) | BP recall | Solvability F1 |
|---|---|---|---|
| Base Qwen2.5-1.5B | low (SPA Table 2) | n/a | n/a |
| Vanilla RL | matches SPA Table 2 | low | ~50% |
| State-Estimation RL | matches SPA Table 2 | low–mid | ~55–65% |
| **SPA + termination SFT (our v1)** | preserves SPA Pass@1 | mid (40–60%) | 60–75% |
| **SPA + termination SFT + RL (our v2)** | matches/beats SPA Pass@1 | high (60–80%) | 75%+ |

The comparison-of-interest is the last two rows against State-Estimation RL and SPA-without-termination — that's where Q1 and Q2 are answered.

---

## 4. In scope vs. out of scope

### In scope (rationale):

| Item | Why |
|---|---|
| **Sudoku** as the sole primary environment | Only SPA-paper env that satisfies §1's predictive-gap criterion. Place a locally-valid number → grid looks normal → unsolvable downstream. Heuristic baseline is weak (constraint propagation needed). This is where our research question bites. |
| **SPA-style world model SFT** (`<observation>` + `<prediction>` tags, self-play with ground-truth replacement) | This is the substrate the project is built on. |
| **Termination tags as our extension** (`<solvable>`, `<breaking_point>`, `<steps_left>`) | Our actual research contribution layered on top of SPA's training corpus. |
| **LLM-policy multi-turn data** | In-distribution states; SPA's self-play recipe. |
| **Random-play data as a CPU-cheap intermediate** | For warm-start / pipeline debugging only. Acknowledged as off-distribution. |
| **Multi-turn SFT with sliding window** | Matches SPA's findings that multi-turn broadens the reasoning frontier. |
| **Single-GPU GRPO / PPO RL** | SPA was single-GPU; our budget is single H800; intentional simplicity over RAGEN/Ray. *v5: vLLM permitted as a local rollout accelerator.* |
| **Asymmetric BP rewards (TP +3.0, FN −2.0, FP −0.5) + format compliance** | Our termination-specific reward shaping; orthogonal to SPA's answer-accuracy reward. |
| **Balanced live-env sampling for RL termination** | Fixes class imbalance at training time. |
| **Baseline set: vanilla RL, State-Estimation RL, VAGEN** | These are the SPA paper's baselines; we must reproduce / report against them. |
| **Heuristic baseline comparison** (`sudoku_baseline.py`, Sokoban deadlock detector) | A heuristic baseline is a *baseline to beat*, not a reason to skip an env. |
| **Ablations**: world-model SFT on/off; termination tags on/off; LLM-policy vs random data; single vs multi-turn | Each ablation directly tests one of §2 Q1–Q5. |
| **Cross-environment evaluation** | Tests Q6 — same recipe across Sokoban/FrozenLake/Sudoku. |
| **Difficulty variation** within an env | Tests recipe robustness; the SPA paper varied grid size to match model capacity. |

### Out of scope (rationale):

| Item | Why excluded |
|---|---|
| **Sokoban** | Fails the §1 predictive-gap criterion. When a Sokoban state is unsolvable, it's because of a visually recognizable deadlock pattern (corner deadlock, frozen box) — doomed-state ≈ deadlock-state, no meaningful gap before formal failure. A simple deadlock detector handles ~all cases. We'd be solving a recognition problem, not a prediction problem. (v2 included it; v3 excludes it on the corrected reason.) |
| **FrozenLake** | Fails the §1 predictive-gap criterion. Termination is per-step: stepping into a hole *is* the failure. There's no "you sealed your fate at step K but kept playing until step K+5." (v2 included it; v3 excludes it.) |
| **Other SPA-paper environments / future SPA extensions** unless they satisfy §1 | The relevant filter is the predictive-gap criterion, not "is it in the SPA paper." If a future env meets §1 we'll consider it. |
| **Partially-observable environments** (POMDPs, hidden-info card games) | Different problem class. Termination under partial observability adds belief-state reasoning that confounds Q1–Q3. |
| **Trivially-always-solvable domains** | "Still solvable?" is degenerate. |
| **Domains without a ground-truth solvability oracle** | We rely on `check_solvability()` to label data and reward RL. No oracle → no project. |
| **Larger models (>3B params)** | SPA showed the gain is largest on small models; our budget is single H800. Scaling laws are not our research question. |
| **Multi-agent / cooperative termination** | Different problem class. |
| **Ray-based distributed coordination (RAGEN-style multi-actor)** | Heavyweight infrastructure not needed for single-GPU. SPA's RAGEN approach intentionally rejected. |
| **FSDP / multi-GPU model parallelism** | Single-H800 constraint; Qwen-1.5B fits comfortably. `sft_trainer.py` (FSDP) is dead code; `simple_sft_trainer.py` is the path. |
| ~~vLLM serving~~ | **Re-permitted in v5.** vLLM as a *local single-process inference accelerator* is in scope (for RL rollouts and Pass@k eval). It's not a distributed system; only Ray is. |
| **`steps_left` in the RL reward** | Trajectory-dependent, prone to reward hacking. SFT-only signal. **Decided.** |
| **Per-step accuracy on imbalanced data as a headline metric** | Trivial baseline gets 90%+ by predicting majority class. |
| **Heavy chain-of-thought as the predicted artifact** | The thinking text is scaffolding; the labels (`<solvable>`, `<breaking_point>`) are what we evaluate and reward. |

### What changed v2 → v3

- **Sokoban + FrozenLake: out** for the corrected reason — neither has a predictive gap (§1). v2 brought them in because they're SPA-paper benchmarks; that was scope-by-association, not scope-by-research-fit.
- **Q6 dropped:** cross-env transfer is no longer a primary research question; at most a future-work bullet.
- **Mission narrowed:** explicitly states the predictive-gap criterion and accepts that our experimental setting is narrower than SPA's three envs by design.
- **Baselines:** still match SPA's set (vanilla RL, State-Estimation RL, VAGEN) — but evaluated on Sudoku only, with our termination metrics added on top of SPA's Pass@1/Pass@k.

---

## 5. Decision rubric — evaluating a new approach or environment

Before adopting a new environment or method, run it through these gates. If any answer is "no" or unclear, the proposal is suspect.

**For new environments — the §1 predictive-gap test is the first gate:**

1. **Predictive gap (the load-bearing test):** does the env have states S at step K where S is locally legal, no action sequence from S reaches the goal, and formal termination arrives at step K+M with M ≥ ~3? → If no, **REJECT**. *This is the question Sokoban and FrozenLake fail.*
2. Does it have a **ground-truth solvability oracle** runnable per-state at data-gen + RL time? → If not, **REJECT**.
3. Is the failure (unsolvability) **non-obvious from a 50-line program** when looking at state S during the gap interval? → If a heuristic catches the doom at step K trivially, the LLM adds little. **REJECT** unless heuristic baseline is part of the experimental contrast.
4. Does the environment have **observable state** an LLM can read as text? → If not (image-only, hidden state), **REJECT** (out of scope).
5. Can we generate data **without human supervision** (random play + oracle, or LLM-policy + oracle)? → If not, **REJECT**.
6. Are breaking points **distributed across the trajectory**, not clustered at one end? → If only at one end, the prediction problem is degenerate.

**For new methods/approaches:**

1. Does it directly address one of §2 Q1–Q6? → If not, defer or **REJECT**.
2. Does it preserve §3's success criteria — in-distribution eval, balanced metrics, single-GPU, SPA-comparable baselines? → If it requires distributed training or a different eval distribution, justify explicitly.
3. Is there a clear **A/B baseline within the project** to compare against? → If not, build one before running.
4. Estimated cost (GPU-hours) ≤ **expected information value**. → A 24h ablation that only marginally addresses a question is a bad trade.
5. Does it introduce backward-compatibility shims, feature flags, or "fallback to old behavior"? → If yes, simplify before adopting; we're in research, not production.
6. **SPA alignment check:** does the change keep our pipeline comparable to SPA's training/eval? Modifications that break comparability (e.g., changing the answer-token loss-mask, changing the SFT objective shape, changing the eval metrics from Pass@1/k) require an explicit case for why the loss of comparability is worth it.

---

## 6. Anti-goals — signs we're off-track

If any of these show up in a writeup or commit, stop and re-evaluate:

- 🚫 "We achieved 93% accuracy" without breaking down by class — that's the imbalance trivially.
- 🚫 RL reward goes up but BP recall stays flat or drops — model is reward-hacking. (Original Sokoban issue, see [progress.md](../progress.md) Step 4.)
- 🚫 SFT or RL evaluated only on the training data distribution — not deployment-relevant.
- 🚫 **Reporting termination metrics without also reporting Pass@1 / Pass@8** on a SPA env. The thesis requires both: a better termination predictor AND a competitive (or better) downstream agent.
- 🚫 **Comparing only against random-policy or heuristic baselines while ignoring vanilla RL / State-Estimation RL / VAGEN.** SPA is our reference frame, not classical baselines.
- 🚫 Adding new components (curriculum, augmentation, multi-stage training) before the simple pipeline is fully eval'd.
- 🚫 Switching to a much larger model "to see if it works better" without first showing the small model's ceiling on the research question.
- 🚫 Bigger eval sets without addressing distribution mismatch.
- 🚫 **Modifying the SPA SFT objective shape (e.g., dropping `<observation>` or `<prediction>` tags, unmasking observation-token loss, etc.) without justifying the break in comparability.**

---

## 7. Decisions: locked vs open

### Locked (don't revisit without strong evidence):
- **Single-environment scope: Sudoku.** Only SPA-paper env satisfying the §1 predictive-gap criterion. Cross-env extension is future work, not current scope.
- **SPA-style training pipeline:** self-play data → world-model SFT (`<observation>`+`<prediction>` tags, observation-token loss masked, ground-truth state replacement) → PPO RL with answer-only loss mask. We extend the tag set with `<solvable>` (termination) but do not change the training shape.
- **Single-step SFT samples** (one (s_t, a_t, s_{t+1}) triple per sample). Prompt is `[system, user_state]`, response is `<think><observation>{s_t}</observation><prediction>{s_{t+1}}</prediction><solvable>{X}</solvable></think><answer>{a_t}</answer>`. *Locked v4 — replaces previous multi-turn lock.*
- **Minimal response tag set:** `<observation>`, `<prediction>`, `<solvable>`, `<answer>`. Dropped `<breaking_point>`, `<terminate_prob>`, `<steps_left>` (see §7.5 v4).
- **Action-conditional `<solvable>` semantics:** `<solvable>` predicts is_solvable(s_{t+1}) given the action in `<answer>`. Useful for "agent self-checks its own next move."
- Single-GPU GRPO/PPO on Qwen2.5-1.5B-Instruct (matches SPA's main reported model).
- Asymmetric `<solvable>` rewards (TP +3.0, FN −2.0, FP −0.5) and format compliance (+0.1/tag) — our termination-specific shaping. *(v4: shifted from BP-tag rewards to solvable-tag rewards since `<breaking_point>` is dropped.)*
- Balanced live-env sampling for the termination part of the RL reward.
- LLM-policy data preferred over random-play for the headline SFT model.

### Open (revisit after first SFT+RL eval):
- Whether to drop **post-BP filler** samples from training (60% of current data, label is trivially "false" regardless of action). Tested as Run B-2 if Run B-1 (single-step format alone) doesn't lift BP recall sufficiently.
- BP-class weighting in SFT cross-entropy loss.
- Difficulty / grid-size choice (we currently use 9×9 easy Sudoku; SPA used 4×4 with 6 empty cells — choose one set for direct comparability).
- RL hyperparameters (KL penalty, learning rate, group size).
- Eval set size and composition.
- Whether to retrain SFT on LLM-policy data, or skip directly to RL after random-play SFT.

---

## 7.5 Format constraints (v4)

After the multi-turn temporal-echo failure (Tier A eval, 2026-04-28), we have **two** format flows. Multi-turn priors as a third flow are removed from this version of the spec.

### A. LLM data-generation output — `<answer>` is the only required tag (locked)
The base model is prompted with a **minimal data-gen system prompt** (`DATA_GEN_SYSTEM_PROMPT_SUDOKU` in `LLMTrajectoryGenerator`) asking only for `<answer>place N at row R col C</answer>`. We don't ask for the full XML during data gen — the base Qwen can't follow it reliably and the wasted decode tokens slow generation 5–20×.

**Quality metric to track:** `parse_failure_rate` per trajectory printed by the data-gen logger. If > 30%, switch back to full prompt or upgrade base model.

### B. SFT target format — minimal XML with ground-truth content (locked v4)
The training target row, built by `format_step()` with variant `sudoku_minimal`, contains exactly:

```
<think>
<observation>{step.state}</observation>             ← env, not LLM (SPA grounding)
<prediction>{step.next_state}</prediction>           ← env, not LLM (SPA grounding)
<solvable>{step.is_solvable}</solvable>              ← oracle, action-conditional
</think>
<answer>{step.action_name}</answer>                  ← action chosen during data gen
```

**Action-conditional semantics:** `<solvable>` is `is_solvable(s_{t+1})` — the solvability *after* the model's chosen action — not the solvability of the current state s_t. This matches our agent-self-checking-its-next-move use case.

**Ground-truth replacement** is required, mirroring SPA §2.2.

**Tags dropped from v3:**
- `<terminate_prob>` — confusing semantics ("how soon will trajectory end" ≠ "is this state doomed")
- `<steps_left>` — SFT-only, not in RL reward, redundant with knowing `done_label`
- `<breaking_point>` — derivable post-hoc from a `<solvable>` time-series at eval (BP at step t = `solvable[t-1]==True AND solvable[t]==False`); explicit tag was empirically not learned (5% recall) and added redundant training signal

**Why locked:** matches SPA's actual SFT shape exactly (their samples are also single-step (s, a, s') triples) plus our one termination-extension tag. Direct comparability.

### Single-turn vs multi-turn — locked: single-step

After Tier A eval revealed multi-turn SFT learned a temporal-echo shortcut (BP recall 5% on multi-turn eval, 0% on single-turn eval), we **dropped multi-turn SFT entirely**:
- Each step in each trajectory is its own training sample.
- Prompt: `[system, user_state]` — no history.
- Response: as in §B above.
- RL stage is still multi-turn rollout (the agent acts in the env multi-turn); the change is only at SFT.

**Why this is the right call:**
- Sudoku is fully observable: the label `is_solvable(s_{t+1})` is a function of `(s_t, a_t)`, not of the trajectory history. History adds zero information about the label.
- Multi-turn priors gave the model a label-history shortcut (echo) that 84.7% of samples could exploit, drowning out the 15.3% gradient signal where the model actually had to look at the grid.
- Single-step removes the shortcut entirely. Every sample requires grid + action conditioning.

### Format-related anti-goals (extends §6)

- 🚫 Adding back any of the dropped tags (`<terminate_prob>`, `<steps_left>`, `<breaking_point>`) without first running an ablation showing the tag improves a metric we care about.
- 🚫 Reverting to multi-turn SFT without a hypothesis explaining why temporal echo wouldn't recur. The empirical result is clear; need new evidence to overturn it.
- 🚫 Changing the `<solvable>` semantics from action-conditional to state-only without updating reward design and eval pipeline together. Mixing semantics across stages is a silent bug source.

| Current activity | Pass/Fail against spec |
|---|---|
| Sudoku as sole target | ✅ §4 in scope; only env satisfying §1 |
| Sokoban work on hold (per [CLAUDE.md](../CLAUDE.md)) | ✅ correct call — out of scope per §4 (no predictive gap) |
| FrozenLake not implemented | ✅ correct call — out of scope per §4 (no predictive gap) |
| Random-play multi-turn data (Track A, done) | ⚠️ generated, but multi-turn structure superseded — kept on disk for reference, not used downstream |
| LLM-policy multi-turn data (Track B, done) | ⚠️ generated, but reformatted to single-step in `data/sudoku_llm_policy_minimal/` for headline training |
| **Single-step SFT (Run B-1, in progress)** | ✅ §7 locked v4 |
| Multi-turn SFT (failed Run B-0) | ❌ deprecated by §7 v4 — kept as cautionary example in [report_2026-04-28_sft_b_diagnosis_and_pivot.md](report_2026-04-28_sft_b_diagnosis_and_pivot.md) |
| Plan to RL with reward v2 (action-conditional `<solvable>` reward) | ✅ §4 in scope, supports §2 Q3 |
| **Plan to compare against vanilla RL, State-Estimation RL, VAGEN** | ❌ Not currently in the plan. Add to remaining-steps in [progress.md](../progress.md). This is a §3 success criterion. |
| Plan to retrain SFT on LLM-policy data | ✅ supports §2 Q4 |
| Plan to compare against `sudoku_baseline.py` heuristic | ✅ §3 nice-to-have, not the load-bearing baseline |
| `simple_sft_trainer.py` over `sft_trainer.py` (FSDP) | ✅ §4 single-GPU constraint |
| `<observation>` + `<prediction>` tags already in our SFT formatter | ✅ matches SPA training shape |

**Concrete deltas this rewrite implies for the plan:**

1. **Stay focused on Sudoku.** Sokoban and FrozenLake are *not* next steps; they're future work after the Sudoku result is in.
2. **Add SPA paper's baselines to the eval pipeline (on Sudoku only):** vanilla RL on base model, State-Estimation-only RL, VAGEN. These produce the comparison numbers that make the thesis testable.
3. **Always report Pass@1 / Pass@8 on Sudoku alongside termination metrics** so the result is comparable to SPA Table 2.
4. The "is the model still a competitive SPA agent" check (Pass@1) is a §6 anti-goal trip-wire — if termination training hurts Pass@1 substantially on Sudoku, that's a failed Q2 and we revisit.

## 9. Reference docs

- [progress.md](../progress.md) — chronological build log
- [CLAUDE.md](../CLAUDE.md) — architecture, design rationale, code-level details (note: pre-dates v2 of this spec)
- [doc/eval_2026-04-28_data_strategy.md](eval_2026-04-28_data_strategy.md) — first eval against this spec (v1 era — Sokoban exclusion noted there is now superseded)
- **Internalizing World Models via Self-Play Finetuning for Agentic RL** (Chen et al. 2025) — the SPA paper, in `doc/`
- **Reinforcement Learning with a Terminator** (Tennenholtz et al. 2022) — the TerMDP paper, in `doc/`
