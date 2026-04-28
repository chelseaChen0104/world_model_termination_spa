# Project Spec — World Model Termination Prediction

**Status:** living document. Update when scope changes.

- **v1** — first draft. Excluded Sokoban for "heuristic detector exists." Wrong reason.
- **v2** — re-included Sokoban + FrozenLake citing SPA paper alignment. Wrong reason.
- **v3** (current, 2026-04-28) — excludes Sokoban + FrozenLake on the *right* reason: neither environment exhibits the **predictive gap** between fatal move and formal failure that our research question requires. Sudoku is the only SPA-paper environment where a state can be "visually solvable but already doomed K steps before failure." See §1 "Predictive gap" for the criterion.

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
4. **Q4 — Does training on LLM-policy data outperform training on random-play data** on an LLM-policy-distribution eval set? (SPA uses self-play; we test the contrast explicitly.)
5. **Q5 — Does multi-turn context (game history) improve BP detection over single-turn** in fully-observable domains? SPA found that multi-turn RL "broadens the reasoning frontier" — does the same hold for termination prediction?

Q6 from v2 ("does the recipe transfer across Sokoban / FrozenLake / Sudoku") is **dropped** because Sokoban and FrozenLake fall outside the predictive-gap criterion (§1). Cross-environment transfer is interesting but is not what our research question is asking — at most, a future-work bullet.

Anything that doesn't help answer one of Q1–Q5 is suspect.

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
| **Single-GPU GRPO / PPO RL** | SPA was single-GPU; our budget is single H800; intentional simplicity over RAGEN/Ray/vLLM. |
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
| **Distributed training (FSDP, Ray, vLLM serving)** | SPA was single-GPU; we mirror that constraint. The existing `sft_trainer.py` (FSDP) is dead code; `simple_sft_trainer.py` is the path. |
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
- **SPA-style training pipeline:** self-play data → world-model SFT (`<observation>`+`<prediction>` tags, observation-token loss masked, ground-truth state replacement) → PPO RL with answer-only loss mask. We extend the tag set with termination tags but do not change the training shape.
- Single-GPU GRPO/PPO on Qwen2.5-1.5B-Instruct (matches SPA's main reported model).
- Asymmetric BP rewards (TP +3.0, FN −2.0, FP −0.5) and format compliance (+0.1/tag) — our termination-specific shaping.
- Balanced live-env sampling for the termination part of the RL reward.
- Multi-turn SFT format with sliding window.
- `steps_left` is SFT-only, not in RL reward.
- LLM-policy data preferred over random-play for the headline SFT model.

### Open (revisit after first SFT+RL eval):
- Sliding window size (currently 10).
- BP-class weighting in SFT cross-entropy loss.
- Difficulty / grid-size choice (we currently use 9×9 easy Sudoku; SPA used 4×4 with 6 empty cells — choose one set for direct comparability).
- RL hyperparameters (KL penalty, learning rate, group size).
- Eval set size and composition.
- Whether to retrain SFT on LLM-policy data, or skip directly to RL after random-play SFT.
- **Multi-turn prior-turn content policy** — see §7.5 below.

---

## 7.5 Format constraints

We have three distinct format flows and they must not be conflated. The actual code is in [src/data/llm_trajectory_generator.py](../src/data/llm_trajectory_generator.py) and [src/data/sft_formatter.py](../src/data/sft_formatter.py); this is the spec the code must satisfy.

### A. LLM data-generation output — `<answer>` is the only required tag (locked)
The base model is prompted with a system prompt describing the full designed format (`<observation>`, `<prediction>`, `<steps_left>`, `<solvable>`, `<breaking_point>`, `<answer>`), but **only `<answer>...</answer>` is required to be parseable** for the action to be extracted. If unparseable, fall back to a random valid action and increment a parse-failure counter.

**Why locked:** the base Qwen has never seen our format. Strict format requirements at this stage would produce mostly random fallbacks and defeat the purpose of using LLM-policy.

**Quality metric to track (not a gate, just monitor):** `parse_failure_rate` per trajectory — if > ~30% across the dataset, the data is effectively random-play and trackB loses its value over Track A. Print this in the per-trajectory log.

### B. SFT target format — full XML with ground-truth content (locked)
The training target row, built by `format_step()`, must contain **all** designed tags:

```
<think>
<observation>{step.state}</observation>             ← env, not LLM
<prediction>{step.next_state}</prediction>           ← env, not LLM
<terminate_prob>...</terminate_prob>                 ← derived from steps_left
<steps_left>{step.steps_left_bucket}</steps_left>    ← oracle
<solvable>{step.is_solvable}</solvable>              ← oracle
<breaking_point>{step.is_breaking_point}</breaking_point>  ← oracle
</think>
<answer>{step.action_name}</answer>                  ← what was actually executed
```

**Ground-truth replacement is required**, mirroring SPA §2.2: "we replace the model's beliefs about current states ŝₜ and future states ŝₜ₊₁ with the ground-truth states." Even if the LLM produced perfectly-formatted XML during data gen, we discard it for the target row and rebuild from oracle.

**Why locked:** comparability with SPA's training shape. Modifying this format (e.g., dropping `<prediction>`) breaks Q1/Q2 measurability.

### C. Multi-turn prior-turn content — open research decision
Prior assistant turns in multi-turn samples currently use `step.llm_raw_response` if present, else fall back to template-generated ground-truth. So a multi-turn sample looks like:

```
[system] {full_format_prompt}
[user] Current state: {state_0}
[assistant] {llm_raw_response_0}    ← LLM output, may be malformed
[user] Action executed. Current state: {state_1}
[assistant] {llm_raw_response_1}    ← LLM output, may be malformed
...
[user] Action executed. Current state: {state_K}
[assistant] {format_step(step_K)}   ← TARGET — clean ground-truth XML
```

**Two valid policies, both defensible:**

1. **Keep LLM-raw priors** (current behavior) — matches inference distribution: the model's own prior outputs at deployment will also be imperfect. Realistic noise.
2. **Replace prior content with ground-truth-formatted XML** — cleaner training signal; prior turns model the format the model is being trained to produce. SPA's RL is single-turn, so they don't take a position on this.

**Decision is open.** Track parse_failure_rate from §A; if it's high (>30%), policy (1) gives the model garbage to condition on and (2) becomes more attractive. If low, (1) is fine.

### Format-related anti-goals (extends §6)

- 🚫 Reporting BP recall on a model whose `<breaking_point>` tag is wrapped in unexpected whitespace, code fences, or alternative casing not handled by the eval parser. Eval must be tolerant of LLM-output noise OR the model must be reliably formatted; pick one and document.
- 🚫 Changing `format_step()`'s tag set or order without re-running all baselines. The XML format IS the API between SFT, RL, and eval — drift is silent and corrosive.
- 🚫 Using LLM-raw multi-turn priors **without measuring `parse_failure_rate`**. We need to know whether the priors are signal or noise.

| Current activity | Pass/Fail against spec |
|---|---|
| Sudoku as sole target | ✅ §4 in scope; only env satisfying §1 |
| Sokoban work on hold (per [CLAUDE.md](../CLAUDE.md)) | ✅ correct call — out of scope per §4 (no predictive gap) |
| FrozenLake not implemented | ✅ correct call — out of scope per §4 (no predictive gap) |
| Random-play multi-turn data (Track A, done) | ✅ §4 acknowledged intermediate |
| LLM-policy multi-turn data (Track B, running) | ✅ §4 in scope; this is SPA's self-play recipe |
| Multi-turn SFT with sliding window | ✅ §4 in scope, supports §2 Q5 |
| Plan to RL with reward v2 | ✅ §4 in scope, supports §2 Q3 |
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
