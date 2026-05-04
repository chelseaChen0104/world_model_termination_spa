# Q&A — SFT Response Tag Design (2026-04-29)

Captures the reasoning behind dropping three response tags from the SFT format. Companion to [spec_project.md](../spec_project.md) §7 ("Locked decisions: minimal response tag set") and [HANDOFF.md](../HANDOFF.md) §4.

**Final tag set (v4 / sudoku_minimal):** `<observation>` + `<prediction>` + `<solvable>` + `<answer>`.
**Dropped tags:** `<breaking_point>`, `<terminate_prob>`, `<steps_left>`.

---

## Q: Why were the three tags dropped?

**Unifying principle: don't train the model to predict things that are derivable from a more primitive prediction.**

The action-conditional `<solvable>` boolean per (s_t, a_t, s_{t+1}) triple is the most primitive signal. Every other "termination-related" tag we considered can be computed at inference time from a sequence of `<solvable>` predictions. Adding them as separate training targets:

- splits gradient signal across redundant targets,
- creates surface metrics that *look* good (high BP-detection accuracy) while masking the actual failure (no `<solvable>` discrimination),
- introduces calibration / definitional pitfalls that don't pay off.

Concrete reasoning per tag below.

---

## Q: Why drop `<breaking_point>`?

A breaking point (BP) = the action that flips solvability from True → False. If you have `<solvable>` per step, the BP is exactly the step where `<solvable>` transitions True→False. Predicting `<breaking_point>` as a separate tag is **predicting the same event twice with a different name** — perfectly correlated within a trajectory.

The bigger reason we removed it: in the v3 multi-turn era, BP-detection accuracy looked like ~95% but it was just **temporal echo** from prior `<solvable>=false` assertions in the conversation history. The "two-metrics" reading masked the actual failure mode (BP recall = 5%, no real `<solvable>` discrimination). With one canonical signal, this kind of confusion can't happen.

**Downstream policy:** any system that needs BPs computes them post-hoc from the `<solvable>` time-series at inference. We do not trust a `<breaking_point>` tag from the model.

---

## Q: Why drop `<terminate_prob>`?

This was supposed to be a continuous probability that the trajectory will terminate. Two problems:

1. **Calibration doesn't come for free from cross-entropy on numeric tokens.** Training the model to emit "0.73" as a sequence of tokens teaches token mimicry, not probability calibration. Achieving real calibration would require a different objective (regression / NLL on a continuous head / quantile loss), which is incompatible with our SFT-on-rendered-text setup.

2. **"Terminate" is semantically ambiguous.** Does it mean *"this rollout will fail eventually"* or *"give up now"*? Two different decisions, conflated.

The cleaner version of the same idea: read **P(true) at the `<solvable>` token directly from logits**. That's a calibrated binary decision (single token), works without changing the SFT objective, and is what our threshold-sweep eval already uses to compute ROC AUC. Free probability without paying the calibration tax.

This is what the [solvable-logprob eval mode](../../evaluate_rl.py) does, and it's what produced B-5's AUC = 0.726 result.

---

## Q: What does `<steps_left>` mean (the dropped tag)?

`<steps_left>` was supposed to predict **how many more steps remain until the trajectory terminates** (by success, deadlock, or hitting max_steps). For each step `t` in a trajectory of length `T`:

| Step `t` | steps_left | Bucket |
|---|---|---|
| 0 (just started) | T | `bucket_20+` |
| 3 (early game) | T−3 | `bucket_11_20` |
| T−2 (one before last) | 2 | `bucket_1_5` |
| T−1 (final step) | 1 | `bucket_1_5` |

In v3 multi-turn the model would emit something like `<steps_left>bucket_6_10</steps_left>` — a prediction of "I think I have 6–10 more steps before this trajectory dies." The four buckets were `bucket_1_5`, `bucket_6_10`, `bucket_11_20`, `bucket_20+`, making it a 4-class classification problem.

**Intent:** complement `<solvable>` ("am I doomed?") with a *horizon* signal ("if doomed, how soon?"). Useful in principle for an agent deciding whether to commit to long-range plans.

---

## Q: Why drop `<steps_left>`?

Three issues, any one of which would have been disqualifying:

1. **Trajectory-dependent label.** The "correct" steps_left at the *current* state depends on what *future* actions the agent will take. Two LLMs (or the same LLM at different temperatures) playing forward from the same state will hit termination at different `t` values. So the label is a fact about *one specific historical rollout*, not a property of the current state alone — unlike `<solvable>`, which is a clean function of `(s_t, a_t)`.

2. **Derivable from `<solvable>` at inference.** If you need a steps-left estimate at runtime, simulate forward (with the same LLM as policy), query `<solvable>` at each step, and count steps until the first `false`. No separate prediction needed.

3. **No RL reward shape.** `<solvable>` has a clean asymmetric reward (TP +3, FN −2, FP −0.5). What's the reward for steps_left being off by 2 buckets? Any ad-hoc reward is arbitrary and disagrees with the trajectory-dependence problem above. So `<steps_left>` would be SFT-only — and would atrophy or drift during RL anyway.

While the *concept* (predict horizon-to-termination) is reasonable, the **implementation as an SFT tag** was problematic on multiple axes.

---

## Q: How does this connect to B-5 working and B-0 failing?

The minimal tag set was a v4 amendment (2026-04-28), made *after* the v3 multi-turn collapse on B-0. It contributed indirectly to B-5's success in two ways:

- With the fuller v3 tag set, the model could (and did) game the easier predictions — copying `<solvable>` from prior turns or echoing `<breaking_point>` post-hoc — without learning the actual discrimination signal. The minimal set removes those gaming opportunities: the `<solvable>` token is the only termination-related decision in the response.

- It makes the loss interpretable. When we observe terminal eval_loss ≈ 0.015 across runs, we know it's mostly format/scaffold tokens (the rendered grid, repeated in `<observation>` and `<prediction>`). We can attribute the discrimination signal cleanly to the threshold-sweep AUC at the `<solvable>` token, not to a tangled mix of correlated tag predictions.

Net effect: fewer ways for the model to game the loss without learning the thing we care about. B-5 reaching AUC 0.726 on 4×4 — vs B-0's multi-turn collapse at BP recall 5% with the fuller tag set on similar data — is consistent with this.

---

## TL;DR

| Dropped tag | One-line reason |
|---|---|
| `<breaking_point>` | Derivable post-hoc from `<solvable>` time-series; previously masked failure via temporal echo. |
| `<terminate_prob>` | Calibration doesn't come from token-level cross-entropy; logprob at `<solvable>` is the cleaner signal. |
| `<steps_left>` | Trajectory-dependent label, derivable from `<solvable>`, and no clean RL reward shape. |

Final SFT response shape:
```xml
<think>
<observation>{rendered s_t}</observation>
<prediction>{rendered s_{t+1}}</prediction>
<solvable>{true|false}</solvable>
</think>
<answer>place {N} at row {R} col {C}</answer>
```
