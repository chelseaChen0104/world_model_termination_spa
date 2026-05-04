# Pass@1 vs Pass@k, Greedy vs Stochastic — Roles in SFT, RL, Rollout, and Eval

A reference for which metric to use when, and how each one is computed in our codebase. Read this if you've ever been confused by a result that says "Pass@1 = 0% but per-batch solve = 53%" — that's regime-1 territory and this doc explains it from first principles.

> **Companion docs**: [eval_2026-05-01_pentomino_greedy_gap.md](../eval_2026-05-01_pentomino_greedy_gap.md) (Pentomino-specific regime-1 mechanism), [handbook_rl_frameworks.md](handbook_rl_frameworks.md) (PPO/GRPO context).

---

## 1. Two orthogonal axes

There are exactly two independent decisions when measuring "did the model solve this puzzle":

### Axis 1 — Decoding mode: greedy vs stochastic

Per-token selection rule when generating the response:

| Mode | Rule | Reproducibility |
|---|---|---|
| **Greedy** (`temperature=0`) | `next_token = argmax(logits)` | deterministic — same input → same output |
| **Stochastic** (`temperature>0`) | `next_token ~ multinomial(softmax(logits/T))` after `top_k`/`top_p` filter | random — different draws each invocation |

Greedy gives one canonical "best guess" per puzzle. Stochastic gives a distribution; each sample is one draw from it.

### Axis 2 — Sampling budget: 1 vs k

How many independent rollouts to spend per puzzle:

| Budget | Notation | Computation |
|---|---|---|
| **k = 1** | "Pass@1" | one rollout per puzzle, success = solve fraction |
| **k > 1** | "Pass@k" | k rollouts per puzzle, success = ANY of k solved |

### The four-way grid

| | Greedy | Stochastic |
|---|---|---|
| **k = 1** | **Pass@1 greedy** — single deterministic attempt | **Pass@1 stochastic** — single random sample (noisy point estimate) |
| **k > 1** | Pass@k greedy is degenerate (same as Pass@1; greedy is deterministic) | **Pass@k stochastic** — most informative; k samples, count puzzles where any solved |

Pass@k greedy doesn't really exist as a distinct metric — running greedy k times gives the identical completion every time. So practically:
- **"Pass@1"** without qualifier usually means Pass@1 greedy.
- **"Pass@k"** without qualifier usually means Pass@k stochastic for some k > 1.
- **"per-batch solve rate"** is our project's specific shorthand: 1-of-K stochastic on the K rollouts collected per puzzle during RL training (we use K=8 by default).

---

## 2. The relationship: Pass@k stochastic is governed by `p`

Let `p` = single-rollout success probability under stochastic sampling for a given puzzle. Under independence:

```
Pass@k stochastic ≈ 1 - (1 - p)^k
```

| `p` (single-rollout) | Pass@1 stochastic | Pass@8 stochastic |
|---:|---:|---:|
| 0.01 | 1.0% | 7.7% |
| 0.05 | 5.0% | 33.7% |
| 0.10 | 10.0% | 56.9% |
| 0.20 | 20.0% | 83.2% |
| 0.50 | 50.0% | 99.6% |

So a model with even a 5% per-rollout success probability will produce Pass@8 ≈ 34% — a 6.7× lift just from sampling more. **This is why per-batch (1-of-8) is dramatically more forgiving than Pass@1 greedy** — and why the regime-1 stochastic-greedy gap can be huge even when the underlying policy isn't catastrophically broken.

---

## 3. The roles per training stage

### 3.1 SFT — neither metric is computed during training

SFT loss is supervised cross-entropy on token sequences. There's no rollout; the model is trained to reproduce the *exact* response in the training data given the prompt. Loss is teacher-forced — at each token position, the model is conditioned on the *true* prefix from the training response, regardless of what the model itself would have generated.

So during SFT:
- No Pass@1 / Pass@k is computed at training time
- The closest thing is `eval_loss` (cross-entropy on a held-out validation parquet)
- A low eval_loss does NOT imply a good policy — the model could be perfectly memorizing prompts (Hidato's 183 unique samples → eval_loss=0.001 → Pass@1=0%)

**After SFT, before RL**, you should measure:
- **Pass@1 greedy** (single deterministic eval) — does the trained policy actually solve puzzles?
- **Pass@k stochastic** (k=8 or higher) — does it have *any* positive signal worth bootstrapping from in RL?

If Pass@k = 0%, RL has no reward signal to climb from. If Pass@1 = 0% but Pass@k > 0%, you're in regime-1 territory — RL can still climb but greedy will lag.

### 3.2 RL training rollout — uses Pass@k stochastic implicitly

GRPO rollout collection samples K completions per prompt with `temperature > 0`. Each rollout's reward feeds the group-relative advantage:

```python
# In our trainer (rl_trainer_v6.py)
for puzzle in batch:
    K_rollouts = sample_k_completions(puzzle, K=group_size, temperature=0.7)
    rewards = [score(r) for r in K_rollouts]
    advantages = (rewards - mean(rewards)) / std(rewards)
```

The "per-batch solve rate" we log every training step is exactly: `(# of rollouts with success=True) / (n_puzzles_per_batch * group_size)`. With group_size=8, this is the empirical 1-of-8 stochastic Pass@k for each puzzle, averaged over the batch.

**Why we use stochastic rollouts during RL**:
- *Exploration*: greedy at every step would visit the same trajectory every time. RL needs to see varied actions to learn which are good.
- *Group-relative advantage requires variance*: if all K rollouts of a puzzle return the same reward, advantage = 0 and the policy doesn't learn anything for that puzzle. Stochasticity ensures variance.
- *Off-policy correction via PPO ratio*: the ratio `exp(new_logp - old_logp)` only makes sense when there's stochasticity to compute logprobs over.

**`temperature=0` during RL is broken**. The trainer forces `do_sample=True` whenever `cfg.temperature > 0`; if you set temperature=0 by mistake, GRPO degenerates to vanilla PG with single rollouts and probably won't train at all.

### 3.3 RL training-time eval — uses Pass@1 greedy

Every `eval_every` steps (default 25), the trainer pauses RL and runs a clean eval:

```python
# rl_trainer_v6.py: quick_pass1()
old_temp = cfg.temperature  # save 0.7
cfg.temperature = 0.0       # switch to greedy
ro = do_rollout(...)         # one greedy rollout per eval puzzle
cfg.temperature = old_temp
```

This gives **Pass@1 greedy** on a fixed eval set (default 30 puzzles). Logged as `pass@1` in the per-step JSONL. This is what our doc tables usually mean by "Pass@1 trajectory across training".

**Why greedy at eval-time** (not stochastic):
- Reproducibility — same Pass@1 every time means we can compare across runs
- Reflects deployment behavior — at inference we usually want deterministic, "best" answers

### 3.4 Final / paper / external comparison eval — both, ideally

For the paper, we want to report numbers comparable to the SPA paper which uses Pass@1 and Pass@k:

- **Pass@1 (greedy)** — the canonical "how good is your model" headline.
- **Pass@k (stochastic, k=8)** — the SPA paper's secondary headline; reflects "how good with extra compute at deployment".

To compute these properly:
- Use a *different* puzzle set than what the model was trained or training-evaled on.
- Use enough puzzles that the standard error is small (≥100 puzzles is a typical paper-quality bar; n=30 has 1-puzzle = 3.3pp noise).
- Use `temperature=0.7` for stochastic Pass@k (matches SPA paper conventions).
- Report the gap between greedy and stochastic — it's a structural property of the env and policy.

---

## 4. The greedy-stochastic gap as a diagnostic

The gap `Pass@k stochastic − Pass@1 greedy` is informative:

| Gap | Interpretation | Examples in our project |
|---:|---|---|
| ~ 0pp | Calibration is fine; argmax matches the high-probability path. | Sudoku Run A: greedy Pass@1 33%, stochastic similar. |
| > 0pp moderate | Some greedy decisions land in low-probability regions; sampling diversifies and recovers. | Hidato B-H1 RL early: greedy 16.7%, stochastic per-batch 25-50%. |
| > 0pp huge | Bimodal collapse / regime-1: argmax is consistently wrong, but the runner-up token has non-trivial probability. | Pentomino B-8 RL: greedy 0%, stochastic per-batch 84%. |
| < 0pp | Implausible; would mean greedy is *better* than the average sample. | Never observed. |

A huge gap usually signals **calibration mismatch in the action policy** (the model knows roughly what to do, but the argmax token is "almost-but-not-quite right"). Fixes that target this:
- v8.2 dual-token KL anchor (RL-stage)
- `--action-quality-bonus` (direct gradient on action quality, RL-stage)
- Larger model (often resolves bimodality without other intervention)

A gap that stays at ~0pp with low absolute Pass@1 means the policy is **fundamentally weak** — RL/data interventions, not calibration tricks, are needed.

---

## 5. Concrete cheat sheet by use case

| Use case | Metric | Why |
|---|---|---|
| "Should I move from SFT to RL?" | Pass@k stochastic (k≥4) > 0 | RL needs positive reward signal to climb |
| "Did RL improve the headline number?" | Pass@1 greedy | Reproducible single-attempt success |
| "Did RL hurt calibration?" | solvable_acc + bp_recall | Per-step viability prediction quality |
| "Is my model in regime-1 collapse?" | Pass@1 greedy ≪ Pass@k stochastic | Diagnostic for bimodal action distribution |
| "Is the SFT data leak fixed?" | Pass@1 greedy lifts off 0% | Removing the doom-text shortcut should restore greedy |
| "Compare against SPA paper" | Pass@1 + Pass@8, both | SPA Table 5 uses both |
| "Deployment ROI of the truncation gate" | (greedy Pass@1, tokens/episode) | Tradeoff curve |

---

## 6. How each metric is computed in our code

| Metric | Where | Function |
|---|---|---|
| Pass@1 greedy (training-time eval) | [rl_trainer_v6.py](../../src/training/rl_trainer_v6.py) | `quick_pass1()` (sets temperature=0, runs `do_rollout` on `eval_n_puzzles`) |
| Per-batch solve rate (1-of-K stochastic) | [rl_trainer_v6.py](../../src/training/rl_trainer_v6.py) | logged as `solved %` in each step's metric line; computed from `do_rollouts_batched` results |
| Pass@1 / Pass@k offline (paper-quality) | [evaluate_rl.py](../../evaluate_rl.py) | `--metric pass-at-k` mode |
| Stochastic Pass@k post-hoc | [scripts/sanity_check_checkpoint.py](../../scripts/sanity_check_checkpoint.py) | Section C — runs `K` stochastic rollouts per puzzle, reports Pass@k |
| Stochastic per-rollout sweep | [scripts/eval_hidato_stochastic_pass.py](../../scripts/eval_hidato_stochastic_pass.py), [scripts/debug_polyomino_one_rollout.py](../../scripts/debug_polyomino_one_rollout.py) | per-puzzle distribution at varying T |

---

## 7. Common pitfalls

### 7.1 "AUC = 1.000 but Pass@1 = 0%"

ROC AUC on logprobs measures the model's *ranking* of `<solvable>=true` vs `=false` across teacher-forced positions. It says nothing about the model's *generative* behavior. A model can perfectly rank a small training distribution (memorized) while collapsing to garbage on greedy decode. Always pair AUC with at least one rollout-based metric.

### 7.2 "Per-batch solve rate is high, why is Pass@1 zero"

You're in regime-1. Per-batch (1-of-8 stochastic) is forgiving by 8× sampling; Pass@1 (1-of-1 greedy) is not. The fix targets the action-policy bimodality, not the sampling budget.

### 7.3 "Greedy Pass@1 is 0% — RL won't work"

Not necessarily. RL works on stochastic rollouts. If per-batch (1-of-K stochastic) is positive, RL can lift the greedy-collapse model. Hidato B-H1 went from greedy Pass@1 16.7% to 66.7% over 157 RL steps even though the SFT was bimodal-prone.

### 7.4 "Stochastic Pass@1 is the same as Pass@1"

Specifically, "Pass@1 stochastic" = 1 stochastic rollout per puzzle. Different invocations give different numbers (the noise is large). To make it useful, you need to either:
- Average over many puzzles (n ≥ 100)
- OR report Pass@k for k ≥ 4 instead

---

## 8. Recommended reporting template

For a new SFT or RL run, report at minimum:

```
              Greedy        Stochastic (k=8, T=0.7)
              ──────       ─────────────────────────
Pass@k:       Pass@1: X%   Pass@8: Y%, per-batch: Z%
solvable_acc: A            (not applicable — per-step metric)
bp_recall:    B            (not applicable)
ROC AUC:      C            (teacher-forced; same regardless of decoding)

n_eval_puzzles: 30 (or 100+ for paper)
eval seed: <seed>
checkpoint:  <path>
```

This gives the calibration (solvable_acc, bp_recall, AUC), the action-policy quality (Pass@1 greedy), and the action-policy ceiling (Pass@8 stochastic), in one glance.

---

## 9. Quick reference table

| Term | What it measures | Default in our code |
|---|---|---|
| `temperature` | per-token sampling temperature | 0.7 (rollout), 0.0 (eval/greedy) |
| `top_k` | only top-k tokens are candidates | 50 |
| `top_p` | nucleus filter | 0.95 |
| `group_size` | K rollouts per puzzle in RL | 8 |
| `n_puzzles_per_batch` | distinct puzzles per RL step | 4 |
| `eval_n_puzzles` | puzzles in the eval set | 30 |
| `eval_every` | RL steps between evals | 25 |

---

## 10. Pointers

- [eval_2026-05-01_pentomino_greedy_gap.md](../eval_2026-05-01_pentomino_greedy_gap.md) — full Pentomino regime-1 walkthrough
- [handbook_rl_frameworks.md](handbook_rl_frameworks.md) — PPO/GRPO context
- [handbook_data_and_model_sanity.md](handbook_data_and_model_sanity.md) — sanity checks for these metrics
- [rl_walkthrough_2026-04-30.md](rl_walkthrough_2026-04-30.md) — RL training step-by-step
- [HuggingFace generation docs](https://huggingface.co/docs/transformers/main_classes/text_generation) — canonical `temperature`, `top_k`, `top_p` semantics
- [Chen et al. 2021 HumanEval](https://arxiv.org/abs/2107.03374) — formal Pass@k definition with the unbiased estimator (we use the simpler "any of k" form)
