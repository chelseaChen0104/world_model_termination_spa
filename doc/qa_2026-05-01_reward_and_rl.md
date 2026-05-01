# Q&A — Reward Design and RL Mechanics (2026-05-01)

Captures the questions and answers from a deep-dive on how the reward function and PPO/GRPO update interact, in particular:
- How reward is assigned to action tokens
- Why we have a per-step reward and a per-rollout total
- What "dynamic calibration drift" means and why it's the actual collapse mechanism
- Why Sudoku doesn't need an action-quality reward but Pentomino-easy does

Companions:
- [qa_2026-04-30_rl_diagnostics.md](qa_2026-04-30_rl_diagnostics.md) — what KL / pg_loss / clipfrac mean.
- [rl_walkthrough_2026-04-30.md](rl_walkthrough_2026-04-30.md) — step-by-step trainer walkthrough.
- [eval_2026-04-30_b7_rl_phase1.md](eval_2026-04-30_b7_rl_phase1.md) — the v6 collapse + v7/v8 redesigns.
- [sanity_2026-04-30_b7_rollout_stats.json](sanity_2026-04-30_b7_rollout_stats.json) — empirical numbers cited below.

---

## Q: How is reward given for the `<answer>...</answer>` token (the action)?

The action token has **no direct per-token reward score**. The reward function never says "your placement at row 1 col 1 was good/bad." Instead, the action affects the rollout's total reward through three indirect paths:

### Path 1 — Format compliance (small, applies to the *tag*, not the content)

```python
# rl_trainer_v6.py:153
def format_reward(text, per_tag):
    return sum(per_tag for t in REQUIRED_TAGS if t.lower() in text.lower())
```

Just having an `<answer>` tag in the response gives +0.05. Doesn't matter what's inside it.

### Path 2 — Indirect via the next state's reward

The action is what causes the env to transition to a new state. That next state has a `gt_solvable` label which then drives the *next step's* reward.

| If the action leads to → | Next state `gt_solvable` | Next step's `calib_reward` (if model predicts correctly) |
|---|---|---|
| solvable next state | True | +1.0 (TN) |
| doom next state | False | +1.0 (TP) |

Both give +1.0 reward as long as the model's `<viability>` prediction is correct. So if calibration is perfect, the model gets the same per-step reward whether it picked a *good* action (leads to solvable) or a *bad* action (leads to doom). **The action's quality is invisible in the per-step reward when calibration is good.**

What does differ between good and bad actions:
- **Trajectory length**: a good action keeps the rollout alive for another step → another chance to earn +1.0.
- **success_bonus**: only good actions can eventually lead to a complete tiling → +3.0 final bonus.

So good actions accumulate *more* total reward over the rollout, even though each individual step gives the same +1.0. That's the only way action quality enters the reward signal indirectly.

### Path 3 — Direct action-quality reward (optional, default off)

The `--action-quality-bonus N` flag added in v8.1:
- +N if the action lands on a solvable state
- −N if the action lands on a doom state

This is the only mechanism that scores the action directly per-step. Off by default; turned on only when v8's calibration anchor isn't enough by itself.

### How action tokens get gradient anyway

Even though action tokens have no direct per-token reward, they still get gradient through PPO + GRPO advantage broadcast:

1. Sum `step_reward` across all steps of the rollout → `final_reward`.
2. GRPO computes per-rollout advantage: `A_i = (final_reward_i − group_mean) / group_std`.
3. PPO loss for *every* response token (action, viability, observation, …) uses **the same scalar `A_i`**.

```python
# rl_trainer_v6.py:732-733
adv_t = torch.full_like(new_logp, adv)   # same scalar for every token
```

So if rollout `i` had advantage `+1.5`, every token of that rollout's responses gets pushed up by gradient proportional to `+1.5 × ratio`. If rollout `j` had `−0.5`, all its tokens get pushed down. The action gradient comes from *whether the rollout this action started ended up above or below the group mean*, not from any per-action scoring.

### Why this is weak on Pentomino-easy

The action gradient does useful work only when rollouts in the same group have meaningfully different rewards. On Pentomino-easy:
- 73% of rollouts are 1-step doom with `final_reward ≈ +1.30`.
- 27% are 2-step with `final_reward ≈ +2.5`.
- 0% reach success → no `+3.0` bonus.

In a typical group of 8: ~6 land at +1.30, ~2 land at +2.5. group_std is small. Advantages are small (~±0.3). The action gradient is weak; the action policy barely moves in 200 RL steps.

That's why **adding `--action-quality-bonus 1.0` is the natural Plan B** if calibration anchoring alone doesn't lift Pass@1 — it gives every doom action a direct −1.0 hit, creating much stronger gradient pressure on action tokens.

---

## Q: Why do we have both per-step reward and per-rollout total reward?

| Quantity | Where it lives | What it represents |
|---|---|---|
| `step.calib_reward` + `step.fmt_reward` + `step.progress_reward` + `step.action_quality_reward` | each `StepRecord` | Reward components attributable to *this single transition*. Set when the step happens during the rollout. |
| `step.step_reward` | each `StepRecord` | Sum of the above components. The "what happened on this turn" scalar. |
| `rollout.final_reward` | each `Rollout` | `sum(step.step_reward for step in steps) + (success_bonus if solved else fail_bonus)` |

### Why per-step decomposition exists

1. **Reward is naturally per-step.** Each step has a real ground-truth label (`info["is_solvable"]`) that arrives only after that action. The calibration reward on step *t* is intrinsically a per-step quantity — you can only assign it where it happened.

2. **v7's class-balance rescaling needs it.** We want to rescale the calibration term by inverse-frequency without disturbing format reward or success bonus, so we keep the components separate:

```python
# rl_trainer_v6.py:599-606  (rebalance_rewards)
for s in ro.steps:
    w = w_solv if s.gt_solvable else w_doom
    s.calib_reward = float(s.calib_reward) * w        # rescale ONE component
    s.step_reward = (s.calib_reward + s.fmt_reward    # recompose step
                     + s.progress_reward + s.action_quality_reward)
```

3. **Logging and diagnostics.** When something goes wrong (calibration collapse, format-breaking, …), per-step decomposition lets us answer: "Did the model lose the format reward, or the calibration reward, or both?" Without it, we'd just see `final_reward` move with no signal for why.

### Why we sum to a single rollout-level scalar for PPO

GRPO is *rollout-level by design*. It doesn't have a per-step value function — instead, it computes the advantage as:

```python
# rl_trainer_v6.py:660-666
group_rewards = np.array([r.final_reward for r in group])
baseline = group_rewards.mean()
std = group_rewards.std() + 1e-8
advantages = [(r - baseline) / std for r in group_rewards]
```

This needs **one scalar per rollout** — that's what gets compared across the group. GRPO uses the variance across rollouts on the *same starting state* as its baseline (instead of a learned value function), and that comparison only makes sense at the rollout level.

### The chain

```
per-step reward components  ──sum──▶  step.step_reward
                                            │
                                            ▼
                                    sum + success_bonus
                                            │
                                            ▼
                                    rollout.final_reward (one scalar / rollout)
                                            │
                                            ▼
                            (group of 8)  ──GRPO──▶  advantage_i
                                                          │
                                                          ▼
                                              broadcast to every
                                              token of every step
                                                          │
                                                          ▼
                                                     PPO loss
```

### The alternative we *didn't* take

PPO with GAE (Generalized Advantage Estimation) uses per-step advantages with a learned value function `V(s)`:

```
per-step reward → bootstrap with V(s_t+1) → per-step TD error → per-step advantage
```

This gives more precise credit assignment. But it requires training a separate value-function head, careful bootstrapping, and more hyperparameters (GAE lambda, value-loss coef). GRPO trades that complexity for simplicity by using same-puzzle group baselines instead. For our scale (1.5B model, ~32 rollouts/step), GRPO is the right tradeoff.

---

## Q: What is "dynamic calibration drift"?

Three pieces:

1. **Calibration** = the model's `<viability>` prediction is well-aligned with ground truth. B-7 SFT predicts True 26.3% of the time, true class is 27%, accuracy 98%. That's calibrated.

2. **Drift** = the policy slowly walks off that calibrated point during RL. Predicted-True rate slides from 26% → 5% → 0% over ~50-100 training steps. Each gradient step nudges it down by a tiny amount; cumulatively the small nudges add up to a complete flip.

3. **Dynamic** = the drift comes from the gradient steps themselves, even though the *static* reward landscape (the function "what reward does this policy get?") says the model started at the global maximum. The optimizer walks downhill from a high point, not because there's a downhill direction it's *trying* to find, but because the local gradient is noisy/biased and the KL leash isn't tight enough to hold.

### The reward landscape picture

Plot per-step expected reward (under v7) on the y-axis vs. policy on the x-axis:

```
 reward
  +1.0 ┤●●●●●●●●●●  ← B-7 SFT lives here. Global max. Oracle calibration.
       │           ╲
       │             ╲
       │               ╲
  +0.5 ┤                  ●●  ← "always False" basin. Local max.
       │                   /
       │                  /
   0.0 ┤                /
       │
  -0.5 ┤●  ← "always True" — never visited; gradient never goes here.
       └──────────────────────────────
        SFT          drift            "always False"
```

Two important features:

- **The top of the hill (SFT/oracle) is flat.** At the global max, the gradient is zero by definition. Gradient steps with noise push the policy in random directions — *any* direction looks downhill from a peak.
- **The "always-False" valley is also a local max.** From inside it, the gradient says "stay here." Predicting True more often would mean some FN errors (predict True on doom, GT False, gives −1.0 in v7), and those errors hurt reward. The local optimizer wants to go *deeper* into "always False," not back up to oracle.

So the trajectory is:
1. Noisy GRPO advantages nudge the policy off the flat peak.
2. KL leash (`kl_coef=0.05`, averaged over all ~150 response tokens) is too weak to pull it back.
3. Policy slides down into the always-False basin.
4. Local gradient now keeps it there.

### The pencil analogy

A pencil balanced on its tip. It's at the highest point a pencil-shaped object can be. Statically, that's the maximum of "tip-height." But it's *unstable* — any tiny breath of wind tips it over, and once tipped, gravity pulls it all the way down to lying flat. The "drift" from upright to flat isn't because lying-flat has higher tip-height; it's because the upright position is an unstable stationary point.

B-7 SFT is the upright pencil. Always-False is the pencil lying flat. The reward landscape says upright is best, but the dynamics of RL training (noisy gradients + weak global KL leash) won't keep the pencil upright over 200 steps.

### Why Fix 1 (tag-specific KL anchor) addresses it

A KL anchor specifically on the `<viability>` tokens is like adding a stronger spring connecting the pencil-tip to the upright position. The action tokens still get gradient (the pencil body can wobble), but the spring on the viability tokens prevents them from drifting. The pencil stays upright because we explicitly hold its tip.

Mathematically: add `+λ · KL(π_policy(viability_token) || π_SFT(viability_token))` to the loss with `λ ≈ 0.5`. The current calibration is by construction the minimum of this KL term — any drift from it incurs penalty proportional to `λ`. With `λ` large enough, the drift signal is overwhelmed and calibration stays locked.

The general PPO loss already has a KL penalty (the `kl_coef * kl` term), but it's averaged over *all* response tokens (~150 per response), so the per-token effect is tiny (~0.05/150 ≈ 0.0003). A viability-tag-specific anchor concentrates the penalty on the 1-3 tokens that matter for calibration, making it much stronger per-token without changing action-policy training.

---

## Q: Why does Sudoku not need an action-quality reward, but Pentomino does?

Sudoku's *structure* gives the policy implicit action-quality signal that Pentomino doesn't.

### Three structural differences

**1. Trajectory length distribution**

Sudoku 4×4 trajectory at training temperature:
- Steps 1–3: empty/sparse board, almost any move is solvable → mostly GT=True.
- Steps 4–5: constraints tighten → mix of True and False.
- Step 5–6: BP often hits → GT=False.

Per-trajectory composition: ~60% GT=True states, ~40% GT=False. Both classes contribute real gradient.

Pentomino-easy trajectory:
- Step 1: 73% terminate as doom → GT=False on a doomed state, then trajectory ends.

Per-trajectory composition: ~73% GT=False, ~27% GT=True.

**2. The "always-False" expected reward edge**

Plug class frequencies into v6's reward (`TP +1.0 / FN -0.7 / FP -0.5 / TN +0.3`):

| | Sudoku (60% True / 40% False) | Pentomino (27% True / 73% False) |
|---|---|---|
| E[r \| always False] | 0.6×(-0.5) + 0.4×(+1.0) = **+0.10** | 0.27×(-0.5) + 0.73×(+1.0) = **+0.60** |
| E[r \| always True] | 0.6×(+0.3) + 0.4×(-0.7) = **-0.10** | 0.27×(+0.3) + 0.73×(-0.7) = **-0.43** |
| Edge of "always False" | **+0.20/step** | **+1.03/step** |

The collapse-attractor for "always False" is **5× stronger on Pentomino**. On Sudoku the edge is small enough that other gradient signals overpower it.

**3. success_bonus actually fires on Sudoku**

This is the biggest one. From Sudoku Run A's eval trajectory (lr=1e-5 continuation):

| step | Sudoku Pass@1 | success_bonus contribution per rollout |
|---|---|---|
| 0 | 6.67% | 6.67% × +3 = **+0.20 expected** |
| 250 (peak) | 36.67% | 36.67% × +3 = **+1.10 expected** |
| 500 (final) | 33.33% | 33.33% × +3 = **+1.00 expected** |

Pentomino-easy at any RL step: Pass@1 = 0% → success_bonus contribution = **+0.00 expected**.

So on Sudoku, the success_bonus is a real gradient signal worth **+1.0 reward** in expectation per rollout once the policy starts solving. That's larger than the per-step viability reward and directly says "find action sequences that complete the puzzle." The policy gets goal-directed pressure for free.

On Pentomino, success_bonus contributes essentially zero because nothing succeeds. Without that, the only gradient signal *is* per-step viability — and per-step viability has the always-False attractor.

### The summary

The thing that makes Sudoku self-regulate is **trajectory length × success rate**. Both bottom out on Pentomino-easy — that's not a reward-shape problem, it's an env-shape problem.

Two ways to fix it:

| Fix | Mechanism |
|---|---|
| **v8 reward** (viability-tag KL anchor) | Synthesize a tag-specific KL pressure that holds calibration regardless of class skew — directly attacks the dynamic drift mechanism. |
| **v8 + action-quality reward**  (`--action-quality-bonus N`) | Add a per-step direct gradient on action quality (±N for non-doom / doom action), since the implicit Sudoku-style signal is missing. |
| **B-9 5×10 / 10-piece** (bigger env) | Restructure the env so trajectory length and success rate naturally rise → success_bonus fires → Sudoku-style self-regulation. |

For the paper, the bigger-env story is more generalizable. For RL on the existing 5×4 board, v8 (and v8 + action-quality if needed) is the cheaper test.

---

## Q: There's no separate `rl_trainer_v8.py` file — why?

There isn't, by design. The trainer is a single file: [src/training/rl_trainer_v6.py](../src/training/rl_trainer_v6.py). The "v6/v7/v8" suffixes refer to **reward configurations** (selected at runtime via `--reward-version`), not trainer file versions. The "v6" in the filename is historical (when the trainer was first written, it implemented v6 reward).

```bash
# All three live in the same file; the flag picks the reward shape:
python src/training/rl_trainer_v6.py --reward-version v6     # original
python src/training/rl_trainer_v6.py --reward-version v7     # symmetric + class balance + progress
python src/training/rl_trainer_v6.py --reward-version v8 \   # v7 + viability-tag KL anchor
                                     --viability-kl-coef 0.5
```

The composable `--action-quality-bonus N` flag works alongside any reward version — for instance, `--reward-version v8 --action-quality-bonus 1.0` gives v8's calibration anchor plus direct action gradient.

The `rl_trainer_v6.py` filename should probably be renamed to just `rl_trainer.py` to remove the misleading version-in-filename pattern; it's a follow-up cleanup task.
