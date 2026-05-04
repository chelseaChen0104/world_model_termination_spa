# RL Training Walkthrough (Step-by-Step, with Code References)

A complete walkthrough of how one RL training step works in this project, mapped to the concepts in PPO/GRPO and the actual lines of code that implement them. Companion docs:
- [qa_2026-04-30_rl_diagnostics.md](qa_2026-04-30_rl_diagnostics.md) — what the per-step log fields mean (KL, pg_loss, clipfrac).
- [eval_2026-04-30_b7_rl_phase1.md](../eval_2026-04-30_b7_rl_phase1.md) — the v6 collapse + v7 reward redesign.
- [plan_2026-04-29_rl_approach.md](../plan_2026-04-29_rl_approach.md) — Phase 1 / Phase 2 plan.

The trainer is one ~800-line file: [src/training/rl_trainer_v6.py](../../src/training/rl_trainer_v6.py). Read this doc with that file open.

---

## 0. The big picture in one diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│ ONE RL TRAINING STEP (out of N_TOTAL_STEPS, default 200)            │
└─────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────┐
  │ 1. Sample 4 puzzles  │ rl_trainer_v6.py:756  (random seeds)
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │ 2. For each puzzle,  │ do_rollouts_batched()  rl_trainer_v6.py:373
  │    run 8 rollouts    │   ─ one model.generate() call per "turn"
  │    in parallel.      │     batched across all alive rollouts.
  │    32 rollouts total │   ─ each rollout: env.reset → loop turns →
  │    × 1.5 avg steps   │     terminate (done / max_steps / invalid).
  │    × ~150 tokens     │
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │ 3. Compute reward    │ in do_rollouts_batched, per-step:
  │    per (rollout,step)│   solvable_reward()   :137
  │    + final bonus     │   format_reward()     :133
  │                      │   progress bonus      :448  (v7 only)
  │                      │ end-of-trajectory:
  │                      │   success_bonus       :466
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │ 4. (v7) Rebalance    │ rebalance_rewards()   :481
  │    per-class rewards │   ─ inverse-frequency weighting
  │                      │ ─ recompute step_reward + final_reward
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │ 5. GRPO advantage:   │ grpo_advantages()     :521
  │    A_i = (r_i - μ)/σ │   ─ μ, σ within each puzzle's group of 8.
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │ 6. PPO update        │ ppo_update()          :537
  │    × 2 epochs:       │   per-token loss:
  │    - new_logp        │     -min(r·A, clip(r)·A)  + kl_coef·KL
  │    - ratio = exp(Δ)  │   ─ backward + optimizer.step
  │    - clip + KL pen   │
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │ 7. Log metrics       │ :784
  │    + every 25 steps: │
  │    - greedy eval     │ quick_pass1()  :622
  │    + every 100 steps:│
  │    - save checkpoint │
  └──────────────────────┘
```

That's one training step. We repeat it 200 times. The whole loop is `for step in range(1, cfg.n_total_steps + 1):` at [rl_trainer_v6.py:761](../../src/training/rl_trainer_v6.py#L761).

---

## 1. Fundamental setup — what is being trained?

We start from a **supervised fine-tuned (SFT) model** — for Pentomino, this is B-7 at `outputs/sft_pentomino_easy_b7_spa_hparams/final` — and update its weights so that its outputs lead to higher rewards on the Pentomino tiling environment.

There are **two copies** of the SFT model in GPU memory:

```python
# rl_trainer_v6.py:723-741
policy = AutoModelForCausalLM.from_pretrained(cfg.sft_checkpoint, ...)  # trainable
policy.train()

ref_policy = AutoModelForCausalLM.from_pretrained(cfg.sft_checkpoint, ...)  # frozen
ref_policy.eval()
for p_ in ref_policy.parameters():
    p_.requires_grad_(False)
```

| name | role |
|---|---|
| `policy` | the model whose weights we're updating. Starts identical to the SFT. |
| `ref_policy` | a frozen copy of the SFT, never updated. Used to compute a KL penalty that anchors the learning policy near its starting behavior. |

We also load the **tokenizer** of the base model (`Qwen/Qwen2.5-1.5B-Instruct`) and create one **environment template** (a `PolyominoEnv` or `SudokuEnv`) that we'll clone into 32 parallel instances every rollout.

---

## 2. The environment — what defines a state, action, reward source?

The env is the source of ground-truth `is_solvable` labels. For Polyomino it's defined in [src/environments/polyomino.py](../../src/environments/polyomino.py).

A **state** is a partial board: which cells are already covered, which pentominoes are still unplaced, and whether the env has terminated. The model never sees this struct directly — it sees a *rendered text* version through `env.reset()` / `env.step()`'s `obs`:

```
Pentomino-easy current board (5x4):
. . . .
. . . .
. . . .
. . . .
. . . .
Pieces remaining: L, P, W, Y
Place a piece using <answer>place {piece} ori={K} at row {R} col {C}</answer>.
```

An **action** is a string the model emits inside `<answer>...</answer>`. The env parses it (action format is env-specific) and either accepts the placement or rejects it as invalid.

**Transitions** are deterministic — given a state and a valid action, there's exactly one next state. The transition function uses an **exact-cover solvability checker** (DLX / Algorithm X) to label whether the resulting state is `is_solvable=True` or `is_solvable=False`. That label is the only source of supervision we have.

The env also reports:
- `done = True` if the puzzle is solved, or no remaining piece can be placed, or the action was invalid.
- `info["is_solvable"]` — the DLX checker's label for the *next* state (after the action).
- `info["is_breaking_point"]` — True if this action transitioned solvable → unsolvable.
- `info["success"]` — True if all pieces are placed (a complete tiling).
- `info["action_is_valid"]` — False if the action couldn't be parsed or the placement is illegal.

---

## 3. The model's response — what tags carry the gradient signal?

For Polyomino, the model is trained (during SFT) to emit responses in the form:

```xml
<observation>...what's currently on the board...</observation>
<next_state>...what the board will look like after my action...</next_state>
<viability>true</viability>
<answer>place L ori=2 at row 1 col 1</answer>
```

The two pieces of this response that drive RL reward:
- **`<viability>true|false`** — the model's prediction of whether the next state is solvable. The reward function compares this to `info["is_solvable"]` from the env.
- **`<answer>...`** — the action that gets executed. Determines what next state the env transitions to.

The whole response is parsed with regexes:

```python
# rl_trainer_v6.py:114-130
def parse_solvable(text):
    m = _re_solvable.search(text)        # <solvable>true|false</solvable> (Sudoku)
    if not m:
        m = _re_viability.search(text)    # <viability>true|false</viability> (Polyomino)
    if not m:
        return None
    return m.group(1).lower() == "true"

def parse_action(text):
    # Sudoku-specific; Polyomino uses _extract_answer() to pull <answer>...</answer>
    ...
```

If `<viability>` is unparseable, that step is treated as worst-case for reward (penalized as if the model predicted the wrong class). This punishes format-breaking outputs.

---

## 4. Rollouts — generating training data on-policy

A **rollout** is one full puzzle attempt: starting from `env.reset()`, the model picks actions turn by turn until the env terminates (success / doom / max-steps / invalid action). We store every step as a `StepRecord`:

```python
# rl_trainer_v6.py:151-170
@dataclass
class StepRecord:
    prompt_text: str       # the user-facing prompt the model saw
    response_text: str     # what the model actually emitted
    response_ids: list     # tokens of just the response (for PPO grad later)
    action: Optional[tuple]
    pred_solvable: Optional[bool]   # parsed from <viability>
    gt_solvable: bool               # from env info["is_solvable"]
    is_breaking_point: bool
    step_reward: float
    old_logps: Optional[list] = None  # cached at rollout time — see §6 PPO bug fix
    calib_reward: float = 0.0         # the per-step <viability>/<solvable> reward
    fmt_reward: float = 0.0           # +0.05 per of 4 required tags
    progress_reward: float = 0.0      # +0.1 per valid step (v7 only)
    action_was_valid: bool = False
```

### 4.1 Why we batch 32 rollouts per step

Per-step we sample `n_puzzles_per_batch=4` puzzle seeds and run `group_size=8` parallel rollouts per puzzle, giving **4 × 8 = 32 rollouts**:
- The `8 per puzzle` is needed for **GRPO's group baseline** (see §7) — to compute the per-puzzle mean reward we average across all 8 rollouts of the same starting board.
- The `4 different puzzles` reduces gradient variance — without it, every gradient step would be specialized to one specific board.

All 32 rollouts are stepped **in parallel**: at each turn t, we collect the prompts of all alive rollouts, batch them through one `model.generate()` call, and route the responses back to the appropriate rollout. This is the speedup that made RL practical — see `do_rollouts_batched` at [rl_trainer_v6.py:373](../../src/training/rl_trainer_v6.py#L373).

### 4.2 Sampling the response

```python
# rl_trainer_v6.py:312-342
@torch.no_grad()
def sample_responses_batched(model, tokenizer, prompts, cfg, device):
    # left-pad prompts (causal LM batched generation)
    inputs = tokenizer(prompts, padding=True, return_tensors="pt", ...).to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=cfg.max_response_tokens,    # 256
        temperature=cfg.temperature,                # 0.7 in training, 0.0 in eval
        do_sample=cfg.temperature > 0,
        top_p=0.95, top_k=50,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
        output_scores=True,                          # needed for old_logps caching
    )
    ...
```

`temperature=0.7` makes generation **stochastic** — each of the 8 rollouts in a group ends up exploring a different action sequence even though they all start from the same board. This stochasticity is what gives GRPO its variance to compute advantages from.

### 4.3 Caching old logprobs at rollout time (the critical PPO bug fix)

Before passing each generated token through `softmax`, we capture the log-probability **the policy assigned to it at the moment it was sampled**:

```python
# rl_trainer_v6.py:362-367
old_logps_i = []
for t in range(cut):
    log_probs = F.log_softmax(out.scores[t][i], dim=-1)
    old_logps_i.append(log_probs[response_block[i, t]].item())
```

This `old_logps` array is stored on the `StepRecord` and reused later by the PPO update. Why: in vanilla PPO, the importance ratio is `r = π_new / π_old`, and `π_old` is supposed to be the policy at sampling time. If you instead recompute `π_old` via a fresh forward pass under the *current* (already-updated) policy, you get `π_new = π_old` by definition, so `r = 1` always, and PPO's clipping never activates — i.e., the optimizer thinks "everything is fine" and silently drifts. This was the bug we fixed transitioning from earlier trainer versions to v6/v7.

---

## 5. Reward — turning rollouts into scalars

The reward is computed **per-step** during the rollout, then summed at end-of-trajectory along with a success bonus.

### 5.1 Per-step reward components (v6)

For every step inside `do_rollouts_batched`:

```python
# rl_trainer_v6.py:451-460
calib_r = solvable_reward(pred, gt_solvable, cfg) if action_was_valid else cfg.fn_reward
progress_r = cfg.progress_bonus_per_step if action_was_valid else 0.0
fmt_r = format_reward(text, cfg.format_per_tag)
step_reward = calib_r + fmt_r + progress_r
```

**`calib_reward`** scores the `<viability>` prediction against the env's ground truth:

| Outcome | v6 | v7 |
|---|---|---|
| TP (pred=False, GT=False) | +1.0 | +1.0 |
| FN (pred=True, GT=False) | −0.7 | −1.0 |
| FP (pred=False, GT=True) | −0.5 | −1.0 |
| TN (pred=True, GT=True) | +0.3 | +1.0 |
| unparseable | worst-case for class | worst-case for class |

(See `solvable_reward` at [rl_trainer_v6.py:144](../../src/training/rl_trainer_v6.py#L144). Magnitudes from `RLConfig` defaults at [rl_trainer_v6.py:79](../../src/training/rl_trainer_v6.py#L79).)

**`fmt_reward`** = +0.05 per of 4 required tags present in the response (max +0.20). Discourages format-breaking outputs.

**`progress_reward`** (v7 only) = +0.1 per valid action that advanced the trajectory. Tries to bias the policy toward actions that don't immediately doom the board.

### 5.2 End-of-trajectory bonus

After a rollout terminates we add:

```python
# rl_trainer_v6.py:478-479
final_reward = sum(s.step_reward for s in r["steps"])
final_reward += cfg.success_bonus if r["is_solved"] else cfg.fail_bonus
```

with `success_bonus = +3.0` (v6 was originally +10, reduced to +3 in v6.1) and `fail_bonus = 0.0`.

### 5.3 v7 post-batch class rebalancing

After all 32 rollouts are collected, [rebalance_rewards()](../../src/training/rl_trainer_v6.py#L483) walks every step, counts `(n_solv, n_doom)`, and scales each step's `calib_reward` by an inverse-frequency weight:

```python
# rl_trainer_v6.py:511-513
w_solv = total / (2.0 * max(n_solv, 1))
w_doom = total / (2.0 * max(n_doom, 1))
w_solv = max(floor, min(cap, w_solv))     # default [0.5, 5.0]
w_doom = max(floor, min(cap, w_doom))
```

Then `step_reward` and `final_reward` are recomputed from the rescaled `calib_reward`. v6 path skips this entirely (`cfg.class_balance == False`).

### 5.4 What the reward is actually scoring

A single rollout's `final_reward` decomposes as:

```
final_reward = Σ_t [ calib_r(pred_t, gt_t) + fmt_r(t) + progress_r(t) ] + success_bonus·[is_solved]
```

That whole scalar is what GRPO will compare across the 8-rollout group.

---

## 6. The optimization target — what we want to push up

We want the policy to assign **higher probability** to response tokens that came from rollouts with **higher reward**, and lower probability to tokens from low-reward rollouts. The naive policy gradient would multiply each token's log-probability by the reward of its rollout:

```
   ∇L_PG = E [ ∇ log π(token) · R_rollout ]
```

This works in principle but has two practical problems:
1. The reward magnitude is arbitrary (some scales destabilize training).
2. Reusing the same rollout to take more than one gradient step requires **importance sampling** — the math breaks down because the policy that produced the rollout is no longer the policy we're updating.

GRPO + PPO solve both:
- **GRPO** normalizes reward to a per-group advantage so the gradient scale is consistent.
- **PPO** uses a clipped importance-sampled objective so we can safely take multiple gradient steps per batch.

---

## 7. GRPO advantage — making rewards comparable

Instead of using raw reward, we compute an **advantage** per rollout:

```python
# rl_trainer_v6.py:521-534
def grpo_advantages(rollouts, group_size):
    advs = []
    for g_start in range(0, len(rollouts), group_size):
        group = rollouts[g_start:g_start + group_size]
        rewards = np.array([r.final_reward for r in group])
        baseline = rewards.mean()
        std = rewards.std() + 1e-8
        for r in rewards:
            advs.append(float((r - baseline) / std))
    return advs
```

For each group of 8 rollouts (sharing the same puzzle seed):
- baseline `μ` = mean reward across the 8.
- standard deviation `σ`.
- advantage of rollout *i* = `(reward_i − μ) / (σ + ε)`.

This gives a **z-score-style scalar** per rollout. The 8 rollouts in a group always sum to advantage 0, so the gradient pushes "above-average" rollouts up and "below-average" rollouts down, **independently of the absolute reward magnitude**.

The advantage `A_i` is then **broadcast to every token of every step in rollout i**. So all ~150 response tokens of step 1 of rollout i, plus all ~150 tokens of step 2 of rollout i, all share the same scalar `A_i`.

---

## 8. PPO update — token-level loss

The actual gradient comes from PPO's **clipped surrogate objective**, computed per response-token. For one step's response, with token-wise old-logprobs `old_logp_t` (cached at rollout time, §4.3) and new-logprobs `new_logp_t` from a fresh forward pass:

```
ratio_t   = exp(new_logp_t - old_logp_t)
unclip_t  = ratio_t · A_i
clip_t    = clip(ratio_t, 1-ε, 1+ε) · A_i
loss_t    = - min(unclip_t, clip_t)
kl_t      = (new_logp_t - ref_logp_t)²
total_t   = loss_t + kl_coef · kl_t
```

Then we sum/mean `total_t` over all response tokens and call `backward()`.

In code, this is `ppo_update()` at [rl_trainer_v6.py:545](../../src/training/rl_trainer_v6.py#L545):

```python
# rl_trainer_v6.py:570-588
old_logp = torch.tensor(step.old_logps, ...)       # cached at rollout time
ref_logp = compute_response_logprobs(ref_policy, ...)  # frozen SFT logprobs
new_logp = compute_response_logprobs(policy, ...)      # current policy, with grad

# length-mismatch guard (tokenization roundtrips can drop tokens)
min_len = min(old_logp.numel(), new_logp.numel(), ref_logp.numel())
old_logp = old_logp[:min_len]
new_logp = new_logp[:min_len]
ref_logp = ref_logp[:min_len]

ratio = torch.exp(new_logp - old_logp)
adv_t = torch.full_like(new_logp, adv)
unclipped = ratio * adv_t
clipped = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_t
pg_loss = -torch.min(unclipped, clipped).mean()

kl = (new_logp - ref_logp).pow(2).mean()  # squared logprob deviation as proxy KL
loss = pg_loss + cfg.kl_coef * kl
(loss / max(1, len(rollouts))).backward()
```

The gradient flows through `new_logp` (which has `requires_grad=True`) back to the model parameters. After we've accumulated gradients across all rollouts × all steps, we call:

```python
# rl_trainer_v6.py:606-608
torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
optimizer.step()
optimizer.zero_grad()
```

### 8.1 Why two PPO epochs?

After computing one set of rollouts, we run the PPO update twice:

```python
# rl_trainer_v6.py:778-779
for _ in range(cfg.ppo_epochs):     # default 2
    ppo_metrics = ppo_update(...)
```

The first epoch updates the policy; the second epoch redoes the same loss computation under the *now-updated* policy. The clipping in PPO's objective ensures that the second epoch can't make changes too large (the importance ratio gets clipped), so we get more mileage from each batch of rollouts at minimal stability risk.

### 8.2 The KL penalty

The `kl_coef * kl` term in the loss is what keeps the policy from drifting too far from the SFT reference. Without it, PPO would happily drive the policy toward maximum reward even at the cost of breaking format compliance, calibration, or coherence. With `kl_coef=0.05` (small but nonzero), every gradient step balances "increase reward" against "stay close to where SFT started."

This is the safety net. When KL itself blows past ~1.0 in the logs, the safety net is being dragged but not snapping — but the policy is already in a regime where calibration tends to break.

---

## 9. Logging & evaluation

### 9.1 Per-step training log

After each training step we collect a metrics dict and emit one stdout line + one JSONL row:

```python
# rl_trainer_v6.py:782-806
log = {
    "step": step,
    "rollout_time_s": ..., "step_time_s": ...,
    "n_rollouts": len(rollouts),
    "reward_mean": float(rewards.mean()),
    "reward_std":  float(rewards.std()),
    "solved_rate": float(solved_rate),
    "adv_min": ..., "adv_max": ...,
    **cb_metrics,    # v7: class_balance_applied, n_solv, n_doom, w_solv, w_doom
    **ppo_metrics,   # pg_loss, kl, clipfrac, n_tokens
}
```

What each field means is documented in [qa_2026-04-30_rl_diagnostics.md](qa_2026-04-30_rl_diagnostics.md).

### 9.2 Periodic greedy eval

Every `eval_every` steps (default 25) we pause training and run a **greedy** eval (temperature 0.0) on a held-out set of 30 fresh puzzles:

```python
# rl_trainer_v6.py:617-654
@torch.no_grad()
def quick_pass1(policy, tokenizer, env, system_prompt, cfg, device, n_puzzles):
    ...
    for i in range(n_puzzles):
        seed = 100000 + i  # disjoint from training seeds
        cfg.temperature = 0.0  # greedy
        ro = do_rollout(policy, tokenizer, env, system_prompt, seed, cfg, device)
        cfg.temperature = old_temp
        ...
    return {"pass@1": ..., "solvable_acc": ..., "bp_recall": ...}
```

Three metrics:
- `pass@1` — fraction of puzzles solved with greedy decoding.
- `solvable_acc` — fraction of `<viability>` predictions that were correct, across all valid steps.
- `bp_recall` — fraction of true breaking points (true→false transitions) that the model identified as `<viability>=False`.

These are the metrics the eval doc tables cite. `solvable_acc` collapsing to 0.0 is the canonical signal that the policy has fallen into the "always False" attractor.

### 9.3 Checkpoint saves

Every `save_every` steps (default 100) and at the end of training, we call `policy.save_pretrained(...)`. The final checkpoint lands in `<output_dir>/final/`. Intermediate checkpoints are useful when a run goes off the rails — we can roll back to the last good one.

---

## 10. End-to-end concrete example: one full step

Putting it all together, with made-up but realistic numbers from the v7 Pentomino run.

### Setup at step 50
- Policy is a slightly-updated B-7 SFT.
- Ref is the original B-7 SFT (frozen).
- Optimizer state is whatever AdamW has accumulated through 50 steps.

### Phase 1 — Rollout (~25 seconds)
- Sample 4 puzzle seeds.
- Spawn 32 `PolyominoEnv` instances, each gets `env.reset(seed=...)`.
- Turn 1: build 32 prompts; one batched `model.generate()` produces 32 responses; for each rollout, parse `<viability>` and `<answer>`, step the env, store a `StepRecord` with `old_logps`. Most rollouts terminate here (doom).
- Turn 2: ~9 alive rollouts; same drill. ~3 reach turn 3.
- Final state: 32 rollouts, total ~48 steps across them, each step has ~150 response tokens.

### Phase 2 — Reward and rebalance (instant, CPU)
- Per-step `calib_reward + fmt_reward + progress_reward` already filled in during Phase 1.
- Rollout `final_reward = sum(step_reward) + success_bonus·is_solved`.
- v7 only: `rebalance_rewards()` counts `n_solv=12, n_doom=36`, computes weights, rescales `calib_reward` accordingly, recomputes `step_reward` and `final_reward`.
- 32 final_reward values: e.g., `[+0.8, +0.6, +0.7, ..., +1.9, +0.5]`.

### Phase 3 — Advantages (instant)
- Group 1 (rollouts 0-7): mean=0.8, std=0.4 → advantages like `[+0.1, -0.4, +0.0, +1.5, ...]`.
- Group 2 (rollouts 8-15): same drill with their own mean/std.
- 32 advantages, one per rollout.

### Phase 4 — PPO update × 2 epochs (~3 seconds)
For epoch 1:
  for each (rollout, advantage) pair:
    for each step in the rollout:
      load `step.response_ids`, `step.old_logps`.
      forward `policy` on `prompt + response` → get `new_logp` per token.
      forward `ref_policy` on same → get `ref_logp` per token.
      compute `ratio = exp(new_logp - old_logp)`, `pg_loss = -min(ratio·adv, clip(ratio)·adv).mean()`.
      compute `kl = (new_logp - ref_logp).pow(2).mean()`.
      `loss = pg_loss + 0.05·kl`. `loss.backward()`.
  After all rollouts: `clip_grad_norm`, `optimizer.step()`, `optimizer.zero_grad()`.

For epoch 2: identical but with the now-updated `policy`. The clipping bounds how much the policy can have moved in epoch 1.

### Phase 5 — Log + (sometimes) eval (instant or +30s)
Emit stdout + JSONL row. If `step % eval_every == 0`, run greedy `quick_pass1` for ~30 seconds.

### Total wall time
~30 seconds per training step on H800 with current config. 200 steps ≈ 100 minutes. 500 steps ≈ 4 hours.

---

## 11. Connection to recent observations

This is where the abstract knobs become specific findings. From the v7 Pentomino run + sanity test ([sanity_2026-04-30_b7_rollout_stats.json](../sanity_2026-04-30_b7_rollout_stats.json), [eval_2026-04-30_b7_rl_phase1.md](../eval_2026-04-30_b7_rl_phase1.md)):

| RL concept (this doc) | What we measured | Implication |
|---|---|---|
| Rollout length distribution (§4.1) | 73% of rollouts are 1-step on Pentomino-easy | The PPO update sees almost no multi-step transitions — most gradient comes from terminal (action, doom) pairs. |
| GRPO advantage (§7) | Within a same-puzzle group, all 8 rollouts often die at step 1 with similar reward | Group baseline `μ` ≈ each rollout's reward → advantage `(r-μ)/σ` ≈ 0 for most tokens. Gradient signal is weak. |
| KL (§8.2) | KL drifted 0.04 → 0.65 in 75 steps under v7 | Policy moved ~16× from where it started — half the v6 drift but still significant. Calibration eventually broke. |
| pg_loss (§8) | Stayed nonzero (-0.07 to -0.30) through the v7 collapse | Gradient *was* flowing — it just flowed in the wrong direction. The collapse isn't a frozen-gradient failure, it's a wrong-direction failure. |
| Counterfactual reward (§5.4) | Oracle (+1.00) >> always_false (+0.46) under v7 | The reward landscape doesn't favor "always False" globally; the collapse is a dynamic drift, not the global minimum. |
| success_bonus (§5.2) | Pass@1 = 0.00% under stochastic sampling | The +3.0 success_bonus literally never fires, so the only gradient is per-step calibration. There's no goal-directed signal. |

The conclusion the data supports: **the trajectory-length distribution is what's killing us, not the reward magnitudes per se.** Fixes that increase trajectory length (bigger board) or decouple calibration from action quality (auxiliary KL anchor on the viability tag) are more promising than further tweaking per-step calibration rewards.

---

## 12. Where to go next in the code

If you want to modify behavior:

| To change… | Edit… |
|---|---|
| Reward magnitudes (TP/FN/FP/TN, success bonus, format bonus) | `RLConfig` defaults [:79-95](../../src/training/rl_trainer_v6.py#L79) or `--*-reward` CLI flags |
| Per-step weighting / rebalancing strategy | `rebalance_rewards()` [:483](../../src/training/rl_trainer_v6.py#L483) |
| Group / batch sizing | `--n-puzzles-per-batch`, `--group-size` CLI flags |
| Sampling temperature / max tokens | `RLConfig.temperature`, `RLConfig.max_response_tokens` |
| KL coefficient | `--kl-coef` CLI flag |
| Add a Phase 2 truncation gate (don't run dead rollouts) | The hook is at [:285-294](../../src/training/rl_trainer_v6.py#L285) (`truncation_mode == 'conservative'`); not yet wired up |
| Add an auxiliary head / per-tag KL | New code path in `ppo_update()`; would need to identify `<viability>` token positions in the response and add a separate KL term computed only over those positions |

For the launch wrappers (which clouds, which env, which output dir), see [scripts/run_sudoku_4x4_rl_v6_phase1.sh](../../scripts/run_sudoku_4x4_rl_v6_phase1.sh) (Sudoku) and [scripts/run_pentomino_5x4_rl_v6_phase1.sh](../../scripts/run_pentomino_5x4_rl_v6_phase1.sh) (Pentomino).
