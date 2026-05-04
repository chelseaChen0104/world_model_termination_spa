# RL Frameworks Handbook — TRL, VERL, and Our Hand-Rolled Trainer

A learning-oriented reference for the three RL approaches in play in this project: HuggingFace's **TRL**, ByteDance's **VERL**, and our own custom **`rl_trainer_v6.py`**. Read top-to-bottom if you're new to LLM RL; jump to §5 if you already know PPO/GRPO and just want to know what we built and why.

> **Scope note**: This handbook focuses on RL *for LLMs* (post-training reinforcement learning of language models against scalar rewards or preference data). Classical deep-RL frameworks like Stable-Baselines3, RLlib, and CleanRL are not covered.

---

## 1. RL fundamentals for LLMs

### 1.1 The objective

Given a base policy `π_ref` (an SFT'd LLM), we want to find a policy `π_θ` that maximizes expected reward on prompts `x`:

```
maximize over θ:    E_{x ~ D, y ~ π_θ(·|x)} [ r(x, y) ]
                    − β · KL[ π_θ(·|x) || π_ref(·|x) ]
```

The KL penalty keeps the new policy close to the reference so we don't catastrophically forget the SFT capabilities. `r(x, y)` can be any scalar function of the (prompt, completion) pair — a learned reward model, a programmatic checker, a human label, or a mix.

### 1.2 Three families of LLM-RL algorithms

| Family | What's needed | Examples | Used in this project? |
|---|---|---|---|
| **Preference-based** | pairs `(y_chosen, y_rejected)` | DPO, IPO, KTO | No |
| **PPO-style on-policy** | scalar reward `r(x,y)`, value estimate, advantage | PPO, REINFORCE++ | Indirectly — we inherit PPO's clipped surrogate |
| **Group-relative** | scalar reward, no value model — uses K samples per prompt as the baseline | **GRPO**, RLOO | **Yes — primary algorithm** |

We use GRPO with PPO's clipped surrogate. The next three subsections explain each in depth, and the relationship between them.

### 1.3 PPO in depth

**Proximal Policy Optimization** (Schulman et al. 2017) was developed for continuous-control RL well before LLMs. It has two parts that LLM training inherits to varying degrees:

#### 1.3.1 The clipped surrogate objective

The classic policy-gradient loss is `−E[log π_θ(y) · A]`. Vanilla PG is on-policy: it requires the data to be drawn from the *current* policy. But during a multi-pass training step, the policy moves between gradient steps — so by mini-batch 2, the data you collected is no longer truly on-policy.

PPO fixes this with **importance sampling + clipping**. Define the per-token ratio:

```
ratio_t = π_θ(y_t | y_<t, x) / π_θ_old(y_t | y_<t, x)
        = exp(log π_θ(y_t | y_<t, x) − log π_θ_old(y_t | y_<t, x))
```

The clipped surrogate loss is:

```
L_PPO = − E[ min( ratio_t · A_t,  clip(ratio_t, 1−ε, 1+ε) · A_t ) ]
```

with `ε ≈ 0.2`. The `min` and `clip` together cap how much a single update can change the policy:

- If `A_t > 0` (good action), the loss wants `ratio_t` *up*. The clip caps it at `1+ε`, so we don't overshoot when the policy has already moved a lot.
- If `A_t < 0` (bad action), the loss wants `ratio_t` *down*. The clip floors it at `1−ε`, same reasoning in reverse.
- The `min(...)` is the "pessimistic" version: take whichever is more restrictive.

Practically: clipping prevents single-batch policy collapses. You can run multiple PPO epochs over the same rollout data and the clip keeps the cumulative drift bounded. **This is the part we inherit from PPO.**

#### 1.3.2 The value function (critic)

Standard PPO needs an *advantage estimate* `A_t` per token. The textbook way is to learn a value function `V_φ(s_t)` and compute:

```
A_t = r_t + γ V_φ(s_{t+1}) − V_φ(s_t)            # one-step
       (or GAE-λ for variance-reduced versions)
```

This requires:
- A separate value-head model `V_φ` (typically the same architecture as the policy, with a scalar head)
- Doubling memory: actor + critic both held in GPU
- A second loss term `L_V = (V_φ(s_t) − returns_t)²` to train the critic
- Hyperparameter tuning for the critic (its own LR, λ for GAE, etc.)

For LLMs this is expensive AND brittle:
- Memory: a 7B actor + 7B critic = 14B parameters → halves the per-GPU batch size.
- Brittleness: LLM rewards are typically sparse (one number per completion), which makes value-function regression hard. The critic chases noise.
- Implementation complexity: requires careful credit assignment across thousands of tokens.

This is why nearly every modern LLM RL paper (DeepSeek, OpenAI o1/R1, etc.) is moving away from explicit critics.

### 1.4 GRPO in depth

**Group Relative Policy Optimization** (Shao et al. 2024, DeepSeekMath) replaces the critic with a much cheaper baseline: sample K completions per prompt and use the **group's mean reward** as the baseline.

#### 1.4.1 The advantage formula

For each puzzle/prompt `x`, sample `K` (we use 8) completions `y_1, …, y_K`. Score each: `r_i = reward(x, y_i)`. Then for the i-th completion, define:

```
A_i = (r_i − mean({r_1, …, r_K})) / (std({r_1, …, r_K}) + ε_std)
```

This is then **broadcast to every token** in completion `y_i` (every action token gets the same per-completion advantage). Some implementations stratify across PPO mini-batches; ours does not — see the trainer code if that matters.

What this buys you:
- **No critic to train** → memory roughly halved vs PPO, simpler optimization
- **Variance reduction**: subtracting the group mean removes the prompt-level reward floor (e.g., "this puzzle is easy, every sample scored ~0.8" → mean=0.8 → advantages center on 0)
- **Standardization**: dividing by std normalizes scale across puzzles (some give 0–10 reward, others 0–100)
- **Naturally on-policy at the *prompt* level**: each puzzle's K rollouts give a self-contained estimate

What it costs you:
- **K× compute per puzzle at rollout time**: you generate K completions per puzzle instead of 1. Vs PPO, this is the new dominant cost.
- **No bootstrapping across prompts**: each puzzle's advantage is local — there's no information transfer between different puzzles' value estimates. For very-similar prompts a critic might help; for diverse prompts (puzzles), GRPO is cleaner.
- **Group must have variance**: if all K rollouts for a puzzle return the same reward, `std=0` → advantage is `0/0` — that puzzle contributes nothing this step. We add a tiny `ε_std` to the denominator and this is rarely a real issue.

#### 1.4.2 GRPO inherits PPO's clipping

GRPO **does not throw away PPO's clipped surrogate** — it just replaces the advantage. The full GRPO objective is:

```
L_GRPO = − E[ min( ratio_t · A_i^{group},  clip(ratio_t, 1−ε, 1+ε) · A_i^{group} ) ]
```

Same clip trick, same importance-sampling reasoning. The only thing that changes vs PPO is how `A` is computed.

(There are also "vanilla" variants without clipping — RLOO, GRPO-no-clip — but they're less stable. The clipping cost is one extra `min/clamp` per token, basically free.)

### 1.5 Why we use GRPO (with PPO's clipped surrogate)

**Concretely, our project uses:**
- **GRPO** for advantage computation (group of K=8 rollouts per puzzle, mean-std normalized)
- **PPO's clipped surrogate** for the per-token loss
- **A separate β·KL(π || π_ref) term** to keep the trained policy close to the SFT initialization (a soft anchor, distinct from PPO's intra-update clipping)

So when we say "we use GRPO", what we technically mean is: GRPO *advantages* + PPO *clipping* + a *fixed-coefficient KL anchor*. This is the standard modern recipe.

**Why this combination for our setup**:

1. **No critic — fits on one GPU.** Our model is 1.5B; a 1.5B critic would push us out of memory on a single 80 GB A800 once you add optimizer state. GRPO sidesteps the critic entirely.

2. **Sparse, programmatic reward → critic doesn't help.** Our reward is computed by running the env (`is_solvable`, `success`) plus per-step shaping. There's no "natural" function for V to learn — value would have to integrate over puzzle randomness which is high-variance.

3. **Group baseline is naturally robust to puzzle difficulty.** Some Sudoku puzzles are easy (every rollout solves), some are hard (none do). GRPO's per-prompt mean handles this without us tuning anything.

4. **PPO clipping protects against off-policy drift.** We do multiple PPO mini-batch passes per rollout (`n_ppo_epochs > 1` in our trainer); without clipping, the second epoch's gradient would be wildly off-distribution.

5. **The KL anchor (β·KL[π||π_ref]) is *separate* from PPO clipping** and serves a different purpose:
   - PPO clip prevents *single-update* policy collapse (intra-step protection)
   - KL anchor prevents *cumulative* drift from the SFT init across many RL steps (inter-step protection)
   They're complementary. Our trainer uses both.

#### 1.5.1 Where the difference shows up in our metrics

In each training-step log line:

```
step  47 | reward +10.60±1.80 | solved 50% | pg_loss +0.000 | kl 0.0013 | clipfrac 0.00 | via_kl 0.0000 | step_t 255s
```

| Field | Comes from |
|---|---|
| `reward ±std` | Group of K=8 rollouts per puzzle (the GRPO group) |
| `solved %` | Fraction of those K rollouts that succeeded — the empirical estimate of `p` |
| `pg_loss` | The clipped-surrogate policy-gradient loss (PPO part) |
| `clipfrac` | Fraction of tokens whose `ratio` got clipped (PPO clip activity) — usually near 0 means policy is still on-policy |
| `kl` | KL divergence between current and reference policy (the soft anchor) |
| `via_kl` | Our v8-specific extra: KL on the `<viability>` token positions only |

If `clipfrac` is consistently > 0.1, the policy is moving fast enough that PPO's clip is binding — could indicate too-high learning rate or too many PPO epochs per rollout.

If `kl` grows monotonically across steps without saturation, the KL anchor coefficient β is too low — the policy is drifting from the reference and we may forget the SFT capabilities.

### 1.6 The on-policy / off-policy / clipped-surrogate machinery (PPO bug fix)

The math above is what makes the importance ratio meaningful: **`log π_θ_old` must be computed at sample time (once, before any gradient step) and cached**, not recomputed during the update under the freshly-updated weights. If you make that mistake — cache nothing, recompute `old_logp` from the model's current weights right before the update — then `ratio = exp(new_logp − old_logp) ≈ 1` always and PPO degenerates into vanilla policy gradient with no clipping benefit.

This is the bug fix our trainer's [`sample_response`](../../src/training/rl_trainer_v6.py) handles: we capture per-token logprobs in the same forward pass that generates the response, store them on the `StepRecord`, and consume them as `old_logps` during PPO updates.

### 1.4 What an LLM RL trainer actually looks like

Strip away the framework wrapper and any LLM RL trainer is doing this loop:

```python
for step in range(N_TOTAL_STEPS):
    # ROLLOUT PHASE
    prompts = sample_prompts()
    completions, old_logps = generate_with_logprobs(policy, prompts, K=group_size)
    rewards = reward_function(prompts, completions)            # custom!
    advantages = group_relative_normalize(rewards)             # GRPO

    # UPDATE PHASE
    for ppo_epoch in range(K_PPO_EPOCHS):
        new_logps = forward_logprobs(policy, prompts, completions)
        ref_logps = forward_logprobs(ref_policy, prompts, completions)
        ratio = (new_logps - old_logps).exp()
        kl    = approximate_kl(new_logps, ref_logps)
        pg_loss  = -torch.min(ratio * adv, clip(ratio,1±ε) * adv)
        total = pg_loss.mean() + β * kl.mean()
        total.backward(); optimizer.step()
```

That's the entire game. Frameworks differ in:
- How they parallelize (single-GPU? FSDP? vLLM-served rollouts? Ray-distributed?)
- How they package the reward function plug-in
- How much they hide vs expose
- What batching/length-normalization assumptions they make

---

## 2. TRL — HuggingFace's Transformer Reinforcement Learning library

[github.com/huggingface/trl](https://github.com/huggingface/trl) | docs: [hf.co/docs/trl](https://huggingface.co/docs/trl)

### 2.1 What it is

TRL is HuggingFace's official RL toolbox built on top of `transformers` and `accelerate`. It's designed to be easy to plug into existing HF workflows (anything you can `from_pretrained()` you can RL-train).

### 2.2 Key trainers

| Trainer | Algorithm | Reward source |
|---|---|---|
| `SFTTrainer` | Supervised fine-tuning (not RL strictly) | — |
| `DPOTrainer` | Direct Preference Optimization | preference pairs `(chosen, rejected)` |
| `KTOTrainer` | Kahneman-Tversky Optimization | binary "good/bad" labels |
| `PPOTrainer` (legacy) | Classical PPO with critic | scalar reward function |
| `GRPOTrainer` | Group Relative Policy Optimization | scalar reward function |
| `OnlineDPOTrainer` | DPO with on-policy generation + judge | LLM judge or reward model |
| `RLOOTrainer` | RLOO (REINFORCE Leave-One-Out) | scalar reward function |

### 2.3 Custom rewards — yes, fully supported

```python
from trl import GRPOTrainer, GRPOConfig

def my_reward_fn(completions, prompts, **kwargs):
    """Return list of floats, one per completion."""
    rewards = []
    for c in completions:
        # arbitrary logic — parse XML, run env, call API, etc.
        rewards.append(float(parse_and_score(c)))
    return rewards

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    reward_funcs=my_reward_fn,         # callable, OR a list of callables, OR a string path
    train_dataset=ds,
    args=GRPOConfig(num_generations=8, beta=0.04, ...),
)
trainer.train()
```

`reward_funcs` can be:
- a single callable (like above)
- a list of callables (rewards summed)
- a string pointing to a HF reward model (e.g. `"OpenAssistant/reward-model-deberta-v3-large-v2"`)
- a `PreTrainedModel` instance

So **any reward you can compute in Python is fair game**. This was the misconception my earlier message gave — TRL absolutely supports arbitrary rewards.

### 2.4 What TRL is great at

- **Single-GPU and small multi-GPU (`accelerate`)** — works out of the box with `accelerate launch`.
- **Tight HF ecosystem integration** — same `model_name_or_path`, `Tokenizer`, `Dataset`, `Trainer`, `TrainingArguments` you already know.
- **LoRA + 4-bit + 8-bit** via `peft` and `bitsandbytes` — supported on every trainer.
- **vLLM rollouts** — recent versions support `use_vllm=True` to offload generation to vLLM for ~5–10× faster rollouts.
- **Many algorithms in one place** — easy to A/B between DPO, GRPO, KTO, etc. by swapping the trainer class.

### 2.5 What TRL hides / makes hard

The framework expects rollouts to be **single-turn prompt → completion**. If your "rollout" is actually a multi-step env-interaction loop (agent picks action, env executes, agent sees next state, agent picks action…), you have to either:

(a) Run the env interaction inside your reward function and condense the whole multi-step trajectory into a single (prompt, completion) pair where "completion" is the agent's first action. You lose the per-step reward shaping, can't gate rollouts mid-trajectory, etc.

(b) Subclass `GRPOTrainer` and override `_generate_and_score_completions()` to inject your env loop. Doable but defeats the "easy" benefit.

You also can't easily:
- Apply KL pressure on **specific token positions** within the response (e.g., the `<solvable>` tag content tokens) without subclassing the loss function.
- Truncate rollout generation **mid-completion** based on parsed partial output (TRL generates full completions then scores).
- Cache old_logps separately from a forward pass on the freshly-rolled-out batch (TRL handles this internally — you don't need to, which is also a black-box hazard if you want to verify).

### 2.6 Verdict

**Use TRL when**: your task is single-turn (instruction-following, math, code), you want a familiar API, you're happy with standard PPO/GRPO/DPO algorithms, and your reward is just a scalar computed once per completion.

**Don't use TRL when**: your reward involves multi-step env interaction with mid-rollout decisions, or you need to apply algorithmic structure (KL anchors, truncation gates) at specific positions inside the response.

---

## 3. VERL — Volcano Engine Reinforcement Learning

[github.com/volcengine/verl](https://github.com/volcengine/verl) | paper: HybridFlow (Sheng et al. 2024)

### 3.1 What it is

VERL is ByteDance's distributed RL framework, built around the **HybridFlow** programming model (a single controller orchestrating multiple worker fleets — one fleet for actor inference via vLLM, one for ref-model logprobs, one for critic, etc.). Designed from the ground up for **large-scale, multi-node** training (think: 70B+ models on 64+ GPUs).

### 3.2 Architecture

```
┌──────────────────────────────────────────────────────┐
│  Single Controller (driver process)                  │
│   - reads YAML config                                │
│   - orchestrates rollout + update across workers     │
└──────────────────────────────────────────────────────┘
       │             │             │            │
       ▼             ▼             ▼            ▼
   ┌─────────┐   ┌─────────┐   ┌─────────┐  ┌──────────┐
   │ Actor   │   │ vLLM    │   │ Ref     │  │ Critic   │
   │ workers │   │ rollout │   │ workers │  │ workers  │
   │ (FSDP)  │   │ workers │   │ (FSDP)  │  │ (FSDP)   │
   └─────────┘   └─────────┘   └─────────┘  └──────────┘
```

Each fleet is a Ray actor pool. The controller dispatches work, collects results, and updates weights via FSDP.

### 3.3 Algorithms supported

PPO, GRPO, REINFORCE++, ReMax, DAPO, RLOO, and a few proprietary variants. Mostly the same algorithm zoo as TRL but with distributed-first implementations.

### 3.4 Custom rewards — yes, also supported

VERL config (YAML) lets you specify a `reward_score` function as a Python module path:

```yaml
# config.yaml
reward_model:
  reward_manager: naive
  reward_score: my_module.my_reward_fn   # or model path for a learned reward model

# my_module.py
def my_reward_fn(data_source, solution_str, ground_truth, extra_info=None):
    # arbitrary scoring
    return float(score)
```

There's also a `RewardManager` interface for batched/parallel reward computation, useful when reward involves heavy compute (running a verifier, calling an API, etc.).

### 3.5 What VERL is great at

- **Multi-node distribution** — scales to 70B models across 8+ nodes; this is its primary differentiator vs TRL.
- **Decoupled rollout** — vLLM workers handle generation, FSDP workers handle gradient updates. Generation speed is dramatically higher than with `model.generate()`.
- **Production-grade batching** — handles long-context, padding, and sequence packing efficiently.
- **Declarative configs** — entire training run specified in YAML; reproducibility is good.
- **Active research community** — DAPO, ReMax, etc. land in VERL early.

### 3.6 What VERL hides / makes hard

- **Setup cost is high** — Ray cluster, vLLM, FSDP, all need to coexist. Setup-by-tutorial; not "import and call".
- **Single-GPU dev iteration is awkward** — VERL's whole point is distributed; running on one GPU works but feels heavy.
- **Custom rollout shapes** — like TRL, VERL assumes single-turn prompt → completion. Multi-step env loops need custom plumbing inside the reward function.
- **Per-token loss surgery** — anchoring KL on specific token positions, conditional rollout truncation, etc., require diving into VERL's worker code.

### 3.7 Verdict

**Use VERL when**: you're training models ≥ 7B at multi-node scale, you want production-grade rollout throughput, and your team has Ray ops experience.

**Don't use VERL when**: you're on one GPU, your model is small (< 3B), or you want fast iteration on the reward/algorithm itself (the framework is heavy to debug).

---

## 4. RAGEN — the SPA paper's framework

For completeness, since SPA (the paper our project extends) builds on RAGEN:

[github.com/RAGEN-AI/RAGEN](https://github.com/RAGEN-AI/RAGEN)

RAGEN is a **multi-turn agentic RL** framework with native support for envs (Sokoban, FrozenLake, Sudoku, others). It provides:
- Multi-turn rollout where each turn is `(env-state → LLM-reasoning → action → env-step → reward)`.
- Built-in env interface compatible with Gym-style envs.
- PPO/GRPO updates via Ray + vLLM (similar architecture to VERL).

It's the right framework for our problem on paper. We didn't use it because:
1. Our target is single-GPU, small model. RAGEN's distributed architecture is overkill.
2. We want very tight control over the per-step reward shaping (truncation gate, viability KL anchor, action-quality bonus). RAGEN's reward configuration is less flexible.
3. RAGEN's env API didn't quite match our `BaseTerminationEnv` signature; would have required adapter code.

So we hand-rolled, taking the SPA recipe (state-estimation + transition-modeling SFT) as the conceptual contribution and reimplementing the RL stage from scratch.

---

## 5. Our hand-rolled trainer — `rl_trainer_v6.py`

### 5.1 Where it lives and what it does

[`src/training/rl_trainer_v6.py`](../../src/training/rl_trainer_v6.py) — single file, ~1300 lines. Pure PyTorch + transformers, no other RL deps.

Implements GRPO with our custom modifications:
- **Group-relative advantage** with K=8 rollouts per prompt
- **PPO clipped surrogate** loss with cached old_logps
- **KL penalty** against a frozen reference policy
- **Custom reward versions** v6, v7, v8 (more in §5.4)
- **Truncation gate** that early-terminates rollouts mid-trajectory based on parsed `<solvable>` predictions
- **Eval-fix flags** for prompt-format consistency: `--prepend-current-state`, `--single-turn-eval`, `--max-response-tokens` (added 2026-05-02 to fix the multi-turn-history bug — see [eval_2026-05-01_truncation_full.md](../eval_2026-05-01_truncation_full.md))

### 5.2 Code map

```
RLConfig (dataclass)              ← all hyperparameters as fields
  ├─ tp_reward / fn_reward / ...   ← per-class rewards for solvable prediction
  ├─ format_per_tag                ← format compliance reward
  ├─ viability_kl_coef             ← v8 KL anchor strength
  ├─ truncation_mode / threshold   ← Phase 2 compute-saving gate
  └─ ... (50+ fields)

REQUIRED_TAGS                      ← canonical XML tag set
parse_solvable / parse_action      ← regex-based parsers
solvable_reward / format_reward    ← per-component reward functions

build_prompt                       ← chat-template wrapper around system + history + user
sample_response (single)           ← generate_one_response + capture old_logps
sample_responses_batched           ← K rollouts in one forward pass

do_rollout / do_rollouts_batched   ← env-interaction loop, builds Rollout(steps)
quick_pass1                        ← greedy Pass@1 eval (called every eval_every steps)

main                               ← argparse + RLConfig assembly + the training loop
```

### 5.3 The training loop (literal pseudocode)

```python
policy = AutoModelForCausalLM.from_pretrained(args.sft_checkpoint)
ref_policy = AutoModelForCausalLM.from_pretrained(args.sft_checkpoint)  # frozen

for step in range(cfg.n_total_steps):
    # ── ROLLOUT ─────────────────────────────────────────
    seeds = sample_unique_seeds(n_puzzles_per_batch)
    rollouts = do_rollouts_batched(
        policy, tokenizer, env_factory, system_prompt,
        seeds, cfg, group_size=cfg.group_size,
    )
    # rollouts is List[Rollout], each Rollout has .steps with cached old_logps,
    # parsed solvable predictions, action validity, env-reward signals.

    # ── COMPUTE ADVANTAGES (GRPO) ───────────────────────
    rewards = [sum(s.step_reward for s in r.steps) for r in rollouts]
    adv = group_relative_normalize(rewards, group_size)

    # ── PPO UPDATE ──────────────────────────────────────
    for ppo_epoch in range(cfg.n_ppo_epochs):
        new_logps = forward_logprobs(policy, rollouts)
        ref_logps = forward_logprobs(ref_policy, rollouts)  # cached or recomputed
        ratio = (new_logps - rollout_old_logps).exp()
        pg = -torch.min(ratio * adv,
                        ratio.clamp(1-eps, 1+eps) * adv)
        kl = (new_logps - ref_logps)            # k1 estimator
        loss = pg.mean() + cfg.kl_coef * kl.mean()
        # v8 viability KL anchor:
        if cfg.viability_kl_coef > 0:
            via_kl = compute_viability_kl_anchor(...)
            loss = loss + cfg.viability_kl_coef * via_kl
        loss.backward(); optimizer.step()

    # ── EVAL + SAVE ─────────────────────────────────────
    if step % cfg.eval_every == 0:
        eval_metrics = quick_pass1(policy, ..., n_puzzles=30)
        log_jsonl({"step": step, "phase": "eval", **eval_metrics})
    if step % cfg.save_every == 0:
        policy.save_pretrained(f"{out}/checkpoint-{step}")
```

Everything else is bookkeeping — JSONL logging, occasional eval, periodic checkpoint saves.

### 5.4 The custom additions

These are the bits that wouldn't have been clean in TRL/VERL and motivated the hand-roll:

#### 5.4.1 Per-class asymmetric reward (v6 baseline)

```python
# In RLConfig:
tp_reward = 1.0    # correctly predicted doom on a doomed state
fn_reward = -0.7   # missed doom (false negative)
fp_reward = -0.5   # false alarm
tn_reward = 0.3    # correctly said solvable
```

The catch-doom-it-matters-more asymmetry isn't expressible as a single scalar reward function in a clean way. It requires per-step reward decomposition that knows ground-truth labels.

#### 5.4.2 v7: class-balance reweighting + progress bonus

When the per-step class is heavily imbalanced (98% doom in some Sudoku random-play data), per-class reward gets dominated by the majority. v7 reweights inversely by class frequency in the batch, capped at `class_balance_cap`. Plus a small `progress_bonus_per_step` for any valid action that advances the trajectory.

#### 5.4.3 v8: viability-tag KL anchor

The novel bit. Standard KL penalty constrains the *whole response* against the reference. v8 *additionally* applies a KL pressure specifically on the token positions where the `<viability>` (or `<solvable>`) tag's content lives:

```python
def compute_viability_kl_anchor(response_ids, viability_token_positions,
                                 new_logps, ref_logps, coef):
    # Only at positions where the tag content lives:
    via_pos = viability_token_positions  # found by find_viability_token_positions()
    diff = new_logps[via_pos] - ref_logps[via_pos]
    return coef * diff.pow(2).mean()
```

This lets RL freely optimize the action policy while keeping calibration locked to the SFT distribution. Crucial because RL rewards naturally pull the model toward "always say doom = collect cheap calibration reward", which destroys the discrimination the SFT learned.

In TRL/VERL this would require subclassing the loss function and parsing each response to find tag positions — not impossible but much more invasive than adding a single function to our trainer.

#### 5.4.4 v8.2: dual-token anchor

Same as v8 but anchors *both* `>true` and `>false` logprobs (not just the sampled one). Designed to fix bimodal-confidence collapse. See [eval_2026-05-01_truncation_full.md](../eval_2026-05-01_truncation_full.md) for results.

#### 5.4.5 Truncation gate

```python
# In do_rollouts_batched:
for t in range(cfg.max_rollout_steps):
    # ... generate response, parse <solvable> ...
    if (cfg.truncation_mode == "conservative"
            and pred is False
            and len(steps) >= cfg.truncation_min_step
            and confidence_above(threshold)):
        rollout.alive = False
        rollout.truncated_early = True
```

Mid-rollout, if the model emits `<solvable>=false` with high confidence (e.g., `P(false) ≥ 0.99`), we stop generation immediately and don't run further env-steps. This saves ~22% rollout tokens at training time. TRL/VERL generate the full completion before scoring, so this kind of reward-time-decision-to-truncate-generation is hard to bolt on.

#### 5.4.6 Skip-solvable-reward CLI flag (baseline mode)

For SPA Table 5 baselines (vanilla RL, SE-only, SPA-full), the model isn't trained to emit `<solvable>` so we don't penalize its absence:

```python
calib_r = 0.0 if cfg.skip_solvable_reward else solvable_reward(pred, gt, cfg)
```

Plus a parallel `--skip-prediction-tag` flag that drops `<prediction>` from the format-reward required-tag set.

### 5.5 What our trainer is missing vs TRL/VERL

Honest list:

- **No FSDP / multi-GPU.** We assume one GPU. Larger than ~3B is impractical.
- **No vLLM rollouts.** We use `model.generate()` directly. Rollout phase is ~5–10× slower than vLLM-served rollouts would be.
- **No LoRA option in RL.** Full-parameter updates only. Memory budget is tight for 1.5B on a 24 GB GPU.
- **No mature checkpoint resume.** We save checkpoints but resume-from-checkpoint is partial.
- **Less battle-tested.** The training loop is correct on the paths we've tested; corner cases (very-short rollouts, batch=1, etc.) may have rough edges.

For our small-model, single-GPU, custom-reward-heavy setup the trade is worth it. For a 7B+ training job, switch to TRL or VERL.

---

## 6. Comparison table

| Dimension | TRL | VERL | Ours (`rl_trainer_v6.py`) |
|---|---|---|---|
| **Lines of code (RL trainer)** | ~3000 (per algorithm) | ~10000 (framework) | ~1300 (single file) |
| **Custom scalar reward** | ✅ `reward_funcs` arg | ✅ `reward_score` config | ✅ direct in code |
| **Custom per-step reward shaping** | ⚠️ requires subclassing | ⚠️ requires subclassing | ✅ native |
| **Token-position-specific KL anchor** | ⚠️ subclass loss | ⚠️ subclass worker | ✅ native (v8) |
| **Mid-rollout generation truncation** | ❌ generates fully | ❌ generates fully | ✅ native (Phase 2 gate) |
| **Multi-step env loops** | ⚠️ flatten to single-turn | ⚠️ flatten to single-turn | ✅ native (do_rollout) |
| **Single-GPU dev** | ✅ great | ⚠️ heavy | ✅ great |
| **Multi-node scale** | ⚠️ via accelerate, limited | ✅ Ray-native | ❌ not supported |
| **vLLM rollouts** | ✅ `use_vllm=True` | ✅ native | ❌ |
| **FSDP** | ✅ via accelerate | ✅ native | ❌ |
| **LoRA / 4-bit** | ✅ via peft + bnb | ✅ native | ❌ in RL (yes in SFT) |
| **Distributed reward computation** | ⚠️ in-process | ✅ RewardManager | ❌ |
| **Setup time** | minutes | hours | minutes |
| **Iteration speed on reward changes** | minutes | minutes (config edit) | seconds |
| **Battle-tested on 70B+** | partially | yes | no |

---

## 7. When to choose which

```
Are you training ≥ 7B with multi-node infra?
├─ YES → VERL (or RAGEN if your env is multi-step agentic)
└─ NO  →
   Is your reward a single scalar, computable from (prompt, completion)?
   ├─ YES → TRL (DPO/GRPO/PPO depending on your data shape)
   └─ NO (multi-step env, mid-rollout decisions, position-specific losses)  →
      Hand-roll on torch + transformers (like ours).
      Use TRL's source code as a reference for the GRPO update math.
```

---

## 8. Learning path

If you want to deeply understand LLM RL, in order:

### 8.1 Read

1. **DeepSeekMath GRPO paper** ([Shao et al. 2024](https://arxiv.org/abs/2402.03300)) — the cleanest exposition of the GRPO update.
2. **Original PPO paper** ([Schulman et al. 2017](https://arxiv.org/abs/1707.06347)) — for the clipped surrogate intuition.
3. **InstructGPT paper** (§3.1 in [Ouyang et al. 2022](https://arxiv.org/abs/2203.02155)) — RLHF as applied to LLMs.
4. **DPO paper** ([Rafailov et al. 2023](https://arxiv.org/abs/2305.18290)) — the preference-based alternative.
5. **HybridFlow paper** ([Sheng et al. 2024](https://arxiv.org/abs/2409.19256)) — VERL's design rationale.

### 8.2 Code

6. **[trl/trainer/grpo_trainer.py](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py)** — TRL's GRPO source; ~1500 lines, very readable. Match its `_compute_loss` against the paper.
7. **[Our `rl_trainer_v6.py`](../../src/training/rl_trainer_v6.py)** — same algorithm, smaller, with project-specific extensions documented inline.
8. **[verl/trainer/ppo](https://github.com/volcengine/verl/tree/main/verl/trainer/ppo)** — VERL's PPO/GRPO implementation; harder to follow but shows distributed structure.

### 8.3 Build

Implement GRPO from scratch (no library) on a toy task — math word problems with regex-checked answers is the canonical exercise. Match GRPO paper's reported numbers on GSM8K-like data.

After that you'll find any of TRL, VERL, or our trainer immediately understandable as small variations on the same skeleton.

---

## 9. Pointers within this project

- [src/training/rl_trainer_v6.py](../../src/training/rl_trainer_v6.py) — main trainer
- [src/training/simple_sft_trainer.py](../../src/training/simple_sft_trainer.py) — SFT entry point (HuggingFace `Trainer`-based)
- [src/training/sft_trainer.py](../../src/training/sft_trainer.py) — older FSDP-based SFT (uses `verl`; dormant)
- [doc/eval_2026-05-01_truncation_full.md](../eval_2026-05-01_truncation_full.md) — Phase 2 truncation-gate experiment writeup
- [doc/spec_project.md](../spec_project.md) §3 — success criteria including SPA-style baseline comparisons
- [doc/runs_reference_2026-05-01.md](../runs_reference_2026-05-01.md) — live ledger of SFT and RL runs and their results

---

## 10. Glossary

- **GRPO** — Group Relative Policy Optimization. PPO without a value model; uses K samples per prompt to estimate the baseline.
- **PPO** — Proximal Policy Optimization. Policy gradient with a clipped surrogate objective for off-policy correction.
- **DPO** — Direct Preference Optimization. Closed-form alternative to PPO that learns directly from preference pairs without a reward model.
- **KL anchor / KL penalty** — `β · KL(π_θ || π_ref)` term that keeps the trained policy close to a reference policy. Essential to prevent reward-hacking-driven catastrophic forgetting.
- **Old logp / new logp** — log probabilities of the sampled tokens under the rollout-time policy (old) vs the current (post-update) policy (new). Their ratio drives the PPO loss.
- **Group-relative advantage** — (`r - mean(r)`) / `std(r)` over the K samples for the same prompt. Replaces the value-function-based advantage in PPO.
- **Reference policy** — frozen copy of the SFT model used for the KL penalty.
- **vLLM** — `vllm` (Kwon et al. 2023) — fast LLM serving framework; can dramatically speed up rollout generation.
- **FSDP** — Fully Sharded Data Parallel; PyTorch's primary distributed-training primitive for large models.
- **HybridFlow** — VERL's controller-and-fleets architectural pattern.
- **RAGEN** — multi-turn agentic RL framework (the SPA paper's basis).
