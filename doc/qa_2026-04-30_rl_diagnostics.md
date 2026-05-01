# Q&A — Reading the RL Training Logs (2026-04-30)

Captures what the per-step training metrics in `logs/rl_b{5,7}_phase1*.log` actually mean, so future sessions don't have to re-derive it. Companion to [eval_2026-04-30_b7_rl_phase1.md](eval_2026-04-30_b7_rl_phase1.md), [plan_2026-04-29_rl_approach.md](plan_2026-04-29_rl_approach.md), and the trainer at [src/training/rl_trainer_v6.py](../src/training/rl_trainer_v6.py).

A typical training log line:

```
step   25 | reward +0.81±0.28 | solved 0% | pg_loss +0.104 | kl 0.0365 | clipfrac 0.02 | step_t 19s
```

`reward`, `solved`, and `step_t` are self-explanatory (mean ± std of `final_reward` across the batch's rollouts; per-batch greedy-style solve rate; wall time per step). The three PPO-specific metrics — **kl**, **pg_loss**, **clipfrac** — are the diagnostic core.

---

## Q: What is KL?

The "how far has the policy drifted from the frozen SFT reference" number. Computed per-token as `(new_logp − ref_logp)²` averaged over all response tokens (a proxy for KL divergence; see `ppo_update` in [rl_trainer_v6.py](../src/training/rl_trainer_v6.py)).

| value | reading |
|---|---|
| **0** | policy is identical to the SFT (start of training, or frozen) |
| **< 0.5** | policy is learning while staying anchored to SFT — desirable |
| **0.5–1.0** | drifting; check whether the calibration metric (greedy `<viability>` acc, AUC) is still healthy |
| **> 1.0** | policy has wandered off; output distribution looks very different from SFT — usually correlates with calibration collapse |

**The KL coefficient `0.05`** in our config (`RLConfig.kl_coef`) is a penalty term added to the loss: `loss = pg_loss + kl_coef · kl`. Higher coefficient → tighter leash on drift. We use 0.05 because it's small enough to allow real learning but large enough that runaway drift gets pulled back.

**Concrete failure example.** In the v6 B-7 Pentomino run, KL hit 1.7 by step 50 and stayed there. That correlated with greedy `<viability>` accuracy collapsing from 1.0 → 0.0 between steps 25 and 50. The policy had moved far enough from the AUC=1.0 SFT reference that it was no longer producing well-calibrated `<viability>` predictions. v7 at step 25 = 0.037 — well within the safe zone.

---

## Q: What is pg_loss?

The PPO surrogate **policy-gradient loss**, averaged over response tokens. The actual loss being minimized is:

```
loss_token = − min( r · A,  clip(r, 1 − ε, 1 + ε) · A )
```

where `r = π_new / π_old` is the importance ratio (see clipfrac below) and `A` is the rollout's advantage (how good *this* rollout was vs. the group baseline; we use GRPO-style normalized advantages — `(reward_i − group_mean) / (group_std + ε)`).

The reported `pg_loss` is this quantity averaged over all response tokens in the batch. Sign and magnitude tell different things:

| value | reading |
|---|---|
| **negative** | on average, the loss is going down — policy is moving toward higher-advantage actions. Healthy. |
| **positive** | average tokens are *increasing* the loss this step. Often happens when the gradient is small or oscillating between PPO epochs (we run 2 epochs per batch). Not alarming on its own. |
| **exactly 0.000** | nothing is moving. **Death signal.** |
| **magnitude ~0.05–0.5** | typical for a healthy run |

**Concrete failure example.** In v6 B-7, `pg_loss` collapsed to `+0.000` from step ~76 onward and stayed there for the next ~125 steps. The policy had stopped learning anything; the gradient signal was zero. v7 at step 25 = +0.104 means non-trivial gradient is still flowing; the sign is less informative than the magnitude.

**Why "surrogate"?** Because PPO doesn't optimize the true policy-gradient objective directly — it optimizes a *clipped, importance-sampled* approximation that's stable to compute over a batch of off-policy rollouts. The clipping is what `clipfrac` measures.

---

## Q: What is clipfrac, and what does "PPO clip activated" mean?

`clipfrac` = fraction of response tokens where the PPO clip activated this step.

To explain "clip activated" we need the importance ratio. PPO computes per-token:

```
r = π_new(token | context) / π_old(token | context)
  = exp(new_logp − old_logp)
```

This says how much the *current* policy boosted (or shrunk) the probability of the token that was actually generated during the rollout, relative to the policy at rollout time.

| ratio r | meaning |
|---|---|
| `1.0` | new policy assigns this token the same probability as old — no change |
| `1.5` | new policy is 50% more likely to emit this token |
| `0.4` | new policy is 60% less likely to emit it |

PPO then **clips** this ratio to a trust region `[1 − ε, 1 + ε]`. With `ε = 0.2` (our `RLConfig.clip_eps`), that's `[0.8, 1.2]`. The objective for the token is:

```
loss_token = − min( r · A,  clip(r, 0.8, 1.2) · A )
```

**"PPO clip activated"** = `r` fell outside `[0.8, 1.2]`. Concretely:

| ratio r | clipped to | clip activated? | what it means |
|---|---|---|---|
| 1.05 | 1.05 (unchanged) | no | policy made a small adjustment to this token; PPO let it through |
| 1.35 | 1.20 | **yes** | policy wanted to boost this token by 35%; PPO capped at 20% |
| 0.55 | 0.80 | **yes** | policy wanted to halve this token's prob; PPO capped at −20% |

`clipfrac = 0.02` = 2% of all response tokens in this batch hit the clip.

### Why the clip exists

Without it, a single token's update could swing its probability by 5× or 10× in one step (especially with a large advantage). That's what PPO's predecessor TRPO tried to prevent with a hard KL constraint; PPO's clipping is the same idea but cheaper. It bounds how much harm one bad update can do, so off-policy rollouts can be reused across multiple gradient epochs without destabilizing the policy.

### Reading clipfrac values

| value | reading |
|---|---|
| **0** | suspicious — the policy isn't actually changing (or the rollout-time logprobs equal the update-time logprobs, which is the v6 PPO bug) |
| **0.01–0.10** | a small share of tokens are pushing the trust-region boundary — healthy, the optimizer is moving but not over-aggressively |
| **0.10–0.30** | meaningful clipping — policy wants to swing further than PPO permits; consider lowering LR or shortening PPO epochs |
| **> 0.30** | most updates are getting clipped, gradient is partly thrown away — definitely too aggressive |

### The v6 PPO bug

In our original v6 trainer, `old_logp` was being recomputed at PPO-update time using the *current* policy. So by definition `old_logp == new_logp`, which means `r ≡ 1.0` for every token, which means `clipfrac == 0` always. The clip was never activated, but not because the policy was learning slowly — because the ratio was *never not 1*.

**The fix** (in [rl_trainer_v6.py](../src/training/rl_trainer_v6.py): `StepRecord.old_logps`): cache per-token logprobs *at rollout time* and reuse them for the entire PPO update phase. Then `new_logp` (recomputed each PPO epoch) genuinely differs from `old_logp` (frozen at sampling time), `r` actually moves, and clipping becomes meaningful.

After the fix, v7 step 25 clipfrac = 0.02 — non-trivial movement, well within the trust region.

---

## Q: How do these three signals interact?

Read them as a 3-light dashboard:

```
KL low (< 0.5)       → policy isn't drifting from SFT reference
pg_loss nonzero      → gradients are flowing
clipfrac > 0         → ratio is moving (no PPO bug)
```

**Healthy run signature**: all three lights green throughout.

**Failure modes** (each maps to a distinct underlying problem):

| symptom | likely cause |
|---|---|
| KL spiking (> 1) → calibration metric drops | reward shape is pulling policy away from SFT in a way that breaks the discrimination signal. *This is what killed v6 B-7.* |
| pg_loss → 0 (sticks at 0.000) | gradient has vanished — usually because every rollout in the batch gets the same reward (zero advantage signal), or because the policy has converged on a deterministic strategy that no longer has gradient pressure |
| clipfrac → 0 (and stays there) | either the v6 PPO bug (old==new logps) or the policy isn't moving at all (LR too low / data too easy) |
| KL low + pg_loss nonzero + Pass@1 stuck at 0% | reward is stable but the env is genuinely too hard — no rollout reaches success, so success_bonus never fires. Different problem; address via env design, not RL config |

The B-7 v6 collapse was a specific combo: KL drift (1.7), pg_loss freeze (0.000), clipfrac freeze (0.01) — gradient died after the policy walked off the SFT manifold. The v7 reward redesign targets the *cause* (per-step reward landscape, see [eval_2026-04-30_b7_rl_phase1.md](eval_2026-04-30_b7_rl_phase1.md) Option D); these three diagnostics will tell us whether the fix worked.

---

## Q: What does the per-step training log NOT show?

The per-step training log shows reward, solve-rate, and the three PPO diagnostics. It does **not** show:

- **Greedy calibration metrics** (`solvable_acc`, BP recall) — those come from the periodic `=== eval at step N ===` blocks (every `eval_every` steps, default 25). Per-step solve rate is computed at training temperature T=0.7 over K=8 rollouts/puzzle; greedy eval is at T=0.0 over 30 fresh puzzles. The two can disagree.
- **Pass@1 trajectory** — only computed at eval blocks.
- **Class-balance reweighting (v7 only)** — emitted as `class_balance_applied`, `n_solv`, `n_doom`, `w_solv`, `w_doom` fields in the JSONL log (`rl_log.jsonl`); not in the per-step stdout summary.

For full per-step state, read the JSONL: `outputs/rl_b7_phase1_v7/rl_log.jsonl`.
