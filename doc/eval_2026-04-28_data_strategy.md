# Evaluation — Multi-turn + Data Generation Strategy (2026-04-28)

Written mid-Track-B run, after Track A (random-play multi-turn) completed and the parquet was inspected directly.

## Glossary

- **Track A** = random-play multi-turn data generation. Random non-conflicting placement at each step. CPU-only. Output: `data/sudoku_multiturn/`. ✅ done.
- **Track B** = LLM-policy multi-turn data generation. Qwen2.5-1.5B-Instruct picks moves. Runs on H800. Output: `data/sudoku_llm_policy/`. ⏳ running.

## Numbers measured (Track A, sample of 5,000 from `wm_train.parquet`)

| Metric | Value |
|---|---|
| Solvable | 6.3% |
| Unsolvable | 93.7% |
| Breaking point | 4.3% |
| Multi-turn depth (msgs/sample) | min 2, max 22, mean 17.6 |
| 60% of samples have full 22-msg context | (10 prior turns + system + current state) |
| Trajectory step span | 0–29, mean 12.2 |
| BP step distribution | clustered at steps 0–10 (early game) |

## 1. Multi-turn strategy

Each sample is `[system, user_state_0, asst_resp_0, ..., user_state_K]`, 10-turn sliding window, loss masked to the final assistant turn only.

**Concerns:**
- **Track A: prior assistant turns are template-generated *correct* labels.** The model trains on `(history of correct predictions) → correct prediction`. At inference its own prior predictions will be noisy — mild distribution shift. Sudoku's full observability mitigates this; the grid carries the load, prior reasoning is mostly redundant.
- **Track B: prior turns are the LLM's actual outputs**, which match inference distribution. Better.
- Sliding window of 10 is fine for Sudoku — grid state is fully visible, so we lose only reasoning chains, not game state.

**Verdict:** multi-turn is justified for Track B; for Track A it's overbuilt. Single-turn would have been ~equivalent for random-play data.

## 2. Data generation strategy

**Track A:** random non-conflicting placement. Mean trajectory length: **12.2 steps** before termination — random play burns out fast.

**Track B:** LLM picks reasonable moves. Expected to survive longer per trajectory → more solvable states per trajectory, fewer BPs proportionally, BPs that look more like real mistakes.

The two tracks produce **structurally different** datasets, not just different sample sizes.

## 3. Data balance

**Two problems with Track A's distribution:**

1. **Trivial baseline = 93.7% accuracy** by predicting "always unsolvable." Headline accuracy will be misleading.
2. **BP timing is unrealistic:** BPs cluster at steps 0–10 because random placements quickly create unsatisfiable constraints. An LLM playing the game will make subtler mistakes 15–25 moves in, where most of the grid is filled. The SFT model trained on Track A will learn "BPs come early in mostly-empty grids" — wrong prior for deployment.

Track B should fix both: better solvable/BP ratio (LLM survives longer), and BPs distributed across the trajectory.

## 4. Expected results

**Track A SFT:**
- Headline accuracy: ~93–95% (mostly from predicting unsolvable)
- BP recall: ~30–55%, biased toward early-game BPs only
- BP precision: low; model over-predicts BP for sparse grids
- Useful as a checkpoint and warm start, **not as a deployable predictor**

**Track B SFT:**
- Lower headline accuracy (~75–85%) because the dataset is harder
- BP recall on LLM-policy eval: 5–20 points higher than Track A is a reasonable hope
- This is the model worth eval-ing seriously

**RL with reward v2 + balanced live sampling:**
- Designed to fix what SFT data couldn't — class imbalance via balanced batches, BP undertraining via asymmetric reward (TP +3.0, FN −2.0).
- Risk: SFT-on-Track-A → balanced-live-env-RL has large distribution shift; initial RL steps will be unstable. Track B SFT mitigates this.

## 5. Recommended tweaks (non-blocking)

1. **Eval both SFT models on the same balanced live-env eval set.** Track A's apparent weakness vs. Track B's apparent strength is the real question to answer.
2. **Add a class-weighted SFT loss.** Weighting BP samples ~5× in cross-entropy is cheap and would measurably lift BP recall. Currently only RL has asymmetric weighting.
3. **Periodically eval Track A SFT on Track B-style states** to quantify distribution shift, not just trust the rationale in [CLAUDE.md](../CLAUDE.md).
