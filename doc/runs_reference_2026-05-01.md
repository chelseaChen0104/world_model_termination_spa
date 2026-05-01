# Training Runs Reference (2026-05-01)

A structured catalog of every training run in this project. Each entry has the
same fields so you can read across runs at a glance:

```
ID:            short identifier (B-N, Run X, Phase Y)
Goal:          what the run was meant to test/produce
Status:        completed / failed (recoverable) / deprecated / in flight / pending
Hparameters:   env, model, lr, epochs/steps, batch, reward, etc.
Data:          dataset path used as input
Checkpoint:    output location (path on cloud + local if synced)
Result:        headline metrics
Notes:         diagnostic insights, links to follow-up runs
```

For chronological context with paragraph-form notes, see
[runs_ledger_2026-04-29.md](runs_ledger_2026-04-29.md). For the master pickup
point, see [HANDOFF.md](HANDOFF.md).

---

## SFT runs (Sudoku)

### B-0 (deprecated)

- **Goal**: Initial SFT with multi-turn full-tag-set format (`<observation>` +
  `<prediction>` + `<terminate_prob>` + `<solvable>` + `<breaking_point>` +
  `<steps_left>`) on Sudoku 9×9 random-play data.
- **Status**: ❌ failed (temporal echo). Greedy BP detection looked ~95% but was
  echoing prior `<solvable>=false` assertions in conversation history; BP recall
  on hold-out was 5%.
- **Hparameters**: 9×9 easy, Qwen2.5-1.5B-Instruct, lr=1e-5, ep=3, bs=32 effective.
- **Data**: `data/sudoku_9x9_termination_*` (random-play multi-turn, 6,221 samples).
- **Checkpoint**: `outputs/sft_termination*` (deprecated).
- **Result**: Greedy BP "accuracy" 95% (artefact); real recall 5%.
- **Notes**: Prompted the v3→v4 pivot to single-step minimal format with reduced
  tag set (drop `<terminate_prob>`, `<breaking_point>`, `<steps_left>`).

### B-1 (deprecated)

- **Goal**: Single-step minimal format on 9×9 Sudoku — sanity check that v4 format
  trains at all.
- **Status**: ❌ greedy collapse to "always False"; AUC ≈ 0.5.
- **Hparameters**: same as B-0.
- **Data**: 9×9 single-step minimal.
- **Checkpoint**: deprecated.
- **Result**: AUC ≈ 0.5, greedy 0% recall on solvable.
- **Notes**: Drove the move from greedy classification metric to ROC AUC for
  evaluation (avoids being misled by collapse direction).

### B-2 / B-3 / B-4 (deprecated)

- **Goal**: Successive 9×9 attempts with class rebalancing (B-3) and SPA-scale
  hparams (B-4) on 9×9 single-step data.
- **Status**: ❌ all plateau at AUC ≈ 0.46-0.47 (chance).
- **Hparameters**: 9×9 easy, lr=1e-5, ep=3, bs=32. B-3 added inverse-frequency
  class rebalancing; B-4 used SPA-scale hparams (lr=1e-4, ep=5, bs=16) but on
  the same 9×9 data.
- **Data**: `data/sudoku_9x9_*` (random-play, single-step, varying filters).
- **Checkpoint**: deprecated.
- **Result**: AUC = 0.46 / 0.46 / 0.45.
- **Notes**: Established that class imbalance and hparams alone don't fix 9×9 —
  task is too hard for Qwen-1.5B SFT at this scale. Pushed move to 4×4.

### B-5 ✅ (canonical Sudoku SFT)

- **Goal**: First-pass-on-the-recipe at smaller scale: 4×4 with SPA-scale hparams
  and SPA-scale data.
- **Status**: ✅ **completed** — first SFT run with real discrimination.
- **Hparameters**: Sudoku 4×4 easy, Qwen2.5-1.5B-Instruct, **lr=1e-4, ep=5, bs=16
  effective**, max_length=1024.
- **Data**: `data/sudoku_4x4_llm_policy_minimal_spa_scale/wm_train_no_post_bp.parquet`
  (LLM-policy data, 25,649 train + 6,413 val samples).
- **Checkpoint**: `outputs/sft_sudoku_4x4_minimal_b5_spa_hparams/final/`
  (autodl, autodl2, **local** ✓).
- **Result**: AUC = 0.726, P(true) separation +0.022, greedy `solvable_acc` =
  0.609. Prec(F) at τ=0.10 = ~65%.
- **Notes**: Closed Q1/Q7/Q8 for SFT. Used as the input policy for all Sudoku
  RL experiments.
- **Detailed report**: [eval_2026-04-29_b5_4x4_spa_replication.md](eval_2026-04-29_b5_4x4_spa_replication.md).

### B-6 (paused)

- **Goal**: Full 9×9 SPA-replication — 9×9 + SPA-scale data + SPA hparams.
- **Status**: ⏸️ paused; 9×9 SPA-scale data gen completed only easy difficulty
  (medium+hard skipped per pause request).
- **Hparameters**: 9×9 easy, SPA hparams.
- **Data**: `data/sudoku_9x9_llm_policy_minimal_spa_scale/wm_train_easy.parquet`.
- **Checkpoint**: not trained.
- **Result**: n/a.
- **Notes**: Would close the negative-result narrative on 9×9. ~12h GPU to resume.

---

## SFT runs (Pentomino)

### B-7 ✅ (canonical Pentomino SFT)

- **Goal**: Test cross-env transfer of the recipe — same SPA-scale recipe on
  Pentomino (5×4 board, `{L, P, W, Y}` pieces). New tag set (`<observation>` +
  `<next_state>` + `<viability>` + `<answer>`).
- **Status**: ✅ **completed**, AUC = 1.000.
- **Hparameters**: Pentomino-easy, Qwen2.5-1.5B-Instruct, **lr=1e-4, ep=5, bs=16
  effective**, max_length=1024.
- **Data**: `data/pentomino_easy_llm_policy_minimal/wm_train_no_post_bp.parquet`
  (2,964 train + 742 val; **80.7% step-0**, **0% step-3 samples** — see
  late-stage scarcity caveat).
- **Checkpoint**: `outputs/sft_pentomino_easy_b7_spa_hparams/final/` (autodl).
- **Result**: AUC = **1.000**, P(true) separation = +0.548 (24× B-5),
  Prec(F)=94.3% at τ=0.10. Pass@1 stochastic = **0/400** (sanity test).
- **Notes**: Recipe transfers cross-env *for discrimination*. But Pass@1 = 0
  reveals the SFT data has zero step-3 examples (model never saw a "place last
  piece" sample). Drives B-8 (augmenter) and B-9 (5×10 with longer trajectories).
- **Detailed reports**: [eval_2026-04-30_b7_pentomino_easy.md](eval_2026-04-30_b7_pentomino_easy.md),
  [sanity_2026-04-30_b7_rollout_stats.json](sanity_2026-04-30_b7_rollout_stats.json).

### B-8 ✅ (completed, augmentation cured Pass@1=0%)

- **Goal**: Test the late-stage scarcity hypothesis — does adding 30× oversampled
  solution-path samples (uniform across step 0-3) lift Pass@1 off 0%?
- **Status**: ✅ **completed on autodl1, 2026-05-01**. First attempt crashed at
  final save (autodl1 disk full); cleaned 28 GB and retrained successfully.
- **Hparameters**: Same as B-7 (lr=1e-4, ep=5, bs=16, max_length=1024).
- **Data**: `data/pentomino_b8_combined/wm_train_no_post_bp.parquet`
  (5,124 train: B-7's 2,964 + 30× × 72 augmented = 2,160. Distribution: step 0
  57%, step 1 21%, step 2 11%, step 3 11% ← **first non-zero step-3 coverage**).
- **Checkpoint**: `outputs/sft_pentomino_b8_augmented/final/` (autodl).
- **Result** (logprob eval + sanity rollout test):
  - AUC = **1.000** (perfect, matches B-7 — augmentation didn't degrade discrimination)
  - Threshold sweep: 100% Acc / Prec(T) / Rec(T) at all τ from 0.10 to 0.95
  - **Pass@1 stochastic (T=0.7) = 22.25%** (89/400) ← was 0/400 on B-7
  - Rollout length distribution: **30% reach step 4 (complete tilings)** vs 0% on B-7
  - First-action doom rate: 39% vs 73% on B-7 (model also picks better first moves)
  - Calibration trade: 98% → 91% viability accuracy (small)
- **Notes**:
  - Augmenter at [src/data/solution_path_augmenter.py](../src/data/solution_path_augmenter.py).
  - Combiner at [scripts/combine_b7_with_augmented.py](../scripts/combine_b7_with_augmented.py).
  - Sanity stats at [doc/sanity_2026-05-01_b8_rollout_stats.json](sanity_2026-05-01_b8_rollout_stats.json).
  - **Headline finding**: B-7's Pass@1=0% was a *data composition* problem, not a
    recipe problem. The augmenter targeting late-stage states cured it by enabling
    coherent multi-step format generation.

### B-9 📋 (planned)

- **Goal**: Larger Pentomino variant (5×10 / 10-piece) targeting trajectory length
  distribution. Tests whether "Pass@1 = 0% kills success_bonus → calibration drift"
  resolves with mechanically longer trajectories.
- **Status**: 📋 planned. P-0 sweep done, LLM-policy data done; augmenter run +
  SFT training pending.
- **Hparameters**: 5×10 board, **`{F, I, L, N, P, T, U, V, Y, Z}`** (10 of 12
  standard pentominoes — locked by P-0 sweep, 4,664 distinct tilings),
  Qwen2.5-1.5B-Instruct, lr=1e-4, ep=5, bs=16 expected.
- **Data**:
  - LLM-policy (done): `data/pentomino_b9_llm_policy_minimal/wm_train_no_post_bp.parquet`
    (4,652 train + 1,163 val, **48.2% solvable / 51.8% BP**, much more balanced
    than B-7's 18.8%/81.2% thanks to longer trajectories on the bigger board).
  - Solution-path augmented (pending): up to 4,664 × 10 = 46,640 samples; will use
    `--max-tilings 1000` to keep ~10K augmented + ratio reasonable.
- **Checkpoint**: pending (`outputs/sft_pentomino_b9_augmented/final/`).
- **Result**: pending.
- **Notes**:
  - P-0 sweep results: [doc/p0_5x10_10piece_sweep.txt](p0_5x10_10piece_sweep.txt).
  - Top-5 piece subsets all exclude X (the rare "+" pentomino, only 1 orientation).
  - **2026-05-01 finding**: B-9 LLM-policy data gen showed 100% parse-failure rate,
    0 successful trajectories. Confirms augmenter is essential for B-9 too.

---

## RL runs (Sudoku)

### Phase 1 v6 (deprecated)

- **Goal**: First Sudoku RL — does v6 reward (asymmetric per-class + format +
  success_bonus=10) lift Pass@1 above 0%?
- **Status**: ❌ no Pass@1 lift; calibration regressed 0.609 → 0.483.
- **Hparameters**: env=sudoku 4×4 easy, source=B-5 SFT, **lr=1e-6**, kl_coef=0.05,
  4 puzzles × 8 rollouts × 200 steps; v6 reward (TP +1.0, FN −0.7, FP −0.5,
  TN +0.3, success_bonus +10, format +0.05/tag).
- **Data**: live env (rollouts).
- **Checkpoint**: deprecated.
- **Result**: Pass@1 = 0%, solvable_acc 0.609 → **0.483**, KL drift 0.005.
- **Notes**: success_bonus=10 dominated rare-success gradient; per-step asymmetry
  caused calibration regression. Drove v6.1 (success_bonus 10→3) and lr bump to
  1e-5 (Run A).

### Phase 1 v6.1 ✅

- **Goal**: Reduced success_bonus (10→3) to give per-step calibration more relative
  weight.
- **Status**: ✅ Pass@1 lifted 0% → 6.67%; calibration recovered 0.483 → 0.620.
- **Hparameters**: same as v6 except success_bonus = 3.0, **lr=1e-5**.
- **Checkpoint**: `outputs/rl_b5_phase1_v6_1/final/` (autodl, autodl2).
- **Result**: Pass@1 = **6.67%**, solvable_acc = 0.620, bp_recall = 1.0.
- **Notes**: Confirmed v6.1 was the right reward shape for Sudoku. Used as input
  to Run A.

### Run A (Sudoku v6.1 continuation, 500 steps) ✅

- **Goal**: Test if longer training at the same lr=1e-5 lifts Pass@1 further.
- **Status**: ✅ **completed** — Pass@1 5× lift.
- **Hparameters**: source = v6.1 endpoint, **lr=1e-5, 500 steps**, otherwise same
  as v6.1.
- **Checkpoint**: `outputs/rl_b5_phase2_continue/final/` (autodl2).
- **Result**: Pass@1 trajectory:
  - step 0 (= v6.1 final): 6.67%
  - step 250 (peak): **36.67%** (11/30)
  - step 500 (final): **33.33%** (10/30)
  - solvable_acc: 0.620 → 0.514 (mild calibration regression, slower than v6
    Pentomino due to longer Sudoku trajectories)
  - bp_recall: 1.000 throughout ✓
- **Notes**: Confirms lr was the bottleneck on Sudoku, not the reward shape.
  Calibration drift smaller than v6 Pentomino because Sudoku's 3-5-step rollouts
  expose both classes to the gradient. Run A's checkpoint is the input to Phase 3
  v8 anchor below.

### Phase 3 v8 anchor (Sudoku) ✅

- **Goal**: Test the v8 viability-tag KL anchor on a known-working RL setup —
  does the anchor restore calibration without hurting Pass@1?
- **Status**: ✅ **completed** — Pass@1 lifted 33.33% → 50.0%, calibration held.
- **Hparameters**: source = Run A endpoint, env=sudoku 4×4 easy, lr=1e-5,
  200 steps, **--reward-version v8 --viability-kl-coef 0.5**.
- **Checkpoint**: `outputs/rl_b5_phase3_v8_anchor/final/` (autodl2, **local** ✓).
- **Result**:
  - Pass@1 trajectory: 33.33% (init) → 43.33% (step 25) → **50.0%** (step 200, 15/30)
  - solvable_acc: 0.514 (init) → **0.509** (held within noise)
  - bp_recall: 1.000 throughout ✓
  - via_kl ≈ 0 throughout when policy hadn't drifted; rose modestly on Sudoku
    (because Run A endpoint was already off-SFT)
  - ROC AUC at final = **0.949** (logprob-eval, 200 solv + 200 unsolv)
- **Notes**: First clean experimental win for the calibration-anchor mechanism.
  The anchor preserved calibration while Pass@1 climbed monotonically — exactly
  the design goal. **This checkpoint is the input to the Phase 2 truncation
  experiment.**

### Phase 2 truncation experiment Option A ✅ (rollout-only, 10 steps)

- **Goal**: Quantify the project's headline value claim — calibrated `<solvable>`
  predictions enable compute savings via early termination of doomed rollouts.
- **Status**: ✅ **completed 2026-05-01** on autodl2.
- **Hparameters**:
  - source = v8 anchor checkpoint (`outputs/rl_b5_phase3_v8_anchor/final`)
  - 10 rollout steps × truncation_mode=off
  - 10 rollout steps × truncation_mode=conservative, **τ=0.99**
  - same seed both conditions
- **τ selection**: threshold sweep on v8 checkpoint (200 solvable + 200 unsolvable
  val samples): P(false)|GT=False mean 0.997, median 1.000;
  P(false)|GT=True mean 0.959. τ=0.99 picks the regime where most doom states
  are truncated and most solvable states are NOT.
- **Result**:
  - Mean rollout time / step: 15.16 s → 14.10 s (**−7.0%**)
  - Mean tokens / step: 16,094 → 12,562 (**−21.9%**)
  - Mean rollout length: 4.50 → 3.52 (**−21.8%**)
  - Truncated rollouts: 0 / 320 → **173 / 320 (54.1%)**
  - Pass@1 / per-batch solve rate: preserved within sampling noise
- **Notes**: The 22% token reduction is the genuine compute (FLOPs) savings.
  Wall-time savings are smaller because batched rollouts are limited by the
  longest-surviving rollout per turn. Detailed report:
  [eval_2026-05-01_truncation_option_a.md](eval_2026-05-01_truncation_option_a.md).

### Phase 2 truncation experiment Option B ✅ (full RL training, 50 steps)

- **Goal**: End-to-end measurement of compute savings during agentic RL training,
  capturing both rollout-phase and PPO-update-phase savings.
- **Status**: ✅ **completed 2026-05-01** on autodl2.
- **Hparameters**: same as Option A but **50 RL steps** per condition (not 10),
  with eval_every=25 to track Pass@1 trajectory.
- **Checkpoints**: `outputs/trunc_exp_b_off/final/`, `outputs/trunc_exp_b_on_tau0.99/final/`.
- **Result**:
  - **Mean step time: 64.81 s → 52.27 s (−19.4%)** ← much bigger than Option A's
    −7% rollout-only number, because PPO update phase ALSO benefits from shorter
    rollouts (fewer tokens to compute logprobs over)
  - Total step time over 50 steps: 3,240 s → **2,613 s** (−10.4 min wall)
  - Mean tokens / step: 16,195 → 12,402 (−23.4%)
  - Mean rollout length: 4.55 → 3.49 (−23.3%)
  - Truncated rollouts: 0/1,600 → **881/1,600 (55.1%)**
  - Pass@1 / Eval — confounded by eval-time truncation:
    - OFF: step 0 = 50.0%, step 25 = 53.3%, step 50 = 53.3%
    - ON:  step 0 = 30.0%, step 25 = 23.3%, step 50 = 20.0%
    - **Methodology note**: `quick_pass1()` eval also runs through the
      truncation gate, so ON's eval Pass@1 underestimates the trained policy's
      true Pass@1 (recoverable rollouts get killed during eval). For an
      apples-to-apples comparison, the ON-final checkpoint should be re-eval'd
      with `truncation_mode=off`.
- **Notes**:
  - Token + wall-time savings are correctly measured (these are rollout-and-update
    metrics independent of eval procedure).
  - Pass@1 number for the paper requires the clean re-eval (~5 min on autodl2).
  - Launcher: [scripts/run_truncation_exp_option_b.sh](../scripts/run_truncation_exp_option_b.sh).

---

## RL runs (Pentomino)

### B-7 RL Phase 1 (v6) ❌ (deprecated)

- **Goal**: Apply the Sudoku RL recipe directly to Pentomino — does v6 reward
  with the same hparams work on the new env?
- **Status**: ❌ failed — calibration collapsed (greedy `<viability>` flipped
  True→False between steps 25 and 50).
- **Hparameters**: source = B-7 SFT, env=polyomino 5×4, lr=1e-5, 200 steps,
  v6 reward (success_bonus=3 from v6.1).
- **Checkpoint**: `outputs/rl_b7_phase1/final/` (deprecated; **deleted from autodl1**
  during disk cleanup 2026-05-01).
- **Result**: Pass@1 = 0%, solvable_acc = 1.000 → **0.000**, KL spiked to 1.7.
- **Notes**: First documented RL failure on Pentomino. Drove the v7 redesign and
  the diagnostic sanity test.
- **Detailed report**: [eval_2026-04-30_b7_rl_phase1.md](eval_2026-04-30_b7_rl_phase1.md).

### B-7 RL Phase 1 (v7) ❌ (deprecated)

- **Goal**: Test v7 reward (symmetric magnitudes + class balancing + progress
  bonus) as the targeted fix for v6 collapse mechanism.
- **Status**: ❌ partial mitigation only — collapse delayed from step ~50 to
  ~75 but not prevented.
- **Hparameters**: source = B-7 SFT, env=polyomino 5×4, lr=1e-5, killed at 100
  steps, **--reward-version v7** (TP=+1.0, FN=−1.0, FP=−1.0, TN=+1.0,
  class_balance=on, progress_bonus=0.1).
- **Checkpoint**: deprecated.
- **Result**: Pass@1 = 0%, solvable_acc = 1.000 → 0.000 (delayed by 25 steps vs v6).
- **Notes**: Made it clear the per-class-asymmetry hypothesis was insufficient.
  Drove the sanity test which revealed the *dynamic* drift mechanism, leading to
  the v8 anchor design.

### B-7 RL Phase 1 (v8) ❌ (deprecated)

- **Goal**: Test v8 = v7 + viability-tag KL anchor on Pentomino.
- **Status**: ❌ failed (anchor mechanism worked but couldn't overcome the
  Pentomino-specific issue: Pass@1 = 0% means success_bonus never fires →
  no goal-directed gradient → calibration drifts in the only direction available).
- **Hparameters**: source = B-7 SFT, env=polyomino 5×4, lr=1e-5, 200 steps,
  **--reward-version v8 --viability-kl-coef 0.5**.
- **Checkpoint**: deprecated.
- **Result**: Pass@1 = 0%, solvable_acc oscillated 0.0/1.0/0.0/1.0/0.0 then
  collapsed to final 0.0. via_kl = 0 throughout (anchor preserves sampled-token
  logp), but action policy + unsampled-token drift caused greedy collapse anyway.
- **Notes**: The contrast with the **successful Sudoku v8 anchor** is informative
  — same anchor mechanism, same code; the difference is whether the env supports
  a non-zero Pass@1. This is the strongest piece of evidence that the Pentomino
  failure is env-shape, not reward-shape.

### B-8 RL with v8 anchor 🔄 (in flight, stochastic-vs-greedy gap)

- **Goal**: Test whether the v8 calibration anchor mechanism generalizes to
  Pentomino once the SFT data composition is fixed (B-8). Sudoku v8 anchor lifted
  Pass@1 33% → 50% with calibration held; does the same recipe work given B-8's
  better starting point?
- **Status**: 🔄 in flight on autodl1 (step 75/200, ~1.5 hr remaining).
- **Hparameters**: source = B-8 SFT, env=polyomino 5×4, lr=1e-5, 200 steps,
  --reward-version v8 --viability-kl-coef 0.5.
- **Checkpoint**: pending (`outputs/rl_b8_v8_anchor/final/`).
- **Result so far** (steps 25, 50, 75):
  - **Per-batch solve rate (T=0.7) climbed 16% → 84% peak → 53-66% steady** — by
    far the strongest Pentomino RL action-policy result we've ever produced.
  - via_kl ≈ 0 throughout (single-token anchor active and effective on sampled
    tokens), KL drift tiny (~0.01).
  - **But greedy Pass@1 = 0% across all 3 evals** (steps 25, 50, 75); greedy
    `solvable_acc` collapsed to 0.0 from initial 1.0.
- **Diagnosis**: this is the **stochastic-vs-greedy gap**. v8 single-token anchor
  preserves logp of the SAMPLED viability token, which keeps stochastic sampling
  on-distribution. But greedy argmax is determined by the relative order of
  >true vs >false logprobs at each viability position; if the unsampled token's
  logp drifts independently, greedy can flip from True→False even with
  via_kl=0. **Action policy and viability head are independently learnable axes
  in our setup; RL improved the action head dramatically while viability greedy
  collapsed.**
- **Implication for paper**: Pass@8 (or stochastic Pass@N) from this checkpoint
  should be huge given 53-84% per-batch solve rate. Greedy Pass@1 needs the
  v8.2 dual-token anchor to be stable.
- **Notes**: Drove the v8.2 implementation. Launcher:
  [scripts/run_rl_b8_v8.sh](../scripts/run_rl_b8_v8.sh).

---

## Pending / proposed runs

### B-8 RL with v8.2 (dual-token anchor) 📋

- **Goal**: Test whether the v8.2 dual-token anchor (anchor BOTH `>true` and
  `>false` logprobs at every viability position regardless of which was sampled)
  closes the stochastic-vs-greedy gap that B-8 RL with single-token v8 exposed.
- **Mechanism**: v8 only constrained logp of the sampled token. Greedy argmax
  depends on the relative order of `>true` vs `>false` at each viability
  position; if the unsampled token drifts, greedy flips. v8.2 anchors both
  tokens by construction → relative ordering is preserved by the loss term.
- **Pipeline**: same as B-8 RL with v8 but `--dual-token-anchor` enabled
  (auto-detects token IDs at startup; verified `>true`=33284, `>false`=30392
  on Qwen2.5).
- **Hparameters**: source = B-8 SFT, env=polyomino 5×4, lr=1e-5, 200 steps,
  --reward-version v8 --viability-kl-coef 0.5 --dual-token-anchor.
- **Checkpoint**: pending (`outputs/rl_b8_v8_2_dual_anchor/final/`).
- **Effort**: ~3-5 hr GPU (same as v8; small overhead for extra forward passes
  on the 1-3 viability positions per response).
- **Success criteria**: greedy Pass@1 ≥ B-8 SFT's 0% (i.e., positive), greedy
  solvable_acc preserved. Per-batch stochastic solve rate stays high (50%+).
- **Launcher**: [scripts/run_rl_b8_v8_2.sh](../scripts/run_rl_b8_v8_2.sh).

### Phase 2 truncation re-eval (clean Pass@1 measurement) 📋

- **Goal**: Re-evaluate the Option B ON-final checkpoint with
  `truncation_mode=off` to disambiguate "training+truncation Pass@1" from
  "eval-time truncation eval-side effect."
- **Pipeline**: load `outputs/trunc_exp_b_on_tau0.99/final/`, run quick_pass1
  on 30 fresh puzzles with truncation off.
- **Effort**: ~5 min on autodl2.
- **What it tells us**: if Pass@1 ≈ 50% under clean eval, truncation didn't
  hurt the trained model and Option B's reported drop was eval-side artifact.
  If still <50%, training-with-truncation actually regressed quality.

### B-9 SFT (5×10 / 10-piece, augmented)

- **Goal**: Bigger Pentomino + late-stage augmentation, on a board where Pass@1
  > 0 is achievable.
- **Pipeline**:
  1. Run augmenter on 5×10 / `{F,I,L,N,P,T,U,V,Y,Z}` with `--max-tilings 1000`
     (~10 min CPU). Outputs ~10K samples uniform across step 0-9.
  2. Combine with B-9 LLM-policy data (4.6K samples).
  3. Train SFT (~1.5 hr GPU).
  4. Eval AUC, Pass@1 stochastic, sanity test rollout distribution.
- **Effort**: ~3-4 hr total.
- **Success criteria**: AUC ≥ 0.95, Pass@1 stochastic > 1% (any non-zero
  validates the trajectory-length-distribution hypothesis), step 8/9 sample
  count > 0.

### B-9 RL with v8 anchor

- **Goal**: Run RL on B-9 SFT with the v8 anchor — if Pass@1 lifts the way
  Sudoku v8 did, the recipe is complete on a harder env.
- **Pipeline**: same as Phase 3 v8 anchor (lr=1e-5, v8 reward, viability_kl_coef
  0.5), 200 steps.
- **Effort**: ~5 hr GPU.
- **Gates**: only run if B-9 SFT achieves Pass@1 stochastic > 1% (success_bonus
  must fire occasionally for the gradient to be meaningful).

### Phase 2 truncation Option B (full RL with truncation)

- **Goal**: Apples-to-apples measurement of the compute-savings claim during
  agentic RL training (vs Option A's rollout-only proxy).
- **Pipeline**: 50 RL steps with truncation OFF, 50 with truncation ON, both
  starting from v8 anchor checkpoint. Measure end-to-end wall time, Pass@1
  trajectory, total tokens.
- **Effort**: ~6 hr GPU.
- **Gate**: only run if Option A shows clear savings (>20%) without Pass@1
  regression.

### B-7 train↔val (s_t, a_t) overlap check

- **Goal**: Disambiguate memorization vs generalization for B-7's AUC=1.000.
- **Pipeline**: deduplicate train and val by (state, action) pairs, rerun eval.
- **Effort**: ~10 min CPU.
- **Notes**: Reduces reviewer attack surface.

### B-6 (9×9 SPA-replication, paused)

- **Goal**: Close the negative-result narrative on 9×9.
- **Pipeline**: resume 9×9 SPA-scale data gen (medium + hard difficulties),
  train SFT.
- **Effort**: ~12 hr GPU.

---

## Sample composition snapshot (Pentomino)

| dataset | total | step 0 | step 1 | step 2 | step 3 | step 4+ |
|---|---|---|---|---|---|---|
| B-7 (LLM-policy only) | 2,964 | 80.7% | 18.0% | 1.3% | **0.0%** | 0% |
| **B-8 (B-7 + 30× augmented)** | **5,124** | 57.2% | 21.0% | 11.3% | **10.5%** | 0% |
| B-9 LLM-policy (5×10) | 4,652 | TBD | TBD | TBD | TBD | mixed |

The B-7 → B-8 jump is the headline: step 3 went from zero coverage to 10.5%.

---

## Glossary

- **Pass@1**: fraction of held-out puzzles the model solves with greedy decoding.
- **`solvable_acc`**: per-step accuracy of `<viability>` / `<solvable>` predictions
  on the *greedy state distribution*.
- **`bp_recall`**: fraction of true breaking points (solvable→doom transitions) the
  model identifies as `<viability>=False`.
- **`Prec(F)`**: among predictions of "False," the fraction that are actually doom.
  The truncation gate threshold metric.
- **`via_kl`**: per-step viability-tag KL deviation from frozen ref policy. v8
  anchor metric. Should stay near 0 if anchor is effective.
- **τ_truncation**: probability threshold for the Phase 2 truncation gate.
  `truncate if logp(false_token) > log(τ)`. Default 0.95; we use 0.99 on Sudoku
  v8 to balance Prec(F) against Rec(F)|GT=True.
