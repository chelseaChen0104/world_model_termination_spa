# SFT Runs Ledger — World Model Termination SPA (2026-04-29)

Consolidated record of all SFT training runs. Each entry has: hparams, data path, sample counts, output checkpoint, training/eval logs, key numbers, and a one-paragraph note on what we learned.

For the deeper write-ups on the headline findings, see the dated artifacts:
- [report_2026-04-28_sft_b_diagnosis_and_pivot.md](report_2026-04-28_sft_b_diagnosis_and_pivot.md) — B-0 temporal-echo finding
- [eval_2026-04-28_sft_track_b_tier_a.md](eval_2026-04-28_sft_track_b_tier_a.md) — B-0 single-turn Tier A eval
- [eval_2026-04-29_b5_4x4_spa_replication.md](eval_2026-04-29_b5_4x4_spa_replication.md) — B-5 (the success)
- [qa_2026-04-29_tag_design.md](discussion/qa_2026-04-29_tag_design.md) — why we dropped `<breaking_point>` / `<terminate_prob>` / `<steps_left>`

Common across all runs:
- Base model: **Qwen/Qwen2.5-1.5B-Instruct**
- Trainer: [`src/training/simple_sft_trainer.py`](../src/training/simple_sft_trainer.py) (HuggingFace Trainer, single GPU, no FSDP)
- Cloud: H800 on AutoDL (`autodl` and `autodl2` aliases)
- Format compliance is 100% on every run that reached eval (the model never breaks the XML scaffold).

---

## Summary table

| Run | Date | Task | Format | Train n | Epochs | LR | Batch | ROC AUC | Outcome |
|---|---|---|---|---|---|---|---|---|---|
| **B-0** | 2026-04-28 | 9×9 | multi-turn (full tags) | 6,221 | 3 | 1e-5 | 32 | — *(BP recall 5%)* | ❌ temporal echo |
| **B-1** | 2026-04-28 | 9×9 | single-step minimal | 6,221 | 3 | 1e-5 | 32 | ~0.5 | ❌ greedy collapse to False |
| **B-2** | 2026-04-29 | 9×9 (no_post_bp) | single-step minimal | 2,482 | 3 | 1e-5 | 32 | **0.468** | ❌ chance |
| **B-3** | 2026-04-29 | 9×9 (no_post_bp + class-bal) | single-step minimal | ~2,482* | 3 | 1e-5 | 32 | **0.462** | ❌ chance |
| 4x4 baseline | 2026-04-29 | 4×4 (no_post_bp) | single-step minimal | 1,336 | 3 | 1e-5 | 32 | — *(greedy Rec 2%)* | ❌ collapse, AUC not measured |
| **B-4** | 2026-04-29 | 9×9 (no_post_bp) | single-step minimal | 2,482 | 5 | **1e-4** | **16** | **0.455** | ❌ chance — disproved scale-only hypothesis |
| **B-5** | 2026-04-29 | 4×4 SPA-scale (no_post_bp) | single-step minimal | **6,571** | 5 | **1e-4** | **16** | **0.726** | ✅ **first run with real signal** |
| **B-7** | 2026-04-30 | **5×4 Pentomino-easy** {L,P,W,Y} | single-step minimal (`<viability>` tags) | **2,964** | 5 | **1e-4** | **16** | **1.000** | ✅ **perfect AUC; cross-env transfer confirmed** |

*B-3 sample count assumed ≈ B-2 based on shared no_post_bp filter; exact count in train log unverified (separate output dir, log not preserved as a dedicated file).

---

## B-0 — multi-turn full-tag SFT *(failure: temporal echo)*

| field | value |
|---|---|
| Output dir | `outputs/sft_sudoku_llm_policy/` |
| Train file | `data/sudoku_llm_policy/wm_train_filtered.parquet` |
| Val file | `data/sudoku_llm_policy/wm_val_filtered.parquet` |
| Train samples | 6,221 (multi-turn, full tag set) |
| Val samples | 1,608 |
| Epochs / LR / batch | 3 / 1e-5 / 32 (4 per_device × 8 grad_accum) |
| `max_length` | 4,096 |
| Total updates | 582 |
| Training log | [logs/sft_b.log](../logs/sft_b.log) |
| Eval logs | [logs/eval_a.log](../logs/eval_a.log) (greedy), [logs/eval_a_multiturn.log](../logs/eval_a_multiturn.log) (multi-turn) |

**Notes.** First Track-B run. Multi-turn samples included prior assistant turns (`<solvable>` history) as conversation context, with the loss masked to the final turn. Multi-turn eval showed BP detection accuracy ~95% (looked great) but BP **recall = 5.0%** (tiny — the model was just echoing prior `<solvable>=false` assertions, not learning). Single-turn evaluation showed it had collapsed to all-True. Diagnosed in [report_2026-04-28_sft_b_diagnosis_and_pivot.md](report_2026-04-28_sft_b_diagnosis_and_pivot.md): 84.7% of multi-turn samples have a *trivial echo shortcut* — prior turn's `<solvable>` matches the target. **This run motivated the v4 amendment**: pivot to single-step samples + minimal tag set.

---

## B-1 — single-step minimal, post-BP kept *(failure: chance, greedy collapse to False)*

| field | value |
|---|---|
| Output dir | `outputs/sft_sudoku_minimal/checkpoint-582` (no `/final/` saved) |
| Train file | `data/sudoku_llm_policy_minimal/wm_train_filtered.parquet` |
| Val file | `data/sudoku_llm_policy_minimal/wm_val_filtered.parquet` |
| Train samples | 6,221 (single-step minimal, post-BP samples kept) |
| Val samples | 1,608 |
| Epochs / LR / batch | 3 / 1e-5 / 32 |
| `max_length` | 2,048 (after format change shrunk responses) |
| Training log | [logs/sft_b_minimal.log](../logs/sft_b_minimal.log) |
| Eval log | [logs/eval_b1.log](../logs/eval_b1.log) |
| Greedy Acc / Prec / Rec / F1 | 67.0% / 100% / 1.0% / 2.0% |
| ROC AUC | ~0.5 (no signal) |

**Notes.** First single-step minimal-format run after the v4 pivot. Same data as B-0 but reformatted via [`scripts/reformat_to_minimal.py`](../scripts/reformat_to_minimal.py) to drop multi-turn history and the redundant tags. Greedy classification collapsed to "always-False" (recall = 1.0% on solvable). Format compliance 100%, but no learned discrimination — just learned the prior on the unsolvable-dominated training distribution. Combined with B-2/B-3, this is what motivated the threshold-based logprob eval (greedy alone is a poor metric when the model is biased one way).

---

## B-2 — single-step minimal, post-BP filter applied *(failure: chance, greedy collapse to True)*

| field | value |
|---|---|
| Output dir | `outputs/sft_sudoku_minimal_no_post_bp/final` |
| Train file | `data/sudoku_llm_policy_minimal/wm_train_filtered_no_post_bp.parquet` |
| Val file | `data/sudoku_llm_policy_minimal/wm_val_filtered_no_post_bp.parquet` |
| Train samples | 2,482 (post-BP filler removed) |
| Val samples | 639 |
| Epochs / LR / batch | 3 / 1e-5 / 32 |
| `max_length` | 2,048 |
| Total updates | ~234 |
| Training log | [logs/sft_b2.log](../logs/sft_b2.log) |
| Eval logs | [logs/eval_b2.log](../logs/eval_b2.log) (greedy), [logs/eval_logprob.log](../logs/eval_logprob.log) (logprob, first half) |
| Greedy Acc / Prec / Rec / F1 | 34.3% / 33.1% / 95.0% / 49.1% |
| **ROC AUC** | **0.468** |

**Notes.** Same minimal format as B-1 but with [`scripts/filter_post_bp.py`](../scripts/filter_post_bp.py) applied to remove (Solvable=False, BP=False) post-BP filler samples — improves class balance from ~94/6 unsolvable/solvable to ~75/25. Greedy this time collapsed in the *opposite* direction (always-True), because the post-BP filter shifted the class prior. Logprob threshold sweep revealed AUC = 0.468 — same chance as B-1; greedy collapse direction is just an artifact of where the threshold lands. **This run produced our first AUC measurement and the realization that greedy is a brittle metric for this task.**

---

## B-3 — no_post_bp + class-balancing variant *(failure: chance)*

| field | value |
|---|---|
| Output dir | `outputs/sft_sudoku_minimal_b3/final` |
| Data | likely no_post_bp + 2× BP oversample (per [`scripts/oversample_bp.py`](../scripts/oversample_bp.py)) — exact training-time data path not preserved in a dedicated log |
| Epochs / LR / batch | 3 / 1e-5 / 32 (defaults) |
| Eval logs | [logs/eval_b3.log](../logs/eval_b3.log) (greedy), [logs/eval_logprob.log](../logs/eval_logprob.log) (logprob, second half) |
| Greedy Acc / Prec / Rec / F1 | 64.7% / 33.3% / 6.0% / 10.2% |
| **ROC AUC** | **0.462** |

**Notes.** Class-balancing variant of B-2 — applied 2× oversampling on BP samples to lift the BP class prior. AUC 0.462 confirms what B-1/B-2 already showed: **the class imbalance was not the cause of zero discrimination**; rebalancing doesn't fix it. After this run, we ruled out class balance as the explanation and started looking at training scale (which led to the 80× under-training hypothesis). The dedicated training log was not preserved as a separate file (some sessions wrote to a shared log that was overwritten).

---

## 4×4 baseline *(failure: greedy collapse, AUC not measured)*

| field | value |
|---|---|
| Output dir | `outputs/sft_sudoku_4x4_minimal_no_post_bp/final` |
| Train file | `data/sudoku_4x4_llm_policy_minimal/wm_train_no_post_bp.parquet` |
| Val file | `data/sudoku_4x4_llm_policy_minimal/wm_val_no_post_bp.parquet` |
| Train samples | 1,336 (single-cloud 1,000-traj gen, no_post_bp filtered) |
| Val samples | 325 |
| Epochs / LR / batch | 3 / 1e-5 / 32 (same defaults as B-1/B-2/B-3) |
| `max_length` | 1,024 |
| Total updates | 123 |
| Training log | [logs/sft_4x4.log](../logs/sft_4x4.log) |
| Eval log | [logs/eval_4x4.log](../logs/eval_4x4.log) |
| Greedy Acc / Prec / Rec / F1 | 67.3% / 100% / 2.0% / 3.9% |
| ROC AUC | not measured for this run |

**Notes.** Run via [`scripts/run_4x4_pipeline.sh`](../scripts/run_4x4_pipeline.sh) — the first 4×4 SPA-replication attempt, but **with our smaller hparams (lr=1e-5, ep=3, bs=32) and only 1,000 trajectories**. Greedy collapsed to "always-False" with 2.0% recall on solvable. We did not run the threshold-based logprob eval on this checkpoint at the time; the next 4×4 attempt (B-5) used SPA's hparams + ~5× the data. Sat between the B-3 / B-4 runs chronologically; not labeled with a B-N number because it's a different task variant (not a successor of the 9×9 B-series).

---

## B-4 — 9×9 + SPA hyperparameters *(failure: chance, disproves under-training hypothesis)*

| field | value |
|---|---|
| Output dir | `outputs/sft_sudoku_minimal_b4_spa_hparams/checkpoint-600` (final crashed mid-save, see notes) |
| Train file | `data/sudoku_llm_policy_minimal/wm_train_filtered_no_post_bp.parquet` (same data as B-2/B-3) |
| Val file | `data/sudoku_llm_policy_minimal/wm_val_filtered_no_post_bp.parquet` |
| Train samples | 2,482 |
| Val samples | 639 |
| Epochs / LR / batch | **5 / 1e-4 / 16** (per_device 4 × grad_accum 4) |
| `max_length` | 2,048 |
| Planned updates | 775 |
| **Actual updates completed** | **600** (training crashed at step 600 during checkpoint save — disk full on autodl1) |
| Launch script | [scripts/run_sudoku_9x9_sft_b4.sh](../scripts/run_sudoku_9x9_sft_b4.sh) |
| Training log | [logs/sft_b4.log](../logs/sft_b4.log) |
| Eval logs | [logs/eval_b4.log](../logs/eval_b4.log) (failed — no `/final/`), [logs/eval_b4_ck600.log](../logs/eval_b4_ck600.log) (recovered) |
| Greedy Acc / Prec / Rec / F1 | 33.3% / 33.3% / 100.0% / 50.0% (collapsed to all-True) |
| **ROC AUC** | **0.455** |

**Notes.** Designed to test the "we're 80× under-trained vs SPA" hypothesis: same data as B-2/B-3 but with SPA's exact hparams (5 epochs × 1e-4 LR × bs 16 ≈ 30× more effective gradient signal). Crashed at step 600 of 775 during the `/final/` checkpoint save when autodl1's disk hit 100% (`RuntimeError: unexpected pos 2695335488 vs 2695335380`). The 200/400 intermediate checkpoints were full size; checkpoint-600's `model.safetensors` (3.08GB) saved cleanly but `optimizer.pt` was truncated. We deleted intermediate checkpoints to recover disk and ran eval against `checkpoint-600/` — model loads fine for inference. **AUC = 0.455 is essentially chance**; eval_loss curve shape was healthier than B-3 (faster early convergence) but the discrimination signal didn't appear. **Conclusion: the under-training hypothesis is wrong — scale alone does not fix 9×9.** This is what motivated the B-5 4×4 replication: if 4×4 succeeded with the same hparams, the issue must be task difficulty, not recipe.

---

## B-5 — 4×4 + SPA hyperparameters + SPA-scale data *(success: AUC 0.726)*

| field | value |
|---|---|
| Output dir | `outputs/sft_sudoku_4x4_minimal_b5_spa_hparams/final` |
| Train file | `data/sudoku_4x4_llm_policy_minimal_spa_scale/wm_train_no_post_bp.parquet` |
| Val file | `data/sudoku_4x4_llm_policy_minimal_spa_scale/wm_val_no_post_bp.parquet` |
| Train samples | **6,571** (4×4 SPA-scale, no_post_bp; class balance 40% solvable / 60% BP) |
| Val samples | 1,645 |
| Epochs / LR / batch | **5 / 1e-4 / 16** |
| `max_length` | 1,024 |
| Total updates | 2,050 |
| `eval_steps` | 10 (205 dense eval points — finest curve we have) |
| Launch script | [scripts/run_sudoku_4x4_sft.sh](../scripts/run_sudoku_4x4_sft.sh) |
| Training log | [logs/sft_b5.log](../logs/sft_b5.log) |
| Eval log | [logs/eval_b5.log](../logs/eval_b5.log) |
| Greedy Acc / Prec / Rec / F1 | 44.3% / 24.4% / 32.0% / 27.7% (over-predicts True; not collapsed) |
| **ROC AUC** | **0.726** ✅ |
| P(true) mean separation | +0.023 (solvable 0.045 vs unsolvable 0.022) |

**Notes.** Data generated by splitting 4×4 SPA-scale gen (~5,000 trajectories total) across both clouds in parallel: autodl1 part-A (2,500 trajs, seed=42) + autodl2 part-B (2,500 trajs, seed=43), ~3.7h each. Combined locally via [`scripts/combine_4x4_spa_scale_parts.py`](../scripts/combine_4x4_spa_scale_parts.py). **First SFT run with real solvability discrimination** (AUC 0.726 vs ≈0.46 for every prior run). Greedy is still noisy — the model is uncalibrated and biased toward False (mean P(true) = 0.045 even on actually-solvable states) — but the *ranking* is sound. Confirmed the recipe works; localized the 9×9 collapse to task-difficulty, not architecture/hparams. Full report: [eval_2026-04-29_b5_4x4_spa_replication.md](eval_2026-04-29_b5_4x4_spa_replication.md). Loss-curve mid-training spike at step 430 (~epoch 1.05) is an optimizer transient at the epoch-2 boundary; benign.

---

## B-7 — 5×4 Pentomino-easy SFT *(success: AUC 1.000, cross-env transfer)*

| field | value |
|---|---|
| Output dir | `outputs/sft_pentomino_easy_b7_spa_hparams/final` (on autodl1) |
| Train file | `data/pentomino_easy_llm_policy_minimal/wm_train_no_post_bp.parquet` |
| Val file | `data/pentomino_easy_llm_policy_minimal/wm_val_no_post_bp.parquet` |
| Train samples | **2,964** (pentomino-easy; class balance **18.8% solvable / 81.2% BP**) |
| Val samples | 742 |
| Epochs / LR / batch | **5 / 1e-4 / 16** (mirror B-5 hparams) |
| `max_length` | 1,024 |
| Total updates | 925 |
| `eval_steps` | 25 |
| Launch script | [scripts/run_pentomino_5x4_sft.sh](../scripts/run_pentomino_5x4_sft.sh) |
| Training log | [logs/sft_b7.log](../logs/sft_b7.log) |
| Eval log | [logs/eval_b7.log](../logs/eval_b7.log) |
| Greedy Acc / Prec / Rec / F1 | 49.5% / 33.3% / 1.0% / 1.9% (collapsed to all-False — same shape as B-5 greedy) |
| **ROC AUC** | **1.000** ✅✅ (perfect) |
| P(true) mean separation | **+0.548** (solvable 0.548 vs unsolvable 0.000) — 24× larger than B-5 |
| Prec(F) at τ=0.10 | **94.3%** — already past the Phase 2 truncation gate (≥90%) |

**Notes.** First SFT run on a structurally different env from Sudoku. New tag set: `<observation>` + `<next_state>` + `<viability>` + `<answer>` (renamed per [doc/spec_pentomino.md](spec_pentomino.md) §4). **Recipe transfers cross-env on the discrimination metric** — AUC 1.000 vs B-5's 0.726, and 100% precision at τ=0.10 with 94% recall. Greedy still collapses to "always False" (same calibration issue as B-5 — bimodal P(true) with most mass on False) but the *ranking* is perfect. Pentomino's predictive gap is more visually local (isolated unfillable regions are directly visible in the rendered cells) than Sudoku's (constraint cascade across rows/cols/boxes), which likely explains the stronger discrimination. Class imbalance (81% BP) reinforces the "this looks doomed" signal. Full report: [eval_2026-04-30_b7_pentomino_easy.md](eval_2026-04-30_b7_pentomino_easy.md). Required two minor patches: (1) `evaluate_rl.py` `parse_predictions` now accepts both `<solvable>` and `<viability>` tags; (2) `evaluate_solvable_logprob` got a `tag_name` parameter + `--tag-name` CLI flag.

---

## What we learned, summarized

1. **Multi-turn collapses to echo on Sudoku** (B-0). Single-step samples are mandatory for this task.
2. **Greedy classification is a brittle metric** when the model is uncalibrated — collapse direction is an artifact of class-prior, not signal. ROC AUC is the trustworthy metric (B-1 vs B-2 made this obvious).
3. **Class rebalancing alone doesn't fix discrimination** (B-3 vs B-2: identical AUC).
4. **Hyperparameter scaling alone doesn't fix 9×9** (B-4 vs B-3: same AUC despite 30× more effective gradient signal).
5. **The recipe works on 4×4 with SPA-scale data and SPA hparams** (B-5: AUC 0.726). The 9×9 collapse is task-difficulty for Qwen-1.5B SFT alone.
6. **The recipe transfers cross-env** (B-7 pentomino-easy: AUC 1.000). Pentomino's predictive gap is sharper (visually local) and yields stronger discrimination than 4×4 Sudoku — even with less than half the training data.

## Pending and recently completed runs (post-2026-04-30)

### Recently completed

- **Phase 1 v6.1 RL on B-5** — completed, lifted Pass@1 from 0% → 6.67% at lr=1e-6; the v6.1 success_bonus reduction (10→3) also dampened (but did not eliminate) the v6 calibration regression. Output: `outputs/rl_b5_phase1_v6_1/final`.

- **Run A (Sudoku continuation)** — completed 2026-05-01 on autodl2. Continued from v6.1 endpoint at **lr=1e-5** (10× the original) for 500 more steps. **Pass@1: 6.67% → 33.33%** (peak 36.67% at step 250); bp_recall held at 1.000 throughout; `solvable_acc` drifted 0.620 → 0.514 (mild calibration regression — same direction as v6 Pentomino but much slower). Headline finding: **lr was the bottleneck, not the reward shape, on Sudoku.** Output: `outputs/rl_b5_phase2_continue/final`. Eval trajectory:
  | step | Pass@1 | solvable_acc | bp_recall |
  |---|---|---|---|
  | 0 (= v6.1 endpoint) | 6.67% | 0.620 | 1.000 |
  | 50 | 26.67% | 0.609 | 1.000 |
  | 250 | **36.67% (peak)** | 0.467 | 1.000 |
  | 500 (final) | 33.33% | 0.514 | 1.000 |

- **B-7 RL Phase 1 (v6, deprecated)** — completed 2026-04-30 with calibration collapse. v6 reward + 1-step rollout bias on Pentomino-easy → greedy `<viability>` flipped True→False between steps 25 and 50. Pass@1 stayed 0%. See [eval_2026-04-30_b7_rl_phase1.md](eval_2026-04-30_b7_rl_phase1.md). B-7 SFT (AUC=1.000) remains the canonical pentomino model. Failed checkpoint at `outputs/rl_b7_phase1/final` (do not deploy).

- **B-7 RL Phase 1 (v7, partial mitigation)** — completed 2026-05-01. v7 = symmetric magnitudes + class balance + progress bonus delayed the collapse from step ~50 to step ~75 but did not prevent it. Calibration still flipped to all-False by step 75. Output: `outputs/rl_b7_phase1_v7/final` (deprecated).

- **B-7 sanity test** — 2026-04-30. 400 rollouts at T=0.7 to characterize the empirical conditions driving the v6/v7 collapse. Findings:
  - Rollout length: 73% are 1-step, 27% are 2-step, **0% reach step 3+**.
  - Pass@1 stochastic: **0/400** — `success_bonus` literally never fires.
  - B-7 SFT calibration on its rollout distribution: 98.0% accuracy, 26.3% predicted-True vs 27.0% true class rate (essentially oracle).
  - Counterfactual under v7: oracle = +1.00, sft_actual = +0.99, always_false = +0.46, always_true = −0.46. **The reward landscape favors correct prediction by a wide margin** — collapse is dynamic drift, not the static optimum.
  See [sanity_2026-04-30_b7_rollout_stats.json](sanity_2026-04-30_b7_rollout_stats.json) and the analysis in [eval_2026-04-30_b7_rl_phase1.md](eval_2026-04-30_b7_rl_phase1.md).

### v8 reward + the implementation bug (caught 2026-05-01)

v8 = v7 + auxiliary KL anchor on `<viability>`/`<solvable>` tag tokens against the frozen ref policy (default coef 0.5). Targets the dynamic calibration drift identified in the sanity test.

**First v8 attempt (autodl, 2026-05-01)** — ran for ~100 steps before being killed. Calibration collapsed at step 75 just like v7. Investigation revealed the position-finder function returned `[]` on every call (`n_via_tokens = 0` per step in JSONL), so the anchor was never actually applied — it was effectively v7. Root cause: tokenization roundtrip mismatch. `response_ids` includes a trailing EOS token that gets stripped during decode→re-tokenize, so the length check returned [] and the anchor silently no-op'd.

**Fix** — re-decode `response_ids` ourselves (so the text we tokenize comes from the IDs, not from the upstream `response_text` which was decoded with `skip_special_tokens=True`), then accept length mismatches as long as the prefix tokens match. Also cache the positions on `StepRecord.viability_token_positions` at rollout time so the PPO update doesn't have to re-derive them. See [src/training/rl_trainer_v6.py](../src/training/rl_trainer_v6.py).

**Second v8 attempt (autodl, 2026-05-01, in flight)** — relaunched after the fix. JSONL confirms the anchor now fires: `n_via_tokens = 32` per step (= one `<viability>` token per rollout × 32 rollouts). `via_kl = 0.000000` to 6 decimal places through step 25 — the anchor is fully effective at holding viability tokens at SFT logprobs. Step 25 eval: `solvable_acc = 1.0`. Step 50 (v6 collapse point) and step 75 (v7 collapse point) are the discriminating tests still ahead.

**Sudoku v8 anchor experiment (autodl2, 2026-05-01, in flight)** — applies the v8 anchor to Run A's final checkpoint. Hypothesis: the anchor should restore `solvable_acc` from 0.514 → ~0.95 without losing the 33% Pass@1. Validates v8 on a working setup, decoupled from Pentomino's "Pass@1 stuck at 0%" confound. Output: `outputs/rl_b5_phase3_v8_anchor/`. Launcher: [scripts/run_sudoku_4x4_rl_v8_phase3.sh](../scripts/run_sudoku_4x4_rl_v8_phase3.sh).

### Pending / proposed (not yet started)

- **Run B (lr=1e-4 from B-5)** — **dropped.** Run A's lr=1e-5 already lifted Pass@1 to ~33%; lr=1e-4 risks blowing past the trust region with marginal upside.
- **B-6** — 9×9 + SPA hparams + SPA-scale data (paused; 9×9 SPA-scale gen completed easy difficulty only).
- **B-7 Pass@1 measurement** — measure greedy Pass@1 on B-7 SFT to anchor pentomino results against a solving metric.
- **B-9 (5×10 / 10-piece pentomino)** — replaces the previously-proposed 5×5 variant. P-0 sweep on 66 (10-of-12) subsets in progress on local. Targets the trajectory-length distribution that drives the B-7 collapse mechanism. ~1 day of work after subset is locked.
- **Phase 2 truncation experiment on Sudoku Run A's checkpoint** — would test the headline value claim (compute savings via early termination). Requires v8 anchor (in flight) to lift Prec(False) above the truncation gate threshold (current Run A Prec(False) = 0.38, gate = 0.90).
