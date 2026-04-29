# Future Steps — World Model Termination Prediction

Living to-do list of work that's queued for later. Distinct from [SPEC.md](SPEC.md) (the research-scope spec) and [pipeline_design.md](pipeline_design.md) (the operational runbook). When an item is done, move it to "Completed" or delete.

Each item has: priority, estimated effort, dependency on prior items, and expected output. Re-read this list at the start of each work session before deciding what to do next.

---

## Immediate next session

### IMM-1 — Eval the just-finished single-step SFT (Run B-1)
- **Priority:** must-do before anything else
- **Depends on:** Run B-1 SFT completing (~25 min on H800)
- **Effort:** ~15 min
- **What to do:**
  ```bash
  ssh autodl 'cd /root/autodl-tmp/world_model_termination_spa && \
    bash scripts/_run_with_env.sh python -u evaluate_rl.py \
      --env sudoku --metric termination --skip-rl \
      --sft-path outputs/sft_sudoku_minimal/final \
      --eval-from-parquet data/sudoku_llm_policy_minimal/wm_val_filtered.parquet \
      --n-per-class 100 \
      2>&1 | tee logs/eval_b1.log'
  ```
- **Decision points:**
  - If BP recall ≥ 40%: temporal-echo hypothesis confirmed, format pivot worked. Proceed to IMM-2.
  - If BP recall 20–40%: pivot helped but more is needed. Run IMM-3 next.
  - If BP recall < 20%: deeper issue (class imbalance dominant). Run IMM-3 immediately.

### IMM-2 — Tier B Pass@k eval on Run B-1
- **Priority:** if IMM-1 looks good
- **Depends on:** IMM-1 produced acceptable BP recall
- **Effort:** ~30 min on H800
- **What to do:**
  ```bash
  bash scripts/_run_with_env.sh python -u evaluate_rl.py \
    --env sudoku --metric pass-at-k --skip-rl --include-base \
    --sft-path outputs/sft_sudoku_minimal/final \
    --n-puzzles 20 --k 1,8
  ```
- **Output:** Pass@1, Pass@8 numbers comparable to SPA Table 2 (small N).
- **Save eval result to** `doc/eval_<date>_b1_passk.md`.

### IMM-3 — Run B-2: drop post-BP filler
- **Priority:** if IMM-1's BP recall is < 40%
- **Depends on:** IMM-1 done
- **Effort:** 10 min reformat + 10 min training
- **What to do:**
  1. Add a `--drop-post-bp` flag to `scripts/reformat_to_minimal.py` (or new script `scripts/filter_post_bp.py`) that filters out rows where `is_solvable=False AND is_breaking_point=False`.
  2. Apply to `data/sudoku_llm_policy_minimal/wm_train_filtered.parquet` → expected ~2,500 train samples (62% pre-BP solvable + 38% BP transitions).
  3. Re-train SFT with same hyperparameters → `outputs/sft_sudoku_minimal_no_post_bp/`.
  4. Re-run eval (same as IMM-1 but pointing at new checkpoint).
- **Decision points:** if BP recall lifts further, post-BP samples were a distractor. If not, class-weighted loss is next.

### IMM-4 — Run B-3: class-weighted SFT loss
- **Priority:** only if IMM-3 is needed AND insufficient
- **Depends on:** IMM-3 done
- **Effort:** ~half-day code change + 30 min training
- **What to do:**
  1. Modify `simple_sft_trainer.py` to accept a `--bp-class-weight N` flag.
  2. In the data collator (or a custom loss function), multiply cross-entropy on `<solvable>X</solvable>` token positions by N when `is_solvable=False` (the minority class).
  3. Tune N: try 5, 10. Re-train, re-eval each.
- **Output:** ablation table showing BP recall at N=1 (uniform), N=5, N=10.

---

## Near-term (next 1–2 weeks, after IMM is settled)

### NEAR-1 — SPA paper baselines (the load-bearing comparison gap)
- **Priority:** required for any publication-quality result per [SPEC.md](SPEC.md) §3
- **Effort:** ~3–6 GPU-hours per baseline
- **Three baselines to run on the same Sudoku setup:**
  1. **Vanilla RL** — base Qwen2.5-1.5B + PPO with task-success reward only. No SFT cold-start. (Code: existing `rl_trainer.py` with reward zeroed except success.)
  2. **State-Estimation-only RL** — SFT with `<observation>` only (no `<prediction>`, no `<solvable>`), then PPO. (Code: add a variant `sudoku_se_only` to `SFTFormatter`.)
  3. **VAGEN** — online world modeling reward during RL, no SFT cold-start. (Code: ~2–3 days to port from RAGEN repo, or cite SPA's published numbers if reproducible.)
- **Output:** baseline column for the eval table in `doc/eval_<date>_full_comparison.md`.

### NEAR-1.4 — RL truncation strategy: when does hard truncation actually help (and when does it hurt)?

There are **three distinct ways** the `<solvable>` tag can interact with RL, and they have different efficiency/quality trade-offs. Document for clarity before running NEAR-1.5.

**Mechanism 1: Reward shaping (always-on, default).** `<solvable>` predictions are scored against ground truth in the reward function (asymmetric: TP +3.0, FN −2.0, FP −0.5). The gradient pulls the model toward better predictions. Does NOT change rollout length — every step plays out. Cost: zero. This is reward v2 as currently designed.

**Mechanism 2: Hard rollout truncation.** Stop the rollout the moment the model emits `<solvable>=false` with high confidence. Save compute on the remaining steps. **Risky early in RL** because the model's predictions are noisy — false positives kill viable trajectories.

**Mechanism 3: Action filtering.** When the model proposes an action with `<solvable>=false`, reject and re-sample. Doesn't truncate the trajectory; spends compute on better action choices instead. Trades compute axis: more per-step cost, fewer wasted trajectories overall.

**The sequence that makes Mechanism 2 viable** — this is the operational protocol once the model is reasonable:

1. **Train SFT to a reasonable termination predictor** — get `<solvable>` accuracy at least above modal-class baseline. (We're here, modulo current failure modes.)
2. **Run RL with reward shaping (Mechanism 1) for many steps** — let the gradient improve `<solvable>` via reward feedback. **Do not truncate.** Predictions are noisy in this phase; trusting them corrupts the rollout data.
3. **Periodically measure precision-recall of `<solvable>` during RL training** — every N steps, run a fresh balanced eval and compute the precision-recall curve at multiple confidence thresholds. Save these as a time-series.
4. **Once precision is high enough at some threshold τ** (target: ≥95% precision for "predicted doomed"), turn on conservative hard-truncation: only truncate rollouts where `<solvable>=false` confidence > τ. Conservative threshold means few false positives, modest compute savings.
5. **Continue RL with truncation enabled** — compute savings compound (faster rollouts → more rollouts per GPU-hour → faster convergence). Model still improves; recheck precision at threshold τ regularly. Tighten τ if precision drifts down, loosen if drifts up.

**Why this sequence:**
- Skipping step 2 (using truncation from step 0) is the failure mode: noisy predictions cause bad terminations, RL gets stuck on a corrupted training signal.
- Skipping step 3 (no measurement) means we'd have no idea when truncation becomes safe.
- Step 4's conservative threshold is intentional: precision matters far more than recall for RL truncation, because the cost of a false positive (losing a viable trajectory) is much higher than the benefit of a true positive (saving wasted moves).

**Open implementation questions to resolve before NEAR-1.5:**
- How often to recompute precision-recall during RL? (every 100 RL steps? every epoch?)
- What confidence threshold τ to start with? (0.95? 0.99?)
- How to extract `<solvable>` confidence from a generation step? (token-level logprob at the `<solvable>X</solvable>` position)
- Action filtering (Mechanism 3) — do we test it as a separate ablation, or only after Mechanism 2 is validated?

### NEAR-1.5 — Compute-budget experiment: does early termination actually save GPU-hours in RL training?
- **Priority:** load-bearing for the project's headline value proposition (early termination → more compute-efficient RL)
- **Depends on:** working SFT checkpoint (IMM tier complete) AND working RL trainer (NEAR-2 in some form)
- **Effort:** ~1–2 days for the experimental harness + GPU time
- **What to do:**
  1. Run RL training **without** the `<solvable>` early-termination cutoff — let every rollout run to its natural end (success or max_steps). Measure: samples/GPU-hour and wall-clock to reach Pass@1 = X.
  2. Run RL training **with** the `<solvable>` early-termination cutoff — when the trained model's `<solvable>=false` exceeds confidence threshold τ, truncate the rollout. Measure: samples/GPU-hour and wall-clock to reach the same Pass@1 = X.
  3. Vary τ (0.5, 0.7, 0.9) and measure the savings vs over-termination penalty (early-truncated trajectories that were actually salvageable).
- **Expected output:** a plot showing compute-savings vs Pass@1 trade-off, ideally demonstrating measurable speedup (20%+) at no significant Pass@1 loss.
- **Why this matters:** without this experiment, "early termination saves compute" is a claim, not a finding. NeurIPS reviewers will ask for it.

### NEAR-2 — RL training (Stage 3) on the working SFT checkpoint
- **Priority:** the core thing that closes the project's pipeline
- **Depends on:** IMM-1 / IMM-3 / IMM-4 producing a working SFT model
- **Effort:** ~6–12 GPU-hours (depending on KL schedule + group size)
- **What to do:** existing `rl_trainer.py` with `LiveEnvTerminationRLTrainer` and reward v2 (action-conditional `<solvable>` rewards now, not BP).
- **Reward shape (v4-updated):**
  - `<solvable>=false` predicted, ground truth false: TP +3.0
  - `<solvable>=true` predicted, ground truth false: FN −2.0
  - `<solvable>=false` predicted, ground truth true: FP −0.5
  - `<solvable>=true` predicted, ground truth true: 0 (or small +0.5)
  - Format compliance: +0.1 per XML tag present
- **Anti-goal trip-wire:** if Pass@1 drops materially during RL, termination training is hurting agent performance — pause and investigate.

### NEAR-3 — Eval set fix: oversample BPs in fresh-generated eval
- **Priority:** any eval that uses `generate_balanced_eval_set` is currently producing eval sets with only ~5/200 BPs. Need to sample BPs explicitly.
- **Effort:** ~1 hour
- **What to do:** modify `generate_balanced_eval_set` in `evaluate_rl.py` to require N_bp BP samples, N_post_bp post-BP samples, N_solvable solvable samples — independently sourced. Currently it just collects "unsolvable" without distinguishing BP from post-BP.

### NEAR-4 — Update `pipeline_design.html` to v4 format
- **Priority:** low — the markdown version is current; HTML is a visual aid
- **Effort:** ~1 hour
- **What to do:** edit the HTML's panel data + flow nodes to reflect single-step samples and the dropped tags.

### NEAR-5 — Save activity logs to doc/
- **Priority:** low; quality-of-life
- **Effort:** scripted, ongoing
- Each completed eval/training run should land a dated writeup in `doc/eval_<date>_<topic>.md` per the saved `doc_folder_convention` memory.

---

## Medium-term (next 1–2 months — NeurIPS-track scope)

### MED-1 — Add Kakuro environment
- **Priority:** load-bearing for the cross-environment story per [SPEC.md](SPEC.md) §5
- **Effort:** ~3–5 days
- **What to do:**
  1. Add `src/environments/kakuro.py` matching `BaseTerminationEnv` interface.
  2. Verify it satisfies the §5 rubric (predictive gap, oracle, observability, etc.) — write a small probe report.
  3. Generate Track A and Track B data on the new env.
  4. Re-run the SFT + eval pipeline.
- **Output:** numbers comparable to Sudoku result. Demonstrates recipe transfers within constraint-satisfaction-puzzle class.

### MED-2 — Test temporal-echo failure mode generalization (the §2 Q6 question)
- **Priority:** if the temporal-echo finding is the headline contribution, this is the primary experiment
- **Depends on:** MED-1 (need a second environment)
- **Effort:** ~2–3 days of analysis + ablation
- **What to do:**
  1. Run multi-turn SFT on Kakuro with the v3-era format (with `<breaking_point>` etc.) deliberately, predict outcome.
  2. Measure if the same temporal-echo collapse occurs.
  3. Apply v4 single-step fix, measure improvement.
  4. Cross-env analysis: does the failure mode + fix work the same across envs?
- **Paper framing:** "Temporal-echo shortcut in multi-turn world-model SFT" — N environments, 1 fix, robust effect.

### MED-3 — Multi-model-size scaling
- **Priority:** strengthens any NeurIPS submission
- **Effort:** ~1 day per model size, ~3–5 sizes total
- **What to do:** repeat the headline result on Qwen2.5-0.5B, 1.5B (done), 3B, and possibly LLaMA3.2-1B. Same SFT recipe, same eval protocol.
- **Output:** scaling table — does the temporal-echo + fix work at all sizes?

### MED-4 — Theory write-up: TerMDP framing
- **Priority:** low (NeurIPS doesn't strictly require theory but it lifts borderline papers)
- **Effort:** 1–2 weeks
- **What to do:** show that our action-conditional `<solvable>` signal corresponds to a TerMDP in some formal sense. Borrow regret-bound machinery from Tennenholtz et al. 2022 if applicable.

---

## Long-term / blue-sky

### LT-1 — Polyomino tiling environment
- Adds topology-flavored predictive gap (unfillable holes). Different "shape" of failure than constraint propagation. Distinct enough to argue cross-domain.
- See [SPEC.md](SPEC.md) §5 Tier 2 candidates.

### LT-2 — Difficulty-curriculum SFT
- Currently Sudoku-easy only. Could vary 4×4 / 6×6 / 9×9 to argue the recipe scales with difficulty.

### LT-3 — Workshop / paper draft
- Earliest-viable outline per [report_2026-04-28_sft_b_diagnosis_and_pivot.md](report_2026-04-28_sft_b_diagnosis_and_pivot.md) §5 implications.
- If Path A (workshop): focus on Sudoku end-to-end with strong eval.
- If Path B (NeurIPS-main): center on temporal-echo failure mode + fix + cross-env validation.

### LT-4 — Open-source release
- Once the headline result lands, package the env + data-gen + training scripts as a clean repo for community use.

---

## Decision checkpoints

These are explicit fork-in-the-road moments. Re-evaluate the project's direction at each:

| Checkpoint | Triggered when | What to decide |
|---|---|---|
| **CP-1: SFT result quality** | After IMM-1 (and possibly IMM-3, IMM-4) | Is the SFT working well enough to proceed to RL? Or is the fundamental approach broken? |
| **CP-2: RL improves SFT?** | After NEAR-2 | Does RL lift BP recall above SFT? If not, reward shaping might be wrong (or the SFT is at the ceiling). |
| **CP-3: Single-env paper vs cross-env paper** | After NEAR-2 + good results | Submit Sudoku-only to a workshop, or invest 1–2 months in MED-1/2/3 for NeurIPS-main? |
| **CP-4: Temporal-echo paper vs termination-prediction paper** | After MED-2 | Frame the contribution around the failure mode (broader impact, riskier) or around termination prediction (narrower, safer). |

---

## Completed / superseded (reference)

- ✅ Track A random-play multi-turn data generation (`data/sudoku_multiturn/`)
- ✅ Track B LLM-policy multi-turn data generation (`data/sudoku_llm_policy/`)
- ✅ Length-filter script ([scripts/filter_long_samples.py](../scripts/filter_long_samples.py))
- ✅ Pass@k eval mode in `evaluate_rl.py`
- ✅ Multi-turn-from-parquet eval mode in `evaluate_rl.py`
- ✅ Reformat script: multi-turn → single-step minimal ([scripts/reformat_to_minimal.py](../scripts/reformat_to_minimal.py))
- ✅ SFTFormatter `sudoku_minimal` variant added
- ⚠️ Multi-turn SFT (Run B-0) — completed but failed eval (BP recall 5%); kept as cautionary example
- 🟡 Single-step SFT (Run B-1) — in progress at time of writing
