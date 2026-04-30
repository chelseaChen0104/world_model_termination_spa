# Project Handoff — World Model Termination SPA

> **For a new Claude session picking up this project.** This is the master pickup point. Read this first, then dive into the cross-referenced docs only when needed.

Last updated: 2026-04-29 (B-5 lands). As the project state changes, update §3 ("Where we are right now") and the run table in §6 — the rest is reasonably stable.

---

## 1. Quick orientation

**Mission (one paragraph):** Train a small LLM (Qwen2.5-1.5B-Instruct) to predict per-state solvability of partially-filled Sudoku puzzles, so an agent can **terminate hopeless episodes early** and save compute during agentic RL. The recipe builds on SPA (Chen et al. 2025) — self-play world-model SFT (state estimation + transition modeling) — and extends it with a `<solvable>` termination tag. The thesis: world-model grounding makes termination prediction reliable in environments with a *predictive gap* (states that look solvable but are already doomed). See [SPEC.md](SPEC.md) for the full research framing.

**Repo:** [github.com/chelseaChen0104/world_model_termination_spa](https://github.com/chelseaChen0104/world_model_termination_spa)

---

## 2. Three things any new session should know first

### Finding 1: Multi-turn world-model SFT has a temporal-echo failure mode (the headline candidate finding)
When training samples include prior assistant turns, 84.7% of samples in our setup have a *trivial echo shortcut* — the previous turn's `<solvable>` label is almost always the same as the current target. Cross-entropy heavily under-trains the 15.3% of samples where the answer flips at the BP transition. Result: BP recall = 5%. We pivoted to single-step SFT samples (one (s, a, s') triple per row) which removes the shortcut. Documented in [report_2026-04-28_sft_b_diagnosis_and_pivot.md](report_2026-04-28_sft_b_diagnosis_and_pivot.md).

### Finding 2: Even single-step SFT shows AUC ≈ 0.46 (no learned discrimination)
After fixing the format and re-training (Run B-2, B-3), threshold-based logprob eval revealed: P(<solvable>=true) for actually-solvable samples ≈ P(<solvable>=true) for actually-unsolvable samples (within noise). The model learned a class prior but not a discriminative signal. Class weighting (B-3 vs B-2) only flipped which class greedy collapsed to, not whether discrimination existed. See [eval_2026-04-28_sft_track_b_tier_a.md](eval_2026-04-28_sft_track_b_tier_a.md).

### Finding 3: 80× under-training is NOT the only issue — task difficulty matters too (updated 2026-04-29)
Located SPA's actual SFT config from their HuggingFace dataset and GitHub:

| | SPA | Us (Run B-3) | Run B-4 (test) | Run B-5 (4×4) |
|---|---|---|---|---|
| Task | 4×4 / 6 empty | 9×9 / ~32 empty | 9×9 / ~32 empty | 4×4 / 6 empty |
| Train samples | 6,060 | 2,482 | 2,482 | 6,571 |
| Epochs / LR / batch | 5 / 1e-4 / 16 | 3 / 1e-5 / 32 | 5 / 1e-4 / 16 | 5 / 1e-4 / 16 |
| Effective grad signal | ~0.189 | ~0.0023 | ~0.0775 | ~0.205 |
| **Final ROC AUC** | (paper not directly comparable) | **0.462** | **0.455** | **0.726** |

**B-4 disproved "scale alone fixes 9×9":** SPA hyperparameters with 30× more grad signal still produced AUC 0.455 (essentially random) on 9×9.

**B-5 confirmed the recipe works on 4×4:** SPA hyperparameters + SPA-scale data on the easier task produced AUC 0.726 — the first SFT run to show real discrimination. See [eval_2026-04-29_b5_4x4_spa_replication.md](eval_2026-04-29_b5_4x4_spa_replication.md).

**Conclusion:** the recipe is sound; **9×9 is genuinely too hard for Qwen-1.5B SFT at this scale**. To make 9×9 work we need either (a) a bigger model, (b) RL on top of SFT, or (c) much more data than 6k samples. The 9×9 SFT path is on hold pending RL.

---

## 3. Where we are right now (2026-04-29 evening)

| Cloud | Task | Status | ETA |
|---|---|---|---|
| **autodl1** (`ssh autodl`) | 9×9 SPA-scale data gen — 3 difficulties × 3,700 trajectories (11,100 total) | Easy phase, ~700/3,700 | ~9–10 hours total |
| **autodl2** (`ssh autodl2`) | Idle — last job: B-5 SFT (complete, AUC 0.726) | — | — |

**Recently completed (2026-04-29):**
- ✅ **Run B-4** — 9×9 + SPA hparams. Crashed on `/final/` save (disk full); recovered via `checkpoint-600`. AUC = 0.455 (no discrimination, essentially random).
- ✅ **4×4 SPA-scale data gen** — 5,000 trajectories split across both clouds → 6,571 single-step samples after no_post_bp filter (above SPA's 6,060 target).
- ✅ **Run B-5** — 4×4 + SPA hparams + SPA-scale data. **AUC = 0.726** — first SFT run with real discrimination. See [eval_2026-04-29_b5_4x4_spa_replication.md](eval_2026-04-29_b5_4x4_spa_replication.md).

**Decision points coming up:**
- After 9×9 SPA-scale gen finishes (~9h) → run B-6 (9×9 + SPA hparams + SPA-scale data, ~6,000 samples). If AUC still 0.46, 9×9 SFT is conclusively too hard at this capacity. If AUC 0.55+, scale matters even on 9×9.
- **Next strategic step:** RL on B-5 4×4 checkpoint to lift the uncalibrated P(true) ≈ 0.045 closer to a useful regime, and to test whether asymmetric rewards (TP+3, FN−2) further raise AUC. RL infra is not built yet (TRL+vLLM per SPEC v5).

---

## 4. Locked decisions (don't relitigate)

From [SPEC.md](SPEC.md) §7. Each was validated or chosen against alternatives:

| Decision | Why locked |
|---|---|
| **Sudoku is the sole primary environment** | Only SPA-paper env satisfying §1 predictive-gap criterion (state visually solvable but already doomed). Sokoban / FrozenLake fail this — explicitly out of scope. |
| **Single-step SFT samples** (one (s, a, s') per row) | Multi-turn version exhibited temporal-echo collapse at BP recall 5%. v4 amendment 2026-04-28. |
| **Minimal response tag set:** `<observation>` + `<prediction>` + `<solvable>` + `<answer>` | Dropped `<breaking_point>` (derivable post-hoc), `<terminate_prob>` (confused semantics), `<steps_left>` (SFT-only, redundant). v4. |
| **Action-conditional `<solvable>` semantics:** is_solvable(s_{t+1}) given the chosen action | Matches SPA's training shape; useful for "agent self-checks its own next move." |
| **LLM-policy data over random play** | In-distribution; SPA-aligned. Their RandSFT row gets 1/3 the Pass@1 of LLM-policy SFT. |
| **Single-GPU GRPO/PPO RL** | Per SPEC §4. No Ray, no distributed coordination. **vLLM as local accelerator was un-banned in v5** since it doesn't require a cluster. |
| **Asymmetric `<solvable>` rewards for RL** (TP +3.0, FN −2.0, FP −0.5) | Catching doom matters more than false alarms. Adapted from old BP rewards in v4. |
| **Mac is source of truth, GitHub is backup, clouds are ephemeral** | See [workflow.md](workflow.md). |

---

## 5. Open questions (where to push next)

From [SPEC.md](SPEC.md) §2 + [future_steps.md](future_steps.md):

| Q | Status | Next experiment |
|---|---|---|
| Q1 — Does SPA-style SFT improve termination prediction? | **Answered: yes, on 4×4** (AUC 0.726) — but **no on 9×9** at this scale (AUC 0.455). Closes the SFT-only question. | (closed) |
| Q2 — Does adding termination tags hurt Pass@1? | Pending | Pass@k mode in `evaluate_rl.py` (already built) |
| Q3 — Does asymmetric-reward RL lift AUC / calibration over SFT? | Pending — RL not yet run | Set up TRL+vLLM on top of B-5 4×4 checkpoint |
| Q4 — LLM-policy vs random-play data | Deferred (Track A SFT dropped to focus compute) | Indirect comparison via heuristic baseline |
| Q5 — Multi-turn vs single-turn helps BP detection? | **Answered: no** — multi-turn collapses to echo on Sudoku | (closed) |
| Q6 — Does temporal-echo failure mode generalize across multi-turn world-model SFT setups? | Untested | Reproduce on Kakuro / Nonogram (future) |
| **Q7 — Is SFT failure data-scale, task-difficulty, or capacity?** | **Mostly answered: task difficulty is the dominant factor.** B-4 (SPA scale on 9×9) failed, B-5 (SPA scale on 4×4) worked. 9×9 SPA-scale gen running to confirm. | Continue 9×9 gen → B-6 |
| **Q8 — Does our recipe replicate SPA's published 4×4 results?** | **Yes — AUC 0.726, real signal.** Calibration is poor (uncalibrated P(true) ≈ 0.045) but ranking is sound. | (closed for SFT; reopens for Pass@k vs SPA's headline numbers) |

---

## 6. Pipeline state (visual)

```
                    ┌──────────────────────────────┐
                    │   SudokuEnv + oracle         │  ✅ done
                    │   (constraint propagation)   │
                    └─────────────┬────────────────┘
                                  │
                    ┌─────────────┴────────────────┐
                    │  Stage 1 — Data generation   │
                    └─────────────┬────────────────┘
                                  │
        ┌─────────────────┬───────┴───────┬─────────────────┐
        ▼                 ▼               ▼                 ▼
   ┌─────────┐      ┌──────────┐    ┌─────────┐       ┌──────────┐
   │ Track A │      │ Track B  │    │ Diverse │       │ 4×4 SPA  │
   │ random  │      │ LLM-     │    │ multi-  │       │ replica  │
   │ play    │      │ policy   │    │ diff    │       │          │
   │ (CPU)   │      │ (H800)   │    │ (H800)  │       │ (H800)   │
   ✅ done           ✅ done       🟡 running         🟡 running
   data/sudoku-     data/sudoku- data/sudoku-     data/sudoku_4x4-
   _multiturn       _llm_policy  _llm_policy_     _llm_policy/
                    _minimal     {easy,medium,
                                  hard}/

                    ┌─────────────────────────────┐
                    │  Stage 2 — SFT (single-step,│
                    │     minimal XML target)     │
                    └─────────────┬───────────────┘
                                  │
   ┌──────┬──────┬──────┬──────┬──────┬──────┐
   │ B-0  │ B-1  │ B-2  │ B-3  │ B-4  │ B-5  │
   │ 9x9  │ 9x9  │ 9x9  │ 9x9  │ 9x9  │ 4x4  │
   │ multi│ minim│ minim│ no   │ SPA  │ SPA  │
   │ -turn│ form │ form │ post │ hpara│ hpara│
   │      │      │      │ -BP  │      │ +SPA │
   │      │      │      │      │      │ scale│
   │ AUC  │ AUC  │ AUC  │ AUC  │ AUC  │ AUC  │
   │ —    │ ~0.5 │ 0.468│ 0.462│ 0.455│ 0.726│
   └──────┴──────┴──────┴──────┴──────┴──────┘
        │                                │
        ▼                                ▼
   ┌──────────────────────────┐    ┌──────────────────────────┐
   │ B-6 (queued)             │    │  RL on B-5 (next)        │
   │ 9x9 + SPA hparams +      │    │  TRL+vLLM, GRPO/PPO,     │
   │ SPA-scale data           │    │  asymmetric <solvable>   │
   │ (after 9x9 gen finishes) │    │  rewards                 │
   └──────────────────────────┘    └──────┬───────────────────┘
                                          │
                                          ▼
                                   ┌──────────────────────────┐
                                   │  Stage 3 — RL training   │  🚧 not built
                                   │  TRL+vLLM (per v5)       │  Setup ~1 day, run ~6-12 hr
                                   └──────┬───────────────────┘
                                          │
                                          ▼
                                   ┌──────────────────────────┐
                                   │  Stage 4 — Evaluation    │  ✅ infra ready
                                   │  termination + Pass@k    │  baselines (vanilla RL,
                                   │                          │  SE-RL, VAGEN) 🚧 todo
                                   └──────────────────────────┘
```

For an interactive version: open [pipeline_design.html](pipeline_design.html).

---

## 7. Operational basics (the 5 things you need to know to operate)

### 7.1 Two cloud instances
- `autodl` (port 23540) — for big runs
- `autodl2` (port 18386) — for parallel quick experiments
- Both are H800, both have repo at `/root/autodl-tmp/world_model_termination_spa/`
- SSH config aliases already set up. Keys are `~/.ssh/id_ed25519_autodl` (no passphrase).

### 7.2 Sync commands
```bash
bash scripts/sync-up.sh                      # push code to both clouds
bash scripts/sync-up.sh --target autodl2     # push to one
bash scripts/sync-down.sh                    # pull data/logs/outputs from both
bash scripts/sync-down.sh --with-models      # also pull model weights
```
Slash commands `/sync-up` and `/sync-down` work the same way.

### 7.3 Each cloud has its own data dirs (disjoint paths)
| autodl1 produces | autodl2 produces |
|---|---|
| `data/sudoku_llm_policy_easy/` (and `medium`, `hard`) | `data/sudoku_4x4_llm_policy/` |
| `outputs/sft_sudoku_minimal*` | `outputs/sft_sudoku_4x4_minimal_no_post_bp/` |
No collision when sync-down merges to local.

### 7.4 GitHub
- Origin: `chelseaChen0104/world_model_termination_spa`
- Latest commit: `dd2ae10` (2026-04-29) — the "pivot to single-step SFT + multi-cloud workflow" checkpoint
- Push commits periodically as code-only backup. Data/outputs/logs gitignored (anchored `/data/` etc).

### 7.5 Cloud essentials
- Python at `/root/miniconda3/bin/python`
- `bash scripts/_run_with_env.sh <cmd>` wraps PATH + network_turbo proxy + `HF_HUB_DISABLE_XET=1`
- tmux for long runs (`tmux new-session -d -s name "..."`)
- Cache redirects to `/root/autodl-tmp/cache/*` (system disk is small)
- `data/sudoku_llm_policy*` parquets are checked in as references (small) but generated data isn't

---

## 8. Where to find things

### Reference docs (living, current)
| Doc | What it is |
|---|---|
| [HANDOFF.md](HANDOFF.md) | **You are here.** Master pickup. |
| [SPEC.md](SPEC.md) | Research scope, locked vs open decisions, success criteria. v5 current. |
| [pipeline_design.md](pipeline_design.md) | How to operate the pipeline (data → SFT → RL → eval) end-to-end. |
| [pipeline_design.html](pipeline_design.html) | Interactive flow chart with click-through node details. |
| [workflow.md](workflow.md) | Sync architecture (Mac ↔ cloud ↔ GitHub). |
| [future_steps.md](future_steps.md) | Prioritized to-do list with decision checkpoints. |
| [CLAUDE.md](../CLAUDE.md) | Architecture + code-level details (note: pre-v4 era, banner at top points here). |
| [progress.md](../progress.md) | Chronological build log of every session. |

### Dated artifacts (eval write-ups, reports)
| Doc | What it is |
|---|---|
| [eval_2026-04-28_data_strategy.md](eval_2026-04-28_data_strategy.md) | Pre-training data inspection — predicted the BP-step skew that contributed to v3's failure |
| [eval_2026-04-28_sft_track_b_tier_a.md](eval_2026-04-28_sft_track_b_tier_a.md) | Single-turn Tier A eval result on Run B-0 (the multi-turn-collapse) |
| [report_2026-04-28_sft_b_diagnosis_and_pivot.md](report_2026-04-28_sft_b_diagnosis_and_pivot.md) | The temporal-echo finding + format pivot rationale |
| [eval_2026-04-29_b5_4x4_spa_replication.md](eval_2026-04-29_b5_4x4_spa_replication.md) ([html](eval_2026-04-29_b5_4x4_spa_replication.html)) | **B-5 results — first SFT run with real discrimination (AUC 0.726).** Closes Q1/Q7/Q8 for SFT. Includes loss plot + SPA-paper comparison. |
| [qa_2026-04-29_tag_design.md](qa_2026-04-29_tag_design.md) | Why `<breaking_point>`, `<terminate_prob>`, `<steps_left>` were dropped from the SFT format. |
| [runs_ledger_2026-04-29.md](runs_ledger_2026-04-29.md) | **Consolidated record of every SFT run (B-0 through B-5 + 4x4 baseline)** — hparams, data paths, sample counts, output dirs, log paths, AUC/F1/recall, and a paragraph note per run. |

### Reference papers (PDFs)
- `INTERNALIZING WORLD MODELS VIA SELF-PLAY FINETUNING FOR AGENTIC RL.pdf` — the SPA paper (Chen et al. 2025)
- `Reinforcement Learning with a Terminator.pdf` — TerMDP paper (Tennenholtz et al. 2022) — theoretical scaffolding for termination as a learnable signal

### Key code paths
| Path | What |
|---|---|
| `src/environments/sudoku.py` + `sudoku_utils.py` | Env + solvability oracle (constraint propagation + bounded backtracking) |
| `src/data/llm_trajectory_generator.py` | LLM-policy trajectory gen with minimal data-gen prompt |
| `src/data/sft_formatter.py` | All SFT format variants including `sudoku_minimal` |
| `src/training/simple_sft_trainer.py` | HF Trainer-based SFT, single-GPU |
| `src/training/rl_trainer.py` | RL trainer (currently v3-era, depends on verl — needs replacement with TRL) |
| `evaluate_rl.py` | Termination eval, Pass@k mode, threshold-based logprob extraction |

### Key scripts
| Script | Purpose |
|---|---|
| `scripts/sync-up.sh` / `sync-down.sh` | Multi-target Mac ↔ cloud sync |
| `scripts/_run_with_env.sh` | AutoDL env wrapper (PATH + proxy + HF_HUB_DISABLE_XET) |
| `scripts/regen_random_multiturn.py` | Track A regeneration |
| `scripts/generate_llm_policy_data_gpu.sh` | Track B data gen |
| `scripts/generate_diverse_data.sh` | 3-difficulty data gen (autodl1 currently running this) |
| `scripts/combine_diverse_to_minimal.sh` | Combine 3-difficulty parquets + reformat |
| `scripts/run_4x4_pipeline.sh` | End-to-end 4×4 SPA replication (autodl2 currently running this) |
| `scripts/run_b4_spa_hparams.sh` | Run B-4 — same data as B-3 with SPA hyperparameters (queued) |
| `scripts/reformat_to_minimal.py` | Multi-turn parquet → single-step minimal |
| `scripts/filter_long_samples.py` | Drop samples > token budget (legacy from multi-turn era) |
| `scripts/filter_post_bp.py` | Drop (Solvable=False, BP=False) post-BP filler |
| `scripts/oversample_bp.py` | 2× BP samples for class weighting (used in B-3) |

---

## 9. Cheat sheet for first 5 minutes of a new session

```bash
# Where are we?
git log --oneline -5
cat doc/HANDOFF.md                # this file — single entry point
cat doc/future_steps.md            # what to do next
ssh autodl 'tmux ls'               # what's running on cloud 1
ssh autodl2 'tmux ls'              # what's running on cloud 2

# Routine sync
bash scripts/sync-up.sh            # after editing code
bash scripts/sync-down.sh          # after a cloud run completes

# Eval the current best SFT
ssh autodl 'cd /root/autodl-tmp/world_model_termination_spa && \
  bash scripts/_run_with_env.sh python -u evaluate_rl.py \
    --env sudoku --metric solvable-logprob --skip-rl \
    --sft-path outputs/sft_sudoku_minimal_no_post_bp/final \
    --eval-from-parquet data/sudoku_llm_policy_minimal/wm_val_filtered.parquet \
    --n-per-class 100'
```

---

## 10. Common confusions and corrections

These came up during prior sessions and are worth knowing:

- **"(s, a, s') triple" doesn't fully specify the format.** Both us and SPA train on (s, a, s'), but SPA's "a" is a *bundle* of moves separated by `||`, ours is a *single* move. Different per-response semantics.

- **"Greedy hides discrimination" was wrong on closer inspection.** Initial temperature-probe interpretation suggested the model had buried discrimination. Threshold-based logprob eval revealed AUC ≈ 0.46 — actually no discrimination at all. The temperature-probe accuracy was sampling noise, not signal.

- **vLLM is not Ray.** Initially banned together; v5 amendment unlocked vLLM as a single-process accelerator. Ray-based distributed coordination remains banned.

- **`.gitignore` `data/` (without leading slash) excluded `src/data/` source code** in the initial commit. Fixed in `dd2ae10`. Anyone cloning before that commit got broken code.

- **Autodl `network_turbo` proxy + new HF xethub backend → 401 errors** on model downloads. Fix: `export HF_HUB_DISABLE_XET=1` in `scripts/_run_with_env.sh`.

- **Track A (random play) was abandoned, not validated.** SPA's RandSFT ablation row shows it's substantially worse than LLM-policy. We dropped Track A SFT to focus compute on the LLM-policy path.

- **BP detection accuracy on multi-turn eval can look high (~95%) without actual learning.** It's just temporal echo from prior `<solvable>=false` assertions. The real test is BP recall + threshold-based logprob.

---

## 11. The minimum context if you only read one paragraph of any doc

> Project trains Qwen-1.5B to predict whether a Sudoku state is doomed (action-conditional `<solvable>` tag), enabling early termination of hopeless RL rollouts. We extend SPA's recipe (Chen et al. 2025) — world-model SFT on (s, a, s') triples — with this termination tag. Earlier multi-turn SFT failed (temporal echo). Pivoted to single-step SFT samples + minimal tags + action-conditional semantics. As of 2026-04-29: **B-5 confirms the recipe works on 4×4 (AUC 0.726)** but **9×9 fails at this scale even with SPA's exact hyperparameters (B-4 AUC 0.455)** — task is too hard for Qwen-1.5B SFT alone. Currently generating 9×9 SPA-scale data (~9h on autodl1) for one final 9×9 SFT test (B-6); after that, the strategic next step is RL on the B-5 4×4 checkpoint.
