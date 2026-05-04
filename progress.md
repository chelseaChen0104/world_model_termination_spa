# Steps Performed

## Session 1: Sokoban Foundation + Initial Pipeline

### Step 1: Sokoban Environment with Random Puzzle Generation
- Created `src/environments/base.py` — `BaseTerminationEnv` abstract class defining the interface: `reset()`, `step()`, `check_solvability()`, `get_all_actions()`, `get_state_info()`
- Rewrote `src/environments/sokoban.py` — `SokobanEnv` with random puzzle generation (reverse-playing algorithm) and `SokobanDeadlockDetector`

### Step 2: Sokoban SFT Data Generation (Random Play)
- Used `src/data/trajectory_generator.py` — `TrajectoryGenerator` generates trajectories with random actions
- Used `src/data/sft_formatter.py` — `SFTFormatter(variant='full')` formats each step as chat messages with XML response tags
- Generated `data/termination_study_v2/`:
  - `wm_train.parquet` — 61,096 samples
  - `wm_val.parquet` — 15,275 samples
  - Known issue: 99.2% solvable per-step (extreme class imbalance)

### Step 3: Sokoban SFT Training
- Ran `src/training/simple_sft_trainer.py` (HuggingFace Trainer, single GPU)
- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Hyperparams: 3 epochs, batch size 4, gradient accumulation 8, lr 1e-5
- Output: `outputs/sft_termination/checkpoint-4500/` (~3GB)
- Training done on AutoDL cloud GPU

### Step 4: Initial RL Training Attempts
- Attempted RL training with `SimpleTerminationRLTrainer` on static parquet dataset
- Discovered reward hacking: model exploited 99.2% solvable class imbalance by always predicting "solvable=true"
- Diagnosed root cause: static dataset RL cannot fix class imbalance at data level

### Step 5: Evaluation and Analysis
- Evaluated SFT and RL models — confirmed reward hacking behavior
- Analyzed original SPA paper to understand their approach (live env, balanced sampling)
- Decision: need balanced sampling from live environment

---

## Session 2: Balanced RL + Sudoku Pivot

### Step 6: Reward Function v2
- Added `compute_termination_reward_v2()` to `src/training/rl_trainer.py`
- Asymmetric BP rewards: TP +3.0, FN -2.0, FP -0.5 (catching deadlocks matters most)
- Solvable prediction: correct +0.5, wrong -0.25
- Format compliance: +0.1 per XML tag present (prevents format forgetting during RL)
- Removed steps_left from RL reward (trajectory-dependent, SFT only)

### Step 7: Live Environment RL Trainer
- Created `src/data/live_trajectory_sampler.py` — `LiveTrajectorySampler` generates balanced batches (~50/50 solvable/unsolvable) from live env
- Created `LiveEnvTerminationRLTrainer` class in `src/training/rl_trainer.py`
- Created `main_live_env()` entry point supporting both sokoban and sudoku envs
- Updated `src/training/config/rl_termination.yaml` with reward v2 weights, environment config, sampling config

### Step 8: Evaluation Rewrite
- Rewrote `evaluate_rl.py` with balanced evaluation from live env
- Added confusion matrices for both solvable and BP predictions
- Added F1 scores, precision, recall
- Added per-deadlock-type recall (zero_candidates, constraint_propagation, no_solution)
- CLI: `python evaluate_rl.py --n-solvable 100 --n-unsolvable 100 [--sft-path PATH] [--rl-path PATH]`

### Step 9: Sudoku Environment
- Created `src/environments/sudoku.py` — `SudokuEnv(BaseTerminationEnv)` with solvability checking
  - Action format: `"place N at row R col C"`
  - `step()` checks placement validity, detects breaking points via solvability transition
  - `get_all_actions()` returns valid placements for all empty cells
  - `check_solvability()` delegates to `SudokuSolvabilityChecker` with caching
- Created `src/environments/sudoku_utils.py` — solvability checker
  - `SudokuSolvabilityChecker` with constraint propagation + bounded backtracking (~2ms/check)
  - `_propagate()`: naked singles + hidden singles
  - `_backtrack()`: MRV heuristic, depth-limited
  - Also: `generate_sudoku_puzzle()`, `is_valid_placement()`, `get_valid_numbers()`, `find_conflicts()`, `format_grid()`
- Updated `src/environments/__init__.py` to export `SudokuEnv`, `SudokuSolvabilityChecker`

### Step 10: Pipeline Updates for Sudoku Support
- Updated `src/data/trajectory_generator.py` — generalized action handling for string actions (Sudoku) vs int actions (Sokoban); added `break` when no valid actions remain
- Updated `src/data/sft_formatter.py` — added `sudoku_full` variant with Sudoku-specific system prompt; skips `format_state_with_coordinates()` for Sudoku
- Updated `scripts/generate_spa_data.py` — added `--grid-size`, `--difficulty` args; added `sudoku_full` variant and `SudokuEnv` branch
- Created `src/training/config/rl_sudoku.yaml` — Sudoku-specific RL config (grid_size 9, difficulty easy, max_steps 30)
- Created `src/evaluation/sudoku_baseline.py` — `SudokuHeuristicBaseline` with simple (candidate counting) and full (constraint propagation) modes

### Step 11: Sudoku SFT Data Generation (Random Play)
- Generated `data/sudoku_termination/` using random actions:
  - `wm_train.parquet` — 25,649 samples
  - `wm_val.parquet` — 6,413 samples
  - Total: 32,062 samples from 1,280 trajectories
  - Class distribution: 6.6% solvable, 93.4% unsolvable, 4.0% breaking points
  - Note: this data uses random play and needs to be regenerated with LLM-policy

### Step 12: Documentation
- Created `CLAUDE.md` with full project documentation:
  - Architecture overview, data generation pipeline, training commands, evaluation guide
  - Key insight: why Sudoku over Sokoban (no simple deadlock detector)
  - Random play vs LLM-policy comparison with example input/output pairs
  - Impact analysis on SFT and RL training

### Step 13: LLM-Policy Trajectory Generator (Implementation)
- Created `src/data/llm_trajectory_generator.py`:
  - `LLMTrajectoryConfig` dataclass (model path, temperature, device, etc.)
  - `LLMTrajectoryGenerator` class — uses base LLM (Qwen2.5-1.5B-Instruct) to play Sudoku
  - `_generate_response()` — applies chat template, runs model.generate()
  - `_parse_action_from_response()` — extracts action from `<answer>` tags with fallback pattern matching
  - `generate_trajectory()` — full rollout with ground-truth annotations from env solvability checker
  - `generate_balanced_dataset()` — balanced dataset generation (success/failure/timeout ratios)
  - `main()` CLI entry point with argparse
  - Falls back to random action if LLM output is unparseable
  - Uses CPU for generation (MPS has cumsum int64 bug with transformers)
- Upgraded `transformers` (4.30.2 → 4.46.3) and `jinja2` (2.x → 3.1.6) to support Qwen2 tokenizer
- Verified imports and action parsing work correctly
- End-to-end model inference test pending (CPU generation is slow on Mac)

### Step 14: Multi-Turn Conversation Support
- Added `llm_raw_response: Optional[str] = None` field to `TrajectoryStep` in `src/data/trajectory_generator.py`
  - Stores the LLM's verbatim response for use as prior-turn context in multi-turn SFT data
  - Backward compatible: defaults to None, existing call sites unchanged
- Updated `src/data/llm_trajectory_generator.py`:
  - Stores raw LLM response in `TrajectoryStep.llm_raw_response`
  - Added `--multi-turn` and `--max-context-turns` CLI args to `main()`
- Refactored `src/data/sft_formatter.py`:
  - `format_trajectory()` now accepts `multi_turn=False` and `max_context_turns=None` params
  - Extracted `_format_single_turn()` (original behavior, unchanged)
  - Added `_format_multi_turn()` — builds growing conversation history per step
    - Step K prompt: [system, user_state_0, assistant_resp_0, ..., user_state_K]
    - Prior assistant content: `llm_raw_response` if available, else template-generated ground truth
    - Sliding window via `max_context_turns` param
  - `create_sft_dataset()` forwards multi-turn params
- Updated `src/training/simple_sft_trainer.py`:
  - Added `elif role == 'assistant'` branch in prompt-building loop
  - Prior assistant turns in prompt are automatically masked with label=-100
  - Only the final response (last assistant turn) has real labels
  - Added `hasattr(prompt_messages, 'tolist')` handling for numpy arrays from parquet
- Tests passed:
  - Single-turn format: 3 steps → 3 samples with 2 messages each ✅
  - Multi-turn format: 3 steps → samples with 2, 4, 6 messages ✅
  - Sliding window (max_context_turns=1): step 2 limited to 4 messages ✅
  - Parquet round-trip: variable-length message lists serialize/deserialize correctly ✅
  - Backward compatibility: old single-turn parquet files work with updated trainer ✅
  - Loss masking: prior assistant turns in prompt, only final response trained on ✅

---

## Session 3: Multi-Turn Data Generation + GPU Script

### Step 15: Multi-Turn SFT Data Generation (Random Play, Intermediate)
- MPS inference blocked by `cumsum int64` bug; CPU too slow (~2+ min for 2 steps, would take days for 1280 trajectories)
- Generated multi-turn SFT data using **random play** as practical intermediate step:
  - Random play trajectories still benefit from multi-turn conversation format
  - Prior assistant turns use template-generated ground truth (correct labels)
  - `max_context_turns=10` sliding window caps sequence length
- Output: `data/sudoku_multiturn/`
  - `wm_train.parquet` — 25,649 samples
  - `wm_val.parquet` — 6,413 samples
  - Total: 32,062 samples from 1,280 trajectories
  - Class distribution: 6.6% solvable, 93.4% unsolvable, 4.0% breaking points
  - Context turns: min=0, max=10, mean=7.8 (60% of samples have full 10-turn context)
  - Prompt sizes: min 2 messages (step 0), max 22 messages (step 10+), mean 17.6
  - Max token estimate: ~3,544 tokens (fits in 4096 max_length)
- Verified:
  - Step-0 samples have 2 messages (system + user) ✅
  - Step-5 samples have 12 messages (system + 5×user/assistant + user) ✅
  - All prior assistant turns contain proper XML tags ✅
  - Trainer builds correct `<|im_start|>` formatted text with all three roles ✅

### Step 16: GPU Data Generation Script
- Created `scripts/generate_llm_policy_data_gpu.sh` — ready-to-run script for AutoDL
  - Uses `--multi-turn --max-context-turns 10` for multi-turn format
  - 1280 trajectories × 30 max steps with LLM-policy (Qwen2.5-1.5B-Instruct)
  - Output: `data/sudoku_llm_policy/`
  - Includes post-generation instructions for SFT training

---

## Session 4: AutoDL Cloud Setup + Track A/B Data Generation

### Step 17: AutoDL deploy + caches + dependencies
- Pushed dedicated SSH key, set up `~/.ssh/config` alias `autodl`
- Installed missing deps: `peft`, `scikit-learn` on top of the existing torch/transformers/pandas/pyarrow/accelerate/datasets stack
- Redirected HF/torch/pip caches to `/root/autodl-tmp/cache/*` to avoid filling system disk
- Set `HF_HUB_DISABLE_XET=1` after the new HF xethub backend hit 401 through AutoDL's network_turbo proxy
- Added `scripts/_run_with_env.sh` wrapper (PATH + network_turbo + HF_HUB_DISABLE_XET)

### Step 18: Track A — random-play multi-turn (CPU)
- Re-ran with `python -u` for unbuffered logs
- Generated 32,062 SFT samples (1,280 trajectories × ~25 samples each) → `data/sudoku_multiturn/`
- Class distribution: 6.6% solvable, 93.4% unsolvable, 4.0% BP — heavily skewed
- BP-step distribution: clustered at steps 0–5 (random play burns out fast)

### Step 19: Track B — LLM-policy multi-turn (H800)
- First attempt with FULL system prompt ran ~1h with no visible progress (Python stdout buffering + low BP rate)
- Diagnosed and added **minimal data-gen prompt** (`DATA_GEN_SYSTEM_PROMPT_SUDOKU`): asks only for `<answer>` tag → ~5–10x speedup, parse rate >90% (vs ~30–50% with full prompt)
- `LLMTrajectoryGenerator.config.use_minimal_data_gen_prompt=True` (default); when True, `step.llm_raw_response` is set to None so multi-turn priors fall back to template-generated XML
- Added per-trajectory progress logging to `generate_balanced_dataset()` with `flush=True`
- Final trackB: 1,280 trajectories in ~75 min → 19,838 samples → `data/sudoku_llm_policy/`
- Class distribution: 10.4% solvable, 6.5% BP, ~83% post-BP filler — better than Track A but still skewed

---

## Session 5: Multi-turn SFT and the Temporal-Echo Discovery

### Step 20: First SFT (Run B-0 — multi-turn, doomed by length)
- Built `scripts/filter_long_samples.py` after first SFT crashed with NaN eval_loss — root cause: 54% of samples exceeded `max_length=4096`, leaving labels all -100 (NaN cross-entropy)
- Filtered training set: 6,221 train + 1,608 val (was 15,870 + 3,968)
- Class balance after filtering improved: 24.6% solvable, 15.3% BP, 60.1% post-BP
- Trained Qwen2.5-1.5B-Instruct, 3 epochs, lr=1e-5, batch=32 effective
- Final eval_loss: 0.0136, train converged

### Step 21: Eval revealed temporal-echo failure mode
- Single-turn eval: model collapsed to **constant `<solvable>true</solvable>`** (recall 100%, specificity 0%)
- Multi-turn eval (matching training distribution): better, but **BP recall = 5%** — model echoed prior turns instead of detecting transitions
- Diagnosis: 84.7% of multi-turn samples have a trivial echo shortcut (pre-BP & post-BP-filler labels match the previous turn). Cross-entropy under-trained the 15.3% BP transitions.
- Documented in `doc/eval_2026-04-28_sft_track_b_tier_a.md` and `doc/report_2026-04-28_sft_b_diagnosis_and_pivot.md`

### Step 22: Format pivot — single-step samples + minimal tags (SPEC v4)
- Modified `SFTFormatter` to add `sudoku_minimal` variant — drops `<terminate_prob>`, `<steps_left>`, `<breaking_point>` tags
- Reformatted Track B parquets via `scripts/reformat_to_minimal.py` → `data/sudoku_llm_policy_minimal/`
- New target: `<observation>` + `<prediction>` + `<solvable>` + `<answer>` only
- `<solvable>` semantics: action-conditional — `is_solvable(s_{t+1})` given the model's chosen action
- `<breaking_point>` derivable post-hoc from `<solvable>` time-series

---

## Session 6: B-1, B-2, B-3 — class-balance probing

### Step 23: Run B-1 (single-step, original 25/75 class balance)
- Trained on full filtered minimal data (6,221 samples), 3 epochs, lr=1e-5
- Greedy eval: model collapsed to **constant `<solvable>=false`** (predicts false 99.7%) — opposite of B-0
- Confirmed: format pivot killed the temporal-echo shortcut

### Step 24: Run B-2 (single-step, post-BP filler kept)
- Same data + hyperparameters as B-1 — actually B-1 retrained on minimal-format, results similar
- Eval results: same constant-class collapse pattern under greedy

### Step 25: Run B-3 (single-step, post-BP filler dropped)
- Built `scripts/filter_post_bp.py` to remove (Solvable=False, BP=False) samples
- Result: 2,482 train (was 6,221), class balance 62/38 → modal class flipped from "false" to "true"
- Eval (greedy): predicts True 96% — collapse just shifted direction; no actual discrimination
- Temperature 0.7 sampling probe: revealed 64/36 split, 49% accuracy → looked like buried discrimination

### Step 26: Threshold-based logprob eval — false-alarm on temperature probe
- Built `evaluate_solvable_logprob()` in `evaluate_rl.py`: teacher-forced single forward pass at the `<solvable>` token, extract P(true)/P(false) directly
- B-2 result: P(true) for solvable samples = 0.105; for unsolvable = 0.107. **Separation = -0.002 (essentially zero)**
- B-3 result: 0.055 vs 0.057 — same essentially-zero separation
- **ROC AUC ≈ 0.46 for both — no learned discrimination at the `<solvable>` token**
- Concluded: temperature probe's apparent 49% accuracy was sampling-induced noise upstream of `<solvable>`, not real discrimination
- The class-balance changes only flipped which side greedy collapsed to; never added discriminative signal

---

## Session 7: SPEC v5, multi-cloud, B-4 queued

### Step 27: SPEC v5 amendment — un-banned vLLM
- v3 had grouped vLLM with Ray/RAGEN as out-of-scope. vLLM is actually a single-process inference accelerator (no Ray needed).
- Permitted vLLM as a local accelerator for RL rollouts and Pass@k eval. Ray-based distributed coordination remains out of scope.

### Step 28: Diverse data generation (autodl1, in progress)
- `scripts/generate_diverse_data.sh`: 3 sequential runs (easy/medium/hard, 1,500 trajectories each) → `data/sudoku_llm_policy_{easy,medium,hard}/`
- `scripts/combine_diverse_to_minimal.sh`: reformat + concat into `data/sudoku_llm_policy_diverse_minimal/`
- Easy phase complete (1,500 trajectories saved), medium running, hard queued

### Step 29: Second AutoDL instance + 4×4 SPA replication
- Added `autodl2` SSH alias for second H800 instance
- Updated `scripts/sync-up.sh` and `sync-down.sh` to support `--target {autodl|autodl2|all}` for multi-cloud workflow
- `scripts/run_4x4_pipeline.sh` matches SPA paper's exact 4×4 setup (6 empty cells)
- Surprise finding: Qwen-1.5B SOLVES 4×4 puzzles 76% of the time (vs ~0% on 9×9) — meaningful policy signal, but BP collection rate only 24%

### Step 30: SPA hyperparameter discovery via SPA's HuggingFace dataset
- Located `tyzhu/SPA-sudoku-data` (7,535 samples, 6,060 train / 1,475 val) — same scale as ours
- Located SPA's `sft/finetune_ft.sh` config:
  - **5 epochs (vs our 3)**
  - **lr=1e-4 (vs our 1e-5 — 10× larger)**
  - **batch_size=16 (vs our 32 — half)**
- Total effective gradient signal: SPA ~80× more than our B-3
- Concluded: our SFT may simply be under-trained, not architecturally broken
- Queued **Run B-4** (`scripts/run_sudoku_9x9_sft_b4.sh`) with SPA's hyperparameters on B-3's data, to launch when GPU frees up

### Step 31: GitHub state restored
- Discovered `.gitignore` had unanchored `data/` pattern — was excluding `src/data/` source code
- Fixed to `/data/`, `/outputs/`, `/logs/` (anchored to top-level only)
- Pushed checkpoint commit `dd2ae10` to GitHub with all post-initial work

---

## Current State

### What Exists
| Component | File | Status |
|-----------|------|--------|
| Sokoban env | `src/environments/sokoban.py` | Done (on hold) |
| Sudoku env | `src/environments/sudoku.py` | Done |
| Solvability checker | `src/environments/sudoku_utils.py` | Done |
| Random trajectory gen | `src/data/trajectory_generator.py` | Done (+ `llm_raw_response` field) |
| LLM-policy trajectory gen | `src/data/llm_trajectory_generator.py` | Done (not yet tested end-to-end on GPU) |
| SFT formatter | `src/data/sft_formatter.py` | Done (single-turn + multi-turn support) |
| Live trajectory sampler | `src/data/live_trajectory_sampler.py` | Done |
| SFT trainer | `src/training/simple_sft_trainer.py` | Done (multi-turn assistant role support) |
| RL trainer (live env) | `src/training/rl_trainer.py` | Done |
| Evaluation | `evaluate_rl.py` | Done |
| Heuristic baseline | `src/evaluation/sudoku_baseline.py` | Done |
| Sokoban SFT data | `data/termination_study_v2/` | Done (on hold) |
| Sudoku SFT data (random, single-turn) | `data/sudoku_termination/` | Done (superseded by multi-turn) |
| Sudoku SFT data (random, multi-turn) | `data/sudoku_multiturn/` | Done (intermediate; LLM-policy preferred) |
| Sokoban SFT model | `outputs/sft_termination/checkpoint-4500/` | Done (on hold) |
| Sudoku SFT data (LLM-policy) | `data/sudoku_llm_policy/` | Not yet generated (GPU script ready) |
| GPU generation script | `scripts/generate_llm_policy_data_gpu.sh` | Done |
| Sudoku SFT model | `outputs/sft_sudoku/` | Not yet trained |
| Sudoku RL model | — | Not yet trained |

### Session 8 (2026-04-29): B-4 disproved, B-5 confirmed recipe on 4×4

- **B-4 ran with SPA hyperparameters on 9×9 B-3 data** (lr=1e-4, ep=5, bs=16). Crashed during `/final/` save (autodl1 disk full at 100%). Recovered the `checkpoint-600` (step 600 of 775 — `model.safetensors` was saved cleanly even though `optimizer.pt` got truncated). Eval: **ROC AUC = 0.455** — model collapsed to "always False", essentially chance. **Disproved the 80× under-training hypothesis** as a sufficient explanation for 9×9 failure.
- **Verified Sudoku env labels are clean.** A label-quality investigation found:
  - The depth-30 backtracking limit in `SudokuSolvabilityChecker` does NOT fire on the dataset (0/1000 samples flipped True→False with depth=200).
  - `is_solvable` labels match the checker run on `s_{t+1}` (after applying the action) on 1000/1000 sampled rows. Labels are correct.
  - Initial agent finding of "16.7% mismatch" was a parser bug (it had checked `s_t` instead of `s_{t+1}`).
- **4×4 SPA-scale data generation, parallel split across two clouds:** autodl1 part-A (2,500 trajs, seed=42) + autodl2 part-B (2,500 trajs, seed=43). ~3.7h each, finished simultaneously. Combined via [scripts/combine_4x4_spa_scale_parts.py](scripts/combine_4x4_spa_scale_parts.py): **6,571 single-step samples after no_post_bp filter** (above SPA's 6,060 target). Class balance: 40% solvable / 60% breaking-point.
- **Run B-5: 4×4 + SPA hparams + SPA-scale data.** Trained on autodl2 — 50 minutes for 2,050 train steps (eval_steps=10 → 205 dense eval points). **ROC AUC = 0.726** — first SFT run with real discrimination. P(true) mean separation: solvable 0.045 vs unsolvable 0.022. Greedy decoding still noisy (over-predicts True), but the logprob ranking is sound. Full report: [doc/eval_2026-04-29_b5_4x4_spa_replication.md](doc/eval_2026-04-29_b5_4x4_spa_replication.md).
- **Conclusion:** the recipe is sound; the 9×9 collapse was task-difficulty, not architecture/hparams. 9×9 SFT alone won't solve this at Qwen-1.5B scale.
- **Currently running:** 9×9 SPA-scale data gen on autodl1 (3,700 per difficulty × 3 = 11,100 trajectories, ~9–10h) for one final 9×9 SFT test (B-6) before pivoting to RL.

### Remaining Steps (as of 2026-04-29 evening)
The next-actions list has moved to [doc/future_steps.md](doc/future_steps.md) — that's the live to-do queue with priorities and decision checkpoints. High-level summary:

1. **9×9 SPA-scale gen (autodl1, running, ~9h)** — produces ~6,000 no_post_bp samples for B-6.
2. **B-6: 9×9 + SPA hparams + SPA-scale data** — final 9×9 SFT test. If still AUC ~0.46, 9×9 SFT alone is conclusively unviable.
3. **RL on B-5 4×4 checkpoint** — set up TRL + vLLM (per SPEC v5). Should lift the uncalibrated P(true) ≈ 0.045 into a useful regime, and test whether asymmetric rewards (TP+3, FN−2) further raise AUC.
4. **Pass@k eval on B-5** — replicate SPA's headline numbers (1.6 → 59.6 Pass@1) to anchor performance scale.

### Key Design Decisions
- **Sudoku over Sokoban**: Sudoku has no simple deadlock detector, making it the strongest case for LLM-based termination prediction (SPEC §4).
- **Single-step SFT samples** (v4 lock): one (s, a, s') triple per training row. Multi-turn was abandoned after temporal-echo failure mode.
- **Minimal target tags** (v4 lock): `<observation>` + `<prediction>` + `<solvable>` + `<answer>`. Dropped `<breaking_point>`, `<terminate_prob>`, `<steps_left>` — redundant or confusing.
- **Action-conditional `<solvable>` semantics**: predicts is_solvable(s_{t+1}) given the chosen action.
- **LLM-policy over random play**: in-distribution training data; matches SPA's recipe.
- **Asymmetric `<solvable>` rewards** for RL: TP +3.0, FN −2.0, FP −0.5 (replaces old BP-tag rewards).
- **vLLM permitted** (v5 lock): single-process accelerator for RL rollouts and Pass@k eval. Ray-based distributed coordination still out of scope.

### Dependency Upgrades Performed
- `transformers`: 4.30.2 → 4.46.3 (Qwen2 tokenizer support)
- `jinja2`: 2.x → 3.1.6 (chat template `pass_eval_context` support)
- `pyarrow`: installed (parquet support for pandas)
- `peft`, `scikit-learn`: added on AutoDL instances
- `HF_HUB_DISABLE_XET=1`: required env var to bypass HF's xethub 401 through AutoDL proxy

---

## Master pickup point

**For a new Claude session picking up this project, start with [doc/HANDOFF.md](doc/HANDOFF.md).** That's the synthesis doc. This `progress.md` is the chronological history; the handoff doc has the current state + decisions + what to do next.
