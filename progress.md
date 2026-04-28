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

### Remaining Steps
1. **Train Sudoku SFT model on multi-turn random-play data** — `data/sudoku_multiturn/` → `outputs/sft_sudoku/` (can start immediately on GPU)
2. **Generate LLM-policy data on GPU** — run `scripts/generate_llm_policy_data_gpu.sh` on AutoDL → `data/sudoku_llm_policy/`
3. **Retrain SFT on LLM-policy data** — for in-distribution training (optional if random-play SFT works well)
4. **Evaluate Sudoku SFT model** — balanced eval with confusion matrices
5. **Run Sudoku RL training** — `LiveEnvTerminationRLTrainer` with reward v2
6. **Evaluate Sudoku RL model** — compare with SFT, measure improvement

### Key Design Decisions
- **Sudoku over Sokoban**: Sudoku has no simple deadlock detector, making it the strongest case for LLM-based termination prediction
- **LLM-policy over random play**: In-distribution training data, harder breaking points, smoother SFT→RL transition
- **Asymmetric BP rewards**: Catching deadlocks (TP) rewarded much more than false alarms penalized (FP)
- **Live env balanced sampling**: ~50/50 solvable/unsolvable per RL batch
- **Single-GPU GRPO**: Simpler than SPA's RAGEN framework (Ray, vLLM, PPO)

### Dependency Upgrades Performed
- `transformers`: 4.30.2 → 4.46.3 (Qwen2 tokenizer support)
- `jinja2`: 2.x → 3.1.6 (chat template `pass_eval_context` support)
- `pyarrow`: installed (parquet support for pandas)
