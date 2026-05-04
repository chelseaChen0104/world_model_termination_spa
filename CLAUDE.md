# World Model Termination Prediction (SPA)

> **CLAUDE.md note (2026-04-30):** This file documents the project as of the v3 multi-turn-SFT era. The training-format descriptions below (multi-turn, full XML tag set) are deprecated; the architecture/code references are still accurate. **For a new session: start with [doc/HANDOFF.md](doc/HANDOFF.md) — it's the master pickup point that points to current SPEC, pipeline, workflow, and what's running.**

## SAVE Data Pipeline (parallel project)

SAVE-related code lives under `scripts/sudoku4_*`, `scripts/pentomino5x4_*`, `scripts/hidato5x4_*`, `scripts/save_*`; output under `data/sudoku4/`, `data/pentomino5x4/`, `data/hidato5x4/`. Do not edit existing `src/environments/sudoku*.py`, `src/environments/polyomino*.py`, or `src/environments/hidato*.py` from SAVE work. Per-env specs: [data_generation_sudoku.md](doc/data_generation_sudoku.md), [data_generation_pentomino.md](doc/data_generation_pentomino.md), [data_generation_hidato.md](doc/data_generation_hidato.md). Multi-env plan + locked decisions: [plan_2026-05-03_save_data_generation.md](doc/plan_2026-05-03_save_data_generation.md).

## Project Goal
Train an LLM to predict whether a game state is **solvable** (action-conditional: `is_solvable(s_{t+1})` after the model's chosen action), enabling early termination of hopeless episodes. **Breaking points** are derived post-hoc from a `<solvable>` time-series rather than predicted as a separate tag (v4). The primary environment is **Sudoku** (Sokoban is out of scope per [doc/spec_project.md](doc/spec_project.md) §4 — fails predictive-gap criterion).

## Python Environment
Use anaconda Python: `/Users/siyunchen/opt/anaconda3/bin/python`
- The project `.venv` has no `bin` directory — do NOT use it
- Default `/opt/homebrew/bin/python3` lacks numpy — do NOT use it

## Architecture

### Environments (src/environments/)
- `base.py` — `BaseTerminationEnv` abstract class (reset, step, check_solvability, get_all_actions, get_state_info)
- `sokoban.py` — `SokobanEnv` with random puzzle generation (reverse-playing algorithm) and deadlock detector
- `sudoku.py` — `SudokuEnv` with `SudokuSolvabilityChecker` (constraint propagation + bounded backtracking)
- `sudoku_utils.py` — Puzzle generation, validity checking, solvability checker

### Data Pipeline (src/data/)
- `trajectory_generator.py` — `TrajectoryGenerator` works with any `BaseTerminationEnv` (Sokoban or Sudoku)
- `sft_formatter.py` — `SFTFormatter` with variants: `full` (Sokoban), `sudoku_full` (Sudoku)
- `live_trajectory_sampler.py` — `LiveTrajectorySampler` for balanced RL batches from live env

### Training (src/training/)
- `simple_sft_trainer.py` — HuggingFace Trainer-based SFT (single GPU, no FSDP)
- `sft_trainer.py` — Full FSDP SFT trainer (requires verl, distributed)
- `rl_trainer.py` — Contains:
  - `compute_termination_reward_v2()` — Asymmetric rewards (BP TP: +3.0, FN: -2.0), format compliance, no steps_left
  - `SimpleTerminationRLTrainer` — Static dataset RL (original, broken by class imbalance)
  - `LiveEnvTerminationRLTrainer` — Live env RL with balanced sampling (the fix)
  - `main_live_env()` — Entry point supporting both sokoban and sudoku envs
- Config: `src/training/config/rl_termination.yaml` (Sokoban), `rl_sudoku.yaml` (Sudoku)

### Evaluation
- `evaluate_rl.py` — Balanced eval from live env, confusion matrices, F1, per-deadlock-type recall
- `src/evaluation/sudoku_baseline.py` — Heuristic baseline comparison

## Data Generation (Step-by-Step)

### Sudoku SFT Data (current focus)

**What was run:**
```bash
/Users/siyunchen/opt/anaconda3/bin/python -c "
import os, sys
sys.path.insert(0, 'src')
from environments.sudoku import SudokuEnv
from data.trajectory_generator import TrajectoryGenerator
from data.sft_formatter import SFTFormatter

env = SudokuEnv(grid_size=9, difficulty='easy', max_steps=30)
gen = TrajectoryGenerator(env)
trajectories = gen.generate_balanced_dataset(
    target_size=1280,
    success_ratio=0.0,
    failure_ratio=1.0,
    timeout_ratio=0.0,
    max_steps=30,
    seed=42,
)
fmt = SFTFormatter(variant='sudoku_full')
df = fmt.create_sft_dataset(trajectories)
fmt.save_dataset(df, 'data/sudoku_termination', split_ratio=0.2)
"
```

**Pipeline:**
1. `SudokuEnv(grid_size=9, difficulty='easy', max_steps=30)` — creates 9x9 Sudoku env, removes 40% of cells
2. `TrajectoryGenerator(env)` — wraps env for trajectory rollouts
3. `generate_balanced_dataset(target_size=1280, failure_ratio=1.0)` — generates 1280 trajectories with random actions; nearly all hit deadlocks
4. For each trajectory step: env calls `check_solvability()` which runs constraint propagation + bounded backtracking (~2ms/check)
5. `SFTFormatter(variant='sudoku_full')` — formats each step as chat messages with XML response tags
6. `create_sft_dataset()` — expands trajectories into per-step samples (prompt + response pairs)
7. `save_dataset()` — 80/20 train/val split, saves as parquet + csv

**Output: `data/sudoku_termination/`**
- `wm_train.parquet` — 25,649 samples
- `wm_val.parquet` — 6,413 samples
- Total: 32,062 samples from 1,280 trajectories
- Class distribution: 6.6% solvable, 93.4% unsolvable, 4.0% breaking points
- Note: Inverted from Sokoban (which was 99.2% solvable) because random Sudoku play creates unsolvable states very quickly

**Data format (each row):**
- `data_source`: "sudoku"
- `prompt`: array of [system_msg, user_msg] — system describes Sudoku rules + XML tags, user provides current grid
- `response`: XML with `<think><observation>...<prediction>...<terminate_prob>...<steps_left>...<solvable>...<breaking_point>...</think><answer>place N at row R col C</answer>`
- `extra_info`: JSON with `{step, is_solvable, is_breaking_point}`

### Sokoban SFT Data (on hold)

**Previously generated:** `data/termination_study_v2/`
- `wm_train.parquet` — 61,096 samples
- `wm_val.parquet` — 15,275 samples
- Used `SFTFormatter(variant='full')` with Sokoban system prompt
- Known issue: 99.2% solvable per-step (extreme class imbalance)

## SFT Training (Step-by-Step)

### Sokoban SFT (completed previously)

**What was run:**
```bash
python src/training/simple_sft_trainer.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --train_file data/termination_study_v2/wm_train.parquet \
  --val_file data/termination_study_v2/wm_val.parquet \
  --output_dir outputs/sft_termination \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5
```

**Output:** `outputs/sft_termination/checkpoint-4500/` (Qwen2.5-1.5B-Instruct, 3B params, ~3GB)

### Sudoku SFT (next step)

**To run:**
```bash
/Users/siyunchen/opt/anaconda3/bin/python src/training/simple_sft_trainer.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --train_file data/sudoku_termination/wm_train.parquet \
  --val_file data/sudoku_termination/wm_val.parquet \
  --output_dir outputs/sft_sudoku \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5
```

## RL Training

Uses `LiveEnvTerminationRLTrainer` with balanced sampling from live env.
Config: `src/training/config/rl_sudoku.yaml`

Key design decisions:
- **Only solvable + breaking_point are rewarded in RL** (steps_left is trajectory-dependent, SFT only)
- **Asymmetric BP rewards:** TP +3.0, FN -2.0, FP -0.5 (catching deadlocks matters most)
- **Format compliance:** +0.1 per XML tag present (prevents format forgetting)
- **Balanced sampling:** ~50% solvable / ~50% unsolvable per batch from live env

## Evaluation

```bash
python evaluate_rl.py --n-solvable 100 --n-unsolvable 100 [--sft-path PATH] [--rl-path PATH]
```

Generates balanced eval set from live env, reports confusion matrices, F1, per-deadlock-type recall.

## Key Insight: Why Sudoku over Sokoban

- **Sokoban:** Simple pattern-matching deadlock detector catches most cases. LLM adds marginal value.
- **Sudoku:** No simple detector exists. Unsolvable states look valid (no visual conflicts). Requires constraint reasoning. This is the strongest case for LLM-based termination prediction.
- Sudoku BP rate: 98% of random trajectories hit a breaking point (vs ~25% for Sokoban)





## Random Play vs LLM-Policy Data Generation

### Overview

The original SPA paper uses **LLM-policy trajectories** for data generation: a base LLM (e.g., Qwen2.5-1.5B-Instruct) plays the game by generating `<think>...</think><answer>action</answer>` responses, and the environment executes the parsed action. Our initial implementation used **random actions** instead, which creates a distribution mismatch.

### How Each Approach Works

**Random Play (current implementation):**
1. Environment resets with a fresh puzzle
2. At each step, a random valid action is chosen uniformly
3. Environment executes the action and records the new state
4. Solvability checker annotates each state with ground truth labels
5. SFTFormatter wraps the state in chat format with XML response tags

**LLM-Policy Play (SPA paper approach):**
1. Environment resets with a fresh puzzle
2. The current grid state is rendered as text and sent to the LLM as a user message
3. The LLM generates a response: `<think><observation>...</observation><prediction>...</prediction></think><answer>place 5 at row 3 col 7</answer>`
4. The action is parsed from the `<answer>` tag and executed in the environment
5. The real next state (not the LLM's prediction) is shown back to the LLM
6. The full multi-turn conversation becomes the training sample
7. Loss is masked to only the last assistant turn (the turn being trained on)

### Example: Sudoku Data Comparison

**Random Play — Input (user message):**
```
Current Sudoku State (9x9):
  C1 C2 C3  C4 C5 C6  C7 C8 C9
R1  5  3  .   .  7  .   .  .  .
R2  6  .  .   1  9  5   .  .  .
R3  .  9  8   .  .  .   .  6  .
R4  8  .  .   .  6  .   .  .  3
R5  4  .  .   8  .  3   .  .  1
R6  7  .  .   .  2  .   .  .  6
R7  .  6  .   .  .  .   2  8  .
R8  .  .  .   4  1  9   .  .  5
R9  .  .  .   .  8  .   .  7  9

Previous action: place 2 at row 1 col 4
```

**Random Play — Response:**
```xml
<think>
<observation>A 2 was placed at R1C4. The grid now has 28 filled cells.</observation>
<prediction>This placement may restrict candidates in row 1, column 4, and box 2.</prediction>
<solvable>true</solvable>
<breaking_point>false</breaking_point>
<steps_left>bucket_20+</steps_left>
</think>
<answer>place 7 at row 3 col 5</answer>
```

The action in `<answer>` is chosen randomly — the LLM never actually decided this move. The thinking text is also synthetic (template-filled from ground truth labels), not generated by the LLM.

**LLM-Policy Play — Input (multi-turn conversation, showing last turn):**
```
[System] You are playing Sudoku. Analyze the board and predict termination signals...

[User Turn 1] Current state: [initial grid]
[Assistant Turn 1] <think>..analysis..</think><answer>place 5 at row 1 col 1</answer>
[User Turn 2] Action executed. Current state: [grid after move 1]
[Assistant Turn 2] <think>..analysis..</think><answer>place 3 at row 1 col 2</answer>
...
[User Turn K] Action executed. Current state: [grid after move K-1]
[Assistant Turn K] <think>
<observation>After placing 8 at R4C2, I see row 4 now has 5,8,6,3 filled.
Column 2 has 3,9,6. The remaining candidates for R2C2 are {2,4,7}.</observation>
<prediction>The grid still appears solvable. Row 4 has good progress
with 4 cells filled. No immediate constraint violations visible.</prediction>
<solvable>true</solvable>
<breaking_point>false</breaking_point>
<steps_left>bucket_11_20</steps_left>
</think>
<answer>place 4 at row 2 col 2</answer>
```

Here, the LLM chose `place 4 at row 2 col 2` based on its own reasoning. The thinking text reflects the model's actual analysis. Loss is computed only on Turn K.

### Key Differences

| Aspect | Random Play | LLM-Policy Play |
|--------|------------|-----------------|
| **Action selection** | Uniform random from valid actions | LLM chooses based on reasoning |
| **Thinking text** | Template-filled from ground truth | LLM-generated (may be wrong) |
| **State distribution** | States reached by random play | States reached by LLM play |
| **Conversation context** | Single turn (just current state) | Multi-turn (full game history) |
| **Breaking point cause** | Random bad moves (immediate) | Strategic mistakes (subtle) |
| **Training signal** | Static prompt-response pairs | Full rollout with history |

### Impact on SFT Training

**Random play creates distribution mismatch:**
- Note: our `get_all_actions()` uses `get_valid_numbers()` which only returns numbers with no immediate row/col/box conflict. So random play does NOT place obviously conflicting numbers — every random placement is locally valid.
- However, random play picks **uniformly** among all non-conflicting placements, while an LLM shows bias toward more constrained cells and more "reasonable" numbers. This creates a different **state distribution**: after 3-5 random moves, wrong numbers are scattered randomly across the grid, whereas an LLM would tend to fill easier/more constrained cells first.
- Breaking points from random play tend to occur **earlier** (1-3 moves in) because random wrong placements quickly eliminate candidates for other cells. An LLM would survive more moves before making a subtle fatal mistake.
- The SFT model learns patterns specific to random-play state distributions (e.g., wrong numbers appearing in low-constraint cells where an LLM would never place them) rather than the patterns it will actually encounter during RL/deployment.

**LLM-policy gives in-distribution training:**
- States match what the model will actually encounter during RL and inference
- Breaking points are harder to detect (the LLM made a "reasonable" move that happens to be wrong)
- The model learns to reason about states that plausible policies produce
- Breaking points occur later in trajectories, giving more solvable-state training examples per trajectory

### Impact on RL Training

**Random play SFT → RL gap:**
- The SFT model was trained on random-play states but RL generates states from the SFT model's own policy
- This distribution shift means the SFT model's predictions are unreliable during early RL
- RL must "unlearn" patterns from random play before it can learn useful ones
- More RL steps needed to bridge the SFT-RL distribution gap

**LLM-policy SFT → smooth RL transition:**
- SFT model already sees LLM-generated states, so RL starts from a better initial policy
- The distribution shift between SFT and RL is smaller
- RL can focus on refining predictions rather than correcting distribution mismatch
- Fewer RL steps needed to achieve the same performance

### Single-Turn vs Multi-Turn Data Format

Our current SFT formatter (`SFTFormatter.format_trajectory()`) creates **single-turn** samples: each step is an independent `[system, user_state] → response` pair with no game history. The SPA paper uses **multi-turn** conversations where all previous states and LLM responses are included.

**Single-turn format (our current implementation):**
```
Messages: [system_prompt, "Current state:\n{grid}"]
Response: "<think>...<solvable>true</solvable>...</think><answer>place 5 at row 3 col 7</answer>"
```
Each step is a standalone sample. The model sees only the current grid — no history of what it placed or why.

**Multi-turn format (SPA paper):**
```
Messages: [
  system_prompt,
  "Current state: {grid_0}",                    # Turn 1 state
  "<think>...</think><answer>place 5 at ...</answer>",  # Turn 1 response
  "Action executed. Current state: {grid_1}",    # Turn 2 state
  "<think>...</think><answer>place 3 at ...</answer>",  # Turn 2 response
  ...
  "Action executed. Current state: {grid_K}",    # Turn K state (current)
]
Response: "<think>...</think><answer>place 4 at ...</answer>"  # Only this turn is trained
```
Loss is masked to only the last assistant turn. The model sees the full game history.

**Impact on SFT:**

| Aspect | Single-Turn | Multi-Turn |
|--------|------------|------------|
| **Context** | Model sees only current grid | Model sees all prior moves and states |
| **Reasoning quality** | Must infer everything from grid alone | Can reference what it placed previously |
| **Sequence length** | Short (~200-400 tokens) | Grows with game (~200-400 × K tokens) |
| **Samples per trajectory** | K independent samples from K steps | K samples, each progressively longer |
| **Training efficiency** | More samples per GPU-hour | Fewer samples but richer signal |
| **Breaking point detection** | Must detect from grid state alone | Can reason "I placed X, which may have caused..." |

For Sudoku specifically, single-turn is less of a limitation than for Sokoban because the grid state is fully observable — all placed numbers are visible. The model doesn't need history to see what's on the board. However, multi-turn helps the model learn to **track its own reasoning** and detect when a prior move was the cause of a constraint violation.

**Impact on RL:**

| Aspect | Single-Turn SFT → RL | Multi-Turn SFT → RL |
|--------|----------------------|---------------------|
| **Format consistency** | If RL also uses single-turn prompts, no format gap | If RL uses multi-turn rollouts, SFT model is already adapted |
| **Credit assignment** | Model can't reason about which prior move caused a deadlock | Model can trace deadlocks back to specific prior moves |
| **Reward attribution** | RL reward applies to current step only | RL reward can be contextualized by full trajectory |
| **Computational cost** | Cheaper per step | Context grows with each step, more memory/compute |

**Our current approach**: SFT uses multi-turn format with `max_context_turns` parameter. RL stays single-turn for now (more complex to change, deferred).

### Implementation Plan

To switch from random play to LLM-policy data generation:

1. **Create `LLMTrajectoryGenerator`** — wraps a base model (Qwen2.5-1.5B-Instruct) to play Sudoku ✅
   - Uses `transformers` for inference (no vLLM needed for data gen)
   - Parses actions from `<answer>` tags
   - Falls back to random action if LLM output is unparseable
   - Stores raw LLM response in `TrajectoryStep.llm_raw_response` for multi-turn context
2. **Regenerate Sudoku SFT data** using LLM-policy trajectories with `--multi-turn`
3. **Retrain SFT model** on the new data
4. **RL training** proceeds as before (LiveEnvTerminationRLTrainer already uses live env)

### Status

- `LLMTrajectoryGenerator` implemented ✅ (`src/data/llm_trajectory_generator.py`)
- Multi-turn SFT formatting implemented ✅ (`SFTFormatter` supports `multi_turn=True`)
- `simple_sft_trainer.py` updated ✅ (handles assistant role in multi-turn prompts)
- `TrajectoryStep.llm_raw_response` field added ✅
- Current data (`data/sudoku_termination/`) was generated with **random play, single-turn** — needs regeneration
- LLM-policy data generation not yet run (requires GPU for practical throughput)
- Command to generate: `python -m src.data.llm_trajectory_generator --multi-turn --max-context-turns 10 --num-trajectories 1280 --output-dir data/sudoku_llm_policy`

## Reference SPA Repo
`/Users/siyunchen/Documents/GitHub/SPA_reproduction/` — RAGEN framework with multi-turn RL (Ray, vLLM, PPO). Our approach is simpler (single-GPU, GRPO).
