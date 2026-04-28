# Step-by-Step Function Call Trace

This document traces exactly what each function does at each stage of the pipeline: **Data Generation**, **SFT Training**, and **RL Training**.

---

## Phase 1: Data Generation

### 1A. Random-Play Data Generation (Intermediate)

**Entry point:** inline script or `TrajectoryGenerator`

```
SudokuEnv(grid_size=9, difficulty='easy', max_steps=30)
```

#### Step 1: Create Environment
- **`SudokuEnv.__init__()`** (`src/environments/sudoku.py:46`)
  - Stores `grid_size=9`, `difficulty='easy'`, `max_steps=30`
  - Creates `SudokuSolvabilityChecker(max_backtrack_depth=30)` (`src/environments/sudoku_utils.py:200`)
  - Initializes `current_grid`, `initial_grid`, `solution_grid` to None

#### Step 2: Create Trajectory Generator
```
TrajectoryGenerator(env)
```
- **`TrajectoryGenerator.__init__()`** (`src/data/trajectory_generator.py:78`)
  - Stores reference to `env`
  - Optionally stores `detector` (for Sokoban; None for Sudoku)

#### Step 3: Generate Balanced Dataset
```
gen.generate_balanced_dataset(target_size=1280, failure_ratio=1.0, max_steps=30, seed=42)
```
- **`TrajectoryGenerator.generate_balanced_dataset()`** (`trajectory_generator.py:216`)
  - Computes targets: `target_success=0`, `target_failure=1280`, `target_timeout=0`
  - Loops up to `target_size * 10` attempts:
    - Calls **`generate_random_trajectory(max_steps=30, seed=attempt)`** for each
    - Categorizes by `meta.termination_reason`: success / deadlock / timeout
    - Stops when all buckets are filled
  - Returns `List[(trajectory, metadata)]`

#### Step 4: Generate Single Random Trajectory (called 1280+ times)
- **`TrajectoryGenerator.generate_random_trajectory()`** (`trajectory_generator.py:89`)

  **4a. Reset environment:**
  - Calls **`SudokuEnv.reset(seed=N)`** (`sudoku.py:72`)
    - Calls **`generate_sudoku_puzzle(grid_size=9, difficulty='easy', seed=N)`** (`sudoku_utils.py:140`)
      - Seeds `random` and `numpy`
      - Creates empty 9x9 grid, fills via recursive backtracking (`fill()` function)
      - Copies solution, removes 40% of cells (easy difficulty = ~32 cells removed)
      - Returns `(puzzle_grid, solution_grid)` as numpy arrays
    - Stores `initial_grid`, `solution_grid`, `current_grid`
    - Calls **`SudokuEnv.render()`** (`sudoku.py:167`) -> **`format_grid()`** (`sudoku_utils.py:115`)
      - Formats 9x9 grid as text: numbers separated by spaces, `|` between boxes, `-` between box rows, `.` for empty cells
    - Returns grid string as initial state

  **4b. Step loop (up to max_steps=30):**
  - Calls **`SudokuEnv.get_all_actions()`** (`sudoku.py:179`)
    - For each empty cell `(r,c)`:
      - Calls **`get_valid_numbers(grid, r, c)`** (`sudoku_utils.py:51`)
        - Starts with candidates {1..9}
        - Removes numbers already in same row, column, 3x3 box
        - Returns remaining valid candidates (no immediate conflict)
      - Creates action string `"place N at row R col C"` for each valid number
    - Returns list of all valid actions (typically 50-200 at start)
  - Picks random action from list: `action = random.choice(all_actions)`
  - Calls **`SudokuEnv.step(action_str)`** (`sudoku.py:85`)

    **Inside `step()`:**
    1. Increments `num_steps`
    2. Calls **`check_solvability()`** (`sudoku.py:171`) to get `prev_solvable`
       - Delegates to **`SudokuSolvabilityChecker.check_solvability(grid)`** (`sudoku_utils.py:203`)
         - **`find_conflicts(grid)`** (`sudoku_utils.py:71`) — checks for duplicate numbers in any row/col/box
         - For each empty cell: **`get_valid_numbers()`** — if any cell has 0 candidates -> `"zero_candidates"`
         - **`_propagate(grid, candidates)`** (`sudoku_utils.py:249`) — constraint propagation loop:
           - **Naked singles:** cells with exactly 1 candidate -> place it, eliminate from peers via **`_eliminate_peers()`** (`sudoku_utils.py:307`)
           - **Hidden singles:** number that can only go in one cell within a row/col/box unit -> place it
           - Repeats until no more progress or contradiction found
         - **`_backtrack(grid, candidates, depth=0)`** (`sudoku_utils.py:370`) — bounded backtracking:
           - Picks cell with fewest candidates (MRV heuristic)
           - Tries each candidate: copy grid, place, propagate, recurse
           - Returns True if any path finds a solution
           - Depth limit = 30 (assumes solvable if exceeded)
       - Returns `(is_solvable: bool, reason: Optional[str])`
    3. Parses action via **`_parse_action(action_str)`** (`sudoku.py:213`)
       - Regex matches "place N at row R col C"
       - Converts to 0-indexed (row-1, col-1)
       - Validates bounds
    4. Checks cell is modifiable (`initial_grid[row,col] == 0`) and empty (`current_grid[row,col] == 0`)
    5. Checks no immediate conflict via **`is_valid_placement(grid, row, col, num)`** (`sudoku_utils.py:32`)
    6. Places number: `current_grid[row, col] = num`
    7. Calls **`check_solvability()`** again for `new_solvable`
    8. Detects breaking point: `prev_solvable and not new_solvable`
    9. Checks correctness: `num == solution_grid[row, col]`
    10. Computes reward: +10 solved, +1 correct, -1 breaking point, +0.5 valid but wrong
    11. Returns `(rendered_state, reward, done, info_dict)`

  - Creates **`TrajectoryStep`** (`trajectory_generator.py:16`) with:
    - `state`, `action`, `action_name`, `next_state`, `reward`, `step`
    - `is_solvable`, `is_breaking_point`, `deadlock_type`
    - `llm_raw_response=None` (random play has no LLM response)

  **4c. Post-hoc annotation** (after loop ends):
  - For each step `i` in trajectory of length `T`:
    - `done_label = 1 if i == T-1 else 0`
    - `steps_left = T - i`
    - `steps_left_bucket =` **`bucket_steps(T-i)`** (`trajectory_generator.py:56`)
      - 1 -> "immediate", 2-3 -> "near", 4-7 -> "medium", 8+ -> "far"
    - If breaking point at step B and `i >= B`: `steps_since_break = i - B`
  - Creates **`TrajectoryMetadata`** (`trajectory_generator.py:45`)

#### Step 5: Format as Multi-Turn SFT Data
```
SFTFormatter(variant='sudoku_full')
fmt.create_sft_dataset(trajectories, multi_turn=True, max_context_turns=10)
```

- **`SFTFormatter.__init__(variant='sudoku_full')`** (`sft_formatter.py:143`)
  - Loads system prompt from `SYSTEM_PROMPTS['sudoku_full']`

- **`SFTFormatter.create_sft_dataset()`** (`sft_formatter.py:336`)
  - For each `(trajectory, metadata)` tuple:
    - Calls **`format_trajectory(traj, meta, multi_turn=True, max_context_turns=10)`** (`sft_formatter.py:210`)
    - Dispatches to **`_format_multi_turn()`** (`sft_formatter.py:260`)

      **For each step K in trajectory:**
      1. Compute history window: `history_start = max(0, K - 10)`
      2. Build message list:
         - `messages[0]` = `{"role": "system", "content": system_prompt}`
         - For each prior step `h` from `history_start` to `K-1`:
           - `messages[2h+1]` = `{"role": "user", "content": "Current state:\n{grid}"}` (or "Action executed. Current state:\n{grid}" if h > 0)
           - `messages[2h+2]` = `{"role": "assistant", "content": ...}`
             - Prefers `step.llm_raw_response` if not None (LLM-policy mode)
             - Falls back to **`format_step(step)`** (random play mode)
         - `messages[-1]` = `{"role": "user", "content": "Action executed. Current state:\n{current_grid}"}` (current turn)
      3. Generate ground-truth response: **`format_step(current_step)`** (`sft_formatter.py:156`)
         - Builds XML: `<think><observation>grid</observation><prediction>next_grid</prediction>`
         - Adds `<terminate_prob>`, `<steps_left>`, `<solvable>`, `<breaking_point>`
         - Adds `<answer>place N at row R col C</answer>`
      4. Returns `{'messages': [...], 'response': xml_string, 'metadata': {...}}`

  - Converts each sample to a DataFrame row:
    - `prompt` = `np.array(messages)` (stored as numpy array in parquet)
    - `response` = XML ground-truth string
    - `extra_info` = JSON with `{step, is_solvable, is_breaking_point, num_context_turns}`

#### Step 6: Save Dataset
```
fmt.save_dataset(df, 'data/sudoku_multiturn', split_ratio=0.2)
```
- **`SFTFormatter.save_dataset()`** (`sft_formatter.py:385`)
  - Splits 80/20 train/val via `sklearn.train_test_split(random_state=42)`
  - Saves `wm_train.parquet`, `wm_val.parquet` (and `.csv` copies)

---

### 1B. LLM-Policy Data Generation (GPU required)

**Entry point:** `python -m src.data.llm_trajectory_generator --multi-turn --max-context-turns 10`

#### Step 1: Create LLM Trajectory Generator
```
LLMTrajectoryGenerator(env, config, variant='sudoku_full')
```
- **`LLMTrajectoryGenerator.__init__()`** (`llm_trajectory_generator.py:50`)
  - Stores env, config, variant
  - Loads system prompt from `SFTFormatter.SYSTEM_PROMPTS['sudoku_full']`
  - Model/tokenizer NOT loaded yet (lazy loading)

#### Step 2: Generate Balanced Dataset
```
generator.generate_balanced_dataset(target_size=1280, failure_ratio=1.0, max_steps=30, seed=42)
```
- **`LLMTrajectoryGenerator.generate_balanced_dataset()`** (`llm_trajectory_generator.py:305`)
  - Same logic as `TrajectoryGenerator.generate_balanced_dataset()` but calls `generate_trajectory()` which uses the LLM

#### Step 3: Generate Single LLM-Policy Trajectory
- **`LLMTrajectoryGenerator.generate_trajectory(max_steps=30, seed=N)`** (`llm_trajectory_generator.py:172`)

  **3a. Reset environment** (same as random play)

  **3b. Step loop:**
  - Calls **`_generate_response(state, step_num)`** (`llm_trajectory_generator.py:109`)
    - First call triggers **`_load_model()`** (`llm_trajectory_generator.py:69`)
      - Loads `AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')`
      - Detects device: CUDA -> bfloat16, else -> float32 CPU
      - Loads `AutoModelForCausalLM.from_pretrained()` to device
      - Sets `model.eval()`
    - Builds messages: `[system_prompt, "Current state:\n{grid}"]`
    - Applies chat template via `tokenizer.apply_chat_template()`
    - Tokenizes and runs **`model.generate()`** with:
      - `max_new_tokens=512`, `temperature=0.7`, `top_p=0.9`, `do_sample=True`
    - Decodes only new tokens (after prompt)
    - Returns raw response text

  - Calls **`_parse_action_from_response(response)`** (`llm_trajectory_generator.py:150`)
    - Regex: `<answer>(.*?)</answer>` -> extracts action string
    - Fallback regex: `place\s+\d+\s+at\s+row\s+\d+\s+col\s+\d+`
    - Returns `None` if unparseable

  - If parse fails and `random_fallback=True`:
    - Falls back to `random.choice(env.get_all_actions())`

  - Calls **`SudokuEnv.step(action_str)`** (same as random play)
    - If `info['action_is_valid'] == False`: retries with random fallback

  - Creates `TrajectoryStep` with **`llm_raw_response=response`** (stores verbatim LLM output)

  **3c. Post-hoc annotation** (same as random play)

#### Step 4-6: Format and Save (same as 1A Steps 5-6)
- Key difference: `_format_multi_turn()` uses `step.llm_raw_response` (real LLM output) for prior assistant turns instead of template-generated ground truth

---

## Phase 2: SFT Training

**Entry point:** `python src/training/simple_sft_trainer.py --train_file ... --val_file ...`

#### Step 1: Parse Arguments and Setup
- **`main()`** (`simple_sft_trainer.py:181`)
  - Parses CLI args (model, data paths, hyperparameters)
  - Loads tokenizer: `AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')`
  - Sets `pad_token = eos_token`

#### Step 2: Load Model
- **`AutoModelForCausalLM.from_pretrained()`**
  - CUDA: bfloat16 + SDPA attention
  - MPS: float16 + eager attention
  - CPU: float32 + eager attention
- Optional LoRA: if `--lora_r > 0`, applies `LoraConfig` to q/k/v/o/gate/up/down projections
- Enables gradient checkpointing

#### Step 3: Load and Tokenize Data
```
load_and_process_data(train_file, tokenizer, max_length)
```
- **`load_and_process_data()`** (`simple_sft_trainer.py:68`)
  - Reads parquet file with `pd.read_parquet()`
  - For each row:

    **3a. Parse prompt messages:**
    - Handles numpy array from parquet: `prompt_messages.tolist()`
    - Handles JSON string: `json.loads()`

    **3b. Build prompt text (Qwen2 ChatML format):**
    ```
    <|im_start|>system\n{system_prompt}<|im_end|>\n
    <|im_start|>user\n{state}<|im_end|>\n
    <|im_start|>assistant\n{prior_response}<|im_end|>\n    # multi-turn only
    <|im_start|>user\n{next_state}<|im_end|>\n              # multi-turn only
    ...
    <|im_start|>assistant\n                                  # generation prompt
    ```
    - All message roles (system, user, assistant) are handled
    - Prior assistant turns become part of the prompt (label = -100)

    **3c. Build response text:**
    - `response_text = row['response'] + "<|im_end|>"`

    **3d. Tokenize and create labels:**
    - `prompt_tokens = tokenizer(prompt_text)` -> gets prompt length
    - `full_tokens = tokenizer(prompt_text + response_text, max_length=max_length)`
    - Labels: `[-100] * prompt_len + full_tokens.input_ids[prompt_len:]`
      - **-100 = ignored by loss** (prompt tokens including prior assistant turns)
      - **Actual token ids = trained on** (final response only)

  - Returns HuggingFace `Dataset` with columns: `input_ids`, `attention_mask`, `labels`

#### Step 4: Data Collation
- **`SFTDataCollator.__call__()`** (`simple_sft_trainer.py:148`)
  - Finds max sequence length in batch (capped at `max_length`)
  - Right-pads each sequence:
    - `input_ids`: padded with `pad_token_id`
    - `attention_mask`: padded with 0
    - `labels`: padded with -100
  - Returns batch tensors

#### Step 5: Training Loop
- **`Trainer.train()`** (HuggingFace Trainer)
  - Standard causal LM training loop:
    - Forward pass: `model(input_ids, attention_mask, labels=labels)`
    - Cross-entropy loss computed only on non-(-100) labels
    - Backward pass + gradient accumulation (8 steps)
    - AdamW optimizer update with cosine LR schedule
  - Eval every `eval_steps` (100): computes `eval_loss` on validation set
  - Saves checkpoint every `save_steps` (500), keeps best 3 by `eval_loss`

#### Step 6: Save Final Model
- Saves to `{output_dir}/final/`:
  - Model weights (pytorch_model.bin or safetensors)
  - Tokenizer files
  - Training config

---

## Phase 3: RL Training

**Entry point:** `python -c "from src.training.rl_trainer import main_live_env; main_live_env()"`
**Config:** `src/training/config/rl_sudoku.yaml`

#### Step 1: Setup
- **`main_live_env()`** (`rl_trainer.py:1177`)
  - Loads Hydra config from `rl_sudoku.yaml`
  - Loads tokenizer from base model (`Qwen/Qwen2.5-1.5B-Instruct`)
  - Loads SFT model from checkpoint (the model trained in Phase 2)
  - Creates environment based on config:
    - For Sudoku: `SudokuEnv(grid_size=9, difficulty='easy', max_steps=81)`
  - Gets system prompt from `SFTFormatter(variant='sudoku_full')`

#### Step 2: Create Live Env RL Trainer
```
LiveEnvTerminationRLTrainer(config, tokenizer, model, env, system_prompt, ...)
```
- **`LiveEnvTerminationRLTrainer.__init__()`** (`rl_trainer.py:821`)
  - Sets reward function to **`compute_termination_reward_v2`** (default)
  - Creates **`LiveTrajectorySampler`** (`live_trajectory_sampler.py:33`)
    - **`LiveTrajectorySampler.__init__()`** (`live_trajectory_sampler.py:44`)
      - Creates `TrajectoryGenerator(env)` for random trajectory generation
      - Calls **`_fill_pool()`** (`live_trajectory_sampler.py:79`)
        - Generates `pool_size=200` random trajectories
        - For each trajectory step, creates **`RLSample`** and indexes by class:
          - `_solvable_steps[]`, `_unsolvable_steps[]`, `_breaking_point_steps[]`
  - Creates AdamW optimizer for model parameters

#### Step 3: Training Loop
- **`TerminationRLTrainer.train()`** (`rl_trainer.py:513`)
  - For each training step:

    **3a. Generate Rollouts:**
    - **`LiveEnvTerminationRLTrainer._generate_rollouts()`** (`rl_trainer.py:864`)
      1. Maybe refresh pool: **`_maybe_refresh_pool()`** (`live_trajectory_sampler.py:134`)
         - Every `refresh_frequency=50` steps, clears pool and refills
      2. Sample balanced batch: **`LiveTrajectorySampler.sample_batch()`** (`live_trajectory_sampler.py:140`)
         - Computes: `n_solvable = batch_size * 0.5`, `n_unsolvable = batch_size * 0.5`
         - Within unsolvable: `n_bp = n_unsolvable * 0.3` from breaking point pool
         - Samples with replacement from indexed pools
         - Shuffles to avoid ordering bias
         - Formats via **`_format_batch()`** (`live_trajectory_sampler.py:190`):
           - For each sample: builds `[system, user_state]` messages
           - Tokenizes with chat template + left-padding
           - Returns `{input_ids, attention_mask, extra_infos, samples}`
      3. Run model generation:
         - `model.generate(input_ids, max_new_tokens=..., temperature=..., do_sample=True)`
         - Decodes generated tokens (after prompt)
      4. Parse predictions from generated text:
         - **`parse_termination_predictions(text)`** (`rl_trainer.py:55`)
           - Regex extracts: `<solvable>`, `<breaking_point>`, `<steps_left>`, `<terminate_prob>`, `<answer>`
           - Returns dict of parsed values (None if not found)
      5. Get ground truth from `RLSample` objects: `is_solvable`, `is_breaking_point`
      6. Package into `DataProto` with:
         - `batch`: `{input_ids, attention_mask, input_lens}`
         - `non_tensor_batch`: `{uid, is_solvable_gt, is_breaking_point_gt, solvable_pred, breaking_point_pred, generated_texts}`

    **3b. Compute Rewards:**
    - **`compute_termination_reward_v2(batch, config)`** (`rl_trainer.py:249`)
      - For each sample in batch:

        **Format compliance reward:**
        - +0.1 for each XML tag present: `<solvable>`, `<breaking_point>`, `<steps_left>`, `<terminate_prob>`, `<answer>`
        - Max +0.5 for all 5 tags

        **Solvability prediction reward:**
        - Correct: +0.5 (`solvable_correct`)
        - Wrong: -0.25 (`solvable_wrong`)

        **Breaking point detection reward (asymmetric):**
        - TP (caught deadlock): +3.0 (`bp_tp`)
        - FN (missed deadlock): -2.0 (`bp_fn`)
        - FP (false alarm): -0.5 (`bp_fp`)
        - TN: 0.0

      - Assigns total reward to last token of response via `response_mask`
      - Returns `reward_tensor` shape `(batch_size, seq_len)`

    **3c. Compute Advantages:**
    - **`compute_advantage(data, adv_estimator='grpo')`** (`rl_trainer.py:446`)
      - First computes **`compute_response_mask(data)`** (`rl_trainer.py:133`)
        - Uses `input_lens` to mask prompt tokens (only response tokens are active)
      - Then **`compute_grpo_outcome_advantage()`** (`rl_trainer.py:394`)
        - Sums per-token rewards to get per-sample scores
        - Groups by `uid` (prompt index)
        - Normalizes: `advantage = (score - mean) / (std + epsilon)` within each group
        - Applies response mask: `advantages = normalized_score * response_mask`

    **3d. Update Policy:**
    - **`LiveEnvTerminationRLTrainer._update_actor(batch)`** (`rl_trainer.py:952`)
      1. Forward pass: `model(input_ids, attention_mask)` -> logits
      2. Compute log probs: `log_softmax(logits[:, :-1])`, gather at target token indices
      3. Policy gradient loss: `loss = -mean(log_probs * advantages, mask=response_mask)`
         - Only response tokens contribute (prompt tokens masked out)
      4. Backward pass
      5. Gradient clipping: `clip_grad_norm_(max_grad_norm=1.0)`
      6. Optimizer step

    **3e. Log and Checkpoint:**
    - Prints metrics every 10 steps
    - Saves checkpoint every `save_freq` steps

---

## Phase 4: Evaluation

**Entry point:** `python evaluate_rl.py --n-solvable 100 --n-unsolvable 100`

#### Step 1: Generate Balanced Eval Set
- **`generate_balanced_eval_set()`** (`evaluate_rl.py:72`)
  - Creates `TrajectoryGenerator(env)`
  - Generates random trajectories, collects steps until:
    - 100 solvable states and 100 unsolvable states
  - Returns shuffled list of `{state, is_solvable, is_breaking_point, deadlock_type}`

#### Step 2: Evaluate Model
- **`evaluate_model(model, tokenizer, eval_samples, system_prompt)`** (`evaluate_rl.py:133`)
  - For each eval sample:
    1. Builds single-turn prompt: `[system, "Current state:\n{grid}"]`
    2. Applies chat template and tokenizes
    3. Runs `model.generate(max_new_tokens=512, temperature=0.1, do_sample=False)` (greedy)
    4. Decodes generated response
    5. Parses predictions via **`parse_predictions(text)`** (`evaluate_rl.py:32`)
    6. Updates confusion matrices:
       - Solvable: TP/FP/FN/TN
       - Breaking point: TP/FP/FN/TN
       - Per-deadlock-type recall

#### Step 3: Compute and Print Results
- **`_compute_results(metrics)`** (`evaluate_rl.py:245`)
  - Format compliance rates
  - Solvable accuracy, precision, recall, F1
  - Breaking point accuracy, precision, recall, F1
  - Per-deadlock-type recall breakdown

- If both SFT and RL models evaluated: prints side-by-side comparison table

---

## Data Flow Diagram

```
                              PHASE 1: DATA GEN
                              =================

SudokuEnv.reset(seed)
    |
    v
generate_sudoku_puzzle() --> (puzzle_grid, solution_grid)
    |
    v
[Step Loop] --------> SudokuEnv.step(action)
    |                      |
    |    Random:           |    LLM:
    |    random.choice()   |    model.generate() -> parse <answer>
    |                      |
    |                      v
    |              check_solvability()
    |                |-> find_conflicts()
    |                |-> get_valid_numbers() per empty cell
    |                |-> _propagate() constraint propagation
    |                |-> _backtrack() bounded search
    |                      |
    |                      v
    |              TrajectoryStep(state, action, is_solvable, is_breaking_point, ...)
    |                      |
    v                      v
[Post-hoc annotation: done_label, steps_left, steps_left_bucket]
    |
    v
SFTFormatter._format_multi_turn()
    |-> For each step K: build [sys, user_0, asst_0, ..., user_K]
    |-> Ground-truth response: format_step() -> XML tags
    |
    v
create_sft_dataset() -> DataFrame
    |
    v
save_dataset() -> wm_train.parquet + wm_val.parquet


                              PHASE 2: SFT
                              ============

load_and_process_data(parquet)
    |-> Parse messages (system/user/assistant roles)
    |-> Build ChatML prompt: <|im_start|>role\ncontent<|im_end|>
    |-> Tokenize full text
    |-> Labels: -100 for prompt, token_ids for response
    |
    v
HuggingFace Trainer.train()
    |-> Forward: model(input_ids, labels) -> cross_entropy on non-(-100) tokens
    |-> AdamW + cosine LR + gradient accumulation
    |-> Save best checkpoint by eval_loss
    |
    v
outputs/sft_sudoku/final/ (fine-tuned Qwen2.5-1.5B-Instruct)


                              PHASE 3: RL
                              ===========

LiveTrajectorySampler._fill_pool()
    |-> Generate 200 random trajectories
    |-> Index steps: solvable / unsolvable / breaking_point pools
    |
    v
[RL Loop] ---------> sample_batch()
    |                    |-> ~50% solvable + ~50% unsolvable (30% BP focus)
    |                    |-> Tokenize with chat template
    |                    v
    |              model.generate() on batch
    |                    |
    |                    v
    |              parse_termination_predictions(response)
    |                    |
    |                    v
    |              compute_termination_reward_v2()
    |                |-> Format: +0.1/tag (max +0.5)
    |                |-> Solvable: +0.5 correct, -0.25 wrong
    |                |-> BP: +3.0 TP, -2.0 FN, -0.5 FP
    |                    |
    |                    v
    |              compute_grpo_outcome_advantage()
    |                |-> score = sum(rewards)
    |                |-> normalize within prompt group
    |                    |
    |                    v
    |              _update_actor()
    |                |-> loss = -mean(log_prob * advantage, response_mask)
    |                |-> clip_grad_norm, optimizer.step
    |
    v
outputs/rl_sudoku/step_N/ (RL-tuned model)
```

---

## Key File Summary

| File | Role | Key Functions |
|------|------|---------------|
| `src/environments/sudoku_utils.py` | Low-level Sudoku logic | `generate_sudoku_puzzle()`, `get_valid_numbers()`, `is_valid_placement()`, `SudokuSolvabilityChecker.check_solvability()` |
| `src/environments/sudoku.py` | Sudoku gym-like environment | `SudokuEnv.reset()`, `.step()`, `.get_all_actions()`, `.check_solvability()` |
| `src/data/trajectory_generator.py` | Random-play trajectories | `TrajectoryGenerator.generate_random_trajectory()`, `.generate_balanced_dataset()`, `TrajectoryStep`, `TrajectoryMetadata` |
| `src/data/llm_trajectory_generator.py` | LLM-policy trajectories | `LLMTrajectoryGenerator._generate_response()`, `._parse_action_from_response()`, `.generate_trajectory()` |
| `src/data/sft_formatter.py` | Format trajectories for SFT | `SFTFormatter.format_step()`, `._format_multi_turn()`, `.create_sft_dataset()`, `.save_dataset()` |
| `src/data/live_trajectory_sampler.py` | Balanced RL batches | `LiveTrajectorySampler._fill_pool()`, `.sample_batch()`, `RLSample` |
| `src/training/simple_sft_trainer.py` | SFT training | `load_and_process_data()`, `SFTDataCollator`, HuggingFace `Trainer` |
| `src/training/rl_trainer.py` | RL training | `compute_termination_reward_v2()`, `LiveEnvTerminationRLTrainer._generate_rollouts()`, `._update_actor()` |
| `evaluate_rl.py` | Balanced evaluation | `generate_balanced_eval_set()`, `evaluate_model()`, confusion matrices |
