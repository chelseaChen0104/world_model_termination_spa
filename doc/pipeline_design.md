# Pipeline Design — Data Gen → SFT → RL → Eval

Operational guide. For research rationale see [SPEC.md](SPEC.md); for architecture see [CLAUDE.md](../CLAUDE.md); for chronological history see [progress.md](../progress.md).

**Visual version:** open [pipeline_design.html](pipeline_design.html) in a browser for an interactive flow chart with click-through node details. Three tabs: Pipeline Overview, Format Flows (the three format concerns), Eval & Baselines (the comparison matrix).

This doc is a **runbook**: what to run in what order, what each step produces, what success looks like at each stage. Sudoku-only per [SPEC.md](SPEC.md) v3. Components not yet implemented are flagged 🚧.

---

## 0. Pipeline overview

```
┌──────────────────────┐    ┌──────────────────────┐
│ Stage 1A: random-play│    │ Stage 1B: LLM-policy │
│   multi-turn data    │    │   multi-turn data    │
│   (CPU, ~1 min)      │    │   (H800, 1.5–3 hr)   │
│                      │    │                      │
│ data/sudoku_multiturn│    │ data/sudoku_llm_policy│
└──────────┬───────────┘    └──────────┬───────────┘
           │                           │
           │  warm-start path          │  headline path
           ▼                           ▼
       ┌───────────────────────────────────────┐
       │ Stage 2: SFT (Qwen2.5-1.5B-Instruct)  │
       │   simple_sft_trainer.py               │
       │   ~hours on H800                      │
       │                                       │
       │ outputs/sft_sudoku{,_llm_policy}/     │
       └───────────────────┬───────────────────┘
                           │
                           ▼
       ┌───────────────────────────────────────┐
       │ Stage 3: RL (LiveEnvTerminationRLTrainer)│
       │   rl_trainer.py + rl_sudoku.yaml      │
       │   reward v2, balanced live sampling   │
       │   ~hours on H800                      │
       │                                       │
       │ outputs/rl_sudoku/                    │
       └───────────────────┬───────────────────┘
                           │
                           ▼
       ┌───────────────────────────────────────┐
       │ Stage 4: Evaluation (evaluate_rl.py)  │
       │ + heuristic baseline + SPA baselines  │
       │ + Pass@1 / Pass@8                     │
       │   confusion matrices, F1, BP recall   │
       └───────────────────────────────────────┘
```

---

## 1. Prerequisites (one-time, on AutoDL)

Already done in this session — listed for reproducibility.

```bash
# 1. SSH alias (~/.ssh/config) → autodl key auth
ssh autodl 'echo OK'

# 2. Repo on data disk
ls /root/autodl-tmp/world_model_termination_spa/

# 3. Caches redirected to data disk (HF, torch, pip)
readlink /root/.cache/huggingface  # → /root/autodl-tmp/cache/huggingface

# 4. Env vars in ~/.bashrc + scripts/_run_with_env.sh
#    PATH=/root/miniconda3/bin:$PATH
#    network_turbo proxy
#    HF_HUB_DISABLE_XET=1   ← disables HF's xethub backend (returns 401 via proxy)

# 5. Python deps installed
ssh autodl '/root/miniconda3/bin/pip list | grep -E "torch|transformers|pandas|pyarrow|accelerate|datasets|sklearn"'
# Expected: torch 2.8+cu128, transformers 4.46.3, pandas, pyarrow, accelerate 1.13, datasets 4.8, scikit-learn

# 6. tmux for detached job runs
ssh autodl 'which tmux'
```

---

## 2. Stage 1: Data Generation

Two parallel tracks. Both write parquet files to `data/<name>/wm_train.parquet` and `wm_val.parquet`.

### 2.1 Stage 1A — Random-play multi-turn (CPU, fast intermediate)

**Purpose:** quick warm-start data + pipeline smoke test. Acknowledged off-distribution per [SPEC.md](SPEC.md) §4.

**Script:** [`scripts/regen_random_multiturn.py`](../scripts/regen_random_multiturn.py)

**Pipeline inside the script:**
```python
SudokuEnv(grid_size=9, difficulty='easy', max_steps=30)
    └─> TrajectoryGenerator(env)
        └─> generate_balanced_dataset(target_size=1280, failure_ratio=1.0, seed=42)
            └─> 1280 random-play trajectories, mean length 12.2 steps
                └─> SFTFormatter(variant='sudoku_full', multi_turn=True, max_context_turns=10)
                    └─> create_sft_dataset()  # ~32k per-step samples
                        └─> save_dataset('data/sudoku_multiturn', split_ratio=0.2)
```

**Run on AutoDL (in tmux, so SSH drops don't kill it):**
```bash
ssh autodl '
cd /root/autodl-tmp/world_model_termination_spa
mkdir -p logs
tmux new-session -d -s trackA "bash -lc \"bash scripts/_run_with_env.sh python -u scripts/regen_random_multiturn.py 2>&1 | tee logs/trackA.log\""
'
```

**Expected output:** `data/sudoku_multiturn/{wm_train,wm_val}.parquet` — ~32,062 samples (25,649 train / 6,413 val).

**Verification:**
```bash
ssh autodl '/root/miniconda3/bin/python -c "
import pandas as pd, json
df = pd.read_parquet(\"/root/autodl-tmp/world_model_termination_spa/data/sudoku_multiturn/wm_train.parquet\")
print(df.shape)
infos = [json.loads(x) if isinstance(x,str) else x for x in df.extra_info.iloc[:5000]]
print(\"solvable:\", sum(1 for i in infos if i[\"is_solvable\"])/len(infos))
print(\"bp:      \", sum(1 for i in infos if i[\"is_breaking_point\"])/len(infos))
"'
```

**Expected:** shape `(25649, 6)`, solvable ~6%, bp ~4%.

**Wall time:** ~1 minute. **CPU only.**

### 2.2 Stage 1B — LLM-policy multi-turn (GPU, headline data)

**Purpose:** in-distribution training data per SPA's self-play recipe. The model whose RL we eventually run is the same Qwen2.5-1.5B that generated this data.

**Script:** [`scripts/generate_llm_policy_data_gpu.sh`](../scripts/generate_llm_policy_data_gpu.sh) → `python -m src.data.llm_trajectory_generator`

**Pipeline inside the script:**
```python
LLMTrajectoryGenerator(model="Qwen/Qwen2.5-1.5B-Instruct", temperature=0.7, device="auto")
    └─> generate_balanced_dataset(target_size=1280, failure_ratio=1.0, seed=42, max_steps=30)
        └─> for each trajectory:
              for each step:
                # System prompt asks the LLM for the FULL designed format:
                # <think><observation>…</observation><prediction>…</prediction>
                #   <steps_left>…</steps_left><solvable>…</solvable><breaking_point>…</breaking_point></think>
                # <answer>place N at row R col C</answer>
                #
                # In practice the BASE Qwen produces variable quality (no SFT yet).
                # We extract ONLY <answer> for the action; if unparseable → random fallback.
                # The full raw response is stored in step.llm_raw_response for use as
                # prior-turn context in multi-turn SFT samples.
                response = LLM.generate(state, system_prompt=full_format_prompt)
                action = parse_answer_tag(response) or random_fallback()
                env.step(action) → ground-truth next state, is_solvable, is_breaking_point
                record TrajectoryStep(state, action, ground-truth labels, llm_raw_response=response)

        └─> SFTFormatter(variant='sudoku_full', multi_turn=True, max_context_turns=10)
            # Target row construction: format_step() replaces LLM hallucinations with
            # ground-truth content, matching SPA's recipe (Chen et al. 2025 §2.2).
            #   <observation> ← step.state           (env, not LLM)
            #   <prediction>  ← step.next_state      (env, not LLM)
            #   <solvable>    ← step.is_solvable     (oracle)
            #   <breaking_point> ← step.is_breaking_point   (oracle)
            #   <steps_left>  ← step.steps_left_bucket      (oracle)
            #   <answer>      ← step.action_name     (what was executed)
            #
            # Multi-turn prior assistant turns use step.llm_raw_response if present,
            # else template-generated ground truth (see SPEC.md §7.5 format constraints).
            └─> save to data/sudoku_llm_policy/
```

**Three format concerns to keep separate (see [SPEC.md](SPEC.md) §7.5):**
1. **LLM generation output** during data gen — only `<answer>` must be parseable; rest is forgiving.
2. **SFT target row** — full XML, **all content from env ground truth**, not from LLM output.
3. **Multi-turn prior-turn content** — currently uses LLM raw response (may be malformed). Open research decision; see SPEC.md.

**Run on AutoDL:**
```bash
ssh autodl '
cd /root/autodl-tmp/world_model_termination_spa
mkdir -p logs
tmux new-session -d -s trackB "bash -lc \"bash scripts/_run_with_env.sh bash scripts/generate_llm_policy_data_gpu.sh 2>&1 | tee logs/trackB.log\""
'
```

**Expected output:** `data/sudoku_llm_policy/{wm_train,wm_val}.parquet` — ~30k samples (varies; LLM survives longer per trajectory than random play).

**Class distribution will differ from Track A** — fewer breaking points proportionally because the LLM survives longer, BPs distributed across the trajectory rather than clustered at steps 0–10.

**Wall time:** 1.5–3 hours on H800.

**Common failure modes:**
- `_PyImport_Init: global import state already initialized` → root cause is HF xethub returning 401 through the AutoDL proxy. Fixed by `HF_HUB_DISABLE_XET=1` in `_run_with_env.sh`.
- HF download timeout → retry; the model is ~3GB and the proxy can be flaky. After first successful download the model is cached at `/root/autodl-tmp/cache/huggingface/`.
- Long silence in log with active GPU util → expected. Python stdout is block-buffered when piped through `tee`; progress prints flush every ~50 trajectories.

---

## 3. Stage 2: SFT Training

**Purpose:** cold-start the policy via supervised finetuning on world-model + termination tags. Matches SPA's recipe; we extend the tag set with `<solvable>`/`<breaking_point>`/`<steps_left>`.

**Script:** [`src/training/simple_sft_trainer.py`](../src/training/simple_sft_trainer.py)

**Per [SPEC.md](SPEC.md) §7 locked decisions:**
- HuggingFace `Trainer` (single-GPU; FSDP `sft_trainer.py` is dead code).
- Loss: cross-entropy on response tokens only. For multi-turn: only the **final** assistant turn has real labels; prior assistant turns in the prompt are masked with `-100`.
- Base model: `Qwen/Qwen2.5-1.5B-Instruct`.

### 3.1 SFT on random-play data (warm-start)

```bash
ssh autodl '
cd /root/autodl-tmp/world_model_termination_spa
mkdir -p outputs logs
tmux new-session -d -s sft_a "bash -lc \"bash scripts/_run_with_env.sh python src/training/simple_sft_trainer.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --train_file data/sudoku_multiturn/wm_train.parquet \
  --val_file data/sudoku_multiturn/wm_val.parquet \
  --output_dir outputs/sft_sudoku \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --max_length 4096 \
  2>&1 | tee logs/sft_a.log\""
'
```

**Expected output:** `outputs/sft_sudoku/checkpoint-N/` (~3 GB, fp16/bf16 weights).

**Wall-time estimate on H800:** ~2–4 hours for 3 epochs over 25k samples (effective batch 32, ~2400 steps).

**Verification:**
- Training loss curve in `logs/sft_a.log` (decreasing, no NaN spikes).
- Validation loss at end of each epoch (should drop, plateau by epoch 2–3).
- Checkpoint loads cleanly: `AutoModelForCausalLM.from_pretrained("outputs/sft_sudoku/checkpoint-N")`.

### 3.2 SFT on LLM-policy data (headline)

Same command, just point at `data/sudoku_llm_policy/` and `outputs/sft_sudoku_llm_policy/`. Run after Stage 1B completes.

```bash
# After Stage 1B finishes:
ssh autodl '
cd /root/autodl-tmp/world_model_termination_spa
tmux new-session -d -s sft_b "bash -lc \"bash scripts/_run_with_env.sh python src/training/simple_sft_trainer.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --train_file data/sudoku_llm_policy/wm_train.parquet \
  --val_file data/sudoku_llm_policy/wm_val.parquet \
  --output_dir outputs/sft_sudoku_llm_policy \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --max_length 4096 \
  2>&1 | tee logs/sft_b.log\""
'
```

**This is the SFT model used downstream** in Stage 3. Stage 3.1's random-play SFT is for ablation comparison only.

---

## 4. Stage 3: RL Training

**Purpose:** lift breaking-point recall above what SFT alone can deliver, using asymmetric reward shaping and balanced live-env sampling.

**Script:** [`src/training/rl_trainer.py`](../src/training/rl_trainer.py) with class `LiveEnvTerminationRLTrainer`

**Config:** [`src/training/config/rl_sudoku.yaml`](../src/training/config/rl_sudoku.yaml)

### 4.1 What the trainer does

```python
LiveEnvTerminationRLTrainer(
    sft_checkpoint = "outputs/sft_sudoku_llm_policy/checkpoint-N",
    env = SudokuEnv(grid_size=9, difficulty='easy', max_steps=30),
    sampler = LiveTrajectorySampler(env, balance_ratio=0.5),  # ~50/50 solvable/unsolvable per batch
    reward_fn = compute_termination_reward_v2,  # asymmetric, see below
    rl_alg = "GRPO",  # group-relative PPO, single-GPU friendly
    ...
)
```

### 4.2 Reward v2 (compute_termination_reward_v2)

Per [SPEC.md](SPEC.md) §7 locked:

| Component | Value | Rationale |
|---|---|---|
| BP true positive (predicted=true, label=true) | **+3.0** | Catching deadlocks is the high-value signal |
| BP false negative (predicted=false, label=true) | **−2.0** | Missing a deadlock is costly |
| BP false positive (predicted=true, label=false) | **−0.5** | Mild penalty so the model can over-predict slightly |
| Solvable correct | **+0.5** | Standard correct-prediction reward |
| Solvable wrong | **−0.25** | Standard wrong-prediction penalty |
| Format compliance per XML tag present | **+0.1** | Prevents tag-format degradation during RL |
| `steps_left` | **0** | Excluded from RL reward (SFT-only signal) |

### 4.3 Run

```bash
ssh autodl '
cd /root/autodl-tmp/world_model_termination_spa
tmux new-session -d -s rl "bash -lc \"bash scripts/_run_with_env.sh python src/training/rl_trainer.py \
  --config src/training/config/rl_sudoku.yaml \
  --sft_checkpoint outputs/sft_sudoku_llm_policy/checkpoint-N \
  --output_dir outputs/rl_sudoku \
  2>&1 | tee logs/rl.log\""
'
```

**Expected output:** `outputs/rl_sudoku/checkpoint-N/`.

**Wall time on H800:** highly dependent on config; expect **multiple hours**.

**Verification (per [SPEC.md](SPEC.md) §6 anti-goals):**
- BP recall on eval *increasing* during training, not just total reward (otherwise model is reward-hacking).
- Pass@1 on Sudoku not collapsing (Q2 trip-wire — termination training shouldn't break the SPA agent).
- Format compliance staying ≥99% (XML tags present in nearly all outputs).

---

## 5. Stage 4: Evaluation

### 5.1 Termination metrics — already implemented

**Script:** [`evaluate_rl.py`](../evaluate_rl.py)

```bash
ssh autodl '
cd /root/autodl-tmp/world_model_termination_spa
bash scripts/_run_with_env.sh python evaluate_rl.py \
  --n-solvable 100 \
  --n-unsolvable 100 \
  --sft-path outputs/sft_sudoku_llm_policy/checkpoint-N \
  --rl-path outputs/rl_sudoku/checkpoint-N \
  2>&1 | tee logs/eval.log
'
```

**Reports:**
- Confusion matrices for solvability + breaking-point predictions
- Precision, recall, F1 for each
- Per-deadlock-type recall (zero_candidates, constraint_propagation, no_solution)
- Comparison: SFT vs RL vs (with `--baseline`) heuristic [`sudoku_baseline.py`](../src/evaluation/sudoku_baseline.py)

### 5.2 SPA-comparable metrics — 🚧 not yet implemented

Per [SPEC.md](SPEC.md) §3 success criteria, headline reporting must include **Pass@1 and Pass@8** for direct comparability with SPA Table 2.

**To add:**
- `evaluate_rl.py --metric pass-at-k --k 1,8 --rollouts-per-puzzle 8` mode
- Generate K rollouts per puzzle, count fraction that solve the puzzle
- Output: Pass@1, Pass@8 numbers comparable to SPA Table 2's Sudoku column (Qwen2.5-1.5B-Instruct row)

### 5.3 SPA paper baselines — 🚧 not yet implemented

Per [SPEC.md](SPEC.md) §3, the headline result must compare against:

1. **Vanilla RL** — base Qwen2.5-1.5B-Instruct + RL with task reward only, no world-model SFT, no termination tags. (Re-implement SPA's "Vanilla RL" row.)
2. **State-Estimation-only RL** — SFT with `<observation>` tags only (no `<prediction>`, no termination tags), then RL. (SPA's "State Estimation RL" row.)
3. **VAGEN-style** — online world modeling during RL, no SFT cold-start. (SPA's VAGEN row.)

Each baseline is a separate Stage 2/3 run with the appropriate ablation. Estimate ~3–6 GPU-hours per baseline.

### 5.4 Output of a complete eval

Single `doc/eval_<date>_<topic>.md` per the [doc folder convention](../doc/), containing:

| Model | Pass@1 | Pass@8 | Solvability F1 | BP recall | BP precision | BP F1 |
|---|---|---|---|---|---|---|
| Heuristic baseline | n/a | n/a | low | low | high | low |
| Vanilla RL | from SPA Table 2 | from SPA Table 2 | low | low | low | low |
| State-Estimation RL | from SPA Table 2 | from SPA Table 2 | mid | low–mid | mid | low–mid |
| VAGEN | from SPA Table 2 | from SPA Table 2 | mid | low–mid | mid | low–mid |
| **Random-play SFT** (ours) | should preserve | — | mid | mid (early-game biased) | low | low–mid |
| **LLM-policy SFT** (ours, headline pre-RL) | should preserve | — | mid–high | mid | mid | mid |
| **LLM-policy SFT + RL** (ours, final) | match/beat SPA | match/beat SPA | high | high | mid | mid–high |

Numbers in the right four columns are *targets*, not commitments. The first two columns must match SPA's published numbers within reproduction noise to be credible.

---

## 6. End-to-end command sequence

Once data is generated and checkpoint paths are known, the happy path is:

```bash
# 1. Data — already running in tmux
ssh autodl 'tmux ls'
# trackA: done (data/sudoku_multiturn/)
# trackB: in progress (data/sudoku_llm_policy/)

# 2. Sync down when each track finishes (auto)
bash scripts/sync-down.sh

# 3. SFT on Track B (after trackB completes)
ssh autodl 'tmux new -d -s sft_b "..."'   # see §3.2

# 4. RL on top of SFT (after sft_b completes)
ssh autodl 'tmux new -d -s rl "..."'      # see §4.3

# 5. Eval all checkpoints
ssh autodl 'bash scripts/_run_with_env.sh python evaluate_rl.py ...'  # see §5.1

# 6. Add SPA baselines + Pass@1/Pass@8 (🚧 implement first)
# 7. Write doc/eval_<date>_<topic>.md with the comparison table
```

---

## 7. Build queue (what's missing for the headline result)

Ordered by load-bearing-ness for [SPEC.md](SPEC.md) §3 success criteria:

1. **Pass@1 / Pass@8 mode in `evaluate_rl.py`** — without this, no SPA comparability. Highest priority.
2. **Vanilla RL baseline run** — straightforward; existing `rl_trainer.py` with reward zeroed except task success.
3. **State-Estimation-only SFT baseline** — Track B with `<prediction>` and termination tags removed from the formatter; existing `simple_sft_trainer.py` works as-is.
4. **VAGEN baseline** — most work; requires implementing online world-modeling reward during RL. Cite SPA's VAGEN reimplementation if reproducible from RAGEN repo.
5. **Heuristic-baseline-only run for ablation column** — `evaluate_rl.py --baseline-only` path, computes metrics on `sudoku_baseline.py` predictions over the same eval set.

Items 1–3 are required for the first paper-quality result; item 4 is required for full SPA comparability; item 5 is a nice-to-have ablation column.
