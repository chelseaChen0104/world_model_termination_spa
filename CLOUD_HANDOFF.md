# Cloud Handoff — Continue on AutoDL

This doc lets a fresh Claude Code session on AutoDL pick up where the local session left off. Read [progress.md](progress.md) for full history; [CLAUDE.md](CLAUDE.md) for architecture and design decisions.

## Where we are

Local-side prep is complete. The repo is on AutoDL, deps are installed, and the project imports + runs. We have NOT generated any Sudoku training data on the remote yet. The plan agreed with the user is **Option 3**: kick off random-play multi-turn data regen on CPU **and** LLM-policy data generation on the H800 in parallel, then SFT-train, eval, RL-train, eval.

## AutoDL environment (verified working)

| Item | Value |
|---|---|
| SSH | `ssh autodl` (alias defined in `~/.ssh/config`; uses dedicated key `~/.ssh/id_ed25519_autodl`). Direct form: `ssh -p <PORT> root@<HOST>`. AutoDL credentials live with the user, not in this repo. |
| Repo path | `/root/autodl-tmp/world_model_termination_spa/` |
| GPU | 1× NVIDIA H800 PCIe (CUDA 12.8) |
| Free disk | 50 GB on `/root/autodl-tmp` (data disk, persists across reboots) |
| Python | `/root/miniconda3/bin/python` (3.12) — base conda env, no other envs |
| Installed deps | torch 2.8.0+cu128, transformers 4.46.3, pandas 3.0.2, pyarrow 24.0.0, accelerate 1.13.0, datasets 4.8.5, jinja2 3.1.6, pyyaml 6.0.2, numpy 2.3.2 |
| Smoke test | `SudokuEnv` reset + `get_all_actions()` works (55 actions on seed 42 easy puzzle) |

## What's missing on the remote

- `data/` — empty. No `data/sudoku_multiturn/`, no `data/sudoku_llm_policy/`. Both must be generated here.
- `outputs/` — empty. No SFT or RL checkpoints yet.
- No `requirements.txt` / `setup.py` exist anywhere — `setup_environment.sh` references one that doesn't exist. Don't run that script; the deps above are already installed manually.

## Plan (Option 3 — parallel data tracks)

### Track A — Random-play multi-turn data (CPU, fast)
Regenerates `data/sudoku_multiturn/` (~32k samples from 1280 trajectories). No GPU needed — pure constraint solving in Python.

```bash
cd /root/autodl-tmp/world_model_termination_spa
/root/miniconda3/bin/python -c "
import sys; sys.path.insert(0, 'src')
from environments.sudoku import SudokuEnv
from data.trajectory_generator import TrajectoryGenerator
from data.sft_formatter import SFTFormatter

env = SudokuEnv(grid_size=9, difficulty='easy', max_steps=30)
gen = TrajectoryGenerator(env)
trajectories = gen.generate_balanced_dataset(
    target_size=1280, success_ratio=0.0, failure_ratio=1.0,
    timeout_ratio=0.0, max_steps=30, seed=42,
)
fmt = SFTFormatter(variant='sudoku_full')
df = fmt.create_sft_dataset(trajectories, multi_turn=True, max_context_turns=10)
fmt.save_dataset(df, 'data/sudoku_multiturn', split_ratio=0.2)
"
```

Expected output: `data/sudoku_multiturn/wm_train.parquet` (~25k rows), `wm_val.parquet` (~6k rows). Class dist: ~6.6% solvable, ~93.4% unsolvable, ~4% breaking points (per progress.md).

### Track B — LLM-policy multi-turn data (H800, slow)
Run the prebuilt script to generate in-distribution data using Qwen2.5-1.5B-Instruct as the play policy.

```bash
cd /root/autodl-tmp/world_model_termination_spa
bash scripts/generate_llm_policy_data_gpu.sh
```

This runs `src.data.llm_trajectory_generator` with `--multi-turn --max-context-turns 10`, 1280 trajectories × up to 30 steps. Output: `data/sudoku_llm_policy/`. Expected wall-time on H800: 1–3 hours (it's the first end-to-end run; benchmark on the first ~50 trajectories before assuming the full run will succeed).

Run it under `nohup` or `tmux` so an SSH drop doesn't kill it:
```bash
tmux new -s llmpolicy
bash scripts/generate_llm_policy_data_gpu.sh 2>&1 | tee llm_policy.log
# Ctrl-B, D to detach
```

## After data generation

### Step 1 — SFT on random-play data (do this as soon as Track A finishes)
```bash
cd /root/autodl-tmp/world_model_termination_spa
/root/miniconda3/bin/python src/training/simple_sft_trainer.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --train_file data/sudoku_multiturn/wm_train.parquet \
  --val_file data/sudoku_multiturn/wm_val.parquet \
  --output_dir outputs/sft_sudoku \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --max_length 4096
```

### Step 2 — Eval SFT
```bash
/root/miniconda3/bin/python evaluate_rl.py --n-solvable 100 --n-unsolvable 100 --sft-path outputs/sft_sudoku
```
Reports confusion matrices, F1, per-deadlock-type recall. Compare against `src/evaluation/sudoku_baseline.py` (heuristic) for sanity.

### Step 3 — Optional retrain SFT on LLM-policy data
After Track B finishes, retrain pointing at `data/sudoku_llm_policy/` to a separate `outputs/sft_sudoku_llmpolicy/`. Compare eval numbers.

### Step 4 — RL training
```bash
/root/miniconda3/bin/python src/training/rl_trainer.py --config src/training/config/rl_sudoku.yaml
```
Uses `LiveEnvTerminationRLTrainer` with reward v2 (asymmetric BP rewards: TP +3.0, FN -2.0, FP -0.5; format compliance +0.1/tag; balanced ~50/50 sampling from live env). Steps_left is SFT-only — not in the RL reward.

### Step 5 — Eval RL vs SFT
```bash
/root/miniconda3/bin/python evaluate_rl.py --n-solvable 100 --n-unsolvable 100 \
  --sft-path outputs/sft_sudoku --rl-path outputs/rl_sudoku
```

## Key context for the new session

- **Why Sudoku not Sokoban:** Sokoban has a simple deadlock detector that already catches most cases. Sudoku has none — unsolvable boards look valid. This is the strongest case for an LLM termination predictor. (Sokoban work is on hold.)
- **Random-play vs LLM-policy:** random play creates a state-distribution mismatch (breaking points happen too early, in cells an LLM would never play). LLM-policy gives in-distribution training and a smoother SFT→RL transition. Random-play data is the practical intermediate.
- **Reward design:** asymmetric BP rewards because catching deadlocks matters more than false alarms. Format compliance reward prevents XML-tag forgetting during RL.
- **Single-GPU GRPO**, not the SPA paper's RAGEN/Ray/vLLM stack — simpler, fits the H800.

## SSH access from local (if needed later)

The local Mac key (`~/.ssh/id_rsa`) has a passphrase, so non-interactive SSH from local fails unless the key is loaded into `ssh-agent`:
```bash
ssh-add ~/.ssh/id_rsa  # one-time per terminal session, type passphrase
```
Or use password auth via expect (see how this session ran rsync). The remote `~/.ssh/authorized_keys` has the local pubkey installed.

## Files to read first in the new session

1. [progress.md](progress.md) — full chronological log of what's been built
2. [CLAUDE.md](CLAUDE.md) — architecture, design decisions, the random-play vs LLM-policy comparison
3. [src/training/config/rl_sudoku.yaml](src/training/config/rl_sudoku.yaml) — RL hyperparams
4. [scripts/generate_llm_policy_data_gpu.sh](scripts/generate_llm_policy_data_gpu.sh) — what Track B actually runs
