"""B-H1 stochastic Pass@k sweep: confirm whether the SFT model has ANY
positive signal in its action policy or whether it's truly stuck at 0.

For each (temperature, k) cell, sample k stochastic rollouts per puzzle and
report:
  - Pass@1  (mean fraction of puzzles where rollout 0 solved)
  - Pass@k  (mean fraction of puzzles where ANY of k rollouts solved)
  - solvable_acc per-cell
  - bp_recall per-cell

Usage:
  python scripts/eval_hidato_stochastic_pass.py \
      --sft-path outputs/sft_hidato_b_h1/final \
      --temperatures 0.7 1.0 1.3 \
      --k 8 \
      --n-puzzles 8

Defaults to the 8-puzzle Hidato bank (1× per puzzle).
"""
from __future__ import annotations

import argparse
import os
import sys

# Repo path for src.* imports
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.environments.hidato import HidatoEnv
from src.data.sft_formatter import SFTFormatter
from src.training.rl_trainer_v6 import (
    RLConfig, do_rollout,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sft-path", required=True)
    p.add_argument("--temperatures", nargs="+", type=float,
                   default=[0.0, 0.7, 1.0, 1.3])
    p.add_argument("--k", type=int, default=8,
                   help="Number of stochastic samples per puzzle.")
    p.add_argument("--n-puzzles", type=int, default=8,
                   help="Distinct puzzle seeds to evaluate (cycles through bank).")
    p.add_argument("--max-rollout-steps", type=int, default=20)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== B-H1 stochastic Pass@k sweep ===")
    print(f"  SFT path:       {args.sft_path}")
    print(f"  temperatures:   {args.temperatures}")
    print(f"  k samples:      {args.k}")
    print(f"  n puzzles:      {args.n_puzzles}")
    print(f"  device:         {device}")
    print()

    # Load model + tokenizer
    print("Loading tokenizer + model...")
    tokenizer = AutoTokenizer.from_pretrained(args.sft_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.sft_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()

    env = HidatoEnv(max_steps=args.max_rollout_steps)
    formatter = SFTFormatter(variant="hidato_minimal")
    system_prompt = formatter.system_prompt

    cfg = RLConfig(
        sft_checkpoint=args.sft_path,
        output_dir="/tmp/eval_dummy",
        max_rollout_steps=args.max_rollout_steps,
        n_total_steps=0,
    )

    print(f"\n{'temp':>6} | {'pass@1':>7} | {'pass@k':>7} | {'solvable_acc':>12} | {'bp_recall':>9} | {'avg_len':>7}")
    print("-" * 65)

    for temp in args.temperatures:
        cfg.temperature = temp
        n_solve_at_1 = 0
        n_solve_at_k = 0
        sol_correct = 0
        sol_total = 0
        n_bp_caught = 0
        n_bp_total = 0
        total_steps = 0
        n_rollouts = 0
        for i in range(args.n_puzzles):
            seed_base = 200000 + i  # distinct from training and eval-greedy seeds
            any_solved = False
            for j in range(args.k):
                seed = seed_base * 100 + j
                ro = do_rollout(model, tokenizer, env, system_prompt, seed, cfg, device)
                n_rollouts += 1
                total_steps += len(ro.steps)
                if ro.is_solved:
                    if j == 0:
                        n_solve_at_1 += 1
                    any_solved = True
                for s in ro.steps:
                    if s.pred_solvable is not None:
                        sol_total += 1
                        if s.pred_solvable == s.gt_solvable:
                            sol_correct += 1
                    if s.is_breaking_point:
                        n_bp_total += 1
                        if s.pred_solvable is False:
                            n_bp_caught += 1
            if any_solved:
                n_solve_at_k += 1
        pass1 = n_solve_at_1 / max(1, args.n_puzzles)
        passk = n_solve_at_k / max(1, args.n_puzzles)
        sa = sol_correct / max(1, sol_total)
        bpr = n_bp_caught / max(1, n_bp_total) if n_bp_total else 0.0
        avg_len = total_steps / max(1, n_rollouts)
        print(f"{temp:>6.2f} | {pass1:>7.3f} | {passk:>7.3f} | {sa:>12.3f} | {bpr:>9.3f} | {avg_len:>7.2f}")

    print()
    print(f"Note: Pass@k samples k={args.k} rollouts per puzzle; Pass@1 uses only the FIRST sample.")


if __name__ == "__main__":
    main()
