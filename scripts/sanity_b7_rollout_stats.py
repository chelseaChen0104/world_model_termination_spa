"""Sanity test: empirically measure the rollout statistics that drive the B-7 RL collapse.

Loads the B-7 SFT checkpoint and runs N batched rollouts on Pentomino-easy at the
RL-training temperature (T=0.7), then reports:

  1. Rollout length distribution — confirms whether ~90% are 1-step.
  2. First-action doom rate — fraction of first moves that lead to GT=False.
  3. Per-step GT class composition — what the gradient signal sees per batch.
  4. Pass@1 (success rate) — confirms whether success_bonus ever fires.
  5. Counterfactual expected reward analysis under v6 / v7 / v8 reward shapes
     for fixed policies (always-True, always-False, ground-truth, B-7 SFT actual).

This is the "did the collapse happen for the reasons we think it did?" check.

Usage (on autodl):
  python scripts/sanity_b7_rollout_stats.py \
      --sft-checkpoint outputs/sft_pentomino_easy_b7_spa_hparams/final \
      --n-puzzles 50 --group-size 8 --temperature 0.7 \
      --output doc/sanity_2026-04-30_b7_rollout_stats.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.environments.polyomino import PolyominoEnv
from src.data.sft_formatter import SFTFormatter
from src.training.rl_trainer_v6 import (
    RLConfig,
    do_rollouts_batched,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sft-checkpoint", required=True)
    p.add_argument("--n-puzzles", type=int, default=50)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-rollout-steps", type=int, default=12)
    p.add_argument("--board-h", type=int, default=5)
    p.add_argument("--board-w", type=int, default=4)
    p.add_argument("--piece-set", default="L,P,W,Y")
    p.add_argument("--seed", type=int, default=4242)  # different from RL training seeds
    p.add_argument("--output", default="doc/sanity_b7_rollout_stats.json")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"=== B-7 SFT rollout sanity test ===")
    print(f"  checkpoint:   {args.sft_checkpoint}")
    print(f"  n_puzzles:    {args.n_puzzles}")
    print(f"  group_size:   {args.group_size}  (rollouts/puzzle)")
    print(f"  temperature:  {args.temperature}  (matches RL training)")
    print(f"  total rollouts: {args.n_puzzles * args.group_size}")
    print(f"  device:       {device}")

    print(f"\nLoading model + tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.sft_checkpoint, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    piece_set = tuple(p_.strip().upper() for p_ in args.piece_set.split(","))
    env_template = PolyominoEnv(
        board_h=args.board_h, board_w=args.board_w,
        piece_set=piece_set, max_steps=args.max_rollout_steps,
    )
    formatter = SFTFormatter(variant="polyomino_minimal")
    system_prompt = formatter.system_prompt

    cfg = RLConfig(
        sft_checkpoint=args.sft_checkpoint,
        max_rollout_steps=args.max_rollout_steps,
        temperature=args.temperature,
        max_response_tokens=256,
        n_puzzles_per_batch=args.n_puzzles,
        group_size=args.group_size,
    )

    # Run rollouts in chunks of 4 puzzles × group_size to stay within memory.
    chunk_size = 4
    all_rollouts = []
    print(f"\nRunning {args.n_puzzles} puzzles × {args.group_size} rollouts in chunks of {chunk_size}...")
    for chunk_start in range(0, args.n_puzzles, chunk_size):
        seeds = [random.randint(0, 2**31 - 1)
                 for _ in range(min(chunk_size, args.n_puzzles - chunk_start))]
        ros = do_rollouts_batched(
            model, tokenizer, env_template, system_prompt,
            seeds, cfg, device, group_size=args.group_size,
        )
        all_rollouts.extend(ros)
        n_done = min(chunk_start + chunk_size, args.n_puzzles)
        print(f"  done {n_done}/{args.n_puzzles} puzzles ({len(all_rollouts)} rollouts)")

    # --- Compute statistics -------------------------------------------------
    n_total = len(all_rollouts)
    lengths = [len(r.steps) for r in all_rollouts]
    length_hist = Counter(lengths)

    # First-action doom rate (only counted on rollouts whose first step had a valid action)
    n_first_with_action = sum(1 for r in all_rollouts if r.steps and r.steps[0].action_was_valid)
    n_first_doom = sum(1 for r in all_rollouts
                       if r.steps and r.steps[0].action_was_valid and not r.steps[0].gt_solvable)
    first_doom_rate = n_first_doom / max(1, n_first_with_action)

    # Per-step GT class composition across ALL valid steps
    n_solv = 0
    n_doom = 0
    for r in all_rollouts:
        for s in r.steps:
            if not s.action_was_valid:
                continue
            if s.gt_solvable:
                n_solv += 1
            else:
                n_doom += 1
    n_total_valid_steps = n_solv + n_doom
    solv_frac = n_solv / max(1, n_total_valid_steps)
    doom_frac = n_doom / max(1, n_total_valid_steps)

    # Pass@1 (rollout-level success rate)
    n_solved = sum(1 for r in all_rollouts if r.is_solved)
    pass1 = n_solved / max(1, n_total)

    # Per-step viability prediction accuracy from B-7 SFT
    n_pred_correct = 0
    n_pred_total = 0
    n_pred_true = 0  # how often the model predicts True
    for r in all_rollouts:
        for s in r.steps:
            if s.pred_solvable is None:
                continue
            n_pred_total += 1
            if s.pred_solvable == s.gt_solvable:
                n_pred_correct += 1
            if s.pred_solvable:
                n_pred_true += 1

    # --- Counterfactual reward analysis -------------------------------------
    # Three reward variants on the actual states observed in the rollouts.
    # We collapse the per-step viability prediction policy and ask: under each
    # reward shape, what's the expected reward of (always-True, always-False,
    # oracle, sft_actual)?
    def per_step_reward(pred, gt, *, tp, fn, fp, tn):
        if pred is None:  # treat unparseable as worst-case
            return fn if not gt else fp
        if not pred and not gt: return tp
        if pred and not gt:     return fn
        if not pred and gt:     return fp
        return tn

    # collect (gt, sft_pred) for every valid step
    pairs = []
    for r in all_rollouts:
        for s in r.steps:
            if not s.action_was_valid:
                continue
            pairs.append((s.gt_solvable, s.pred_solvable))

    def policy_pred(gt, sft, mode):
        if mode == "always_true": return True
        if mode == "always_false": return False
        if mode == "oracle": return gt
        return sft  # "sft_actual"

    reward_variants = {
        "v6": dict(tp=+1.0, fn=-0.7, fp=-0.5, tn=+0.3),
        "v7": dict(tp=+1.0, fn=-1.0, fp=-1.0, tn=+1.0),
    }
    cf = {}
    for v_name, v_params in reward_variants.items():
        cf[v_name] = {}
        for mode in ("always_true", "always_false", "oracle", "sft_actual"):
            rs = [per_step_reward(policy_pred(gt, sft, mode), gt, **v_params)
                  for gt, sft in pairs]
            cf[v_name][mode] = round(float(np.mean(rs)) if rs else 0.0, 4)

    # --- Format and save ----------------------------------------------------
    out = {
        "config": {
            "checkpoint": args.sft_checkpoint,
            "n_puzzles": args.n_puzzles,
            "group_size": args.group_size,
            "n_rollouts": n_total,
            "temperature": args.temperature,
            "board": f"{args.board_h}x{args.board_w}",
            "piece_set": list(piece_set),
            "seed": args.seed,
        },
        "rollout_lengths": {
            "histogram": {str(k): v for k, v in sorted(length_hist.items())},
            "mean": round(float(np.mean(lengths)), 3),
            "median": int(np.median(lengths)),
        },
        "first_action_doom_rate": {
            "n_rollouts_with_first_action": n_first_with_action,
            "n_first_action_doom": n_first_doom,
            "rate": round(first_doom_rate, 4),
        },
        "per_step_class_composition": {
            "n_solv": n_solv,
            "n_doom": n_doom,
            "solv_frac": round(solv_frac, 4),
            "doom_frac": round(doom_frac, 4),
        },
        "pass_at_1": {
            "n_solved": n_solved,
            "n_total": n_total,
            "rate": round(pass1, 4),
        },
        "viability_prediction": {
            "accuracy": round(n_pred_correct / max(1, n_pred_total), 4),
            "n_total_predictions": n_pred_total,
            "frac_predicted_true": round(n_pred_true / max(1, n_pred_total), 4),
        },
        "counterfactual_expected_per_step_reward": cf,
    }

    output_path = os.path.join(_REPO_ROOT, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)

    # --- Print summary to stdout --------------------------------------------
    print(f"\n=== RESULTS ===")
    print(f"\nRollout length distribution (n={n_total}):")
    for k in sorted(length_hist):
        bar = "#" * int(50 * length_hist[k] / n_total)
        print(f"  {k:2d} steps: {length_hist[k]:4d}  {bar}")
    print(f"  mean = {out['rollout_lengths']['mean']}, median = {out['rollout_lengths']['median']}")

    print(f"\nFirst-action doom rate: {first_doom_rate*100:.1f}%  ({n_first_doom}/{n_first_with_action})")

    print(f"\nPer-step class composition (across {n_total_valid_steps} valid steps):")
    print(f"  GT=solvable: {n_solv} ({solv_frac*100:.1f}%)")
    print(f"  GT=doom:     {n_doom} ({doom_frac*100:.1f}%)")

    print(f"\nPass@1 (rollout-level): {pass1*100:.2f}%  ({n_solved}/{n_total})")

    print(f"\nB-7 SFT viability prediction:")
    print(f"  accuracy: {out['viability_prediction']['accuracy']*100:.1f}%")
    print(f"  predicted True: {out['viability_prediction']['frac_predicted_true']*100:.1f}% of the time")

    print(f"\nCounterfactual expected per-step reward by policy × reward variant:")
    print(f"{'policy':>15s} | {'v6':>8s} | {'v7':>8s}")
    for mode in ("always_true", "always_false", "oracle", "sft_actual"):
        v6_r = cf["v6"][mode]
        v7_r = cf["v7"][mode]
        print(f"  {mode:>13s} | {v6_r:+8.3f} | {v7_r:+8.3f}")
    print(f"\n--> If 'always_false' >> other policies under v6/v7, the collapse attractor is real.")
    print(f"--> If 'oracle' >> 'always_false' under both, then a calibration-anchoring fix could work without changing the env.")

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
