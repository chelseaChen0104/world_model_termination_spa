"""Greedy Pass@1 eval for a base policy π_θ.

Two levels:
  (a) Per-step accuracy on a held-out (state, action) val set:
      - format_ok: response parses to ActionStruct via env regex
      - local_valid: action is locally legal at the prompt state
      - exact_match: action == oracle solver's action at this step
  (b) Full-episode Pass@1 by subset:
      - From an empty board with each subset, greedy-rollout the policy
      - Episode succeeds iff the board is fully tiled (is_goal) within K_pieces steps
      - Report success rate per subset and aggregate.

Usage:
  python scripts/eval_pi_theta_pass1.py \\
      --model /tmp/sft_pentomino5x6_pi_theta/final \\
      --val   data/pentomino5x6/pi_theta_sft/val.jsonl \\
      --board-h 5 --board-w 6 --k-pieces 6 \\
      --episodes-per-subset 1 \\
      --max-val-samples 200
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.pentomino_env import (
    ActionStruct, render_state_b8, parse_action_text,
    is_local_valid, apply_action, is_goal, empty_board,
)
from scripts.pentomino_solver import PentominoSolver
from scripts.generate_pi_theta_sft_pentomino import (
    SYSTEM_PROMPT, render_user_message, find_valid_subsets,
)


def greedy_generate(model, tok, prompt: str, max_new_tokens: int = 32) -> str:
    """Single greedy decode."""
    inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


def per_step_eval(model, tok, samples: list, max_n: int) -> dict:
    """Loop over (state, oracle_action) pairs; greedy-decode and score."""
    n_format_ok = 0
    n_local_valid = 0
    n_exact = 0
    n_total = 0
    per_step_acc: Counter = Counter()
    per_step_total: Counter = Counter()

    for s in samples[:max_n]:
        prompt = s["prompt"]  # already chat-template-formatted
        oracle = s["response"]
        step = s["metadata"]["step"]
        board = s["metadata"]["board_at_step"]
        remaining = s["metadata"]["remaining_pieces_at_step"]

        gen = greedy_generate(model, tok, prompt)
        # Strip trailing junk after first newline if any
        first_line = gen.split("\n")[0].strip()
        action = parse_action_text(first_line)
        n_total += 1
        per_step_total[step] += 1

        if action is not None:
            n_format_ok += 1
            if is_local_valid(board, remaining, action):
                n_local_valid += 1
                per_step_acc[step] += 1
            if first_line == oracle:
                n_exact += 1
        else:
            # try parsing the whole gen
            action = parse_action_text(gen)
            if action is not None:
                n_format_ok += 1
                if is_local_valid(board, remaining, action):
                    n_local_valid += 1
                    per_step_acc[step] += 1

    return {
        "n_total": n_total,
        "format_ok_rate": n_format_ok / max(n_total, 1),
        "local_valid_rate": n_local_valid / max(n_total, 1),
        "exact_match_rate": n_exact / max(n_total, 1),
        "per_step_local_valid_rate": {
            t: per_step_acc[t] / per_step_total[t] for t in sorted(per_step_total)
        },
    }


def episode_pass1(model, tok, board_h: int, board_w: int, subset: tuple,
                    k_pieces: int) -> Tuple[bool, int, str]:
    """Greedy-rollout from empty board with `subset` pieces.
    Returns (succeeded, steps_taken, fail_reason)."""
    board = empty_board(board_h, board_w)
    remaining = list(subset)
    for step in range(k_pieces):
        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{render_user_message(board, remaining)}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        gen = greedy_generate(model, tok, prompt)
        first_line = gen.split("\n")[0].strip()
        action = parse_action_text(first_line) or parse_action_text(gen)
        if action is None:
            return False, step, f"parse_fail: {gen[:60]!r}"
        if not is_local_valid(board, remaining, action):
            return False, step, f"local_invalid: {first_line!r}"
        board, remaining = apply_action(board, remaining, action)
    if is_goal(board):
        return True, k_pieces, "ok"
    return False, k_pieces, "incomplete"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--board-h", type=int, default=5)
    p.add_argument("--board-w", type=int, default=6)
    p.add_argument("--k-pieces", type=int, default=6)
    p.add_argument("--max-val-samples", type=int, default=200)
    p.add_argument("--episodes-per-subset", type=int, default=1)
    p.add_argument("--max-subsets", type=int, default=172,
                   help="Cap number of subsets evaluated for episode Pass@1")
    p.add_argument("--output", default=None,
                   help="Optional path to save the JSON results")
    args = p.parse_args()

    print(f"=== π_θ greedy eval ===")
    print(f"  model: {args.model}")
    print(f"  val:   {args.val}")
    print(f"  board: {args.board_h}x{args.board_w}, k_pieces={args.k_pieces}")

    print("\nLoading model...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).cuda().eval()

    print(f"\n--- Per-step eval (up to {args.max_val_samples} samples) ---")
    samples = []
    with open(args.val) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    step_results = per_step_eval(model, tok, samples, args.max_val_samples)
    print(f"  n_total: {step_results['n_total']}")
    print(f"  format_ok:    {step_results['format_ok_rate']:.3f}")
    print(f"  local_valid:  {step_results['local_valid_rate']:.3f}")
    print(f"  exact_match:  {step_results['exact_match_rate']:.3f}")
    print(f"  local_valid by step: {step_results['per_step_local_valid_rate']}")

    print(f"\n--- Episode Pass@1 (up to {args.max_subsets} subsets, "
          f"{args.episodes_per_subset} episodes each) ---")
    valid_subsets = find_valid_subsets(args.board_h, args.board_w, args.k_pieces)[:args.max_subsets]
    succ_count = 0
    total_episodes = 0
    fail_reasons: Counter = Counter()
    per_subset_results = []
    for i, subset in enumerate(valid_subsets):
        sub_succ = 0
        for _ in range(args.episodes_per_subset):
            ok, steps, reason = episode_pass1(model, tok, args.board_h, args.board_w,
                                                subset, args.k_pieces)
            total_episodes += 1
            if ok:
                succ_count += 1
                sub_succ += 1
            else:
                fail_reasons[reason.split(":")[0]] += 1
        per_subset_results.append({"subset": ",".join(subset),
                                     "succ": sub_succ,
                                     "n_eps": args.episodes_per_subset})
        if (i + 1) % 20 == 0:
            print(f"  ...{i+1}/{len(valid_subsets)} subsets, "
                  f"running Pass@1 = {succ_count/total_episodes:.3f}")
    pass1 = succ_count / max(total_episodes, 1)
    print(f"\n  Pass@1: {pass1:.3f} ({succ_count}/{total_episodes})")
    print(f"  fail reasons: {dict(fail_reasons)}")

    out = {
        "model": args.model,
        "val": args.val,
        "board_h": args.board_h, "board_w": args.board_w, "k_pieces": args.k_pieces,
        "per_step": step_results,
        "episode_pass1": {
            "pass1": pass1, "succ_count": succ_count, "n_episodes": total_episodes,
            "fail_reasons": dict(fail_reasons),
            "per_subset": per_subset_results,
            "max_subsets_evaluated": len(valid_subsets),
        },
    }
    if args.output:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  saved: {args.output}")


if __name__ == "__main__":
    main()
