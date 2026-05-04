"""Phase 0 gating decision: does rl_b5's behavior depend on the buggy prompt?

The B-5 SFT system prompt (sft_formatter.py:144) claims "numbers 1-9, 3x3 box"
but the data is 4×4 / 1-4 / 2x2. Plan_2026-05-03 §"B-5 prompt wart" defers the
prompt-version decision to this empirical check:

    1. Generate N fresh 4×4 Sudoku puzzles (n_empty=10, deterministic seeds).
    2. Run greedy Pass@1 under the BUGGY prompt (verbatim sudoku_minimal).
    3. Run greedy Pass@1 under the CORRECTED prompt (4x4-aware).
    4. Decide:
         |Pass@1_fixed - Pass@1_buggy| <= GAP_THRESHOLD  -> SHIP_FIXED
         else                                            -> SHIP_BUGGY

Output:
    - data/sudoku4/_phase0_prompt_decision.json (verdict + both numbers)
    - one-line stdout summary

Runs entirely with our local sudoku4_env + sudoku4_solver (additivity:
no touching src/environments/sudoku*.py).

Usage:
    python scripts/sanity_check_rl_b5_under_corrected_prompt.py \\
        --policy-checkpoint outputs/rl_b5_phase3_v8_anchor/final \\
        --n-puzzles 30 --seed-base 100000

Expected wall: ~10 min on H800 for n=30 (~5 placements × 30 puzzles × 2 prompts).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import List, Optional, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from scripts.sudoku4_env import (
    GRID_N, ENV_VERSION,
    render_state_b5, parse_action_text, is_local_valid, apply_action,
    is_goal, generate_root_puzzle,
)
from scripts.sudoku4_solver import Sudoku4Solver


# --- Prompts under test ---

SYSTEM_PROMPT_BUGGY = """You are solving a Sudoku puzzle. Fill in empty cells (shown as .) with numbers 1-9 so that each row, column, and 3x3 box contains each number exactly once.

Grid format: Numbers separated by spaces, | separates 3x3 boxes, - separates rows of boxes.

In your reasoning:
1. Describe the current state in <observation>
2. Predict the next state after your move in <prediction>
3. Assess whether the resulting state will still be solvable in <solvable>: true/false

Then provide your action in <answer> using format: place N at row R col C"""


SYSTEM_PROMPT_FIXED = """You are solving a 4x4 Sudoku puzzle. Fill in empty cells (shown as .) with numbers 1-4 so that each row, column, and 2x2 box contains each number exactly once.

Grid format: Numbers separated by spaces, | separates the left and right 2x2 boxes within each row, --------- separates the top and bottom row halves.

In your reasoning:
1. Describe the current state in <observation>
2. Predict the next state after your move in <prediction>
3. Assess whether the resulting state will still be solvable in <solvable>: true/false

Then provide your action in <answer> using format: place N at row R col C"""


PROMPT_VERSION_BUGGY = "sudoku_minimal_b5_legacy"
PROMPT_VERSION_FIXED = "sudoku_minimal_4x4_corrected_v1"


# --- Rollout machinery ---

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


def build_chat(tokenizer, system_prompt: str, state_text: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Current state:\n{state_text}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_answer(text: str) -> Optional[str]:
    m = ANSWER_RE.search(text)
    return m.group(1).strip() if m else None


def greedy_rollout(model, tokenizer, system_prompt: str, grid: List[List[int]],
                    solver: Sudoku4Solver, max_steps: int, max_new_tokens: int,
                    device: str) -> Tuple[bool, dict]:
    """Run a greedy rollout from `grid` until solved or terminal.

    Returns (solved, debug_dict). solved=True iff the goal state is reached.
    Terminal conditions:
      - is_goal(grid) reached -> solved
      - max_steps exceeded
      - model fails to emit <answer>
      - parsed action is locally illegal at current grid
      - resulting state is not solvable per oracle (we still continue greedy
        but record this so we can attribute failures)
    """
    g = [row[:] for row in grid]
    n_steps = 0
    n_invalid = 0
    n_doomed = 0
    n_no_answer = 0
    while n_steps < max_steps:
        if is_goal(g):
            return True, {"n_steps": n_steps, "n_invalid": n_invalid,
                          "n_doomed": n_doomed, "n_no_answer": n_no_answer}
        chat_prompt = build_chat(tokenizer, system_prompt, render_state_b5(g))
        input_ids = tokenizer(chat_prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = out[0, input_ids.shape[1]:]
        text = tokenizer.decode(gen, skip_special_tokens=True)
        answer = extract_answer(text)
        if not answer:
            n_no_answer += 1
            return False, {"n_steps": n_steps, "n_invalid": n_invalid,
                           "n_doomed": n_doomed, "n_no_answer": n_no_answer,
                           "fail_reason": "no_answer", "last_response_head": text[:200]}
        action = parse_action_text(answer)
        if action is None or not is_local_valid(g, action):
            n_invalid += 1
            return False, {"n_steps": n_steps, "n_invalid": n_invalid,
                           "n_doomed": n_doomed, "n_no_answer": n_no_answer,
                           "fail_reason": "invalid_action",
                           "answer": answer}
        g = apply_action(g, action)
        n_steps += 1
        # Check oracle viability AFTER the action (informational only — we
        # continue greedy regardless; rollout fails when goal isn't reached).
        if not solver.is_viable(g):
            n_doomed += 1
    return is_goal(g), {"n_steps": n_steps, "n_invalid": n_invalid,
                         "n_doomed": n_doomed, "n_no_answer": n_no_answer,
                         "fail_reason": "max_steps_no_goal"}


def pass1_under_prompt(model, tokenizer, system_prompt: str,
                        n_puzzles: int, seed_base: int, n_empty: int,
                        max_steps: int, max_new_tokens: int, device: str,
                        prompt_label: str) -> dict:
    """Run greedy Pass@1 across `n_puzzles` fresh easy 4×4 puzzles."""
    solver = Sudoku4Solver()
    n_solved = 0
    fail_breakdown = {"no_answer": 0, "invalid_action": 0, "max_steps_no_goal": 0}
    debug = []
    t0 = time.perf_counter()
    for i in range(n_puzzles):
        seed = seed_base + i
        grid = generate_root_puzzle(seed=seed, n_empty=n_empty)
        # Skip puzzles that are unsolvable from the start (very rare with random gen);
        # we only want to evaluate solvable starts.
        if not solver.is_viable(grid):
            continue
        solved, info = greedy_rollout(
            model, tokenizer, system_prompt, grid, solver,
            max_steps=max_steps, max_new_tokens=max_new_tokens, device=device,
        )
        if solved:
            n_solved += 1
        else:
            reason = info.get("fail_reason", "unknown")
            fail_breakdown[reason] = fail_breakdown.get(reason, 0) + 1
        debug.append({"i": i, "seed": seed, "solved": solved, **info})
    elapsed = time.perf_counter() - t0
    pass1 = n_solved / max(1, n_puzzles)
    return {
        "prompt_label": prompt_label,
        "n_puzzles": n_puzzles,
        "n_solved": n_solved,
        "pass1": pass1,
        "fail_breakdown": fail_breakdown,
        "elapsed_sec": elapsed,
    }


# --- Main ---

GAP_THRESHOLD = 0.05  # Pass@1 difference at or below which we ship the corrected prompt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy-checkpoint", required=True,
                   help="Path to the rl_b5 (or other base policy) checkpoint")
    p.add_argument("--n-puzzles", type=int, default=30)
    p.add_argument("--n-empty", type=int, default=10,
                   help="Number of cells to empty per puzzle")
    p.add_argument("--seed-base", type=int, default=100000)
    p.add_argument("--max-steps", type=int, default=15)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--output", default="data/sudoku4/_phase0_prompt_decision.json")
    args = p.parse_args()

    print(f"=== Phase 0 prompt-decision sanity check ===")
    print(f"  policy_checkpoint: {args.policy_checkpoint}")
    print(f"  n_puzzles={args.n_puzzles}, n_empty={args.n_empty}, "
          f"seed_base={args.seed_base}, max_steps={args.max_steps}")
    print(f"  GAP_THRESHOLD: {GAP_THRESHOLD} (|fixed - buggy| <= this => SHIP_FIXED)")
    print()

    print("Loading model + tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.policy_checkpoint)
    model = AutoModelForCausalLM.from_pretrained(args.policy_checkpoint,
                                                  torch_dtype=torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    print(f"  loaded; device={device}")
    print()

    common = dict(model=model, tokenizer=tokenizer,
                  n_puzzles=args.n_puzzles, seed_base=args.seed_base,
                  n_empty=args.n_empty, max_steps=args.max_steps,
                  max_new_tokens=args.max_new_tokens, device=device)

    print(f"--- A. Buggy prompt ('{PROMPT_VERSION_BUGGY}') ---")
    res_buggy = pass1_under_prompt(system_prompt=SYSTEM_PROMPT_BUGGY,
                                    prompt_label="buggy", **common)
    print(f"   Pass@1: {res_buggy['pass1']*100:.1f}% "
          f"({res_buggy['n_solved']}/{res_buggy['n_puzzles']}) "
          f"in {res_buggy['elapsed_sec']:.1f}s")
    print(f"   Failures: {res_buggy['fail_breakdown']}")
    print()

    print(f"--- B. Fixed prompt ('{PROMPT_VERSION_FIXED}') ---")
    res_fixed = pass1_under_prompt(system_prompt=SYSTEM_PROMPT_FIXED,
                                    prompt_label="fixed", **common)
    print(f"   Pass@1: {res_fixed['pass1']*100:.1f}% "
          f"({res_fixed['n_solved']}/{res_fixed['n_puzzles']}) "
          f"in {res_fixed['elapsed_sec']:.1f}s")
    print(f"   Failures: {res_fixed['fail_breakdown']}")
    print()

    gap = abs(res_fixed["pass1"] - res_buggy["pass1"])
    if gap <= GAP_THRESHOLD:
        verdict = "SHIP_FIXED"
        verdict_reason = (f"|fixed - buggy| = |{res_fixed['pass1']:.3f} - "
                          f"{res_buggy['pass1']:.3f}| = {gap:.3f} <= {GAP_THRESHOLD}")
    else:
        verdict = "SHIP_BUGGY"
        verdict_reason = (f"|fixed - buggy| = |{res_fixed['pass1']:.3f} - "
                          f"{res_buggy['pass1']:.3f}| = {gap:.3f} > {GAP_THRESHOLD}")

    decision = {
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "gap": gap,
        "gap_threshold": GAP_THRESHOLD,
        "buggy": {**res_buggy, "prompt_version": PROMPT_VERSION_BUGGY,
                  "prompt_text": SYSTEM_PROMPT_BUGGY},
        "fixed": {**res_fixed, "prompt_version": PROMPT_VERSION_FIXED,
                  "prompt_text": SYSTEM_PROMPT_FIXED},
        "policy_checkpoint": args.policy_checkpoint,
        "env_version": ENV_VERSION,
        "n_puzzles": args.n_puzzles,
        "n_empty": args.n_empty,
        "seed_base": args.seed_base,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(decision, f, indent=2)

    print(f"=== VERDICT: {verdict} ===")
    print(f"  {verdict_reason}")
    print(f"  Decision written to {args.output}")


if __name__ == "__main__":
    main()
