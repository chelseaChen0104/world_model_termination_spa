"""Quickly diagnose what SAVE scorer assigns to K candidates at real Q4 states.

For each of N test states:
  - sample K candidates from π_θ
  - run f_φ generate-then-read on each
  - also compute oracle viability (solver) for ground truth
  - print score, oracle, and whether filter (τ_keep) keeps it

If most scores cluster above τ_keep regardless of oracle viability, the
deployed scorer has lost discrimination — confirming our hypothesis.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from sudoku4_env import apply_action  # noqa: E402
from sudoku4_solver import Sudoku4Solver  # noqa: E402
from q4_methods import PolicyClient, SaveScorer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, type=Path)
    ap.add_argument("--save_phi", required=True, type=Path)
    ap.add_argument("--calibration", type=Path, default=None)
    ap.add_argument("--eval", required=True, type=Path)
    ap.add_argument("--n_states", type=int, default=5)
    ap.add_argument("--K", type=int, default=8)
    args = ap.parse_args()

    pol_tok = AutoTokenizer.from_pretrained(args.policy, local_files_only=True)
    if pol_tok.pad_token_id is None:
        pol_tok.pad_token = pol_tok.eos_token
    pol_model = AutoModelForCausalLM.from_pretrained(
        args.policy, torch_dtype=torch.bfloat16, local_files_only=True,
    ).to("cuda").eval()
    policy = PolicyClient(pol_model, pol_tok)

    save_tok = AutoTokenizer.from_pretrained(args.save_phi, local_files_only=True)
    if save_tok.pad_token_id is None:
        save_tok.pad_token = save_tok.eos_token
    save_model = AutoModelForCausalLM.from_pretrained(
        args.save_phi, torch_dtype=torch.bfloat16, local_files_only=True,
    ).to("cuda").eval()

    cal_T = 1.0
    tau_keep = 0.5
    if args.calibration:
        cal = json.loads(args.calibration.read_text())
        cal_T = float(cal["temperature"])
        tau_keep = float(cal["tau_keep"]["tau"])
    print(f"Calibration: T={cal_T:.3f}  τ_keep={tau_keep:.3f}\n")

    scorer = SaveScorer(save_model, save_tok, temperature=cal_T)
    solver = Sudoku4Solver()

    # Load N states
    states = []
    seen = set()
    with args.eval.open() as f:
        for line in f:
            rec = json.loads(line)
            if not rec["state"]["state_viable"]:
                continue
            h = rec["state"]["state_hash"]
            if h in seen:
                continue
            seen.add(h)
            states.append(rec["state"]["state_struct"]["grid"])
            if len(states) >= args.n_states:
                break

    # Stats accumulators
    all_scores = []
    score_when_viable = []
    score_when_doomed = []
    n_candidates_total = 0
    n_kept = 0

    for i, grid in enumerate(states):
        print(f"=== State #{i} ===")
        print('\n'.join(' '.join(str(x) if x else '.' for x in row) for row in grid))
        cands, _ = policy.sample_k(grid, K=args.K)
        print(f"  K={len(cands)} legal candidates")
        for c in cands:
            score, _ = scorer.score(grid, c.action)
            next_g = apply_action(grid, c.action)
            oracle_viable = solver.is_viable(next_g)
            kept_flag = "✓keep" if score >= tau_keep else " drop"
            mark = "[OK]" if oracle_viable else "[DOOM]"
            print(f"  {mark}  {c.action}  score={score:.4f}  {kept_flag}  log_pol={c.generation_logprob:.2f}")
            all_scores.append(score)
            (score_when_viable if oracle_viable else score_when_doomed).append(score)
            n_candidates_total += 1
            if score >= tau_keep:
                n_kept += 1
        print()

    def stats(name, xs):
        if not xs:
            print(f"  {name}: (empty)")
            return
        import statistics
        print(f"  {name}: n={len(xs)} mean={statistics.mean(xs):.3f} "
              f"min={min(xs):.3f} max={max(xs):.3f} "
              f"median={statistics.median(xs):.3f}")

    print("=== AGGREGATE ===")
    stats("all scores", all_scores)
    stats("scores when oracle viable", score_when_viable)
    stats("scores when oracle doomed", score_when_doomed)
    print(f"  fraction kept (score ≥ τ_keep={tau_keep:.3f}): "
          f"{n_kept}/{n_candidates_total} = {n_kept/max(1,n_candidates_total):.3f}")


if __name__ == "__main__":
    main()
