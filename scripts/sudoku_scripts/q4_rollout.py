"""Q4 main rollout entry — runs methods on a Sudoku puzzle set, outputs Table 4.

Loads:
  - π_θ from outputs/rl_b5_phase3_v8_anchor/final
  - f_φ from outputs/save_sudoku4_f_phi/final  (only if SAVE is in --methods)
  - Calibration JSON for τ_keep / τ_fb (optional)
  - Solver

Test puzzle source: t=0 (or any) state_struct.grid from test_natural_policy.jsonl.
We pick the first N viable distinct states.

Per (method, puzzle): one Episode → one EpisodeResult.
Aggregated metrics → outputs Table 4 row.

Usage (smoke run, 4 methods, 20 puzzles):
    python scripts/sudoku_scripts/q4_rollout.py \\
        --policy outputs/rl_b5_phase3_v8_anchor/final \\
        --save_phi outputs/save_sudoku4_f_phi/final \\
        --calibration outputs/save_sudoku4_f_phi/calibration.json \\
        --eval data/sudoku4/test_natural_policy.jsonl \\
        --n_puzzles 20 \\
        --methods policy_top1,best_of_k,save,oracle \\
        --output_json outputs/q4_smoke.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from sudoku4_solver import Sudoku4Solver  # noqa: E402
from q4_episode import Episode  # noqa: E402
from q4_methods import (  # noqa: E402
    PolicyClient, SaveScorer, PromptedScorer, LearnedProgressScorer,
    PolicyTop1Method, BestOfKMethod, LocalProgressMethod, SAVEMethod, OracleMethod,
    PromptedScoreOnlyMethod, LearnedProgressScoreMethod,
    NoTerminationMethod, GreedyTerminationMethod, SAVERetryMethod,
    RandomMatchedRateMethod, OracleTerminationMethod,
)


def load_puzzles(eval_path: Path, n: int) -> List[Dict]:
    """Load up to n viable distinct starting puzzles from test JSONL."""
    seen_hashes = set()
    puzzles = []
    with eval_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            state = rec["state"]
            if not state.get("state_viable"):
                continue
            h = state.get("state_hash")
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            puzzles.append({
                "sibling_set_id": rec["sibling_set_id"],
                "state_hash": h,
                "grid": state["state_struct"]["grid"],
            })
            if len(puzzles) >= n:
                break
    return puzzles


def build_method(name: str, policy: PolicyClient, scorer, prompted_scorer,
                 progress_scorer, solver, K: int, tau_keep: float, tau_fb: float):
    if name == "policy_top1":
        return PolicyTop1Method(policy)
    if name == "best_of_k":
        return BestOfKMethod(policy, K=K)
    if name == "local_progress":
        return LocalProgressMethod(policy, K=K)
    if name == "save":
        if scorer is None:
            raise RuntimeError("SAVE method needs --save_phi")
        return SAVEMethod(policy, scorer, tau_keep=tau_keep, tau_fb=tau_fb, K=K)
    if name == "oracle":
        return OracleMethod(policy, solver, K=K)
    if name == "prompted_score_only":
        if prompted_scorer is None:
            raise RuntimeError("prompted_score_only needs --base_model")
        return PromptedScoreOnlyMethod(policy, prompted_scorer, K=K)
    if name == "learned_progress_score":
        if progress_scorer is None:
            raise RuntimeError("learned_progress_score needs --g_psi")
        return LearnedProgressScoreMethod(policy, progress_scorer, K=K)
    if name == "no_termination":
        if scorer is None:
            raise RuntimeError("no_termination needs --save_phi")
        return NoTerminationMethod(policy, scorer, tau_keep=tau_keep, K=K)
    if name == "greedy_termination":
        if scorer is None:
            raise RuntimeError("greedy_termination needs --save_phi")
        return GreedyTerminationMethod(policy, scorer, tau_keep=tau_keep, K=K)
    if name == "save_retry":
        if scorer is None:
            raise RuntimeError("save_retry needs --save_phi")
        return SAVERetryMethod(policy, scorer, tau_keep=tau_keep, tau_fb=tau_fb, K=K)
    if name == "random_matched":
        # Caller should set tau_fb_override to convey term_rate; here we
        # interpret tau_fb as the desired termination rate (e.g., 0.05).
        return RandomMatchedRateMethod(policy, term_rate=tau_fb, K=K)
    if name == "oracle_termination":
        return OracleTerminationMethod(policy, solver, K=K)
    raise ValueError(f"Unknown method: {name}")


def aggregate(method_name: str, results: List, top1_policy_tokens_per_ep: float | None) -> Dict:
    n = len(results)
    if n == 0:
        return {"method": method_name, "n": 0}
    pass_at_1 = sum(r.pass_at_1 for r in results) / n
    de_entered_rate = sum(r.de_entered for r in results) / n
    de_steps = [r.first_de_step for r in results if r.first_de_step is not None]
    median_first_de = statistics.median(de_steps) if de_steps else None
    total_steps = sum(r.steps_taken for r in results)
    total_non_viable = sum(r.non_viable_selected for r in results)
    non_viable_rate = (total_non_viable / total_steps) if total_steps > 0 else 0.0
    avg_policy_tokens = sum(r.policy_tokens for r in results) / n
    avg_eval_tokens = sum(r.eval_tokens for r in results) / n
    avg_total_tokens = avg_policy_tokens + avg_eval_tokens
    terminated_rate = sum(r.terminated for r in results) / n

    out = {
        "method": method_name,
        "n_puzzles": n,
        "pass_at_1": pass_at_1,
        "de_entered_rate": de_entered_rate,
        "median_first_de_step": median_first_de,
        "non_viable_selection_rate": non_viable_rate,
        "avg_steps_taken": total_steps / n,
        "avg_policy_tokens": avg_policy_tokens,
        "avg_eval_tokens": avg_eval_tokens,
        "avg_total_tokens": avg_total_tokens,
        "terminated_rate": terminated_rate,
    }
    if top1_policy_tokens_per_ep is not None and top1_policy_tokens_per_ep > 0:
        out["net_compute"] = avg_total_tokens / top1_policy_tokens_per_ep
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, type=Path)
    ap.add_argument("--save_phi", type=Path, default=None)
    ap.add_argument("--base_model", type=Path, default=None,
                    help="Path/HF id of base model for prompted_score_only")
    ap.add_argument("--g_psi", type=Path, default=None,
                    help="Path of trained learned-progress checkpoint")
    ap.add_argument("--calibration", type=Path, default=None)
    ap.add_argument("--eval", required=True, type=Path)
    ap.add_argument("--n_puzzles", type=int, default=20)
    ap.add_argument("--methods", default="policy_top1,best_of_k,save,oracle")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=20)
    ap.add_argument("--policy_temperature", type=float, default=1.0)
    ap.add_argument("--policy_top_p", type=float, default=0.95)
    ap.add_argument("--policy_max_new_tokens", type=int, default=256)
    ap.add_argument("--save_max_new_tokens", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_json", type=Path, default=None)
    ap.add_argument("--tau_keep_override", type=float, default=None,
                    help="Override τ_keep from calibration (e.g. for ablations)")
    ap.add_argument("--tau_fb_override", type=float, default=None,
                    help="Override τ_fb (default 0.0). Set equal to τ_keep for single-thresh ablation")
    ap.add_argument("--no_calib", action="store_true",
                    help="Disable temperature scaling (T=1.0)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    print(f"[init] methods: {methods}")
    print(f"[init] n_puzzles: {args.n_puzzles}, K: {args.K}, max_steps: {args.max_steps}")

    # --- Load puzzles ---
    puzzles = load_puzzles(args.eval, args.n_puzzles)
    print(f"[load] {len(puzzles)} puzzles from {args.eval}")
    if not puzzles:
        print("No puzzles found. Aborting.")
        return

    # --- Load π_θ ---
    print(f"[load] π_θ from {args.policy}")
    pol_tok = AutoTokenizer.from_pretrained(args.policy, local_files_only=True)
    if pol_tok.pad_token_id is None:
        pol_tok.pad_token = pol_tok.eos_token
    pol_model = AutoModelForCausalLM.from_pretrained(
        args.policy, torch_dtype=torch.bfloat16, local_files_only=True,
    ).to("cuda").eval()
    policy = PolicyClient(
        pol_model, pol_tok, device="cuda",
        temperature=args.policy_temperature, top_p=args.policy_top_p,
        max_new_tokens=args.policy_max_new_tokens,
    )

    # --- Load f_φ if needed ---
    scorer = None
    prompted_scorer = None
    progress_scorer = None
    tau_keep, tau_fb = 0.5, 0.0
    if "save" in methods:
        if args.save_phi is None:
            raise SystemExit("--save_phi required when SAVE method is requested")
        print(f"[load] f_φ from {args.save_phi}")
        save_tok = AutoTokenizer.from_pretrained(args.save_phi, local_files_only=True)
        if save_tok.pad_token_id is None:
            save_tok.pad_token = save_tok.eos_token
        save_model = AutoModelForCausalLM.from_pretrained(
            args.save_phi, torch_dtype=torch.bfloat16, local_files_only=True,
        ).to("cuda").eval()
        cal_T = 1.0
        if args.calibration is not None:
            cal = json.loads(args.calibration.read_text())
            cal_T = float(cal["temperature"])
            tau_keep = float(cal["tau_keep"]["tau"])
            tau_fb = 0.0
        if args.no_calib:
            cal_T = 1.0
            print("[load] --no_calib: T=1.0")
        if args.tau_keep_override is not None:
            tau_keep = args.tau_keep_override
            print(f"[load] τ_keep override → {tau_keep}")
        if args.tau_fb_override is not None:
            tau_fb = args.tau_fb_override
            print(f"[load] τ_fb override → {tau_fb}")
        print(f"[load] effective T={cal_T:.3f} tau_keep={tau_keep:.3f} tau_fb={tau_fb:.3f}")
        scorer = SaveScorer(save_model, save_tok, temperature=cal_T,
                            max_new_tokens=args.save_max_new_tokens)

    if "prompted_score_only" in methods:
        if args.base_model is None:
            raise SystemExit("--base_model required for prompted_score_only")
        print(f"[load] base model from {args.base_model}")
        base_tok = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
        if base_tok.pad_token_id is None:
            base_tok.pad_token = base_tok.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, local_files_only=True,
        ).to("cuda").eval()
        prompted_scorer = PromptedScorer(base_model, base_tok)

    if "learned_progress_score" in methods:
        if args.g_psi is None:
            raise SystemExit("--g_psi required for learned_progress_score")
        print(f"[load] g_ψ from {args.g_psi}")
        g_tok = AutoTokenizer.from_pretrained(args.g_psi, local_files_only=True)
        if g_tok.pad_token_id is None:
            g_tok.pad_token = g_tok.eos_token
        g_model = AutoModelForCausalLM.from_pretrained(
            args.g_psi, torch_dtype=torch.bfloat16, local_files_only=True,
        ).to("cuda").eval()
        progress_scorer = LearnedProgressScorer(g_model, g_tok)

    # --- Setup ---
    solver = Sudoku4Solver()
    episode_runner = Episode(solver=solver, max_steps=args.max_steps)

    # --- Run ---
    all_results: Dict[str, List] = {}
    t_total = time.perf_counter()
    for method_name in methods:
        method = build_method(
            method_name, policy, scorer, prompted_scorer, progress_scorer,
            solver, K=args.K, tau_keep=tau_keep, tau_fb=tau_fb,
        )
        print(f"\n[{method_name}] running on {len(puzzles)} puzzles...")
        t0 = time.perf_counter()
        results = []
        for i, puz in enumerate(puzzles):
            r = episode_runner.run(puz["grid"], method)
            results.append(r)
            if (i + 1) % 5 == 0 or i + 1 == len(puzzles):
                pass_so_far = sum(rr.pass_at_1 for rr in results) / len(results)
                de_so_far = sum(rr.de_entered for rr in results) / len(results)
                print(f"  [{i+1}/{len(puzzles)}] pass={pass_so_far:.3f} de={de_so_far:.3f}")
        elapsed = time.perf_counter() - t0
        print(f"[{method_name}] done in {elapsed:.1f}s")
        all_results[method_name] = results

    # --- Aggregate ---
    top1_avg_pol = None
    if "policy_top1" in all_results:
        top1_avg_pol = sum(r.policy_tokens for r in all_results["policy_top1"]) / len(all_results["policy_top1"])

    aggregated = []
    for method_name, results in all_results.items():
        aggregated.append(aggregate(method_name, results, top1_avg_pol))

    print("\n=== TABLE 4 ROW (Sudoku) ===")
    fields = ["method", "pass_at_1", "de_entered_rate", "non_viable_selection_rate",
              "avg_total_tokens", "net_compute", "terminated_rate"]
    print(" | ".join(f"{f:>20}" for f in fields))
    print("-" * 145)
    for a in aggregated:
        row = []
        for f in fields:
            v = a.get(f)
            if v is None:
                s = "N/A"
            elif isinstance(v, float):
                s = f"{v:.4f}"
            else:
                s = str(v)
            row.append(f"{s:>20}")
        print(" | ".join(row))

    out = {
        "config": {
            "policy": str(args.policy),
            "save_phi": str(args.save_phi) if args.save_phi else None,
            "n_puzzles": len(puzzles),
            "K": args.K,
            "max_steps": args.max_steps,
            "policy_temperature": args.policy_temperature,
            "policy_top_p": args.policy_top_p,
            "tau_keep": tau_keep,
            "tau_fb": tau_fb,
        },
        "table_4_rows": aggregated,
        "per_episode": {
            method: [r.to_dict() for r in results]
            for method, results in all_results.items()
        },
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(out, indent=2))
        print(f"\n[save] {args.output_json}")
    print(f"\n[total time] {time.perf_counter()-t_total:.1f}s")


if __name__ == "__main__":
    main()
