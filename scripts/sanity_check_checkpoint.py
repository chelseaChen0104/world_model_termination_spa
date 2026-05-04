"""Runtime model-behavior sanity checks for an SFT or RL checkpoint.

Catches the issues we hit in this project:
  - bare-answer collapse at turn 1+ (multi-turn-history bug)
  - viability-tag bimodal collapse ("always-false" greedy)
  - response truncation at max_response_tokens
  - <answer> emitted but env rejects (action-format drift)
  - prompt-format mismatch (training had "Current state:" prefix, eval doesn't)

What it does:
  1. Load the checkpoint.
  2. Run a single GREEDY rollout with verbose printing — dumps prompt + response
     + parsed (action, viability, gt_solvable) at every step, plus rollout outcome.
  3. Run K STOCHASTIC rollouts on N puzzles for Pass@k coverage.
  4. Report aggregate metrics + a list of pass/fail health checks.

Usage:
  python scripts/sanity_check_checkpoint.py \\
      --sft-path outputs/sft_hidato_b_h1/final --env hidato \\
      --prepend-current-state --reset-history-per-step --max-new-tokens 512

Exit code 0 if all health checks pass, 1 if any fail (under --strict).
"""
from __future__ import annotations

import argparse
import os
import sys
import re
from collections import Counter

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


ENV_TAGS = {
    "sudoku":     {"viability_tag": "solvable",  "next_tag": "prediction"},
    "polyomino":  {"viability_tag": "viability", "next_tag": "next_state"},
    "hidato":     {"viability_tag": "solvable",  "next_tag": "prediction"},
}


def make_env(env_name, args):
    if env_name == "sudoku":
        from src.environments.sudoku import SudokuEnv
        from src.data.sft_formatter import SFTFormatter
        env = SudokuEnv(grid_size=args.grid_size, difficulty=args.difficulty,
                        max_steps=args.max_steps)
        formatter = SFTFormatter(variant="sudoku_minimal")
    elif env_name == "polyomino":
        from src.environments.polyomino import PolyominoEnv
        from src.data.sft_formatter import SFTFormatter
        piece_set = tuple(p.strip().upper() for p in args.piece_set.split(","))
        env = PolyominoEnv(board_h=args.board_h, board_w=args.board_w,
                           piece_set=piece_set, max_steps=args.max_steps)
        formatter = SFTFormatter(variant="polyomino_minimal")
    elif env_name == "hidato":
        from src.environments.hidato import HidatoEnv
        from src.data.sft_formatter import SFTFormatter
        env = HidatoEnv(max_steps=args.max_steps)
        formatter = SFTFormatter(variant="hidato_minimal")
    else:
        raise ValueError(env_name)
    return env, formatter


def run_one_rollout(model, tokenizer, env, sys_prompt, args, seed: int,
                    temperature: float, verbose: bool, viability_tag: str):
    obs = env.reset(seed=seed)
    history = []
    n_steps = 0
    n_valid_actions = 0
    n_full_xml = 0
    n_truncated = 0
    via_pred_counts = Counter()
    is_solved = False

    for t in range(args.max_steps):
        if args.prepend_current_state:
            user_msg = f"Current state:\n{obs}" if t == 0 else f"Action executed. Current state:\n{obs}"
        else:
            user_msg = obs

        msgs = [{"role": "system", "content": sys_prompt}]
        if not args.reset_history_per_step:
            for u, a in history:
                msgs.append({"role": "user", "content": u})
                msgs.append({"role": "assistant", "content": a})
        msgs.append({"role": "user", "content": user_msg})
        prompt_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=0.95 if temperature > 0 else None,
                pad_token_id=tokenizer.eos_token_id,
            )
        resp_ids = out_ids[0][inputs["input_ids"].shape[1]:]
        resp_text = tokenizer.decode(resp_ids, skip_special_tokens=True)

        # Did the response hit the token budget?
        if len(resp_ids) >= args.max_new_tokens - 1:
            n_truncated += 1

        # Format + parse checks
        has_full_xml = (
            "<think>" in resp_text and "</think>" in resp_text
            and "<observation>" in resp_text
            and ("<prediction>" in resp_text or "<next_state>" in resp_text)
            and f"<{viability_tag}>" in resp_text
            and "<answer>" in resp_text
        )
        if has_full_xml:
            n_full_xml += 1

        m_via = re.search(rf"<{viability_tag}>\s*(true|false)\s*</{viability_tag}>", resp_text, re.IGNORECASE)
        if m_via:
            via_pred_counts[m_via.group(1).lower()] += 1

        m_act = re.search(r"<answer>(.*?)</answer>", resp_text, re.DOTALL)
        if not m_act:
            if verbose:
                print(f"  STEP {t}: NO <answer> — stopping. Response head: {resp_text[:200]!r}")
            break
        action_str = m_act.group(1).strip()

        next_obs, _r, done, info = env.step(action_str)
        n_steps += 1
        if info.get("action_is_valid", True):
            n_valid_actions += 1
        if info.get("success", False):
            is_solved = True

        if verbose:
            print(f"  STEP {t}: action={action_str!r} valid={info.get('action_is_valid')} "
                  f"is_solvable={info.get('is_solvable')} success={info.get('success')} "
                  f"<{viability_tag}>={m_via.group(1).lower() if m_via else None} "
                  f"resp_tokens={len(resp_ids)} full_xml={has_full_xml}")

        history.append((user_msg, resp_text))
        obs = next_obs
        if done:
            break

    return {
        "is_solved": is_solved,
        "n_steps": n_steps,
        "n_valid_actions": n_valid_actions,
        "n_full_xml": n_full_xml,
        "n_truncated": n_truncated,
        "via_pred_counts": via_pred_counts,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sft-path", required=True)
    p.add_argument("--env", required=True, choices=list(ENV_TAGS.keys()))
    p.add_argument("--prepend-current-state", action="store_true")
    p.add_argument("--reset-history-per-step", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--n-puzzles", type=int, default=8,
                   help="Number of distinct puzzle seeds for stochastic Pass@k.")
    p.add_argument("--k", type=int, default=4,
                   help="Stochastic samples per puzzle.")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--strict", action="store_true",
                   help="Exit non-zero if any health check fails.")
    # Sudoku-specific
    p.add_argument("--grid-size", type=int, default=4)
    p.add_argument("--difficulty", default="easy")
    # Polyomino-specific
    p.add_argument("--board-h", type=int, default=5)
    p.add_argument("--board-w", type=int, default=10)
    p.add_argument("--piece-set", default="F,I,L,N,P,T,U,V,Y,Z")
    args = p.parse_args()

    print(f"=== checkpoint sanity check: {args.sft_path} (env={args.env}) ===")
    print(f"  prepend_current_state={args.prepend_current_state}")
    print(f"  reset_history_per_step={args.reset_history_per_step}")
    print(f"  max_new_tokens={args.max_new_tokens}")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.sft_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.sft_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()

    cfg = ENV_TAGS[args.env]
    via_tag = cfg["viability_tag"]
    env, formatter = make_env(args.env, args)
    sys_prompt = formatter.system_prompt

    # ── Verbose greedy run on puzzle 0 ───────────────────────────────
    print(f"# A. Verbose greedy rollout on puzzle 0 (seed=100000)")
    g = run_one_rollout(model, tokenizer, env, sys_prompt, args,
                        seed=100000, temperature=0.0, verbose=True,
                        viability_tag=via_tag)
    print(f"  → solved={g['is_solved']} n_steps={g['n_steps']} valid_actions={g['n_valid_actions']}/{g['n_steps']} "
          f"full_xml={g['n_full_xml']}/{g['n_steps']} truncated={g['n_truncated']}/{g['n_steps']} "
          f"via_preds={dict(g['via_pred_counts'])}")
    print()

    # ── Greedy on N puzzles for Pass@1 ───────────────────────────────
    print(f"# B. Greedy Pass@1 on {args.n_puzzles} puzzles")
    p1_solved = 0
    p1_full_xml_steps = 0
    p1_total_steps = 0
    p1_truncated = 0
    p1_via_counts = Counter()
    for i in range(args.n_puzzles):
        r = run_one_rollout(model, tokenizer, env, sys_prompt, args,
                            seed=100000 + i, temperature=0.0, verbose=False,
                            viability_tag=via_tag)
        if r["is_solved"]:
            p1_solved += 1
        p1_full_xml_steps += r["n_full_xml"]
        p1_total_steps += r["n_steps"]
        p1_truncated += r["n_truncated"]
        p1_via_counts.update(r["via_pred_counts"])
    pass1 = p1_solved / max(1, args.n_puzzles)
    full_xml_frac = p1_full_xml_steps / max(1, p1_total_steps)
    trunc_frac = p1_truncated / max(1, p1_total_steps)
    print(f"  Pass@1:                 {pass1*100:.1f}% ({p1_solved}/{args.n_puzzles})")
    print(f"  full-XML step fraction: {full_xml_frac*100:.1f}%")
    print(f"  token-budget hits:      {p1_truncated}/{p1_total_steps} steps ({trunc_frac*100:.1f}%)")
    print(f"  viability predictions:  {dict(p1_via_counts)}")
    print()

    # ── Stochastic Pass@k on N puzzles ───────────────────────────────
    print(f"# C. Stochastic Pass@k on {args.n_puzzles} puzzles, k={args.k}, T={args.temperature}")
    pk_any_solved = 0
    pk_total_rollouts = 0
    pk_solved_rollouts = 0
    for i in range(args.n_puzzles):
        any_solved = False
        for j in range(args.k):
            r = run_one_rollout(model, tokenizer, env, sys_prompt, args,
                                seed=200000 + i * 100 + j, temperature=args.temperature,
                                verbose=False, viability_tag=via_tag)
            pk_total_rollouts += 1
            if r["is_solved"]:
                pk_solved_rollouts += 1
                any_solved = True
        if any_solved:
            pk_any_solved += 1
    pass_k = pk_any_solved / max(1, args.n_puzzles)
    per_batch = pk_solved_rollouts / max(1, pk_total_rollouts)
    print(f"  Pass@{args.k}:          {pass_k*100:.1f}% ({pk_any_solved}/{args.n_puzzles})")
    print(f"  per-batch solve rate: {per_batch*100:.1f}% ({pk_solved_rollouts}/{pk_total_rollouts})")
    print()

    # ── Health checks ────────────────────────────────────────────────
    print(f"# D. Health checks")
    failures = []
    def check(name, ok, detail):
        marker = "✅" if ok else "❌"
        print(f"  {marker} {name}: {detail}")
        if not ok:
            failures.append(name)

    check("Greedy rollout produces FULL XML at every step",
          full_xml_frac >= 0.95,
          f"{full_xml_frac*100:.1f}% (low → multi-turn-history bug or token-budget too low)")
    check("Token-budget not hit at any step",
          trunc_frac < 0.05,
          f"{trunc_frac*100:.1f}% (high → bump --max-new-tokens)")
    check("Viability predictions show both classes",
          len(p1_via_counts) >= 2,
          f"{dict(p1_via_counts)} (single-class → bimodal collapse / regime-1)")
    check("Stochastic per-batch solve > 0",
          per_batch > 0.0,
          f"{per_batch*100:.1f}% (zero → SFT lacks positive signal; RL cannot bootstrap)")
    check("Greedy Pass@1 > 0",
          pass1 > 0.0,
          f"{pass1*100:.1f}% (zero with positive per-batch → greedy collapse, possibly fixable)")

    print()
    if failures:
        print(f"=== ⚠️ {len(failures)} health check(s) failed: {', '.join(failures)} ===")
        if args.strict:
            sys.exit(1)
    else:
        print("=== ✅ all health checks passed ===")


if __name__ == "__main__":
    main()
