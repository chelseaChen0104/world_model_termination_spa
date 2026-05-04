"""SAVE sibling-set data generator (env-agnostic dispatcher).

Implements doc/data_generation_sudoku.md §4-§6 (and corresponding sections
of pentomino/hidato docs). Single entry point that takes `--env` and routes
to per-env adapters: solver, env (state/action utils), progress formula.

Usage:
    python scripts/generate_save_data.py \\
        --env sudoku \\
        --role train_balanced \\
        --policy-model outputs/rl_b5_phase3_v8_anchor/final \\
        --policy-checkpoint-id rl_b5_phase3_v8_anchor_final \\
        --n-target 30 \\
        --output data/sudoku4/smoke_test.jsonl \\
        --seed 42

Per-env imports:
  - sudoku   → scripts/sudoku4_{solver,env}.py + scripts/progress_sudoku4.py
  - polyomino → scripts/pentomino5x4_{solver,env}.py + scripts/progress_pentomino5x4.py
  - hidato   → scripts/hidato5x4_{solver,env}.py   + scripts/progress_hidato5x4.py     (TBW)

For toy run, only `sudoku` is fully wired up; pentomino + hidato adapters
exist but the dispatcher will refuse them until `progress_hidato5x4.py` and
the hidato_*_env modules land. (TODO: enable once those are written.)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _resolve_git_commit() -> str:
    """Cached at module import; called once instead of per-record. Reads
    .git/HEAD directly to avoid spawning a `git` subprocess that fails
    noisily on cloud machines that aren't a git checkout."""
    try:
        head_path = os.path.join(_REPO_ROOT, ".git", "HEAD")
        if not os.path.isfile(head_path):
            return "unknown"
        with open(head_path) as f:
            head = f.read().strip()
        if head.startswith("ref: "):
            ref_path = os.path.join(_REPO_ROOT, ".git", head[5:])
            if os.path.isfile(ref_path):
                with open(ref_path) as f:
                    return f.read().strip()
            return "unknown"
        return head  # detached HEAD: it's already a SHA
    except Exception:
        return "unknown"


_GIT_COMMIT = _resolve_git_commit()


import torch

from scripts.save_schema import SiblingSetRecord
from scripts.policy_sampler import (
    load_model, build_chat_prompt, batched_sample,
    policy_eval_logprob, batched_policy_eval_logprob, Sample,
)


# ---------------------------------------------------------------------------
# Per-env dispatch
# ---------------------------------------------------------------------------

def get_env_adapter(env_name: str):
    """Return a dict of env-specific functions and constants.

    Common interface (all adapters expose the same keys):
      env_short, env_version, state_text_version, system_prompt,
      policy_eval_prompt_version, ActionStruct, solver_version, solver,
      render_state(state), state_hash(state), is_goal(state),
      enumerate_legal_actions(state), is_local_valid(state, action),
      apply_action(state, action), parse_action(text), action_text(action),
      canonical_action(action), action_hash(action),
      action_struct_to_dict(action), state_struct_for_record(state),
      generate_root_puzzle(seed, **kwargs), solver_solve(state),
      solver_is_viable(state), solver_find_one_solution(state),
      apply_sol_step(state, sol_step), compute_progress(state)
    """
    if env_name == "sudoku":
        from scripts import sudoku4_env as env_mod
        from scripts import sudoku4_solver as solver_mod
        from scripts import progress_sudoku4 as progress_mod
        solver = solver_mod.Sudoku4Solver()
        AS = env_mod.ActionStruct

        def _to_action(a):
            return AS(**a) if isinstance(a, dict) else a

        def _apply_sol_step(state, sol_step):
            r0, c0, v = sol_step
            return {"grid": env_mod.apply_action(state["grid"], AS(row=r0+1, col=c0+1, value=v))}

        return {
            "env_short": "sudoku4",
            "env_version": env_mod.ENV_VERSION,
            "state_text_version": env_mod.TEXT_VERSION_B5,
            "system_prompt": _SUDOKU_SYSTEM_PROMPT_FIXED,
            "policy_eval_prompt_version": "sudoku_minimal_4x4_corrected_v1",
            "render_state": lambda s: env_mod.render_state_b5(s["grid"]),
            "state_struct_for_record": lambda s: {"grid": s["grid"]},
            "parse_action": env_mod.parse_action_text,
            "action_text": lambda a: env_mod.action_text(_to_action(a)),
            "canonical_action": lambda a: env_mod.canonical_action(_to_action(a)),
            "action_hash": lambda a: env_mod.action_hash(_to_action(a)),
            "action_struct_to_dict": lambda a: a.to_dict() if hasattr(a, "to_dict") else a,
            "is_local_valid": lambda s, a: env_mod.is_local_valid(s["grid"], a),
            "apply_action": lambda s, a: {"grid": env_mod.apply_action(s["grid"], a)},
            "is_goal": lambda s: env_mod.is_goal(s["grid"]),
            "enumerate_legal_actions": lambda s: env_mod.enumerate_legal_actions(s["grid"]),
            "state_hash": lambda s: env_mod.state_hash(s["grid"]),
            "generate_root_puzzle": lambda seed, **kw: {"grid": env_mod.generate_root_puzzle(seed=seed, n_empty=kw.get("n_empty", 6))},
            "solver": solver,
            "solver_version": solver_mod.SOLVER_VERSION,
            "solver_solve": lambda s: solver.solve(s["grid"]),
            "solver_is_viable": lambda s: solver.is_viable(s["grid"]),
            "solver_find_one_solution": lambda s: solver.find_one_solution(s["grid"]),
            "apply_sol_step": _apply_sol_step,
            "compute_progress": lambda s: progress_mod.compute_progress(s["grid"]),
            "ActionStruct": AS,
            "perturbation_kind": "sudoku",   # for sample_prt routing
        }

    elif env_name == "polyomino":
        from scripts import pentomino_env as env_mod
        from scripts import pentomino_solver as solver_mod
        from scripts import progress_pentomino as progress_mod
        from src.environments.polyomino_utils import ALL_PIECES
        import itertools, random as _random
        # 5×6 with 6-piece subsets is the current target
        _BOARD_H = int(os.environ.get("PENT_BOARD_H", "5"))
        _BOARD_W = int(os.environ.get("PENT_BOARD_W", "6"))
        _K_PIECES = (_BOARD_H * _BOARD_W) // 5
        solver = solver_mod.PentominoSolver(board_h=_BOARD_H, board_w=_BOARD_W)

        # Cache of valid k-piece subsets that tile the board (computed lazily).
        _valid_subsets_cache: list = []
        def _get_valid_subsets():
            if _valid_subsets_cache:
                return _valid_subsets_cache
            cap1 = solver_mod.PentominoSolver(
                board_h=_BOARD_H, board_w=_BOARD_W,
                solution_cap=1, node_cap=200_000)
            empty = env_mod.empty_board(_BOARD_H, _BOARD_W)
            for combo in itertools.combinations(ALL_PIECES, _K_PIECES):
                if cap1.solve(empty, list(combo)).solvable:
                    _valid_subsets_cache.append(combo)
            return _valid_subsets_cache

        def _pentomino_root(seed: int, h: int, w: int, k_pieces: int) -> dict:
            """Pick a random valid k-piece subset for the (h, w) board."""
            subsets = _get_valid_subsets()
            if not subsets:
                raise RuntimeError(
                    f"No valid {k_pieces}-piece subsets tile {h}×{w}")
            rng = _random.Random(seed)
            subset = list(rng.choice(subsets))
            return {"board": env_mod.empty_board(h, w), "remaining_pieces": subset}
        AS = env_mod.ActionStruct
        SYS_PENT = (
            "You are solving a pentomino tiling puzzle. The board is a rectangular grid; "
            "you must place the given pentomino pieces so that every cell is covered exactly once, "
            "with no overlaps and no piece extending outside the board.\n\n"
            "Pieces use the standard letters: F, I, L, N, P, T, U, V, W, X, Y, Z. "
            "Each piece is 5 unit squares. Pieces can be rotated and reflected, giving multiple "
            "orientations per piece (orientation IDs 0..N-1, deterministic per piece).\n\n"
            "Board format: each cell shows '.' for empty or the piece-letter that occupies it. "
            "Remaining pieces are listed below the board.\n\n"
            "In your reasoning:\n"
            "1. Describe the current state in <observation>\n"
            "2. Predict the board after your placement in <next_state>\n"
            "3. Assess whether the resulting board is still tileable with the remaining pieces in <viability>: true/false\n\n"
            "Then provide your action in <answer> using format: place {piece} ori={K} at row {R} col {C}\n"
            "where {piece} is one of the remaining pieces, {K} is the orientation id, and (R, C) "
            "are 1-indexed anchor coordinates."
        )

        def _to_action(a):
            if isinstance(a, dict):
                return AS(**a)
            return a

        def _apply_sol_step(state, sol_step):
            piece, ori_id, ar, ac = sol_step
            new_state, new_remaining = env_mod.apply_action(
                state["board"], state["remaining_pieces"],
                AS(piece=piece, ori=ori_id, row=ar+1, col=ac+1)
            )
            return {"board": new_state, "remaining_pieces": new_remaining}

        return {
            "env_short": f"pentomino{_BOARD_H}x{_BOARD_W}",
            "env_version": env_mod.ENV_VERSION,
            "state_text_version": env_mod.TEXT_VERSION,
            "system_prompt": SYS_PENT,
            "policy_eval_prompt_version": "polyomino_minimal_v1",
            "render_state": lambda s: env_mod.render_state_b8(s["board"], s["remaining_pieces"]),
            "state_struct_for_record": lambda s: {"board": s["board"], "remaining_pieces": list(s["remaining_pieces"])},
            "parse_action": env_mod.parse_action_text,
            "action_text": lambda a: env_mod.action_text(_to_action(a)),
            "canonical_action": lambda a: env_mod.canonical_action(_to_action(a)),
            "action_hash": lambda a: env_mod.action_hash(_to_action(a)),
            "action_struct_to_dict": lambda a: a.to_dict() if hasattr(a, "to_dict") else a,
            "is_local_valid": lambda s, a: env_mod.is_local_valid(s["board"], s["remaining_pieces"], a),
            "apply_action": lambda s, a: dict(zip(["board", "remaining_pieces"], env_mod.apply_action(s["board"], s["remaining_pieces"], a))),
            "is_goal": lambda s: env_mod.is_goal(s["board"]),
            "enumerate_legal_actions": lambda s: env_mod.enumerate_legal_actions(s["board"], s["remaining_pieces"]),
            "state_hash": lambda s: env_mod.state_hash(s["board"], s["remaining_pieces"]),
            "generate_root_puzzle": lambda seed, **kw: _pentomino_root(seed, _BOARD_H, _BOARD_W, kw.get("k_pieces", 6)),
            "solver": solver,
            "solver_version": solver_mod.SOLVER_VERSION,
            "solver_solve": lambda s: solver.solve(s["board"], s["remaining_pieces"]),
            "solver_is_viable": lambda s: solver.is_viable(s["board"], s["remaining_pieces"]),
            "solver_find_one_solution": lambda s: solver.find_one_solution(s["board"], s["remaining_pieces"]),
            "apply_sol_step": _apply_sol_step,
            "compute_progress": lambda s: progress_mod.compute_progress(s["board"]),
            "ActionStruct": AS,
            "perturbation_kind": "polyomino",
        }

    elif env_name == "hidato":
        from scripts import hidato5x4_env as env_mod
        from scripts import hidato5x4_solver as solver_mod
        from scripts import progress_hidato5x4 as progress_mod
        solver = solver_mod.Hidato5x4Solver()
        AS = env_mod.ActionStruct
        SYS_HID = (
            "You are solving a Hidato (number-path) puzzle. The board is a rectangular grid where "
            "you must fill in numbers from 1 to N (where N = rows × cols) so that consecutive "
            "numbers (k and k+1) are placed in cells that share an edge (orthogonally adjacent — "
            "up, down, left, or right).\n\n"
            "Grid format: each cell shows its placed number, or '.' for empty.\n\n"
            "You place numbers in sequential order (1, then 2, then 3, ...). Some cells are "
            "pre-filled (givens) and don't need to be placed; the env will skip past them. "
            "Each step you place the next required number into an empty cell that's adjacent to "
            "the previous number's cell.\n\n"
            "In your reasoning:\n"
            "1. Describe the current state in <observation>\n"
            "2. Predict the next state after your move in <prediction>\n"
            "3. Assess whether the resulting state will still be solvable in <solvable>: true/false\n\n"
            "Then provide your action in <answer> using format: place {N} at row {R} col {C}"
        )

        def _to_action(a):
            return AS(**a) if isinstance(a, dict) else a

        def _apply_sol_step(state, sol_step):
            r0, c0, v = sol_step
            return env_mod.apply_action(state, AS(row=r0+1, col=c0+1, value=v))

        def _state_struct_for_record(s):
            # Convert assignment keys to "r,c" strings for JSON serializability
            return {
                "rows": s["rows"],
                "cols": s["cols"],
                "assignment": {f"{r},{c}": v for (r, c), v in s["assignment"].items()},
                "puzzle_id": s.get("puzzle_id", "unknown"),
            }

        return {
            "env_short": "hidato5x4",
            "env_version": env_mod.ENV_VERSION,
            "state_text_version": env_mod.TEXT_VERSION,
            "system_prompt": SYS_HID,
            "policy_eval_prompt_version": "hidato_minimal_v1",
            "render_state": env_mod.render_state_hidato,
            "state_struct_for_record": _state_struct_for_record,
            "parse_action": env_mod.parse_action_text,
            "action_text": lambda a: env_mod.action_text(_to_action(a)),
            "canonical_action": lambda a: env_mod.canonical_action(_to_action(a)),
            "action_hash": lambda a: env_mod.action_hash(_to_action(a)),
            "action_struct_to_dict": lambda a: a.to_dict() if hasattr(a, "to_dict") else a,
            "is_local_valid": env_mod.is_local_valid,
            "apply_action": env_mod.apply_action,
            "is_goal": env_mod.is_goal,
            "enumerate_legal_actions": env_mod.enumerate_legal_actions,
            "state_hash": env_mod.state_hash,
            "generate_root_puzzle": lambda seed, **kw: env_mod.get_root_puzzle(seed=seed),
            "solver": solver,
            "solver_version": solver_mod.SOLVER_VERSION,
            "solver_solve": solver.solve,
            "solver_is_viable": solver.is_viable,
            "solver_find_one_solution": solver.find_one_solution,
            "apply_sol_step": _apply_sol_step,
            "compute_progress": progress_mod.compute_progress,
            "ActionStruct": AS,
            "perturbation_kind": "hidato",
        }
    else:
        raise ValueError(f"Unknown env: {env_name}")


_SUDOKU_SYSTEM_PROMPT_FIXED = """You are solving a 4x4 Sudoku puzzle. Fill in empty cells (shown as .) with numbers 1-4 so that each row, column, and 2x2 box contains each number exactly once.

Grid format: Numbers separated by spaces, | separates the left and right 2x2 boxes within each row, --------- separates the top and bottom row halves.

In your reasoning:
1. Describe the current state in <observation>
2. Predict the next state after your move in <prediction>
3. Assess whether the resulting state will still be solvable in <solvable>: true/false

Then provide your action in <answer> using format: place N at row R col C"""


# ---------------------------------------------------------------------------
# Per-role sampling protocols (per spec §6.3)
# ---------------------------------------------------------------------------

SAMPLING_PROTOCOLS = {
    "train_balanced": {
        "K_total": 12, "K_lt": 3, "K_ht": 3, "K_rand": 0, "K_sol": 3, "K_prt": 3,
        "lt_temperature": 0.3, "ht_temperature": 1.0, "top_p": 0.95,
    },
    "val_natural_calibration": {
        "K_total": 8, "K_lt": 4, "K_ht": 4, "K_rand": 0, "K_sol": 0, "K_prt": 0,
        "lt_temperature": 0.3, "ht_temperature": 1.0, "top_p": 0.95,
    },
    "test_natural_policy": {
        "K_total": 8, "K_lt": 4, "K_ht": 4, "K_rand": 0, "K_sol": 0, "K_prt": 0,
        "lt_temperature": 0.3, "ht_temperature": 1.0, "top_p": 0.95,
    },
}


# ---------------------------------------------------------------------------
# Action regex for extracting <answer>...</answer>
# ---------------------------------------------------------------------------

import re
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


def extract_answer(text: str) -> Optional[str]:
    m = _ANSWER_RE.search(text)
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# Candidate sources (lt / ht / sol / prt)
# ---------------------------------------------------------------------------

def sample_lt_or_ht(model, tokenizer, prompt: str, K: int, temperature: float, top_p: float,
                     adapter: dict) -> List[dict]:
    """Sample K candidates from policy at given temperature.

    Returns list of dicts with {source, action_text_raw, generation_logprob,
    parsed_action_struct (or None if parse fail)}. Caller dedups by canonical hash.
    """
    if K == 0:
        return []
    samples = batched_sample(model, tokenizer, prompt, K=K,
                              temperature=temperature, top_p=top_p)
    out = []
    parse = adapter["parse_action"]
    for s in samples:
        ans_text = extract_answer(s.text)
        if not ans_text:
            out.append({
                "source_text": ans_text,
                "raw_response": s.text,
                "generation_logprob": s.generation_logprob,
                "parsed_action": None,
            })
            continue
        parsed = parse(ans_text)
        out.append({
            "source_text": ans_text,
            "raw_response": s.text,
            "generation_logprob": s.generation_logprob,
            "parsed_action": parsed,
        })
    return out


def sample_sol(state: dict, K: int, adapter: dict) -> List[Any]:
    """K candidates from the solver's solution path. Returns ActionStruct objects
    that are LOCALLY VALID at the current state.

    Why we filter:
      - Sudoku: sol_path[0] is at the current state. sol_path[1+] are placements
        at hypothetical successor states; they may or may not be locally valid
        at the current state (different cell + value).
      - Hidato: env enforces value == state.next_n. Only sol_path[0] satisfies
        this at the current state; sol_path[1] has value == next_n+1 and fails
        is_local_valid.
      - Pentomino: sol_path[0] is at the current state. sol_path[1+] places a
        DIFFERENT piece (already used remaining_pieces shrunk), so they're not
        locally valid.

    The fix: filter through is_local_valid. For sequential games (Hidato), this
    naturally collapses to {sol_path[0]}.
    """
    if K == 0:
        return []
    sol_path = adapter["solver_find_one_solution"](state)
    if not sol_path:
        return []
    AS = adapter["ActionStruct"]
    kind = adapter["perturbation_kind"]
    is_local_valid = adapter["is_local_valid"]
    out = []
    for entry in sol_path:
        if kind == "polyomino":
            piece, ori, ar, ac = entry
            cand = AS(piece=piece, ori=ori, row=ar + 1, col=ac + 1)
        else:
            # sudoku / hidato: (r0, c0, v)
            r0, c0, v = entry
            cand = AS(row=r0 + 1, col=c0 + 1, value=v)
        if is_local_valid(state, cand):
            out.append(cand)
            if len(out) >= K:
                return out
    return out


def sample_prt(state: dict, sol_actions: List[Any], K: int, adapter: dict) -> List[Any]:
    """K perturbed-solver-path actions: locally legal but globally doomed.

    Per-env perturbation strategy:
      - Sudoku: at sol[0]'s (row, col), try alternative values v' ∈ {1..4}
        that are locally legal but make the state unsolvable.
      - Hidato: at sol[0]'s (row, col), there's only one valid value (next_n);
        instead, try OTHER adjacent cells (alternative placements of next_n)
        that are legal-but-doomed.
      - Pentomino: at sol[0]'s anchor, try other (piece, ori, anchor) combos
        that fit at the same focus cell but lead to unsolvable boards.
    """
    if K == 0 or not sol_actions:
        return []
    AS = adapter["ActionStruct"]
    is_local_valid = adapter["is_local_valid"]
    apply_act = adapter["apply_action"]
    solver_is_viable = adapter["solver_is_viable"]
    kind = adapter["perturbation_kind"]
    out = []
    seen = set()

    if kind == "sudoku":
        for sol_action in sol_actions:
            r, c, original_v = sol_action.row, sol_action.col, sol_action.value
            for v_prime in range(1, 5):
                if v_prime == original_v: continue
                if (r, c, v_prime) in seen: continue
                cand = AS(row=r, col=c, value=v_prime)
                if not is_local_valid(state, cand): continue
                if not solver_is_viable(apply_act(state, cand)):
                    out.append(cand); seen.add((r, c, v_prime))
                    if len(out) >= K: return out
            if len(out) >= K: return out

    elif kind == "hidato":
        # Hidato: at the next-n anchor, try alternative empty adjacent cells
        # that are legal but doomed. value is forced (= next_n).
        legal = adapter["enumerate_legal_actions"](state)
        sol_target_cells = {(a.row, a.col) for a in sol_actions}
        for cand in legal:
            if (cand.row, cand.col) in sol_target_cells: continue
            if (cand.row, cand.col, cand.value) in seen: continue
            if not solver_is_viable(apply_act(state, cand)): continue
            out.append(cand); seen.add((cand.row, cand.col, cand.value))
            if len(out) >= K: return out

    elif kind == "polyomino":
        # Pentomino: try all legal placements; pick those that result in unsolvable
        # boards and aren't already in sol_actions.
        legal = adapter["enumerate_legal_actions"](state)
        sol_keys = {adapter["canonical_action"](a) for a in sol_actions}
        for cand in legal:
            ck = adapter["canonical_action"](cand)
            if ck in sol_keys or ck in seen: continue
            if not solver_is_viable(apply_act(state, cand)): continue
            out.append(cand); seen.add(ck)
            if len(out) >= K: return out

    return out


# ---------------------------------------------------------------------------
# Main sibling-set assembly
# ---------------------------------------------------------------------------

def _action_to_struct_dict(a, adapter) -> Optional[Dict[str, Any]]:
    if a is None:
        return None
    return adapter["action_struct_to_dict"](a)


def _get_state_solver_block(state: dict, adapter: dict) -> Tuple[dict, bool, bool]:
    """Return (state_solver dict, state_viable, state_is_goal)."""
    res = adapter["solver_solve"](state)
    state_viable = res.solvable
    state_is_goal = adapter["is_goal"](state)
    return ({
        "num_solutions": res.num_solutions,
        "nodes": res.nodes,
        "backtracks": res.backtracks,
        "solve_time_ms": res.solve_time_ms,
    }, state_viable, state_is_goal)


def _action_space_stats(state: dict, adapter: dict) -> dict:
    """Counts over the FULL legal action space at state s."""
    apply_act = adapter["apply_action"]
    solver_is_viable = adapter["solver_is_viable"]
    legal = adapter["enumerate_legal_actions"](state)
    n_viable = 0
    for a in legal:
        new_state = apply_act(state, a)
        if solver_is_viable(new_state):
            n_viable += 1
    return {
        "num_legal_actions": len(legal),
        "num_legal_viable_actions": n_viable,
        "num_legal_doomed_actions": len(legal) - n_viable,
    }


def _classify_candidate(parse_valid: bool, local_valid: bool,
                         next_is_goal: bool, next_viable: bool) -> str:
    if not parse_valid:
        return "parse_invalid"
    if not local_valid:
        return "local_invalid"
    if next_is_goal:
        return "goal_reaching"
    if next_viable:
        return "valid_viable"
    return "valid_doomed"


def _build_candidate(
    cand_idx: int,
    source: str,
    source_meta: dict,
    raw_action_struct: Any,            # ActionStruct or None
    raw_action_text: Optional[str],     # text the policy produced; None for sol/prt
    raw_response: Optional[str],        # full LLM response text; None for sol/prt
    generation_logprob: Optional[float],
    state: dict,
    adapter: dict,
    model, tokenizer, eval_prompt: str,
    policy_eval_model_path: str,
    policy_eval_prompt_version: str,
) -> dict:
    """Build one candidate dict per save_sibling_set_v1.2 schema."""
    # parse_valid: did we end up with a parseable action?
    parse_valid = raw_action_struct is not None
    if parse_valid:
        action_struct = raw_action_struct
        action_text = adapter["action_text"](action_struct)
        action_canonical = adapter["canonical_action"](action_struct)
        action_hash_val = adapter["action_hash"](action_struct)
    else:
        action_struct = None
        action_text = raw_action_text or ""
        action_canonical = ""
        action_hash_val = "sha1:" + ("0" * 40)  # placeholder for parse-invalid

    # local_valid + transition + solver labels
    if parse_valid:
        local_valid = adapter["is_local_valid"](state, action_struct)
    else:
        local_valid = False

    next_state_dict = None
    candidate_solver = None
    candidate_progress = None
    score_labels = None
    next_is_goal = False
    next_viable = False
    candidate_class = "parse_invalid"

    if parse_valid and local_valid:
        next_state = adapter["apply_action"](state, action_struct)
        next_state_text = adapter["render_state"](next_state)
        next_state_hash = adapter["state_hash"](next_state)
        next_is_goal = adapter["is_goal"](next_state)

        solver_res = adapter["solver_solve"](next_state)
        next_viable = solver_res.solvable

        next_state_dict = {
            "next_state_hash": next_state_hash,
            "next_state_struct": adapter["state_struct_for_record"](next_state),
            "next_state_text": next_state_text,
            "next_state_text_version": adapter["state_text_version"],
            "next_is_goal": next_is_goal,
            "next_viable": next_viable,
        }
        candidate_solver = {
            "solvable": solver_res.solvable,
            "num_solutions": solver_res.num_solutions,
            "nodes": solver_res.nodes,
            "backtracks": solver_res.backtracks,
            "solution_depth": solver_res.solution_depth,
            "solve_time_ms": solver_res.solve_time_ms,
            "solver_version": adapter["solver_version"],
        }
        progress_dict = adapter["compute_progress"](next_state)
        candidate_progress = progress_dict
        score_labels = {
            "local_progress": {
                "value": progress_dict["local_progress_score"],
                "use_for_main_score_baseline": True,
            },
            "solver_residual_difficulty": {
                "value": -math.log(1 + solver_res.nodes),
                "formula": "-log(1 + solver_nodes)",
                "use_for_main_score_baseline": False,
            },
        }
        candidate_class = _classify_candidate(parse_valid, local_valid,
                                                next_is_goal, next_viable)
    else:
        candidate_class = _classify_candidate(parse_valid, local_valid, False, False)

    # policy_eval_logprob: ALWAYS computed for every candidate (spec §4.2).
    # We compute eval_response here but DEFER the logprob call so that all K
    # responses for a sibling set can be batched in one forward pass (see
    # generate_one_sibling_set). _eval_response is a private hand-off field
    # consumed there and stripped before the candidate is finalized.
    if raw_response is not None and raw_response.strip():
        eval_response = raw_response
    elif action_text:
        eval_response = f"<answer>{action_text}</answer>"
    else:
        eval_response = ""
    eval_lp = float("nan")  # filled in by batched call later

    return {
        "candidate_id": f"c{cand_idx:03d}",
        "action_hash": action_hash_val,
        "display_rank": cand_idx,
        "source": source,
        "source_meta": source_meta,
        "action_text": action_text,
        "action_text_canonical": action_canonical,
        "action_struct": _action_to_struct_dict(action_struct, adapter),
        "_eval_response": eval_response,  # private; stripped by caller after batched logprob
        "logprobs": {
            "generation_logprob": generation_logprob,
            "policy_eval_logprob": eval_lp,
            "policy_eval_model": policy_eval_model_path,
            "policy_eval_prompt_version": policy_eval_prompt_version,
        },
        "parse_valid": parse_valid,
        "local_valid": local_valid,
        "transition_valid": parse_valid and local_valid,
        "candidate_class": candidate_class,
        "eligible_for_viability_eval": local_valid,
        "next_state": next_state_dict,
        "solver": candidate_solver,
        "progress": candidate_progress,
        "score_labels": score_labels,
    }


def _set_stats(candidates: List[dict]) -> dict:
    """Aggregate candidate counts into set_stats per spec §2.2."""
    n = len(candidates)
    n_parse_invalid = sum(1 for c in candidates if c["candidate_class"] == "parse_invalid")
    n_local_invalid = sum(1 for c in candidates if c["candidate_class"] == "local_invalid")
    n_valid_viable = sum(1 for c in candidates if c["candidate_class"] == "valid_viable")
    n_valid_doomed = sum(1 for c in candidates if c["candidate_class"] == "valid_doomed")
    n_goal_reaching = sum(1 for c in candidates if c["candidate_class"] == "goal_reaching")
    mixed = (n_valid_viable > 0) and (n_valid_doomed > 0)
    all_viable = (n_valid_viable + n_goal_reaching) == n - n_parse_invalid - n_local_invalid and n > 0
    all_doomed = (n_valid_doomed) == n and n > 0

    sources = ["lt", "ht", "rand", "sol", "prt"]
    breakdown = {}
    for src in sources:
        srcs = [c for c in candidates if c["source"] == src]
        if not srcs:
            continue
        breakdown[src] = {
            "selected": len(srcs),
            "valid_viable": sum(1 for c in srcs if c["candidate_class"] == "valid_viable"),
            "valid_doomed": sum(1 for c in srcs if c["candidate_class"] == "valid_doomed"),
            "invalid": sum(1 for c in srcs if c["candidate_class"] in ("parse_invalid", "local_invalid")),
            "goal_reaching": sum(1 for c in srcs if c["candidate_class"] == "goal_reaching"),
        }

    return {
        "num_candidates": n,
        "num_parse_invalid": n_parse_invalid,
        "num_local_invalid": n_local_invalid,
        "num_valid_viable": n_valid_viable,
        "num_valid_doomed": n_valid_doomed,
        "num_goal_reaching": n_goal_reaching,
        "mixed": mixed,
        "all_viable": all_viable,
        "all_doomed": all_doomed,
        "source_breakdown": breakdown,
    }


def _mine_deceptive_pairs(candidates: List[dict]) -> List[dict]:
    """Spec §2.2: emit pairs (a+, a-) where a+ valid_viable, a- valid_doomed,
    and a-.progress >= a+.progress."""
    out = []
    pluses = [c for c in candidates if c["candidate_class"] == "valid_viable"]
    minuses = [c for c in candidates if c["candidate_class"] == "valid_doomed"]
    pair_idx = 0
    for plus in pluses:
        plus_prog = plus["progress"]["local_progress_score"]
        for minus in minuses:
            minus_prog = minus["progress"]["local_progress_score"]
            if minus_prog >= plus_prog:
                gap = minus_prog - plus_prog
                out.append({
                    "pair_id": f"pair_{pair_idx:03d}",
                    "a_plus_candidate_id": plus["candidate_id"],
                    "a_minus_candidate_id": minus["candidate_id"],
                    "condition": {
                        "plus_next_viable": True,
                        "minus_next_viable": False,
                        "both_local_valid": True,
                        "minus_progress_ge_plus": True,
                        "progress_gap": gap,
                    },
                })
                pair_idx += 1
    return out


def _mix_score(candidates: List[dict]) -> float:
    n = len(candidates)
    if n == 0:
        return 0.0
    n_v = sum(1 for c in candidates if c["candidate_class"] == "valid_viable")
    n_d = sum(1 for c in candidates if c["candidate_class"] == "valid_doomed")
    return min(n_v, n_d) / float(n)


def generate_one_sibling_set(
    state: dict, adapter: dict, sampling_protocol: dict,
    model, tokenizer, system_prompt: str,
    policy_model_path: str, policy_checkpoint_id: str,
    policy_eval_prompt_version: str, training_phase: str,
    set_seed: int,
    root_id: str, trajectory_id: str, sibling_set_id: str,
    t: int, dataset_role: str, split: str,
) -> Optional[dict]:
    """Generate one full sibling-set record. Returns dict or None if ineligible
    (e.g. state is goal or no legal actions)."""

    state_solver_block, state_viable, state_is_goal = _get_state_solver_block(state, adapter)
    if state_is_goal:
        return None  # don't sample sibling sets at the goal

    state_text = adapter["render_state"](state)
    user_msg = f"Current state:\n{state_text}"
    chat_prompt = build_chat_prompt(tokenizer, system_prompt, user_msg)

    legal = adapter["enumerate_legal_actions"](state)
    if not legal:
        return None  # dead end, no sibling set

    K_total = sampling_protocol["K_total"]
    K_lt = sampling_protocol["K_lt"]
    K_ht = sampling_protocol["K_ht"]
    K_sol = sampling_protocol["K_sol"]
    K_prt = sampling_protocol["K_prt"]

    # 1) Gather candidates from all sources
    raw = []  # list of (source, source_meta, action_struct_or_None, raw_text, raw_response, gen_logprob)
    rng = random.Random(set_seed)

    if K_lt > 0:
        lt_results = sample_lt_or_ht(model, tokenizer, chat_prompt, K_lt,
                                      sampling_protocol["lt_temperature"],
                                      sampling_protocol["top_p"], adapter)
        for r in lt_results:
            raw.append(("lt",
                        {"temperature": sampling_protocol["lt_temperature"],
                         "top_p": sampling_protocol["top_p"],
                         "is_oracle_injected": False},
                        r["parsed_action"], r["source_text"], r["raw_response"],
                        r["generation_logprob"]))
    if K_ht > 0:
        ht_results = sample_lt_or_ht(model, tokenizer, chat_prompt, K_ht,
                                      sampling_protocol["ht_temperature"],
                                      sampling_protocol["top_p"], adapter)
        for r in ht_results:
            raw.append(("ht",
                        {"temperature": sampling_protocol["ht_temperature"],
                         "top_p": sampling_protocol["top_p"],
                         "is_oracle_injected": False},
                        r["parsed_action"], r["source_text"], r["raw_response"],
                        r["generation_logprob"]))

    sol_actions = sample_sol(state, K_sol, adapter) if K_sol > 0 else []
    for a in sol_actions:
        raw.append(("sol",
                    {"temperature": None, "top_p": None, "is_oracle_injected": True},
                    a, None, None, None))

    if K_prt > 0:
        prt_actions = sample_prt(state, sol_actions, K_prt, adapter)
        for a in prt_actions:
            raw.append(("prt",
                        {"temperature": None, "top_p": None, "is_oracle_injected": True},
                        a, None, None, None))

    pool_size_before_dedup = len(raw)

    # 2) Dedup by canonical action hash. Parse-invalid candidates dedup by raw_text.
    seen_hashes = set()
    deduped = []
    for entry in raw:
        source, source_meta, action, raw_text, raw_resp, gen_lp = entry
        if action is not None:
            h = adapter["action_hash"](action)
        elif raw_text:
            import hashlib as _hl
            h = "raw:" + _hl.sha1(raw_text.encode()).hexdigest()
        else:
            h = "empty"
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        deduped.append(entry)

    # 3) Shuffle (per spec §2.2 candidate_order = "random_shuffled")
    rng.shuffle(deduped)
    pool_size_after_dedup = len(deduped)

    # 4) Build candidate dicts (validate, transition, solver-label, progress, eval_logprob)
    candidates = []
    for i, (source, source_meta, action, raw_text, raw_resp, gen_lp) in enumerate(deduped):
        cand = _build_candidate(
            cand_idx=i,
            source=source,
            source_meta=source_meta,
            raw_action_struct=action,
            raw_action_text=raw_text,
            raw_response=raw_resp,
            generation_logprob=gen_lp,
            state=state,
            adapter=adapter,
            model=model, tokenizer=tokenizer, eval_prompt=chat_prompt,
            policy_eval_model_path=policy_model_path,
            policy_eval_prompt_version=policy_eval_prompt_version,
        )
        candidates.append(cand)

    # 4.5) Batched policy_eval_logprob across all candidates (spec §4.2).
    # Replaces N sequential forward passes with 1 batched forward pass.
    eval_responses = [c.pop("_eval_response", "") for c in candidates]
    if any(eval_responses):
        lps = batched_policy_eval_logprob(model, tokenizer, chat_prompt, eval_responses)
    else:
        lps = [float("nan")] * len(candidates)
    for c, resp, lp in zip(candidates, eval_responses, lps):
        c["logprobs"]["policy_eval_logprob"] = lp if resp else float("nan")

    # 5) Aggregate stats
    set_stats = _set_stats(candidates)
    deceptive_pairs = _mine_deceptive_pairs(candidates)
    mix = _mix_score(candidates)

    # 6) Build the State block
    state_text_full = adapter["render_state"](state)
    state_hash_val = adapter["state_hash"](state)
    action_space_stats = _action_space_stats(state, adapter)

    state_block = {
        "state_hash": state_hash_val,
        "state_struct": adapter["state_struct_for_record"](state),
        "state_text": state_text_full,
        "state_text_version": adapter["state_text_version"],
        "state_viable": state_viable,
        "state_is_goal": state_is_goal,
        "state_solver": state_solver_block,
        "action_space_stats": action_space_stats,
    }

    # 7) Provenance — git commit was cached at module import (see _GIT_COMMIT below)
    git_commit = _GIT_COMMIT

    record = {
        "schema": "save_sibling_set_v1.2",
        "env": adapter["env_short"],
        "dataset_role": dataset_role,
        "split": split,
        "root_id": root_id,
        "trajectory_id": trajectory_id,
        "sibling_set_id": sibling_set_id,
        "t": t,
        "state": state_block,
        "sampling_protocol": {
            "K_total": K_total,
            "K_lt": K_lt, "K_ht": K_ht, "K_rand": sampling_protocol["K_rand"],
            "K_sol": K_sol, "K_prt": K_prt,
            "lt_temperature": sampling_protocol["lt_temperature"],
            "ht_temperature": sampling_protocol["ht_temperature"],
            "top_p": sampling_protocol["top_p"],
            "policy_model": policy_model_path,
            "policy_checkpoint": policy_checkpoint_id,
            "training_phase": training_phase,
            "candidate_order": "random_shuffled",
            "shuffle_seed": set_seed,
            "set_seed": set_seed,
            "dedup_policy": "canonical_action_hash",
            "candidate_pool_size_before_dedup": pool_size_before_dedup,
            "candidate_pool_size_after_dedup": pool_size_after_dedup,
        },
        "candidates": candidates,
        "set_stats": set_stats,
        "deceptive_pairs": deceptive_pairs,
        "selection_criteria": {
            "mix_score": mix,
            "mix_score_formula": "min(num_valid_viable, num_valid_doomed) / num_candidates",
            "is_boundary": mix >= 0.3,
            "boundary_threshold": 0.3,
            "selection_reason": ([
                "mixed_sibling_set" if set_stats["mixed"] else "non_mixed",
                "boundary_state" if mix >= 0.3 else "non_boundary"
            ]),
        },
        "provenance": {
            "env_version": adapter["env_version"],
            "solver_version": adapter["solver_version"],
            "generator_version": "save_data_gen_v1",
            "git_commit": git_commit,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    }

    return record


# ---------------------------------------------------------------------------
# Trajectory-level orchestration
# ---------------------------------------------------------------------------

def is_boundary_state_simple(state: dict, adapter: dict, threshold: float = 0.3) -> bool:
    """Per spec §4.3: count viable vs doomed legal actions; mix >= threshold."""
    legal = adapter["enumerate_legal_actions"](state)
    if not legal:
        return False
    apply_act = adapter["apply_action"]
    solver_is_viable = adapter["solver_is_viable"]
    n_viable = 0
    for a in legal:
        new_state = apply_act(state, a)
        if solver_is_viable(new_state):
            n_viable += 1
    n_doomed = len(legal) - n_viable
    mix = min(n_viable, n_doomed) / float(len(legal))
    return mix >= threshold


def generate_dataset(
    env_name: str,
    role: str,
    n_target: int,
    output_path: str,
    policy_model_path: str,
    policy_checkpoint_id: str,
    seed: int,
    n_root_puzzles: int = 500,
    sibling_sets_per_root: int = 3,
    boundary_threshold: float = 0.3,
    boundary_oversampling_rate: float = 0.5,
    n_empty_sudoku: int = 6,
):
    """Top-level generation driver. Writes JSONL to `output_path`."""
    adapter = get_env_adapter(env_name)
    sampling_protocol = SAMPLING_PROTOCOLS[role]

    print(f"=== generate_save_data: env={env_name} role={role} target={n_target} ===")
    print(f"  policy: {policy_model_path}")
    print(f"  output: {output_path}")
    print(f"  K mixture: lt={sampling_protocol['K_lt']} ht={sampling_protocol['K_ht']} "
          f"sol={sampling_protocol['K_sol']} prt={sampling_protocol['K_prt']}")

    print("Loading model + tokenizer ...")
    model, tokenizer, device = load_model(policy_model_path)
    print(f"  loaded; device={device}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n_emitted = 0
    n_skipped_state_goal = 0
    n_skipped_no_legal = 0
    n_skipped_non_boundary = 0
    rng = random.Random(seed)

    split_name = "train" if role == "train_balanced" else ("val" if "val" in role else "test")

    t0 = time.perf_counter()
    with open(output_path, "w") as f:
        # Determine puzzle index range based on split for leakage prevention
        # Per spec §4.4: 70/15/15 by puzzle hash. For toy run, use seed ranges:
        if split_name == "train":
            puzzle_seed_range = (0, int(n_root_puzzles * 0.7))
        elif split_name == "val":
            puzzle_seed_range = (int(n_root_puzzles * 0.7), int(n_root_puzzles * 0.85))
        else:
            puzzle_seed_range = (int(n_root_puzzles * 0.85), n_root_puzzles)

        for root_idx in range(*puzzle_seed_range):
            if n_emitted >= n_target:
                break
            # Generate root puzzle (env-aware; Sudoku uses n_empty, others ignore it)
            puzzle_seed = seed * 1000 + root_idx
            try:
                state = adapter["generate_root_puzzle"](seed=puzzle_seed,
                                                         n_empty=n_empty_sudoku)
            except Exception as e:
                print(f"  skip root_idx={root_idx}: {e}")
                continue

            # Verify root is solvable
            if not adapter["solver_is_viable"](state):
                continue

            root_id = f"{adapter['env_short']}_{split_name}_{root_idx:06d}"
            # Use the solver's path as the trajectory
            sol_path = adapter["solver_find_one_solution"](state)
            if not sol_path:
                continue
            trajectory_id = f"{root_id}_solver_00"

            # Walk along the solver's path. At each intermediate state, optionally
            # generate a sibling set (skip non-boundary with prob boundary_oversampling_rate).
            current_state = state
            sets_in_this_root = 0
            for t_idx, sol_step in enumerate(sol_path):
                if n_emitted >= n_target:
                    break
                if sets_in_this_root >= sibling_sets_per_root:
                    break

                # Skip last 2 steps: too forced to be informative
                if t_idx >= len(sol_path) - 2:
                    continue

                # Boundary check
                is_bdry = is_boundary_state_simple(current_state, adapter, boundary_threshold)
                if not is_bdry and rng.random() < boundary_oversampling_rate:
                    n_skipped_non_boundary += 1
                    current_state = adapter["apply_sol_step"](current_state, sol_step)
                    continue

                set_seed = (seed * 1_000_000 + root_idx * 1000 + t_idx) % (2**31)
                sibling_set_id = f"{trajectory_id}_t{t_idx:03d}_set00"

                record = generate_one_sibling_set(
                    state=current_state,
                    adapter=adapter,
                    sampling_protocol=sampling_protocol,
                    model=model, tokenizer=tokenizer,
                    system_prompt=adapter["system_prompt"],
                    policy_model_path=policy_model_path,
                    policy_checkpoint_id=policy_checkpoint_id,
                    policy_eval_prompt_version=adapter["policy_eval_prompt_version"],
                    training_phase="rl",
                    set_seed=set_seed,
                    root_id=root_id, trajectory_id=trajectory_id, sibling_set_id=sibling_set_id,
                    t=t_idx, dataset_role=role, split=split_name,
                )
                if record is None:
                    if adapter["is_goal"](current_state):
                        n_skipped_state_goal += 1
                    else:
                        n_skipped_no_legal += 1
                else:
                    # Pydantic-validate before emit (Sudoku only — Pentomino/Hidato
                    # use Dict[str, Any]-like state_struct that doesn't match the
                    # spec-strict schema yet. Will add per-env subclass schemas later.)
                    if adapter["env_short"] == "sudoku4":
                        try:
                            SiblingSetRecord.model_validate(record)
                        except Exception as e:
                            print(f"  schema validation failed for {sibling_set_id}: {e}")
                            continue
                    f.write(json.dumps(record) + "\n")
                    f.flush()
                    n_emitted += 1
                    sets_in_this_root += 1
                    if n_emitted % 5 == 0:
                        elapsed = time.perf_counter() - t0
                        print(f"  emitted {n_emitted}/{n_target} "
                              f"(elapsed {elapsed:.0f}s, rate {n_emitted/max(1,elapsed):.2f}/s)")

                # Advance state along the trajectory
                current_state = adapter["apply_sol_step"](current_state, sol_step)

    elapsed = time.perf_counter() - t0
    print(f"\n=== Done: emitted={n_emitted}/{n_target} in {elapsed:.0f}s ===")
    print(f"  skipped: goal={n_skipped_state_goal} no_legal={n_skipped_no_legal} "
          f"non_boundary={n_skipped_non_boundary}")
    print(f"  output: {output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=["sudoku", "polyomino", "hidato"])
    p.add_argument("--role", required=True, choices=list(SAMPLING_PROTOCOLS.keys()))
    p.add_argument("--n-target", type=int, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--policy-model", required=True)
    p.add_argument("--policy-checkpoint-id", required=True,
                   help="Stable identifier (e.g., 'rl_b5_phase3_v8_anchor_final')")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-root-puzzles", type=int, default=500)
    p.add_argument("--sibling-sets-per-root", type=int, default=3)
    p.add_argument("--boundary-threshold", type=float, default=0.3)
    p.add_argument("--boundary-oversampling-rate", type=float, default=0.5)
    p.add_argument("--n-empty-sudoku", type=int, default=6)
    args = p.parse_args()

    generate_dataset(
        env_name=args.env, role=args.role,
        n_target=args.n_target, output_path=args.output,
        policy_model_path=args.policy_model,
        policy_checkpoint_id=args.policy_checkpoint_id,
        seed=args.seed,
        n_root_puzzles=args.n_root_puzzles,
        sibling_sets_per_root=args.sibling_sets_per_root,
        boundary_threshold=args.boundary_threshold,
        boundary_oversampling_rate=args.boundary_oversampling_rate,
        n_empty_sudoku=args.n_empty_sudoku,
    )


if __name__ == "__main__":
    main()
