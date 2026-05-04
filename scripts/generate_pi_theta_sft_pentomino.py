"""Generate base-policy SFT data for Pentomino 5×6 (multi-piece-set).

Schema: pi_theta_sft_v1 (the spec for training a non-OOD π_θ).

Strategy:
  1. Enumerate all valid k-piece subsets that tile the (h, w) board.
     Default 5×6 with k=6 → 172 valid subsets.
  2. For each subset, find up to `max_tilings_per_subset` distinct tilings.
  3. For each tiling, walk the solution path step-by-step. At each step,
     emit one (state, action) sample:
       prompt = render(board, remaining_pieces)
       response = "place {piece} ori={K} at row {R} col {C}"
  4. Split at (subset, tiling) level for leakage prevention: 90% train, 10% val.

Output format: pi_theta_sft_v1 schema — JSONL with {prompt, response, metadata}.

Plain action response format (NO XML, NO viability tag) per user directive
2026-05-04: π_θ has one job (pick actions). The viability prediction is f_φ's
job, separately. See doc/SAVE_handoff.md §3 + the SFT spec discussion.

Usage (5×6 default, ~172 subsets × 10 tilings × 6 steps ≈ 10K samples):
  python scripts/generate_pi_theta_sft_pentomino.py \\
      --board-h 5 --board-w 6 \\
      --k-pieces 6 \\
      --output-dir data/pentomino5x6/pi_theta_sft \\
      --max-tilings-per-subset 10 \\
      --val-fraction 0.1 \\
      --seed 42
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.environments.polyomino_utils import ALL_PIECES, PIECE_ORIENTATIONS, placement_cells
from scripts.pentomino_solver import PentominoSolver
from scripts.pentomino_env import (
    ActionStruct, render_state_b8, action_text, action_hash,
)


# --- Board symmetry augmentation ---
#
# 5x6 (non-square) has 4 symmetries: identity, 180° rot, h-flip, v-flip.
# (90° rot would yield 6x5 which isn't our board shape.)
# For each symmetry, transform each (piece, ori, anchor_r, anchor_c) placement
# into the (piece, ori', anchor_r', anchor_c') that produces the symmetric cell
# set. Piece identity is preserved — symmetries permute orientations and shift
# anchors but never relabel pieces.

SYMMETRIES = ["identity", "rot180", "vflip", "hflip"]


def _transform_cell(r: int, c: int, h: int, w: int, sym: str) -> Tuple[int, int]:
    if sym == "identity":
        return r, c
    if sym == "rot180":
        return h - 1 - r, w - 1 - c
    if sym == "vflip":   # top-bottom
        return h - 1 - r, c
    if sym == "hflip":   # left-right
        return r, w - 1 - c
    raise ValueError(sym)


def _find_equivalent_placement(piece: str, ori: int, ar: int, ac: int,
                                 h: int, w: int, sym: str
                                 ) -> Optional[Tuple[str, int, int, int]]:
    """Find (piece, ori', anchor_r', anchor_c') whose cells equal the symmetry-
    transformed cells of the original placement. Returns None if no match
    (shouldn't happen for valid placements + valid symmetries)."""
    cells = placement_cells(piece, ori, ar, ac)
    if cells is None:
        return None
    target = frozenset(_transform_cell(r, c, h, w, sym) for r, c in cells)
    n_oris = len(PIECE_ORIENTATIONS[piece])
    for new_ori in range(n_oris):
        for nr in range(h):
            for nc in range(w):
                test = placement_cells(piece, new_ori, nr, nc)
                if test is None:
                    continue
                if frozenset(test) == target:
                    return (piece, new_ori, nr, nc)
    return None


def _transform_tiling(tiling: List[Tuple[str, int, int, int]],
                       h: int, w: int, sym: str
                       ) -> Optional[List[Tuple[str, int, int, int]]]:
    """Apply symmetry to every placement in a tiling. Preserves placement order
    (the model still walks the same logical sequence — just on a flipped board).
    Returns None if any placement can't be remapped."""
    out = []
    for piece, ori, ar, ac in tiling:
        eq = _find_equivalent_placement(piece, ori, ar, ac, h, w, sym)
        if eq is None:
            return None
        out.append(eq)
    return out


def _augment_tilings(tilings: List[List[Tuple[str, int, int, int]]],
                      h: int, w: int
                      ) -> List[Tuple[str, List[Tuple[str, int, int, int]]]]:
    """For each tiling, emit (sym_name, transformed_tiling) for each symmetry,
    deduplicating tilings that map to themselves under symmetry."""
    out = []
    for tiling in tilings:
        seen = set()
        for sym in SYMMETRIES:
            tr = _transform_tiling(tiling, h, w, sym)
            if tr is None:
                continue
            key = tuple(tr)
            if key in seen:
                continue
            seen.add(key)
            out.append((sym, tr))
    return out


SCHEMA_VERSION = "pi_theta_sft_v1"


SYSTEM_PROMPT = """You are solving a pentomino tiling puzzle. The board is a rectangular grid; you must place the given pentomino pieces so that every cell is covered exactly once, with no overlaps and no piece extending outside the board.

Pieces use the standard letters: F, I, L, N, P, T, U, V, W, X, Y, Z. Each piece is 5 unit squares. Pieces can be rotated and reflected, giving multiple orientations per piece.

Board format: each cell shows '.' for empty or the piece-letter that occupies it.

Respond with ONE move in the format: place {piece} ori={K} at row {R} col {C}
where {piece} is one of the remaining pieces, {K} is the orientation id, and (R, C) are 1-indexed anchor coordinates."""


def render_user_message(board: List[List[str]], remaining: List[str]) -> str:
    """User message: rendered state + question prompt."""
    state_text = render_state_b8(board, remaining)
    return f"Current state:\n{state_text}\n\nWhat is your next move?"


def render_response(action: ActionStruct) -> str:
    """Plain action response — NO XML, NO viability tag."""
    return action_text(action)


def find_valid_subsets(board_h: int, board_w: int, k_pieces: int) -> List[Tuple[str, ...]]:
    """Enumerate all k-piece subsets that tile the (h, w) board."""
    solver = PentominoSolver(board_h=board_h, board_w=board_w,
                              solution_cap=1, node_cap=200_000)
    empty = [["."] * board_w for _ in range(board_h)]
    valid = []
    total = 0
    for combo in itertools.combinations(ALL_PIECES, k_pieces):
        total += 1
        r = solver.solve(empty, list(combo))
        if r.solvable:
            valid.append(combo)
    print(f"  {len(valid)}/{total} subsets ({100*len(valid)/total:.1f}%) tile {board_h}x{board_w}")
    return valid


def find_all_tilings(subset: Tuple[str, ...], board_h: int, board_w: int,
                      max_tilings: int = 20) -> List[List[Tuple[str, int, int, int]]]:
    """Find up to max_tilings distinct tilings of the empty board with this piece subset.

    Returns list of tilings, each tiling = list of (piece, ori_id, anchor_r, anchor_c).
    """
    solver = PentominoSolver(board_h=board_h, board_w=board_w,
                              solution_cap=max_tilings, node_cap=500_000)
    empty = [["."] * board_w for _ in range(board_h)]

    # The solver returns the FIRST solution as solution_path. To get multiple
    # distinct tilings, we'd need to modify the solver to return all paths.
    # For simplicity here: use a fresh search that records all paths up to cap.
    tilings = []
    seen_canonical = set()

    # Run-and-vary approach: search with different piece-orderings to get diversity.
    # Pentomino solver already shuffles (uses MRV heuristic) — but it's deterministic
    # for a given input. To get many tilings we need a multi-solution-collecting solver.
    # Simplest: re-enumerate placements with varying piece priorities.

    # Pragmatic implementation: directly write a multi-solution search here.
    from src.environments.polyomino_utils import PIECE_ORIENTATIONS

    def search_all(remaining_pieces, board, trail, all_tilings, cap):
        """Enumerate up to `cap` distinct tilings of `board` with `remaining_pieces`."""
        if len(all_tilings) >= cap:
            return
        if not remaining_pieces:
            # All pieces placed; check board is fully covered (should be)
            if all(c != "." for row in board for c in row):
                all_tilings.append(tuple(trail))
            return

        # Pick the topmost-leftmost empty cell to cover (MRV-ish)
        target = None
        for r in range(board_h):
            for c in range(board_w):
                if board[r][c] == ".":
                    target = (r, c)
                    break
            if target:
                break
        if target is None:
            return
        tr, tc = target

        # Enumerate all (piece, ori, anchor) that cover (tr, tc)
        for piece in remaining_pieces:
            for ori_id, ori in enumerate(PIECE_ORIENTATIONS[piece]):
                for dr, dc in ori:
                    anchor_r = tr - dr
                    anchor_c = tc - dc
                    cells = placement_cells(piece, ori_id, anchor_r, anchor_c)
                    if cells is None:
                        continue
                    if not all(0 <= cr < board_h and 0 <= cc < board_w
                                and board[cr][cc] == "."
                                for cr, cc in cells):
                        continue
                    # Try this placement
                    for cr, cc in cells:
                        board[cr][cc] = piece
                    trail.append((piece, ori_id, anchor_r, anchor_c))
                    new_remaining = [p for p in remaining_pieces if p != piece]
                    search_all(new_remaining, board, trail, all_tilings, cap)
                    trail.pop()
                    for cr, cc in cells:
                        board[cr][cc] = "."
                    if len(all_tilings) >= cap:
                        return

    board = [["."] * board_w for _ in range(board_h)]
    search_all(list(subset), board, [], tilings, max_tilings)
    return tilings


def trajectory_to_samples(tiling: List[Tuple[str, int, int, int]],
                            subset: Tuple[str, ...],
                            board_h: int, board_w: int,
                            puzzle_id: str) -> List[dict]:
    """Walk the tiling step-by-step; emit one (prompt, response) sample per step."""
    board = [["."] * board_w for _ in range(board_h)]
    remaining = list(subset)
    samples = []
    for step_idx, (piece, ori_id, ar, ac) in enumerate(tiling):
        # Render BEFORE the action: state at this decision step
        action = ActionStruct(piece=piece, ori=ori_id, row=ar + 1, col=ac + 1)
        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{render_user_message(board, remaining)}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        response = render_response(action)

        sample = {
            "prompt": prompt,
            "response": response,
            "metadata": {
                "schema": SCHEMA_VERSION,
                "puzzle_id": puzzle_id,
                "step": step_idx,
                "subset": list(subset),
                "remaining_pieces_at_step": list(remaining),
                "board_at_step": [row[:] for row in board],
                "action_struct": action.to_dict(),
                "action_hash": action_hash(action),
                "is_solver_action": True,
                "board_h": board_h,
                "board_w": board_w,
            },
        }
        samples.append(sample)

        # Apply action to advance state
        cells = placement_cells(piece, ori_id, ar, ac)
        for cr, cc in cells:
            board[cr][cc] = piece
        remaining = [p for p in remaining if p != piece]
    return samples


def split_train_val(all_samples: List[dict], val_fraction: float, rng: random.Random) -> Tuple[List[dict], List[dict]]:
    """Split at puzzle_id level to prevent leakage."""
    by_puzzle: Dict[str, List[dict]] = {}
    for s in all_samples:
        by_puzzle.setdefault(s["metadata"]["puzzle_id"], []).append(s)
    puzzle_ids = sorted(by_puzzle.keys())
    rng.shuffle(puzzle_ids)
    n_val = max(1, int(len(puzzle_ids) * val_fraction))
    val_ids = set(puzzle_ids[:n_val])
    train_samples = []
    val_samples = []
    for pid in puzzle_ids:
        if pid in val_ids:
            val_samples.extend(by_puzzle[pid])
        else:
            train_samples.extend(by_puzzle[pid])
    return train_samples, val_samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--board-h", type=int, default=5)
    p.add_argument("--board-w", type=int, default=6)
    p.add_argument("--k-pieces", type=int, default=6,
                   help="Number of pieces per subset (must equal H*W/5)")
    p.add_argument("--max-tilings-per-subset", type=int, default=10)
    p.add_argument("--output-dir", default="data/pentomino5x6/pi_theta_sft")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-samples", type=int, default=None,
                   help="If set, truncate train set to this size after split")
    p.add_argument("--max-val-samples", type=int, default=None,
                   help="If set, truncate val set to this size after split")
    p.add_argument("--augment", action="store_true",
                   help="Apply 4-fold dihedral augmentation (identity + 180° + "
                        "v-flip + h-flip) to each tiling, deduping symmetric "
                        "self-images. Up to 4× sample count.")
    args = p.parse_args()

    rng = random.Random(args.seed)

    print(f"=== Pentomino π_θ SFT data generation ===")
    print(f"  board: {args.board_h}x{args.board_w}, {args.k_pieces} pieces per subset")
    print(f"  max_tilings_per_subset: {args.max_tilings_per_subset}")
    print(f"  output: {args.output_dir}")
    print()

    print("Step 1: enumerating valid piece subsets...")
    valid_subsets = find_valid_subsets(args.board_h, args.board_w, args.k_pieces)
    print()

    print(f"Step 2: finding all tilings for each subset (cap={args.max_tilings_per_subset})...")
    if args.augment:
        print(f"  augmentation: ON — applying {len(SYMMETRIES)}-fold dihedral per tiling")
    all_samples = []
    n_subsets_processed = 0
    n_tilings_total = 0
    n_augmented_tilings_total = 0
    t0 = time.perf_counter()
    for subset in valid_subsets:
        tilings = find_all_tilings(subset, args.board_h, args.board_w,
                                     args.max_tilings_per_subset)
        if args.augment:
            sym_tilings = _augment_tilings(tilings, args.board_h, args.board_w)
        else:
            sym_tilings = [("identity", t) for t in tilings]
        for tiling_idx, (sym, tiling) in enumerate(sym_tilings):
            puzzle_id = (
                f"pent_{args.board_h}x{args.board_w}_"
                f"{''.join(subset)}_t{tiling_idx:03d}_{sym}"
            )
            samples = trajectory_to_samples(tiling, subset, args.board_h, args.board_w,
                                             puzzle_id)
            all_samples.extend(samples)
        n_subsets_processed += 1
        n_tilings_total += len(tilings)
        n_augmented_tilings_total += len(sym_tilings)
        if n_subsets_processed % 5 == 0:
            extra = (f", {n_augmented_tilings_total} after aug" if args.augment else "")
            print(f"  ...{n_subsets_processed}/{len(valid_subsets)} subsets, "
                  f"{n_tilings_total} base tilings{extra}, {len(all_samples)} samples "
                  f"({time.perf_counter()-t0:.0f}s)")
    if args.augment:
        print(f"  done: {n_tilings_total} base tilings × ~{n_augmented_tilings_total/max(n_tilings_total,1):.1f} sym "
              f"= {n_augmented_tilings_total} tilings → {len(all_samples)} samples "
              f"in {time.perf_counter()-t0:.0f}s")
    else:
        print(f"  done: {n_tilings_total} tilings → {len(all_samples)} (state, action) samples "
              f"in {time.perf_counter()-t0:.0f}s")
    print()

    print(f"Step 3: split train/val at puzzle level...")
    train_samples, val_samples = split_train_val(all_samples, args.val_fraction, rng)
    print(f"  before truncation: {len(train_samples)} train + {len(val_samples)} val")

    if args.max_train_samples and len(train_samples) > args.max_train_samples:
        rng.shuffle(train_samples)
        train_samples = train_samples[:args.max_train_samples]
    if args.max_val_samples and len(val_samples) > args.max_val_samples:
        rng.shuffle(val_samples)
        val_samples = val_samples[:args.max_val_samples]
    print(f"  after truncation: {len(train_samples)} train + {len(val_samples)} val")
    print()

    print(f"Step 4: writing JSONL files...")
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.jsonl")
    val_path = os.path.join(args.output_dir, "val.jsonl")
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    inspection_path = os.path.join(args.output_dir, "sample_inspection.txt")

    with open(train_path, "w") as f:
        for s in train_samples:
            f.write(json.dumps(s) + "\n")
    with open(val_path, "w") as f:
        for s in val_samples:
            f.write(json.dumps(s) + "\n")

    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                              cwd=_REPO_ROOT, text=True).strip()
    except Exception:
        git_commit = "unknown"

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "env": f"pentomino{args.board_h}x{args.board_w}",
        "n_train_samples": len(train_samples),
        "n_val_samples": len(val_samples),
        "n_subsets": len(valid_subsets),
        "n_tilings_total": n_tilings_total,
        "max_tilings_per_subset": args.max_tilings_per_subset,
        "k_pieces": args.k_pieces,
        "board_h": args.board_h,
        "board_w": args.board_w,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_commit,
        "solver_version": "pentomino_solver_v2",
        "system_prompt": SYSTEM_PROMPT,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Sample inspection: first 5 train samples
    with open(inspection_path, "w") as f:
        f.write(f"=== first 5 train samples ===\n\n")
        for i, s in enumerate(train_samples[:5]):
            f.write(f"--- sample {i} ---\n")
            f.write(f"puzzle_id: {s['metadata']['puzzle_id']}\n")
            f.write(f"step: {s['metadata']['step']}\n")
            f.write(f"--- prompt ---\n{s['prompt']}\n")
            f.write(f"--- response ---\n{s['response']}\n\n")

    print(f"  wrote: {train_path}, {val_path}, {metadata_path}, {inspection_path}")
    print()
    print(f"=== DONE ===")
    print(f"  train: {len(train_samples)} samples")
    print(f"  val:   {len(val_samples)} samples")


if __name__ == "__main__":
    main()
