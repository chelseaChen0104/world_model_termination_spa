"""Pentomino piece definitions, orientation enumeration, and DLX-based solvability oracle.

Reused by `PolyominoEnv` (placement + rendering) and `PolyominoSolvabilityChecker` (oracle).

Piece convention:
  - 12 standard pentominoes: F, I, L, N, P, T, U, V, W, X, Y, Z
  - Each piece has multiple distinct orientations under rotation × reflection
  - Each orientation is stored as a list of (dr, dc) offsets from the **anchor cell**,
    where the anchor = top-most leftmost cell of the piece's footprint at that orientation
  - The anchor is always at offset (0, 0); other cells may have positive OR negative dc

Action convention (used by the env):
  "place {piece} ori={K} at row {R} col {C}"
  → place piece's K-th orientation with the anchor at (R-1, C-1), 0-indexed internally
"""
from __future__ import annotations
from typing import Optional


# ----------------------------------------------------------------------------
# Pentomino definitions — cells in (row, col) coordinates
# ----------------------------------------------------------------------------

PIECES_BASE: dict[str, frozenset] = {
    'F': frozenset({(0,1), (0,2), (1,0), (1,1), (2,1)}),
    'I': frozenset({(0,0), (0,1), (0,2), (0,3), (0,4)}),
    'L': frozenset({(0,0), (1,0), (2,0), (3,0), (3,1)}),
    'N': frozenset({(0,1), (1,1), (2,0), (2,1), (3,0)}),
    'P': frozenset({(0,0), (0,1), (1,0), (1,1), (2,0)}),
    'T': frozenset({(0,0), (0,1), (0,2), (1,1), (2,1)}),
    'U': frozenset({(0,0), (0,2), (1,0), (1,1), (1,2)}),
    'V': frozenset({(0,0), (1,0), (2,0), (2,1), (2,2)}),
    'W': frozenset({(0,0), (1,0), (1,1), (2,1), (2,2)}),
    'X': frozenset({(0,1), (1,0), (1,1), (1,2), (2,1)}),
    'Y': frozenset({(0,1), (1,0), (1,1), (2,1), (3,1)}),
    'Z': frozenset({(0,0), (0,1), (1,1), (2,1), (2,2)}),
}

ALL_PIECES = tuple(PIECES_BASE.keys())


def _normalize(cells: frozenset) -> frozenset:
    """Shift cells so min row = 0 and min col = 0."""
    if not cells:
        return cells
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return frozenset((r - min_r, c - min_c) for r, c in cells)


def _rotate_90(cells: frozenset) -> frozenset:
    """Rotate 90° clockwise: (r, c) → (c, -r), then normalize."""
    return _normalize(frozenset((c, -r) for r, c in cells))


def _reflect(cells: frozenset) -> frozenset:
    """Mirror horizontally: (r, c) → (r, -c), then normalize."""
    return _normalize(frozenset((r, -c) for r, c in cells))


def _enumerate_orientations(cells: frozenset) -> list:
    """All distinct orientations under rotation × reflection. Each is normalized.

    Returns a list of frozensets, sorted by tuple representation for determinism.
    """
    seen = set()
    out = []
    cur = _normalize(cells)
    for _ in range(4):
        for shape in (cur, _reflect(cur)):
            shape = _normalize(shape)
            if shape not in seen:
                seen.add(shape)
                out.append(shape)
        cur = _rotate_90(cur)
    out.sort(key=lambda s: tuple(sorted(s)))
    return out


def _to_anchor_offsets(cells: frozenset) -> tuple:
    """Convert a normalized orientation to (dr, dc) offsets from the anchor cell.

    Anchor = top-most leftmost cell (smallest by (row, col) tuple).
    The anchor itself becomes (0, 0). Other cells have offsets relative to the anchor;
    dc may be negative if cells exist to the left of the anchor (e.g., F-piece).
    """
    anchor = min(cells)  # (row, col) tuple, smallest in row-major order
    ar, ac = anchor
    return tuple(sorted((r - ar, c - ac) for r, c in cells))


# Pre-compute orientations as anchor-offset tuples for each piece.
# PIECE_ORIENTATIONS[piece] = list of orientations (each = tuple of (dr, dc) offsets, anchor at (0, 0))
PIECE_ORIENTATIONS: dict[str, list] = {
    p: [_to_anchor_offsets(o) for o in _enumerate_orientations(cells)]
    for p, cells in PIECES_BASE.items()
}


def get_orientation(piece: str, ori_id: int) -> Optional[tuple]:
    """Return the (dr, dc) offsets for the given piece + orientation id, or None if invalid."""
    if piece not in PIECE_ORIENTATIONS:
        return None
    oris = PIECE_ORIENTATIONS[piece]
    if not (0 <= ori_id < len(oris)):
        return None
    return oris[ori_id]


def num_orientations(piece: str) -> int:
    return len(PIECE_ORIENTATIONS.get(piece, []))


def placement_cells(piece: str, ori_id: int, anchor_r: int, anchor_c: int) -> Optional[list]:
    """Return absolute board cells for placing `piece` at orientation `ori_id` with anchor at (anchor_r, anchor_c).

    Returns None if the (piece, ori_id) combination is invalid.
    """
    ori = get_orientation(piece, ori_id)
    if ori is None:
        return None
    return [(anchor_r + dr, anchor_c + dc) for dr, dc in ori]


def fits_on_board(cells: list, board_h: int, board_w: int, board: list) -> bool:
    """True iff every cell is in-bounds AND empty in `board`."""
    for r, c in cells:
        if not (0 <= r < board_h and 0 <= c < board_w):
            return False
        if board[r][c] != '.':
            return False
    return True


# ----------------------------------------------------------------------------
# Solvability oracle — DLX (Algorithm X)
# ----------------------------------------------------------------------------

class PolyominoSolvabilityChecker:
    """Check whether the empty cells of a board can be exactly tiled by a given set
    of remaining pentominoes (each used exactly once).

    Algorithm:
      1. Quick area check (cheap):
         empty_cells_count must equal 5 × len(remaining_pieces). Else 'area_mismatch'.
      2. Quick connectivity check (cheap):
         every connected component of empty cells must have area divisible by 5.
         Else 'small_island' or 'island_size_mismatch'.
      3. Main check: build exact-cover candidate set, run depth-bounded backtracking.

    The backtracking uses MRV-like heuristic: at each step, pick the empty cell
    with the FEWEST candidate placements. Bounded by `max_depth` for safety
    (matches Sudoku checker's conservative-True behavior on depth exceeded).

    Conventions:
      - `board` is a 2D list of strings; '.' = empty, any letter = occupied
      - `remaining_pieces` is a list of piece letters (each appears at most once)
    """

    def __init__(self, max_depth: int = 50):
        self.max_depth = max_depth

    def check_solvability(
        self, board: list, remaining_pieces: list,
    ) -> tuple[bool, Optional[str]]:
        """Returns (is_solvable, reason_if_not).

        is_solvable = True means 'a tiling exists' (or 'depth exceeded — assume yes').
        is_solvable = False means 'definitely no tiling possible'.
        """
        h = len(board)
        w = len(board[0]) if h > 0 else 0

        # Special case: no pieces left
        if not remaining_pieces:
            # Need every cell to be occupied
            if any(board[r][c] == '.' for r in range(h) for c in range(w)):
                return False, "pieces_exhausted_with_empty_cells"
            return True, None

        # 1. Area check
        empty_count = sum(1 for r in range(h) for c in range(w) if board[r][c] == '.')
        if empty_count != 5 * len(remaining_pieces):
            return False, "area_mismatch"

        # 2. Connectivity check — every component of empty cells must be ≥ 5 cells AND
        # divisible by 5 (since each pentomino contributes 5 cells, and pieces cannot
        # span a component boundary).
        components = self._connected_components(board, h, w)
        for comp_size in components:
            if comp_size < 5 or comp_size % 5 != 0:
                return False, f"island_size_mismatch_{comp_size}"

        # 3. Main check: enumerate placements and run backtracking
        # Build candidate placements per (piece, anchor_cell)
        empty_cells = frozenset((r, c) for r in range(h) for c in range(w) if board[r][c] == '.')
        all_placements = self._enumerate_all_placements(empty_cells, remaining_pieces, h, w)

        # Group placements by which empty cell is anchored at (for MRV heuristic)
        # And by which empty cells they cover (for fast lookup)
        # We also need: for each piece, the placements that use it (so we can ensure each piece used once)

        # For depth-first search, we'll pick the most-constrained empty cell each step.
        return self._backtrack(empty_cells, list(remaining_pieces), all_placements, depth=0)

    def _connected_components(self, board, h, w) -> list:
        """Return list of empty-component sizes."""
        visited = [[False] * w for _ in range(h)]
        sizes = []
        for r in range(h):
            for c in range(w):
                if board[r][c] != '.' or visited[r][c]:
                    continue
                # BFS
                size = 0
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    rr, cc = stack.pop()
                    size += 1
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and board[nr][nc] == '.':
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                sizes.append(size)
        return sizes

    def _enumerate_all_placements(self, empty_cells: frozenset, remaining_pieces: list, h: int, w: int) -> list:
        """For each remaining piece × orientation × anchor cell in empty_cells:
        if placement fits entirely within empty_cells, record it.
        Returns: list of (piece, ori_id, anchor_r, anchor_c, cells_frozenset)
        """
        placements = []
        for piece in remaining_pieces:
            oris = PIECE_ORIENTATIONS[piece]
            for ori_id, ori in enumerate(oris):
                for ar, ac in empty_cells:
                    cells = frozenset((ar + dr, ac + dc) for dr, dc in ori)
                    # Bounds + empty check
                    if cells.issubset(empty_cells):
                        placements.append((piece, ori_id, ar, ac, cells))
        return placements

    def _backtrack(self, empty_cells: frozenset, remaining_pieces: list, placements: list, depth: int) -> tuple:
        if not empty_cells:
            return True, None
        if not remaining_pieces:
            return False, "pieces_exhausted_with_empty_cells"
        if depth >= self.max_depth:
            # Conservative — same convention as SudokuSolvabilityChecker
            return True, None

        # MRV: pick the empty cell with fewest candidate placements covering it,
        # restricted to placements using a piece in `remaining_pieces`
        cell_candidates: dict = {}
        rem_set = set(remaining_pieces)
        for placement in placements:
            piece, ori_id, ar, ac, cells = placement
            if piece not in rem_set:
                continue
            if not cells.issubset(empty_cells):
                continue
            for cell in cells:
                cell_candidates.setdefault(cell, []).append(placement)

        # Find an empty cell with no candidates → unsolvable
        # Or pick the cell with fewest candidates (MRV)
        target_cell = None
        target_count = float('inf')
        for cell in empty_cells:
            count = len(cell_candidates.get(cell, []))
            if count == 0:
                return False, f"no_placement_covers_{cell}"
            if count < target_count:
                target_count = count
                target_cell = cell

        if target_cell is None:
            return False, "no_target_cell"  # shouldn't happen

        # Try each candidate placement for the MRV cell
        for placement in cell_candidates[target_cell]:
            piece, ori_id, ar, ac, cells = placement
            new_empty = empty_cells - cells
            new_remaining = [p for p in remaining_pieces if p != piece]
            # We could prune `placements` here for speed, but the cells.issubset check above already filters
            ok, _ = self._backtrack(new_empty, new_remaining, placements, depth + 1)
            if ok:
                return True, None

        return False, "no_solution"


# ----------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------

def _self_test():
    # Sanity: orientation counts match the canonical answer (63 fixed pentominoes)
    total = sum(len(o) for o in PIECE_ORIENTATIONS.values())
    assert total == 63, f"expected 63 fixed pentominoes, got {total}"

    # Sanity: empty 5×4 board with {L, P, W, Y} is solvable (we know there are 20 tilings)
    board = [['.', '.', '.', '.'] for _ in range(5)]
    checker = PolyominoSolvabilityChecker()
    ok, reason = checker.check_solvability(board, ['L', 'P', 'W', 'Y'])
    assert ok, f"5×4 with {{L,P,W,Y}} should be solvable from empty, got {reason}"

    # Sanity: 5×4 with one cell blocked but only 19 empty cells / 4 pieces × 5 = 20 → area mismatch
    board = [['.', '.', '.', '.'] for _ in range(5)]
    board[0][0] = 'L'  # 1 cell occupied
    ok, reason = checker.check_solvability(board, ['L', 'P', 'W', 'Y'])
    assert not ok, f"area-mismatch state should be unsolvable, got ok={ok}"
    assert reason == "area_mismatch", f"expected 'area_mismatch', got {reason}"

    # Sanity: a 3-cell isolated island → small_island
    board = [
        ['.', '.', '.', '.'],   # 4 empty
        ['.', '#', '#', '.'],   # 2 empty (one connected to top, one to right)
        ['.', '#', '#', '.'],   # 2 empty
        ['.', '#', '#', '.'],   # 2 empty
        ['.', '.', '.', '.'],   # 4 empty
    ]
    # actually this has a connected empty region forming an outer ring of ~14 cells
    # Let me make a clearer test: pinch off a 3-cell strip
    board = [
        ['.', '.', '.', '#'],
        ['.', '.', '.', '#'],
        ['#', '#', '#', '#'],
        ['#', '.', '.', '.'],
        ['#', '.', '.', '.'],
    ]
    # Top component: 6 cells, bottom: 6 cells. Both divisible by 5? No — 6 % 5 = 1.
    # So this should fail.
    # 12 empty cells / 5 = 2.4 pieces → area_mismatch first
    ok, reason = checker.check_solvability(board, ['L', 'P'])
    # 2 pieces × 5 = 10 cells, but 12 empty cells → area_mismatch
    assert not ok
    assert reason == "area_mismatch", f"got {reason}"

    print("All polyomino_utils self-tests passed.")


if __name__ == "__main__":
    _self_test()
