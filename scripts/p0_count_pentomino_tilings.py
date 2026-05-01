"""P-0: count distinct tilings of an HxW board for candidate 4-pentomino subsets.

Used to confirm the easy-variant piece set (provisional: {L, P, T, Y}) admits
enough tilings to be a non-degenerate puzzle, per spec §7.1 / §8 item 3.
Criterion: subset must admit ≥10 distinct tilings of the 5×4 board.

Run: python scripts/p0_count_pentomino_tilings.py
"""
from __future__ import annotations
import itertools


# ----------------------------------------------------------------------------
# Pentomino definitions — cells in row,col coordinates (anchor implied at the
# top-most leftmost cell of each shape).
# ----------------------------------------------------------------------------

PIECES_BASE: dict[str, frozenset] = {
    # F
    'F': frozenset({(0,1), (0,2), (1,0), (1,1), (2,1)}),
    # I
    'I': frozenset({(0,0), (0,1), (0,2), (0,3), (0,4)}),
    # L
    'L': frozenset({(0,0), (1,0), (2,0), (3,0), (3,1)}),
    # N
    'N': frozenset({(0,1), (1,1), (2,0), (2,1), (3,0)}),
    # P
    'P': frozenset({(0,0), (0,1), (1,0), (1,1), (2,0)}),
    # T
    'T': frozenset({(0,0), (0,1), (0,2), (1,1), (2,1)}),
    # U
    'U': frozenset({(0,0), (0,2), (1,0), (1,1), (1,2)}),
    # V
    'V': frozenset({(0,0), (1,0), (2,0), (2,1), (2,2)}),
    # W
    'W': frozenset({(0,0), (1,0), (1,1), (2,1), (2,2)}),
    # X
    'X': frozenset({(0,1), (1,0), (1,1), (1,2), (2,1)}),
    # Y
    'Y': frozenset({(0,1), (1,0), (1,1), (2,1), (3,1)}),
    # Z
    'Z': frozenset({(0,0), (0,1), (1,1), (2,1), (2,2)}),
}


def normalize(cells: frozenset) -> frozenset:
    """Shift cells so min row = 0 and min col = 0."""
    if not cells:
        return cells
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return frozenset((r - min_r, c - min_c) for r, c in cells)


def rotate_90(cells: frozenset) -> frozenset:
    """Rotate 90° clockwise: (r, c) → (c, -r)."""
    return normalize(frozenset((c, -r) for r, c in cells))


def reflect(cells: frozenset) -> frozenset:
    """Mirror horizontally: (r, c) → (r, -c)."""
    return normalize(frozenset((r, -c) for r, c in cells))


def all_orientations(cells: frozenset) -> list:
    """All distinct orientations under rotation × reflection. Each is normalized."""
    seen = set()
    out = []
    cur = normalize(cells)
    for _ in range(4):
        for shape in (cur, reflect(cur)):
            shape = normalize(shape)
            if shape not in seen:
                seen.add(shape)
                out.append(shape)
        cur = rotate_90(cur)
    return out


# Pre-compute orientations for each piece
PIECE_ORIENTATIONS: dict[str, list] = {p: all_orientations(cells) for p, cells in PIECES_BASE.items()}


# ----------------------------------------------------------------------------
# Tiling counter — backtracking over (first empty cell × remaining piece × orientation)
# ----------------------------------------------------------------------------

def count_tilings(board_h: int, board_w: int, piece_set: tuple) -> int:
    """Count distinct tilings of a board_h × board_w board using each piece in piece_set
    exactly once. Pieces can be in any orientation (rotations + reflections)."""
    if sum(5 for _ in piece_set) != board_h * board_w:
        return 0  # area mismatch — impossible
    board = [[0] * board_w for _ in range(board_h)]
    return _backtrack(board, board_h, board_w, list(piece_set))


def _first_empty(board, h, w):
    for r in range(h):
        for c in range(w):
            if board[r][c] == 0:
                return (r, c)
    return None


def _backtrack(board, h, w, remaining):
    if not remaining:
        return 1
    pos = _first_empty(board, h, w)
    if pos is None:
        return 0  # board full but pieces remain
    r, c = pos

    count = 0
    for i, piece in enumerate(remaining):
        for ori in PIECE_ORIENTATIONS[piece]:
            # Compute placement so the top-most leftmost cell of this orientation
            # lands at (r, c). Since orientations are already normalized to
            # min row = min col = 0, the (0, 0) cell may or may not be in `ori`.
            # The orientation's anchor (top-most leftmost in row-major) is
            # min(ori, key=(r, c)).
            anchor = min(ori)
            dr, dc = r - anchor[0], c - anchor[1]
            cells = [(rr + dr, cc + dc) for rr, cc in ori]
            # Bounds + overlap check
            valid = True
            for ar, ac in cells:
                if not (0 <= ar < h and 0 <= ac < w):
                    valid = False
                    break
                if board[ar][ac] != 0:
                    valid = False
                    break
            if not valid:
                continue
            # Place
            for ar, ac in cells:
                board[ar][ac] = i + 1  # 1-indexed marker
            # Recurse
            count += _backtrack(board, h, w, remaining[:i] + remaining[i+1:])
            # Unplace
            for ar, ac in cells:
                board[ar][ac] = 0
    return count


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def search_best_subsets(board_h, board_w, n_pieces, top_k=10):
    """Sweep ALL C(12, n_pieces) subsets and return the top_k by tiling count."""
    all_pieces = list(PIECES_BASE.keys())
    results = []
    for ps in itertools.combinations(all_pieces, n_pieces):
        n = count_tilings(board_h, board_w, ps)
        if n > 0:
            results.append((ps, n))
    results.sort(key=lambda x: -x[1])
    return results[:top_k]


def main():
    print(f"=== Pentomino orientation counts ===")
    for p, oris in PIECE_ORIENTATIONS.items():
        print(f"  {p}: {len(oris)} unique orientations")
    total = sum(len(o) for o in PIECE_ORIENTATIONS.values())
    print(f"  total fixed pentominoes: {total} (canonical answer: 63)")
    print()

    # Sweep various small board configs to find a sweet spot
    configs = [
        (3, 5, 3),    # 15 cells / 3 pieces
        (4, 5, 4),    # 20 cells / 4 pieces
        (5, 4, 4),    # 20 cells / 4 pieces (transposed)
        (5, 5, 5),    # 25 cells / 5 pieces
        (6, 5, 6),    # 30 cells / 6 pieces — classic
        (5, 6, 6),    # 30 cells / 6 pieces (transposed)
    ]

    print(f"=== Top 4-piece subsets per board config (target: ≥10 tilings for non-degenerate puzzle) ===\n")
    for h, w, n in configs:
        print(f"--- Board {h}×{w} with {n} pieces ({h*w} cells) ---")
        best = search_best_subsets(h, w, n, top_k=8)
        if not best:
            print(f"  No subset tiles this board.")
        else:
            for ps, count in best:
                flag = '✓' if count >= 10 else ('~' if count >= 5 else '✗')
                print(f"  {flag} {{{', '.join(ps)}}}: {count} tilings")
        print()


if __name__ == "__main__":
    main()
