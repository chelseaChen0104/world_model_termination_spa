# Pentomino Tiling Env — Spec & Plan (2026-04-29)

Second environment for the "world model SFT for termination prediction" recipe. Companion to [doc/SPEC.md](SPEC.md) (project-level scope) and [doc/plan_2026-04-29_rl_approach.md](plan_2026-04-29_rl_approach.md) (RL approach). Diversifies the project beyond Sudoku by adding a structurally different problem class — geometric packing instead of value-placement.

This doc:
1. Pins the puzzle rules and our specific configuration (§1–§3)
2. Locks the SFT response format with renamed tags (§4)
3. Specifies the env API + solvability oracle (§5–§6)
4. Lays out the data-gen + training plan (§7)
5. Lists open questions and decisions we are explicitly deferring (§8)

---

## 1. Game rules

A **polyomino** is a shape formed by joining unit squares edge-to-edge. We use **pentominoes** — the 12 distinct 5-square shapes. Standard letter labels: **F, I, L, N, P, T, U, V, W, X, Y, Z**.

```
F:    .##     I:  #####    L:  #.        N:  .#       P:  ##
      ##.                      #.            .#           ##
      .#.                      #.            ##           #.
                               #.            #.
                               ##

T:  ###     U:  #.#       V:  #..       W:  #..      X:  .#.
    .#.         ###           #..           ##.          ###
    .#.                       ###           .##          .#.

Y:  .#       Z:  ##.
    ##           .#.
    .#           .##
    .#
```

Each piece has multiple distinct orientations under rotation (0°, 90°, 180°, 270°) and reflection (mirror). Effective unique orientations per piece:

| Piece | Unique orientations | Note |
|---|---|---|
| **X** | 1 | full symmetry |
| **I** | 2 | rotational only |
| **T, U, V, W, Z** | 4 each | reflection-symmetric |
| **F, L, N, P, Y** | 8 each | chiral (reflections give new shapes) |

Total placements (board-position × orientation) for a 6×10 board: ~2,000 candidate placements. For a 5×4 board: ~150 candidates.

**Goal of a tiling puzzle:**
1. Cover **every** cell of the board with pieces (no gaps)
2. **No overlap** between pieces
3. **No piece extends outside** the board
4. Each piece in the available set is used **exactly once** (in the variants we'll use)

**Predictive gap** (the property that makes pentomino tilings interesting for our recipe):
- Place a piece in a position that leaves a region of < 5 empty cells → no pentomino fits → **unsolvable, immediately**
- Place a piece that leaves a region whose **shape** doesn't match any remaining piece (after orientation) → **unsolvable, more subtle**
- Place a piece whose remaining-pieces' parity (chessboard-coloring) constraint is violated → **unsolvable, very subtle** (these are the "interesting" cases — invisible to local pattern matching)

The oracle detects all three cases. The third case especially is where the LLM's `<viability>` prediction can carry signal.

---

## 2. Our two configurations

We mirror the Sudoku scale ladder (4×4 easy, 9×9 full) with two pentomino configs.

### 2.1 "Easy" variant — 5×4 board, 4 pentominoes (matches 4×4 Sudoku scale)

- **Board:** 5 rows × 4 columns = 20 cells
- **Pieces:** 4 distinct pentominoes (4 × 5 = 20 cells, exact cover)
- **Piece set (locked 2026-04-29 by P-0 sweep):** **{L, P, W, Y}** — **20 distinct tilings**
- **Why this set** (P-0 results, full sweep across all C(12,4) = 495 candidates):
  | Set | Tilings on 5×4 |
  |---|---|
  | **{L, P, W, Y}** ✓ | **20** |
  | {L, P, T, V} | 16 |
  | {F, L, U, Y} | 12 |
  | {L, P, V, Y} | 12 |
  | {L, P, T, Y} (original spec proposal) | 4 — too low |
  Most 4-piece subsets don't tile 5×4 at all (0 tilings). 5×4 is genuinely tight.
- **Trajectory length:** ≤ 4 actions per puzzle (one per piece)
- **Use as:** SFT-replication target on autodl1, comparable scale to 4×4 Sudoku B-5
- **Predictive gap:** the W piece (staircase shape) creates strong constraints on neighboring cells, and L/P/Y's asymmetry means orientation choice cascades. Expect ~70-80% BP rate from random play.

### 2.2 "Full" variant — 6×10 board, all 12 pentominoes (matches 9×9 Sudoku scale)

- **Board:** 6 rows × 10 columns = 60 cells
- **Pieces:** all 12 standard pentominoes (12 × 5 = 60 cells, exact cover)
- **Trajectory length:** ≤ 12 actions per puzzle
- **Solution count:** 2,339 distinct tilings (well-known)
- **Use as:** the harder generalization test; SPA-scale data + B-5-style hparams

The two variants run on the same env code with different `(board_h, board_w, piece_set)` config.

---

## 3. State, action, terminology

### 3.1 State

The board is a 2D grid of cells. Each cell is either:
- **Empty** (`.`)
- **Occupied** by a placed piece, with the piece's letter as the cell's label (`F`, `I`, ..., `Z`)

The full state `s` = (board grid, set of remaining pieces, set of placed pieces with their (piece, orientation, anchor) records).

Render example (mid-game, 5×4 easy variant):
```
Current board (5x4):
L L L L
. . . L
. P P .
. P P .
P P . .

Remaining pieces: T, Y
```

The render in the user message will include the board AND the list of remaining pieces (so the LLM can reason about what's left).

### 3.2 Action

An action is a placement: **`place {piece} ori={ori_id} at row {R} col {C}`**

- `piece` ∈ {F, I, L, N, P, T, U, V, W, X, Y, Z}
- `ori_id` ∈ {0, 1, …, 7}, mapped to a canonical fixed-orientation table per piece (orientation IDs 0..N where N depends on the piece's symmetry — see §6.2)
- `R, C` are **1-indexed** anchor coordinates (consistent with our Sudoku convention)
- The **anchor** is the cell with the smallest `(row, col)` tuple in the piece's footprint when oriented (i.e., the topmost-leftmost cell of the piece's occupancy in row-major order). Choosing this convention makes the anchor uniquely defined per (piece, orientation) — no ambiguity.

Action validity at state `s`:
1. `piece` is in `remaining_pieces(s)`
2. `ori_id` is a valid orientation index for `piece`
3. The placement (anchor + orientation) results in a footprint where:
   - All 5 cells are within `[0, board_h)` × `[0, board_w)`
   - All 5 cells are empty in the current board

Invalid actions return -0.1 reward and `info.action_is_valid=False`, mirroring the Sudoku env's behavior.

### 3.3 Terminal conditions

A rollout ends when one of:
- **Solved**: `remaining_pieces(s)` is empty AND every cell is occupied. `info.success = True`.
- **Deadlock**: no valid action exists at the current state (can be detected by the oracle returning `is_solvable=False`). `info.success = False`, `info.deadlock_type = 'no_valid_placement'`.
- **Step limit**: `max_steps` exceeded (~ 2× piece count in the config; e.g., 8 for easy, 24 for full — generous safety margin).

---

## 4. SFT response format (NEW tags — locked here)

Per direction in conversation 2026-04-29: pentomino (and the eventual MKD env) use a **renamed tag set**, so we don't conflate Sudoku-era runs (`<solvable>`) with the new envs.

```xml
<think>
<observation>
{rendered current board, e.g., 5x4 with piece-letter cells}
Remaining pieces: P, T, Y
</observation>
<next_state>
{predicted board after applying the chosen action}
Remaining pieces: T, Y
</next_state>
<viability>true|false</viability>
</think>
<answer>place L ori=2 at row 1 col 1</answer>
```

Tag-by-tag:

| Tag | Role | Renamed from Sudoku format? |
|---|---|---|
| `<observation>` | Render of current state (board + remaining pieces) | No — same name |
| `<next_state>` | Predicted state after action | **Yes** — was `<prediction>` in Sudoku |
| `<viability>` | Action-conditional binary: is the predicted next state still tileable? | **Yes** — was `<solvable>` in Sudoku |
| `<answer>` | The chosen action | No — same name |

Semantics of `<viability>`: same as `<solvable>` — action-conditional, evaluated on `s_{t+1}`. The model is asserting "after I place this piece, the board is **still tileable** with the pieces I have left." `false` = "this placement creates a dead-end."

**Note on Sudoku format**: Sudoku stays on `<solvable>` for backwards compatibility with B-0..B-5. Only new envs use the new names.

---

## 5. Env API

`src/environments/polyomino.py` — `PolyominoEnv(BaseTerminationEnv)`:

```python
class PolyominoEnv(BaseTerminationEnv):
    def __init__(
        self,
        board_h: int = 5,
        board_w: int = 4,
        piece_set: Tuple[str, ...] = ('L', 'P', 'T', 'Y'),
        max_steps: int = 8,
        max_dlx_depth: int = 50,   # oracle backtracking budget
    ): ...

    def reset(self, seed: Optional[int] = None) -> str:
        """Initialize board (all empty), shuffle piece order if applicable. Returns rendered state."""

    def step(self, action: Union[str, int]) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Parse action, validate, place piece, check solvability of s_{t+1}, return (obs, reward, done, info).

        info keys (matching SudokuEnv conventions):
          action_is_valid: bool
          action_is_effective: bool
          success: bool                    # tiling complete
          is_solvable: bool                # solvability of s_{t+1} (the new key naming for the env layer)
          is_breaking_point: bool          # prev_solvable AND NOT new_solvable
          deadlock_type: Optional[str]     # 'no_pentomino_fits' | 'parity_violation' | 'no_solution_in_dlx_depth' | None
          action_name: str
        """

    def render(self) -> str: ...
    def get_all_actions(self) -> List[str]: ...
    def get_state_info(self) -> Dict[str, Any]: ...
    def check_solvability(self) -> Tuple[bool, Optional[str]]: ...   # delegates to oracle
```

The env uses `info["is_solvable"]` at the env level (convention shared with `SudokuEnv`). The SFTFormatter is what maps this to the `<viability>` tag in the rendered response. **The env does not know the tag name** — it only reports the boolean.

---

## 6. Solvability oracle — DLX (Algorithm X)

`src/environments/polyomino_utils.py` — `PolyominoSolvabilityChecker`:

```python
class PolyominoSolvabilityChecker:
    def __init__(self, max_depth: int = 50): ...
    def check_solvability(self, board: np.ndarray, remaining_pieces: List[str]) -> Tuple[bool, Optional[str]]:
        """Return (is_solvable, reason).

        Algorithm:
          1. Quick checks (cheap):
             - Total empty area must equal 5 × len(remaining_pieces). If not → 'area_mismatch'
             - Each connected component of empty cells must have area ≥ 5. Else → 'small_island'
             - Optionally: chessboard-parity bound (sum of empty cell colors vs achievable per remaining pieces)
                                                                 → 'parity_violation'
          2. Main check: build exact-cover matrix and run Knuth's Algorithm X with bounded recursion depth.
             - Rows: all (piece, orientation, anchor) placements that fit the empty cells, for each remaining piece
             - Columns: each empty cell + each remaining piece (a piece column ensures piece is used exactly once)
             - DLX with MRV-like column selection (pick column with fewest covering rows)
          3. Return True if a solution is found within max_depth, False otherwise.

        Failure modes (analogous to Sudoku checker):
          - Sound: returns False only if the state is genuinely unsolvable (or DLX depth exceeded — see caveat)
          - Caveat: if max_depth exceeded, return True conservatively (matches Sudoku checker behavior).
            For 5×4 / 4-piece variant this never fires. For 6×10 / 12-piece, may fire on hard instances.
        """
```

**Pre-computation** (done once at env init or first oracle call, cached):
- For each piece in `{F, I, L, N, P, T, U, V, W, X, Y, Z}`: enumerate all unique orientations (4 or 8 per piece) and store as `Set[Tuple[int, int]]` of relative cell offsets.
- For each (piece, orientation, anchor_position): pre-check that all 5 cells fit on the board.

This pre-computation is cheap (~ms total) and keeps per-state oracle calls fast.

**Performance target**: oracle returns in < 10ms for 5×4 / 4-piece states; < 100ms worst-case for 6×10 / 12-piece states. If the latter is too slow we'll add a cache or tighten `max_depth`.

---

## 7. Data-gen + training plan

### 7.1 Implementation milestones

| Stage | Deliverable | Effort | Blocker for |
|---|---|---|---|
| **P-0** | Lock the easy-variant piece set (verify ≥10 tilings exist for 5×4 with chosen 4 pieces). Just a small enumeration script. | ~30 min | env init values |
| **P-1** | `src/environments/polyomino.py` — `PolyominoEnv` + render + step + action parser | ~1.5 days | data gen |
| **P-2** | `src/environments/polyomino_utils.py` — `PolyominoSolvabilityChecker` (DLX) + piece-orientation tables | ~1 day | env step (calls oracle) |
| **P-3** | `src/data/sft_formatter.py` — new variant `polyomino_minimal` with `<observation>` + `<next_state>` + `<viability>` + `<answer>` | ~1 hour | SFT data |
| **P-4** | `src/data/llm_trajectory_generator.py` — verify env-agnostic; add `--env polyomino` flag | ~half day | LLM-policy data gen |
| **P-5** | `scripts/generate_pentomino_easy.sh` + `scripts/generate_pentomino_full.sh` | ~30 min | data gen on cloud |
| **P-6** | Smoke test: 50 random-policy trajectories on 5×4 easy, manually inspect | ~1 hour | confidence |
| **P-7** | LLM-policy trajectory gen on autodl1: 5×4 easy variant, ~3,000 trajectories → ~6,000 SFT samples | ~3-4h on H800 | SFT training |
| **P-8** | Reformat + filter (`reformat_to_minimal.py`, `filter_post_bp.py` — should be env-agnostic, verify) | ~30 min | training |
| **P-9** | SFT training: same hparams as B-5 (lr=1e-4, ep=5, bs=16, max_length=1024). Output `outputs/sft_pentomino_easy_b7_spa_hparams/` | ~30-60 min on H800 | eval |
| **P-10** | Eval: termination + logprob on held-out val. Report AUC, Pass@1 on tilings. | ~15 min | replication done |

**Total dev effort: ~3-4 days of code before any data gen** (P-0 through P-5). Then pipeline mirrors B-5 exactly with env swapped.

### 7.2 Data-gen specifics

For the LLM-policy trajectory generator on the 5×4 easy variant:
- **N_TRAJECTORIES**: 3,000 (target ~6,000 SFT samples after no_post_bp filter, matching SPA scale)
- **max_steps**: 8 (2× piece count for safety)
- **temperature**: 0.7 (matching Sudoku gen)
- **max_context_turns**: 4 (matching trajectory length; multi-turn in raw form, single-step in minimal reformat)
- **seed**: 42
- **Output dir**: `data/pentomino_easy_llm_policy/`

Same parallel-cloud split as 4x4 Sudoku is possible if needed for speed — but rollouts are short (≤ 8 steps × ~1.5s each ≈ 12s/traj), so a single-cloud run on autodl1 should finish in ~3-4 hours.

### 7.3 Training plan

Mirror of B-5 setup, with env swapped:

```bash
python -u src/training/simple_sft_trainer.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --train_file data/pentomino_easy_llm_policy_minimal/wm_train_no_post_bp.parquet \
    --val_file data/pentomino_easy_llm_policy_minimal/wm_val_no_post_bp.parquet \
    --output_dir outputs/sft_pentomino_easy_b7_spa_hparams \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --max_length 1024 \
    --eval_steps 25
```

Eval analogous to B-5: greedy + logprob threshold sweep on `<viability>` (token-level logit lookup at the `true`/`false` position). Implementation note: `evaluate_rl.py --metric solvable-logprob` currently hardcodes the `<solvable>` tag; need to parameterize for `<viability>` (small change).

### 7.4 Success criteria for this env

The key research question this env answers: **does the recipe transfer to a structurally different problem class?** Specifically:

- **AUC > 0.6 on `<viability>` discrimination** — recipe transfers (success).
- **AUC ≈ 0.5** — the recipe fails on geometric tasks; specific to constraint-propagation puzzles. Would be an interesting negative finding, but a project blocker.
- **Pass@1 > 0%** — model can compose at least some valid tilings. (Not a primary metric here; reuses Sudoku's framing.)

If AUC > 0.6 on the easy variant, proceed to the full variant (6×10 with all 12 pentominoes) as a stretch goal.

---

## 8. Open questions / explicit deferrals

These are flagged here so they don't block P-1/P-2 from starting. Decisions can be made when they become blockers, not now.

1. **Reflection orientations: include or not?** Standard pentomino convention includes reflections (8 orientations for chiral pieces). Decision: **include them** (default convention) — gives the LLM a richer action space to reason over. Cost: doubles the action space size, but DLX handles it fine.

2. **Anchor convention.** **Decision: top-most leftmost occupied cell** (row-major minimum of the piece's footprint at the chosen orientation). Deterministic and easy to teach the LLM via examples in the system prompt.

3. **Easy-variant piece set.** ~~Provisional `{L, P, T, Y}`.~~ **Locked 2026-04-29 by P-0 sweep: `{L, P, W, Y}` (20 tilings on 5×4).** The original proposal `{L, P, T, Y}` had only 4 tilings — insufficient. Full enumeration of C(12,4)=495 subsets confirmed `{L, P, W, Y}` is optimal for this board. Fallback: `{L, P, T, V}` (16 tilings) if `{L, P, W, Y}` reveals issues. See [scripts/p0_count_pentomino_tilings.py](../scripts/p0_count_pentomino_tilings.py).

4. **Render format for placed pieces.** **Decision: piece letters** (e.g., `L L L L / . . . L`). Letters are LLM-friendly (Qwen tokenizer treats them as single tokens) and preserve piece identity.

5. **System-prompt piece diagrams.** Should the system prompt include ASCII art of each pentomino's shape and orientations? **Decision: yes for "easy" variant** (4 pieces, manageable). For "full" variant, system prompt would be too long with all 12 — defer the answer until we get there.

6. **Order constraint on actions.** Should the LLM choose any remaining piece each turn, or follow a fixed order (e.g., always place the next remaining piece in alphabetical order)? **Decision: any remaining piece** — gives the LLM real action flexibility, matches how human solvers think.

7. **Multi-turn in raw form vs single-step in minimal.** Same as Sudoku: raw trajectory is multi-turn; reformat to single-step minimal for SFT (the temporal-echo lesson from B-0 still applies).

8. **Hard variant (6×10) trip-wires.** If `<viability>` AUC on hard variant is < 0.55, the recipe is showing diminishing returns at scale — flag as a "hard generalization" failure. Not a blocker; just good to anticipate.

---

## 9. Cross-references

- **Project SPEC**: [doc/SPEC.md](SPEC.md) §1 (predictive-gap criterion), §2 Q6 (cross-env generalization)
- **RL approach** (will reuse for pentomino once SFT is working): [doc/plan_2026-04-29_rl_approach.md](plan_2026-04-29_rl_approach.md) — same v6 reward shape, same phased training, just `<solvable>` → `<viability>` everywhere
- **Tag-design rationale**: [doc/qa_2026-04-29_tag_design.md](qa_2026-04-29_tag_design.md) — why we keep the 4-tag minimal set
- **Runs ledger**: [doc/runs_ledger_2026-04-29.md](runs_ledger_2026-04-29.md) — pentomino runs will append here as `B-7+` once SFT lands
