# SAVE Data Generation Spec — Pentomino 5×4 LPWY

**Schema version**: save_sibling_set_v1.2 (shared with Sudoku)
**Target**: Generate sibling-set training/eval data for SAVE on Pentomino tiling, 5×4 board with piece set {L, P, W, Y}.
**Last updated**: 2026-05-04

> **This doc inherits structure and constraints from [data_generation_sudoku.md](data_generation_sudoku.md).** Sections that are unchanged for Pentomino reference the Sudoku doc by section number rather than restate it. Pentomino-specific deviations are spelled out below.

## Three-stage scaling roadmap

| Stage | train / val / test | Purpose | Promotion gate |
|---|---|---|---|
| **Toy** | 1500 / 500 / 500 | Pipeline validation | SFT loss decreasing + sanity checks pass |
| **Pilot** | 3000 / 1000 / 1000 | Paper trend validation | Viability AUC differentiates by model size + SAVE ≥ score baseline on deceptive subset |
| **Paper-final** | 8000 / 1500 / 1500 | Stable submission numbers | Paper claims hold; numbers stable across reseeded runs |

Pentomino-specific scaling caveat: at 5×4 LPWY only ~3500 reachable states exist in the search tree. Records-per-state ratios at each stage:
- Toy 2500 / ~300 unique states ≈ 8 (acceptable)
- Pilot 5000 / ~500 unique states ≈ 10 (high but OK)
- Paper-final 11000 / ~700 unique states ≈ 16 (memorization risk)

If paper-final ratio becomes a problem, expansion paths (in priority order):
1. Pre-placed-piece variants (1-piece-given starting states) → ~160 distinct roots, no retraining needed
2. Train a multi-piece-set generalist π_θ (5×4, all 26 valid 4-piece subsets) → ~1-2 days SFT+RL
3. Move to 5×6 with 6-piece subsets (172 valid configurations) → requires generalist π_θ retraining

---

## 1. Goal & Context

Same as Sudoku §1 (sibling-set training data feeding SAVE SFT, learned progress baseline, threshold calibration, and paper §3.2-§3.7 evaluations).

### Why Pentomino is a useful second env

Per the SAVE paper §3.1: "In Pentomino tiling, a placement may cover area and look locally compact while leaving an empty region that cannot be tiled by the remaining pieces." This is a topology-flavored predictive gap distinct from Sudoku's constraint-propagation gap, which strengthens the cross-env generalization claim.

### Paper-relevant constraints — same as Sudoku §1

The three hard constraints (calibration set ≠ solver-injected, deceptive subset = `valid_doomed`, score labels ≠ viability relabel) are env-agnostic.

---

## 2. Output Schema (`save_sibling_set_v1.2`)

Identical to Sudoku §2, with three Pentomino-specific field meanings:

| Field | Pentomino value |
|---|---|
| `env` | `"pentomino5x4"` (was `"sudoku4"`) |
| `state.state_text_version` | `"pentomino_text_b8_v1"` (B-8 board format MINUS the doom-suffix leak) |
| `state.state_struct` | `{"board": [[".",".",".",".",...]], "remaining_pieces": ["L","P","W","Y"]}` |
| `candidates[].action_struct` | `{"piece": "L", "ori": 0, "row": 1, "col": 1}` (1-indexed row/col) |
| `candidates[].action_text` | `"place L ori=0 at row 1 col 1"` |
| `candidates[].action_text_canonical` | `"L:0:R1C1"` |

Everything else (state_solver, action_space_stats, sampling_protocol, candidate_class decision tree, set_stats, deceptive_pairs, selection_criteria, provenance) is unchanged.

### State rendering (`pentomino_text_b8_v1`)

```
Current board (5x4):
. . . .
. . . .
. . . .
. . . .
. . . .

Remaining pieces: L, P, W, Y
```

Differences from B-8 training-data format:
- **No `Last action: ...` line** — that's where the doom-suffix leak (`— board now unsolvable (...)`) lived. SAVE renders state from scratch each turn; "Last action" provides no value at single-turn data-gen.
- Always shows the `Current board (HxW):` header (matches B-8).
- Cells separated by single space, '.' for empty, piece letter for filled.

### State hash

`sha1(json.dumps([board, remaining_pieces]))`. Both inputs to the hash because two different remaining-piece sets at the same board are different states.

---

## 3. Three Dataset Roles

Same as Sudoku §3:

- `train_balanced`: 1500 sibling sets, K_total=12, sources `lt:ht:rand:sol:prt = 3:3:0:3:3`
- `val_natural_calibration`: 500 sibling sets, K_total=8, sources `lt:ht:rand:sol:prt = 4:4:0:0:0` (no oracle)
- `test_natural_policy`: 500 sibling sets, same protocol as val
- (Optional `*_balanced_diagnostic`: skip for toy run)

The boundary-state target (≥60% mixed in `train_balanced`) is harder for Pentomino because trajectories are short (4 steps); see §6.3.

---

## 4. Sampling Protocol — Pentomino-specific notes

### 4.1 Five candidate sources

`lt`, `ht`: same as Sudoku §4.1, sampling from `rl_pentomino_5x4_no_leak_v8_aq/final` with `temperature=0.3` and `temperature=1.0`.

`rand`: skip for toy run (`K_rand=0`).

`sol`: solver-derived viable continuation. Use `PentominoSolver().find_one_solution(board, remaining_pieces)` to get one valid tiling sequence. **Filter through `is_local_valid` and emit only candidates that ARE locally valid at the current state** (not future-state placements). For Pentomino, sol_path entries 1, 2, 3 happen to also be locally valid at s_0 because they place different pieces at non-overlapping anchors, so up to `K_sol` candidates pass the filter. At later states, fewer pass.

`prt`: perturbed solver-path action. For Pentomino, the perturbation is more subtle than Sudoku's "pick a different value at the same cell". Strategy:

```python
# Take sol[0]. It places piece P at orientation O at anchor (R, C), covering cells {(r1,c1)...}.
# A perturbation: pick a DIFFERENT (piece', ori', anchor') combination such that:
#   (a) piece' ∈ remaining_pieces
#   (b) placement is legal (in-bounds, no overlap)
#   (c) board AFTER this placement is unsolvable per solver
candidates_perturbed = []
for piece_prime in remaining_pieces:
    for ori_prime in range(num_orientations(piece_prime)):
        for ar in range(BOARD_H):
            for ac in range(BOARD_W):
                # skip the actual sol[0] placement
                if (piece_prime, ori_prime, ar, ac) == sol[0]:
                    continue
                cells = placement_cells(piece_prime, ori_prime, ar, ac)
                if cells is None or not fits_on_board(cells, BOARD_H, BOARD_W, board):
                    continue
                test_board = apply_action(board, remaining_pieces, ActionStruct(piece_prime, ori_prime, ar+1, ac+1))[0]
                test_remaining = [p for p in remaining_pieces if p != piece_prime]
                if not solver.is_viable(test_board, test_remaining):
                    candidates_perturbed.append((piece_prime, ori_prime, ar, ac))
                    if len(candidates_perturbed) >= K_prt:
                        break
```

If fewer than `K_prt` perturbations exist (rare on the empty board, more common at later states), record actual count and proceed. Spec §11 fallback: try perturbing later positions in the sol path.

### 4.2 `policy_eval_logprob` computation

Same protocol as Sudoku §4.2: **bare `action_text` for ALL sources** (not the LLM's full XML response). Logprobs comparable across `lt`/`ht`/`sol`/`prt` for CVCP's `arg max log π_θ(a|s)` tie-break.

Base policy: `outputs/rl_pentomino_5x4_no_leak_v8_aq/final`.
Prompt: `polyomino_minimal_v1` — derived from `SFTFormatter.SYSTEM_PROMPTS["polyomino_minimal"]` verbatim (no wart; this prompt was written for the env).
Records: `policy_eval_prompt_version = "polyomino_minimal_v1"`.

### 4.3 Boundary state preference

For Pentomino with 4-step trajectories, the boundary criterion changes:

```python
def is_boundary_state(board, remaining_pieces, threshold=0.3):
    legal = enumerate_legal_actions(board, remaining_pieces)
    if not legal:
        return False
    # Sample a small subset (full enumeration may be expensive at early states)
    n_sample = min(20, len(legal))
    rng = random.Random(...)
    sampled = rng.sample(legal, n_sample)
    viable_count = sum(1 for a in sampled
                        if solver.is_viable(*apply_action(board, remaining_pieces, a)))
    mix = min(viable_count, n_sample - viable_count) / n_sample
    return mix >= threshold
```

Trajectories of 4 placements means at most 4 candidate states per trajectory (after step 0, 1, 2, 3). With 1500 sibling-set target, we need ~1500/4 = 375 trajectories. Each trajectory comes from a fresh sampling under `π_θ`; root puzzle is always the empty 5×4 LPWY (§5.4).

If boundary rate drops below 60% in `train_balanced`, lower threshold to 0.2 (per spec §11).

### 4.4 Trajectory walking + split for leakage prevention

Pentomino has only ONE root puzzle (empty 5×4 LPWY), so trajectory walks under `π_θ` are how we get state diversity. Each trajectory is a stochastic policy walk:

```python
for traj_seed in range(N_trajectories):
    rng = Random(traj_seed)
    state = get_root_puzzle()  # always (empty board, [L, P, W, Y])
    for t in range(MAX_TRAJ_LEN):  # at most 4 placements for 5×4 LPWY
        # Sample one action via lt-style (T=0.3) — matches sec 4.3 boundary check distribution
        sample = batched_sample(model, tok, build_chat(state), K=1, temperature=0.3)
        action = parse(sample.text)
        if action is None or not is_local_valid(state, action):
            break  # walk dies; OK
        # If state is a boundary state per §4.3, anchor a sibling set here
        if t > 0 and is_boundary_state(state):
            emit_sibling_set(state, traj_seed, t)
        state = apply_action(state, action)
```

**Why stochastic walks**: deterministic solver-path walks always traverse the same canonical states from the empty root, giving only ~4 unique states. Stochastic walks under `π_θ` produce different trajectories per `traj_seed` → different intermediate states explored.

Splits by `traj_seed` for leakage prevention:
- seeds 0…1049 → train (70%)
- seeds 1050…1274 → val (15%)
- seeds 1275…1499 → test (15%)
- Sibling sets in each role come ONLY from trajectories in that role's seed range

Leakage can also occur at **state level** (two different trajectories visiting the same intermediate state). The §6.4 leakage check uses `state_hash` to flag this; Pentomino has ~3500 reachable states from the LPWY root, so some collision is expected at scale. Worth verifying empirically.

---

## 5. Pentomino 5×4 Environment Spec

### 5.1 State and action representation

State (`state.state_struct`):
```json
{
  "board": [[".", ".", ".", "."],
            [".", ".", ".", "."],
            [".", ".", ".", "."],
            [".", ".", ".", "."],
            [".", ".", ".", "."]],
  "remaining_pieces": ["L", "P", "W", "Y"]
}
```

Cells are strings: `"."` for empty, single letter `L|P|W|Y` for filled. (Sudoku used `int`; Pentomino uses `str`. Schema's `StateStruct.grid` field changes type accordingly — handled by Pydantic via `List[List[str]]` in the Pentomino-specific subclass.)

Action structure:
```python
{"piece": "L", "ori": 0, "row": 1, "col": 1}
```

`piece ∈ {L, P, W, Y}`, `ori` is the orientation ID (0 to N-1, deterministic per piece per `PIECE_ORIENTATIONS`), `(row, col)` is the **1-indexed anchor** (top-most leftmost cell of the piece's footprint at orientation `ori`).

Action text: `"place L ori=0 at row 1 col 1"` (matches B-8 training format).
Action canonical: `"L:0:R1C1"` (used for dedup hash).

### 5.2 Local progress formula (`pentomino_local_progress_v1`)

Per the paper §3.4, the Pentomino progress heuristic should "rank candidates by filled area / locally compact placements". Concrete formula:

```python
def local_progress(board):
    h, w = BOARD_H, BOARD_W
    filled_cells = sum(1 for r in range(h) for c in range(w) if board[r][c] != '.')
    filled_normalized = filled_cells / (h * w)
    
    # Holes: connected components of '.' cells with size NOT divisible by 5
    # (any pentomino covers exactly 5 cells; a region whose size isn't a multiple
    # of 5 is provably untileable, regardless of remaining pieces).
    n_holes = sum(1 for sz in connected_components(board) if sz % 5 != 0)
    
    score = filled_normalized - 0.1 * n_holes
    return score, {
        "filled_cells": filled_cells,
        "filled_normalized": filled_normalized,
        "n_holes": n_holes,
        ...
    }
```

This formula:
- Rewards covered area (more pieces placed → higher progress)
- Penalizes obvious untileable regions (component sizes 1, 2, 3, 4, 6, 7, 8, 9, etc.)
- **Does NOT consult the solver** — connectivity check is O(h*w) BFS, not full tiling search

This matters for the deceptive benchmark (paper §3.4): a placement that fills more cells than a viable alternative should *score higher* under progress, even if it makes the board globally untileable for subtler reasons (e.g., a 5-cell region that happens to require the piece W which is blocked elsewhere). The progress formula's coarseness — only flagging "unsalvageable" via component-size — is the entire point: progress and viability disagree exactly when subtle reachability matters.

Schema `progress.formula_id`: `"pentomino_local_progress_v1"`.

### 5.3 Solver interface

`PentominoSolver(board_h=5, board_w=4)` — see [scripts/pentomino5x4_solver.py](../scripts/pentomino5x4_solver.py).

```python
class PentominoSolver:
    version = "pentomino5x4_solver_v1"
    
    def solve(self, board, remaining_pieces) -> SolverResult:
        # Returns: solvable, num_solutions, nodes, backtracks,
        #          solution_depth, solve_time_ms, solution_path
        ...
    
    def is_viable(self, board, remaining_pieces) -> bool: ...
    def find_one_solution(self, board, remaining_pieces) -> Optional[List[Placement]]: ...
```

`solution_path` is a list of `(piece, ori_id, anchor_r_0idx, anchor_c_0idx)` placements. Run exhaustively up to `solution_cap=8` for `solve()` so unsolvable states get a meaningful `nodes` value.

### 5.4 Initial puzzle generation

Pentomino has ONE root puzzle, not many:
```python
def get_root_puzzle():
    return [['.'] * 4 for _ in range(5)], list("LPWY")
```

Diversity comes from sampling diverse trajectories under `π_θ` from this same starting state (different seeds → different `lt`/`ht` action sequences → different intermediate states).

---

## 6. Generation Pipeline

### 6.1 High-level flow

1. Call `get_root_puzzle()` to get the canonical empty 5×4 LPWY board.
2. For each trajectory (seeded by `traj_seed`):
   a. From the root, sample one full trajectory under `π_θ` greedy (or `lt` at T=0.3).
   b. Each intermediate state along the trajectory (steps 0, 1, 2 — skipping step 3 which is "1 piece left, trivial") is a candidate sibling-set anchor.
   c. Skip non-boundary states with probability 0.5 per spec §4.3.
3. For each chosen state s_t, sample one sibling set (12 candidates: 3 lt + 3 ht + 3 sol + 3 prt).
4. Validate, label, dedup, emit JSONL line.
5. Continue until target size reached.

### 6.2 Pseudocode

Same as Sudoku §6.2. Calls into env-specific `apply_action`, `is_local_valid`, `solver.solve`, `compute_progress`, `parse_action_text`, `state_hash`.

### 6.3 Hyperparameters (toy-run defaults)

```python
TOY_RUN_CONFIG_PENTOMINO = {
    "n_trajectories": 800,                 # ~3 sibling sets per traj × 800 = 2400 candidates
    "split_seed_ranges": {"train": (0, 559), "val": (560, 679), "test": (680, 799)},
    "sibling_sets_per_traj": {
        "train_balanced": 3,
        "val_natural_calibration": 4,
        "test_natural_policy": 4,
    },
    "target_size": {
        "train_balanced": 1500,
        "val_natural_calibration": 500,
        "test_natural_policy": 500,
    },
    "boundary_threshold": 0.3,
    "boundary_oversampling_rate": 0.5,
    "max_attempts_per_set": 3,
}

SAMPLING_PROTOCOLS = {
    "train_balanced": {"K_total": 12, "K_lt": 3, "K_ht": 3, "K_rand": 0, "K_sol": 3, "K_prt": 3,
                       "lt_temperature": 0.3, "ht_temperature": 1.0, "top_p": 0.95},
    "val_natural_calibration": {"K_total": 8, "K_lt": 4, "K_ht": 4, "K_rand": 0, "K_sol": 0, "K_prt": 0,
                                "lt_temperature": 0.3, "ht_temperature": 1.0, "top_p": 0.95},
    "test_natural_policy": {"K_total": 8, "K_lt": 4, "K_ht": 4, "K_rand": 0, "K_sol": 0, "K_prt": 0,
                            "lt_temperature": 0.3, "ht_temperature": 1.0, "top_p": 0.95},
}
```

### 6.4 Performance hints

Pentomino is **cheaper than Sudoku** for solver calls (4-step trajectory, small board) but candidate enumeration is more expensive (each piece has up to 8 orientations × ~20 valid anchor cells = ~160 placements per piece). Per-state legal-action enumeration:
- Empty 5×4 LPWY: 172 legal placements (verified via env smoke test)
- After 1 placement: ~80
- After 2: ~30
- After 3: ~5

Solver: Pentomino exhaustive solver completes in <10ms per state for 4×5 LPWY. For 1500 × 12 candidates × ~3 solver calls each = ~54k solver calls = ~10 minutes CPU. LLM forward passes dominate (~36k = 30-60 min on H800).

---

## 7. Sanity Checks

Same as Sudoku §7. Pentomino-specific assertions:

- Schema validation uses `state_struct.board` field (not `state_struct.grid`)
- For `train_balanced`: at least 60% mixed sibling sets (may need to lower boundary threshold to 0.2 if not hit)
- Doom-suffix leak check: NO record's `state_text` or `next_state_text` contains the substring `"— board now unsolvable"` (this would indicate the leak crept back in)
- `action_struct.piece` is one of `{L, P, W, Y}` for all candidates
- `action_struct.ori` is in valid range for each piece

### 7.5 Distributional sanity

Print histograms for:
- Distribution of `set_stats.mixed` per role
- Distribution of `next_viable` per role
- Distribution of `candidate_class` (expect goal_reaching candidates at step 3 — those are the placements that complete the tiling)
- Histogram of `solver.nodes` per candidate
- Histogram of `progress.local_progress_score` for viable vs doomed candidates (overlap is the deceptive-pair signal)
- **Per-piece and per-orientation distribution** in `lt` and `ht` candidates — check for piece-collapse (model only ever picks L first, etc.)

---

## 8. File Layout

```
data/
  pentomino5x4/
    train_balanced.jsonl              # 1500 records
    val_natural_calibration.jsonl     # 500 records
    test_natural_policy.jsonl         # 500 records
    metadata.json                     # generation metadata
    sanity_check_report.txt
    _phase0_prompt_decision.json      # Phase 0 sanity check verdict (likely SHIP_FIXED for Pentomino since polyomino_minimal has no wart)

scripts/
  generate_save_data.py               # SHARED: parameterized by --env
  validate_dataset.py                 # SHARED
  pentomino5x4_solver.py              # ✅ done
  pentomino5x4_env.py                 # ✅ done
  progress_pentomino5x4.py            # TBW
  policy_sampler.py                   # SHARED: HF wrapper for lt/ht + logprob eval
  save_schema.py                      # SHARED: Pydantic models
  sanity_check_rl_b5_under_corrected_prompt.py  # SHARED: parameterizable to --env polyomino
```

`metadata.json` follows Sudoku §8 with `env: "pentomino5x4"`, `policy_model: "outputs/rl_pentomino_5x4_no_leak_v8_aq/final"`.

---

## 9. Pydantic Schema

Schema is shared (`save_sibling_set_v1.2`). The `state.state_struct` field is `List[List[str]]` (vs Sudoku's `List[List[int]]`). The schema accommodates both via `Any` typing OR a discriminated union. For toy run, the simpler choice: Sudoku and Pentomino each use their own custom `StateStruct` subclass, dispatched per `env` field. Both validate against the parent `State` class.

---

## 10. Test Run Protocol (do this first)

Before running 1500-set generation, do a 30-set smoke test:

```bash
python scripts/generate_save_data.py \
    --env polyomino \
    --role train_balanced \
    --n_target 30 \
    --output data/pentomino5x4/smoke_test.jsonl \
    --policy_model outputs/rl_pentomino_5x4_no_leak_v8_aq/final \
    --seed 42
python scripts/validate_dataset.py data/pentomino5x4/smoke_test.jsonl
```

Inspect manually per Sudoku §10. Pentomino-specific checks:
- A `goal_reaching` candidate appears in records at step 3 (the final placement that completes a tiling)
- All `action_text` values match the regex `^place [LPWY] ori=\d+ at row \d+ col(?:umn)? \d+$`
- No `state_text` contains `"— board now unsolvable"`
- `state_struct.board[r][c]` is `"."` or one of `{"L","P","W","Y"}`

Only after smoke test passes manually, scale to 1500/500/500.

---

## 11. Common Failures and Recovery

Same as Sudoku §11, plus Pentomino-specific:

| Symptom | Likely cause | Fix |
|---|---|---|
| `set_stats.mixed=false` for >60% of train_balanced | At step 0 of a 4-step trajectory, most placements are still viable; mix score low | Sample sibling sets at step 1 and step 2, not step 0 |
| Few `prt` candidates available | At empty 5×4 LPWY, hard to find a locally-legal-but-doomed perturbation | Use later positions in the sol path; if still starved, drop K_prt and increase K_sol |
| Doom-suffix leak detected by sanity check | Bug in `render_state_b8` (accidentally including the "Last action" line) | Re-check render — should NEVER append "Last action: ... — board now unsolvable" |
| Piece-collapse in `lt` candidates (always picks L first) | Model is overly confident at empty-board state | Increase `lt_temperature` slightly (0.3 → 0.5); or rely on `ht` for diversity |

---

## 12. Hand-off Checklist

Same as Sudoku §12, with Pentomino-specific additions:

- [ ] All three role files exist at target sizes
- [ ] `validate_dataset.py` passes for all three files
- [ ] No leakage between any two roles (state-level + trajectory-level)
- [ ] `train_balanced` has ≥60% mixed sibling sets
- [ ] `train_balanced` has ≥100 deceptive pairs total
- [ ] No record's `state_text` contains the doom-suffix `"— board now unsolvable"`
- [ ] `val_natural_calibration` and `test_natural_policy` contain zero sol/prt candidates
- [ ] `metadata.json` records `policy_checkpoint = "rl_pentomino_5x4_no_leak_v8_aq_final"`
- [ ] First 5 records of each file have been visually inspected
- [ ] A backup of all generated data exists

---

## End of spec
