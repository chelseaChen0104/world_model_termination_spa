# SAVE Data Generation Spec — Hidato 5×4

**Schema version**: save_sibling_set_v1.2 (shared with Sudoku/Pentomino)
**Target**: Generate sibling-set training/eval data for SAVE on Hidato (Numbrix variant), 5×4 grid, **200-puzzle algorithmically-expanded bank**.
**Last updated**: 2026-05-04

> **This doc inherits structure and constraints from [data_generation_sudoku.md](data_generation_sudoku.md).** Sections that are unchanged for Hidato reference the Sudoku doc by section number. Hidato-specific deviations are spelled out below.

## Three-stage scaling roadmap

| Stage | train / val / test | Purpose | Promotion gate |
|---|---|---|---|
| **Toy** | 1500 / 500 / 500 | Pipeline validation | SFT loss decreasing + sanity checks pass |
| **Pilot** | 3000 / 1000 / 1000 | Paper trend validation | Viability AUC differentiates by model size + SAVE ≥ score baseline on deceptive subset |
| **Paper-final** | 8000 / 1500 / 1500 | Stable submission numbers | Paper claims hold; numbers stable across reseeded runs |

### State coverage and scaling — Hidato uses a fixed puzzle bank

Hidato uses an **algorithmically generated puzzle bank** under `data/hidato_bank_5x4_<ver>/bank.py`. The bank is loaded by [scripts/hidato5x4_env.py](../scripts/hidato5x4_env.py); env var `HIDATO_BANK=<ver>` selects which bank version to use (default `v3` if present, else `v2`, else legacy 8-puzzle fallback).

Bank versions in use as of 2026-05-04:
- **v2** (200 puzzles, seed=42): used for toy data (`hidato5x4_env_v2_200puzzles`)
- **v3** (600 puzzles, seed=42): used for pilot data (`hidato5x4_env_v3_600puzzles`). v3 is a strict superset of v2 — the algorithm is deterministic, so v3[0..199] == v2[0..199] byte-for-byte; v3 adds 400 new puzzles beyond v2.

**Bank ceilings** (records achievable, given `K_max_sibling_sets_per_puzzle=12` and the 70/15/15 split):
- v2 (200): train 1680, val 360, test 360 → caps toy val/test at 360 (we observed 346/345)
- v3 (600): train 5040, val 1080, test 1080 → comfortable for pilot 3000/1000/1000
- For paper-final 8000/1500/1500: train and val/test ceilings of v3 (5040 / 1080) are below paper-final targets. **A v4 bank with ≥1000 puzzles will be needed** (1000×0.7×12=8400 train, 1000×0.15×12=1800 val/test).

**Train/val/test split is at the trajectory level, not the puzzle level.** The split uses `root_idx` ranges (70/15/15), but `puzzle = bank[root_idx % len(bank)]` — meaning train and val/test cycle through the same puzzle pool. Leakage prevention works at the **state level**: different trajectory seeds produce different stochastic walks from the same puzzle, so val/test sample different boundary-state anchors than train.

**Toy/pilot puzzle overlap**: because v3 is a superset of v2, the toy dataset (v2) shares its first 200 puzzles with the pilot dataset (v3). When evaluating, train and eval should both use pilot's val/test (or both use toy's val/test) — don't cross-evaluate toy-trained vs pilot-eval or vice versa, as that would have puzzle-level leakage on the shared 200.

To expand the bank: `scripts/expand_hidato_bank.py --rows 5 --cols 4 --n-puzzles N --seed 42 --output data/hidato_bank_5x4_v<N>/bank.py`. Runs in ~5 min for N=1000.

This contrasts with Sudoku (procedural generation, no bank, scales naturally) and Pentomino (math-capped at 172 valid subsets at 5×6, requires per-trajectory expansion).

---

## 1. Goal & Context

Same as Sudoku §1.

### Why Hidato is a useful third env

Per the SAVE paper §3.1: "In Hidato, a number placement may satisfy local adjacency while preventing any completion of the required consecutive path." This is a Hamiltonian-path predictive gap — distinct from both Sudoku's constraint propagation and Pentomino's tiling. Three orthogonal types of "local-valid but globally doomed" actions strengthen the cross-env claim.

### Hidato variant used

This project uses the **Numbrix** variant: 4-connectivity (orthogonal adjacency only, not diagonal). The puzzle is to fill a partially-given grid with numbers `1..N` such that consecutive numbers `(k, k+1)` occupy edge-adjacent cells.

### Paper-relevant constraints — same as Sudoku §1

The three hard constraints (calibration set ≠ solver-injected, deceptive subset = `valid_doomed`, score labels ≠ viability relabel) are env-agnostic.

---

## 2. Output Schema (`save_sibling_set_v1.2`)

Identical to Sudoku §2, with three Hidato-specific field meanings:

| Field | Hidato value |
|---|---|
| `env` | `"hidato5x4"` (was `"sudoku4"`) |
| `state.state_text_version` | `"hidato_text_v1"` (matches the existing HidatoEnv.render format minus history lines) |
| `state.state_struct` | `{"rows": 5, "cols": 4, "assignment": {(r,c): N, ...}, "next_n": int, "anchor_cell": [r, c]}` |
| `candidates[].action_struct` | `{"row": 2, "col": 3, "value": 5}` (1-indexed; value = next_n) |
| `candidates[].action_text` | `"place 5 at row 2 col 3"` |
| `candidates[].action_text_canonical` | `"R2C3=5"` |

The `value` field is REDUNDANT with `state.state_struct.next_n` — every action at state `s_t` places exactly `next_n`. We keep it explicit in `action_struct` for symmetry with Sudoku's schema and for parser robustness.

### State rendering (`hidato_text_v1`)

```
Hidato puzzle (5x4):
 1 .. .. ..
 2 .. .. ..
 9 .. .. ..
.. .. .. ..
17 .. .. ..

Already placed: [1, 2, 9, 17]
Next number to place: 3
Must be adjacent to 2 at row 2 col 1.
```

Notes:
- The grid uses two-character cell width (`..` for empty, right-padded number for filled). Matches the existing `HidatoEnv.render` format.
- Below the grid, three auxiliary info lines:
  - `Already placed: [...]` — sorted list of all placed numbers
  - `Next number to place: K` — the next sequential number
  - `Must be adjacent to K-1 at row R col C` — the cell where the predecessor lives
- **No `Last action: ...` line in SAVE rendering** (the existing trajectory format includes one but we strip it for state-only single-turn use).

The auxiliary info lines are CRITICAL for Hidato — without them the model can't reason about which placement is legal (the env's adjacency-to-previous constraint isn't visually obvious from the grid alone).

### State hash

`sha1(json.dumps(sorted(assignment.items())))`. Just the cell→number mapping; everything else (next_n, anchor_cell) is derivable.

---

## 3. Three Dataset Roles

Same as Sudoku §3 with size targets. **Bank expansion to 200 algorithmically-generated 5×4 puzzles is REQUIRED** (no longer a future step):

- 200 puzzles × ~17 placements/puzzle × ~stochastic trajectory variations → ~3000 unique states reachable
- 70/15/15 puzzle-level split gives 140 train / 30 val / 30 test puzzles — sufficient for proper leakage isolation

Bank-expansion script: [scripts/expand_hidato_bank.py](../scripts/expand_hidato_bank.py). Output: `data/hidato_bank_5x4_v2/bank.py` (drop-in replacement for the legacy 8-puzzle bank).

**Why 200 puzzles**: at toy-run 1500 records / ~500 unique states ≈ records-per-state ratio 3 (healthy). At paper-final 8000 records / ~500 states ≈ ratio 16 (high; consider further expansion to ~500 puzzles).

---

## 4. Sampling Protocol — Hidato-specific notes

### 4.1 Five candidate sources

`lt`, `ht`: same as Sudoku §4.1. Sample from `outputs/sft_hidato_no_leak/final` with `temperature=0.3` and `temperature=1.0`. **Note: `π_θ` for Hidato is SFT-only, not RL** — RL training was stopped at step 46/200 (user directive 2026-05-03). Pass@1 ≈ 16.7% (vs 60% for the leaked-RL alternative we chose not to use). Implications below in §6.3.

`rand`: skip for toy run.

`sol`: solver-derived viable continuation. Use `Hidato5x4Solver().find_one_solution(state)` to get one valid Hamiltonian-path completion. **Filter through `is_local_valid` and emit only locally-valid candidates.** For Hidato, env enforces `value == state.next_n`, so only `sol_path[0]` passes the filter at any given state — `K_sol` effectively collapses to 1.

`prt`: perturbed solver-path action. **For Hidato, the action space at any state is at most 4 cells** (the cells adjacent to the predecessor of `next_n`). Of those 4, typically 0-2 are pre-existing givens (skipped by env) or already filled. So legal placements = 1-4 cells. Perturbation strategy:

```python
# Take sol[0]: places next_n at the chosen adjacent cell.
# A perturbation: pick a DIFFERENT adjacent cell that is locally legal but doomed.
candidates_perturbed = []
for adj_r, adj_c in adjacent_cells(prev_n_row, prev_n_col, R, C):
    if (adj_r, adj_c) == (sol[0].row, sol[0].col):
        continue
    if state.assignment.get((adj_r, adj_c)) is not None:
        continue  # cell already filled
    test_state = apply_action(state, ActionStruct(adj_r+1, adj_c+1, next_n))
    if not solver.is_viable(test_state):
        candidates_perturbed.append((adj_r, adj_c, next_n))
```

**Hidato `prt` will starve more often than Sudoku's** because the action space is so constrained. At many states, only ONE legal placement exists (forced moves) — so no `prt` is possible. Mitigation per spec §11: drop `K_prt` to 0 at forced-move states; rely on `K_sol=3` as the only oracle source. Boundary states (where multiple moves exist and some are doomed) are exactly the cases where `prt` succeeds.

### 4.2 `policy_eval_logprob` computation

Same protocol as Sudoku §4.2: **bare `action_text` for ALL sources** (not the LLM's full XML response). Logprobs comparable across `lt`/`ht`/`sol`/`prt`.

Base policy: `outputs/sft_hidato_no_leak/final` on autodl1.
Prompt: `hidato_minimal_v1` — derived from `SFTFormatter.SYSTEM_PROMPTS["hidato_minimal"]` verbatim.
Records: `policy_eval_prompt_version = "hidato_minimal_v1"`.

### 4.3 Boundary state preference

For Hidato, the boundary criterion: states where ≥2 legal placements exist AND ≥1 is doomed AND ≥1 is viable.

```python
def is_boundary_state(state, threshold=0.3):
    legal = enumerate_legal_actions(state)
    if len(legal) < 2:
        return False  # forced moves can't be boundary
    viable_count = sum(1 for a in legal if solver.is_viable(apply_action(state, a)))
    doomed_count = len(legal) - viable_count
    mix = min(viable_count, doomed_count) / len(legal)
    return mix >= threshold
```

Hidato has FEWER boundary states than Sudoku per puzzle (most steps are forced). But when boundary states do occur, they're high-quality (genuinely 2-4 way decisions). Expect oversampling rate to need increase: the `boundary_threshold` may need to drop to **0.25** instead of 0.3.

### 4.4 Trajectory walking + split for leakage prevention

**Trajectory walks**: stochastic policy sampling under `π_θ` (T=0.3), same as Sudoku §4.3-§4.4 protocol. Each `(puzzle_idx, traj_seed)` pair produces one walk; intermediate states with `is_boundary_state == True` get anchored as sibling-set sources.

**Splits at puzzle level** (now feasible with 200-puzzle bank):
- puzzles 0…139 → train (70%)
- puzzles 140…169 → val (15%)
- puzzles 170…199 → test (15%)
- Sibling sets in each role come ONLY from trajectories on puzzles in that role's range

**Why puzzle-level split**: with 200 puzzles, the 70/15/15 split gives 140/30/30 puzzles per role — enough for the 1500/500/500 record targets without sharing puzzles across roles. Eliminates state-level leakage that the old 8-puzzle bank forced.

---

## 5. Hidato 5×4 Environment Spec

### 5.1 State and action representation

State (`state.state_struct`):
```json
{
  "rows": 5,
  "cols": 4,
  "assignment": {"0,0": 1, "1,0": 2, "2,0": 9, "4,0": 17},
  "next_n": 3,
  "anchor_cell": [1, 0]
}
```

The `assignment` keys are `"r,c"` strings (0-indexed) for JSON serializability. `anchor_cell` is the 0-indexed cell where `next_n - 1` lives (the predecessor of `next_n`).

Action structure:
```json
{"row": 2, "col": 2, "value": 3}
```

`(row, col)` are **1-indexed**, `value = next_n` (always — the env strictly enforces sequential placement).

Action text: `"place 3 at row 2 col 2"`.
Action canonical: `"R2C2=3"`.

### 5.2 Local progress formula (`hidato_local_progress_v1`)

Per the paper §3.4, the Hidato progress heuristic should "rank candidates by path coverage / contiguity". Concrete formula:

```python
def local_progress(state):
    R, C = state["rows"], state["cols"]
    n_total = R * C
    n_placed = len(state["assignment"])
    placed_normalized = n_placed / n_total
    
    # Penalty: number of "isolated givens" — pre-filled cells whose value v
    # has neither v-1 nor v+1 currently placed AND the path has already
    # passed v's expected position.
    # Simpler proxy (no future-lookup): count empty cells that have NO empty
    # neighbor AND are not adjacent to any placed value within ±1 of next_n.
    isolated_empties = sum(
        1 for (r, c) in empty_cells(state)
        if no_useful_neighbor(state, r, c)
    )
    
    score = placed_normalized - 0.05 * isolated_empties
    return score, {
        "n_placed": n_placed,
        "placed_normalized": placed_normalized,
        "isolated_empties": isolated_empties,
        ...
    }
```

This formula:
- Rewards path progress (more numbers placed → higher score)
- Penalizes obvious dead-end cells via local adjacency proxies
- **Does NOT consult the solver** — only checks neighbor structure

For deceptive examples (paper §3.4): a placement that visually advances the path (more cells filled) but disconnects future givens scores HIGH on this progress metric, while the solver labels it doomed. This is exactly the dissociation SAVE needs.

Schema `progress.formula_id`: `"hidato_local_progress_v1"`.

### 5.3 Solver interface

`Hidato5x4Solver()` — see [scripts/hidato5x4_solver.py](../scripts/hidato5x4_solver.py) (TBW).

```python
class Hidato5x4Solver:
    version = "hidato5x4_solver_v1"
    
    def solve(self, state) -> SolverResult:
        # Returns: solvable, num_solutions, nodes, backtracks,
        #          solution_depth, solve_time_ms, solution_path
        ...
    
    def is_viable(self, state) -> bool: ...
    def find_one_solution(self, state) -> Optional[List[Action]]: ...
```

Wraps `src.environments.hidato_utils.is_solvable` with instrumentation. Backtracking with adjacency + reachability pruning per the existing implementation. Run exhaustively up to `solution_cap=8`.

### 5.4 Initial puzzle selection

Hidato uses the existing 8-puzzle bank from [src/environments/hidato_puzzle_bank.py](../src/environments/hidato_puzzle_bank.py):

| ID | Size | Cells | Givens | Empty (= rollout length) |
|---|---|---|---|---|
| 3x3_snake / 3x3_u / 3x3_spiral | 3×3 | 9 | 2 | 7 |
| 4x3_snake | 4×3 | 12 | 2 | 10 |
| 5x3_snake | 5×3 | 15 | 3 | 12 |
| 4x4_boustrophedon / 4x4_spiral | 4×4 | 16 | 2-3 | 13–14 |
| 5x4_snake | 5×4 | 20 | 3 | 17 |

```python
def get_root_puzzle(seed):
    """Pick a puzzle by seed (deterministic)."""
    bank = PUZZLES  # from hidato_puzzle_bank
    return bank[seed % len(bank)]
```

`provenance.env_version`: `"hidato5x4_env_v1_8puzzles"`.

> **Known limitation**: 8 puzzles is small. Per [future_steps.md](future_steps.md) NEAR-6, expanding the bank algorithmically to ~200 is queued. This affects held-out generalization claims; for the toy run, the 8-puzzle bank is acceptable but the limitation is documented in the paper.

---

## 6. Generation Pipeline

### 6.1 High-level flow

1. For trajectory_seed in 0..N_TRAJ:
   a. Pick a puzzle: `puzzle = bank[traj_seed % 8]`
   b. From the puzzle's initial state, sample one full trajectory under `π_θ` greedy or `lt` (T=0.3)
   c. Each intermediate state along the trajectory (skipping the first step and the last 2) is a sibling-set anchor candidate
   d. Skip non-boundary states with probability 0.5
2. For each chosen state, sample one sibling set (12 candidates: 3 lt + 3 ht + 3 sol + 3 prt; reduce K_prt as needed at forced-move states)
3. Validate, label, dedup, emit JSONL line
4. Continue until target reached

### 6.2 Pseudocode

Same as Sudoku §6.2 with env-specific dispatch.

### 6.3 Hyperparameters (toy-run defaults)

```python
TOY_RUN_CONFIG_HIDATO = {
    "n_trajectories": 1000,
    "split_seed_ranges": {"train": (0, 699), "val": (700, 849), "test": (850, 999)},
    "sibling_sets_per_traj_avg": {  # avg because traj length varies (7-17 steps)
        "train_balanced": 2.5,
        "val_natural_calibration": 4,
        "test_natural_policy": 4,
    },
    "target_size": {
        "train_balanced": 1500,
        "val_natural_calibration": 500,
        "test_natural_policy": 500,
    },
    "boundary_threshold": 0.25,        # lower than Sudoku/Pentomino (0.3) due to forced-move dominance
    "boundary_oversampling_rate": 0.5,
    "max_attempts_per_set": 3,
}
```

**Special handling for forced-move states**: at states with only 1 legal action, skip — sibling set isn't meaningful.

**Adjusted K mixture for early-traj states with action space ≤ 4**:
- If `len(enumerate_legal_actions(state)) < K_total` after dedup: cap K_total at `len(legal_actions)`
- Record the actual K achieved in `sampling_protocol.candidate_pool_size_after_dedup`

### 6.4 Performance hints

Hidato is **fastest of the three games** for solver calls:
- 4×4 / 5×4 grids: bounded-depth backtracking <5ms per state
- 1500 × 12 candidates × ~3 solver calls = ~54k solver calls = ~5 minutes CPU

LLM forward passes still dominate: ~36k forward passes = 30-60 min on H800.

**`π_θ` quality concern**: SFT-only Hidato hits ~16.7% greedy Pass@1 on these puzzles. So `lt` candidates have a non-trivial doomed rate even from valid starting states. This is OK — `lt`/`ht` failing is exactly what makes mixed sibling sets via `sol`/`prt`.

---

## 7. Sanity Checks

Same as Sudoku §7. Hidato-specific assertions:

- Schema validation uses `state_struct.assignment` (dict) field
- For `train_balanced`: target ≥50% mixed (lower than Sudoku's 60% due to forced-move dominance — accept as Hidato-structural)
- Doom-suffix leak check: NO record's `state_text` contains `"unsolvable"` or `"deadlock"` substrings
- For every `next_state`, verify `next_state.assignment` differs from `state.assignment` by exactly one key (the placed cell)
- For every action: `action_struct.value == state.state_struct.next_n` (the value is forced by the env)

---

## 8. File Layout

```
data/
  hidato5x4/
    train_balanced.jsonl              # 1500 records
    val_natural_calibration.jsonl     # 500 records
    test_natural_policy.jsonl         # 500 records
    metadata.json
    sanity_check_report.txt
    _phase0_prompt_decision.json

scripts/
  generate_save_data.py               # SHARED
  validate_dataset.py                 # SHARED
  hidato5x4_solver.py                 # TBW
  hidato5x4_env.py                    # TBW
  progress_hidato5x4.py               # TBW
  policy_sampler.py                   # SHARED
  save_schema.py                      # SHARED
```

`metadata.json`: `env: "hidato5x4"`, `policy_model: "outputs/sft_hidato_no_leak/final"`, `policy_checkpoint: "sft_hidato_no_leak_final"`, with a note: `"policy_is_sft_only: true, rl_training_stopped_at_step_46_per_user_directive_2026-05-03"`.

---

## 9. Pydantic Schema

Schema is shared (`save_sibling_set_v1.2`). Hidato's `state.state_struct` is `{rows, cols, assignment, next_n, anchor_cell}` — different shape than Sudoku/Pentomino. Either:
- (A) Use `state_struct: dict[str, Any]` in the parent schema (least typing safety, simplest)
- (B) Define `HidatoStateStruct(BaseModel)` and switch via `env` field (more validation)

Recommend (A) for the toy run — Pydantic still validates the env+state_text_version pair; the inner shape is checked downstream by `validate_dataset.py` per-env.

---

## 10. Test Run Protocol (do this first)

Before running 1500-set generation, do a 30-set smoke test:

```bash
python scripts/generate_save_data.py \
    --env hidato \
    --role train_balanced \
    --n_target 30 \
    --output data/hidato5x4/smoke_test.jsonl \
    --policy_model outputs/sft_hidato_no_leak/final \
    --seed 42
python scripts/validate_dataset.py data/hidato5x4/smoke_test.jsonl
```

Inspect manually per Sudoku §10. Hidato-specific checks:
- Records span all 8 puzzle IDs (or subset if some don't make boundary states)
- For each record, `state.state_text` contains the auxiliary info lines (`Already placed:`, `Next number to place:`, `Must be adjacent to:`)
- `action_struct.value` matches `state.state_struct.next_n` for all candidates
- `prt`-source candidates exist for at least some boundary states (else the deceptive benchmark starves)

---

## 11. Common Failures and Recovery

Same as Sudoku §11, plus Hidato-specific:

| Symptom | Likely cause | Fix |
|---|---|---|
| `K_prt=0` realized in many records | Forced-move states (only 1 legal placement) dominate | Drop `K_prt` to 0 at forced states; over-recruit `K_lt`+`K_ht` from those states; or skip forced states entirely |
| `set_stats.mixed=false` for most records | At forced moves, only 1 candidate is legal — can't be mixed | Skip forced-move states from sibling-set anchors |
| State-level leakage between train/val/test | Two different (puzzle, traj_seed) pairs produce same intermediate state | Documented limitation. Quantify in sanity check; if >5% leakage, expand the puzzle bank (NEAR-6) |
| `lt` candidates almost all doomed | π_θ is SFT-only, ~16.7% Pass@1 — its argmax often picks bad cells | Expected; rely on `K_sol=3` for viable candidates. If sibling sets dominated by sol+prt, the "natural occupancy" claim weakens — but for `train_balanced`, this is acceptable |
| `solver.solution_path` = None on a viable state | Solver early-terminates without storing path | Bug in solver; fix to record first complete solution |

---

## 12. Hand-off Checklist

Same as Sudoku §12, with Hidato-specific additions:

- [ ] All three role files exist at target sizes
- [ ] `validate_dataset.py` passes for all three files
- [ ] No leakage between any two roles (acknowledged state-level overlap quantified ≤5%)
- [ ] `train_balanced` has ≥50% mixed sibling sets (lower target than Sudoku's 60% due to forced-move structure)
- [ ] `train_balanced` has ≥100 deceptive pairs total
- [ ] No record's `state_text` contains `"unsolvable"` or `"deadlock"` substrings
- [ ] `val_natural_calibration` and `test_natural_policy` contain zero sol/prt candidates
- [ ] `metadata.json` records `policy_checkpoint = "sft_hidato_no_leak_final"` AND notes the SFT-only origin
- [ ] First 5 records of each file have been visually inspected
- [ ] A backup of all generated data exists

---

## End of spec
