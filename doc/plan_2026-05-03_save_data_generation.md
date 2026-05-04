# SAVE Data Generation — Plan & Decisions Log (2026-05-03)

**Status:** planned, awaiting Phase 0 kickoff approval.
**Spec:** [data_generation_sudoku.md](data_generation_sudoku.md)
**Paper:** [paper_SAVE.pdf](paper_SAVE.pdf)
**CLAUDE.md anchor:** "## SAVE Data Pipeline (parallel project)" subsection.

---

## Scope

Generate sibling-set training/eval data for SAVE across **all three paper environments**: Sudoku 4×4, Pentomino 5×4 LPWY, and Hidato 5×4. Each env produces three role files at 1500 / 500 / 500 (JSONL, `save_sibling_set_v1.2` schema). Sudoku is built first to validate the pipeline; Pentomino and Hidato follow with thin per-env adapters once Phase 1 is verified.

**Per-env specs** (each derived from spec §1-§12 of the Sudoku doc):
- Sudoku: [data_generation_sudoku.md](data_generation_sudoku.md)
- Pentomino: [data_generation_pentomino.md](data_generation_pentomino.md)
- Hidato: [data_generation_hidato.md](data_generation_hidato.md)

**Hard constraint (additivity):** All SAVE code lives under `scripts/sudoku4_*`, `scripts/pentomino5x4_*`, `scripts/hidato5x4_*`, `scripts/save_*`; outputs under `data/sudoku4/`, `data/pentomino5x4/`, `data/hidato5x4/`. **Do not edit existing `src/environments/sudoku*.py`, `src/environments/polyomino*.py`, `src/environments/hidato*.py`, or `src/data/sft_formatter.py`.** Existing SFT/RL pipelines must remain untouched and operational.

---

## Locked-in decisions

| # | Decision | Value | Rationale |
|---|---|---|---|
| 1 | Sudoku `π_θ` | `outputs/rl_b5_phase3_v8_anchor/final` (autodl2) | 50% Pass@1 — non-collapsed, has real preferences for CVCP tie-break to leverage |
| 1b | Pentomino `π_θ` | `outputs/rl_pentomino_5x4_no_leak_v8_aq/final` (autodl3) | 100% greedy Pass@1 (verified). No-leak training. |
| 1c | Hidato `π_θ` | `outputs/sft_hidato_no_leak/final` (archived to `/Volumes/yy_drive/SPA_termination/autodl1_outputs/sft_hidato_no_leak_final/`) | SFT-only no-leak (RL training stopped at step 46/200). Weaker than rl'd would be (likely ~16% greedy Pass@1) but 100% leak-free. |
| 2 | Puzzle complexity | Sudoku **6 empty cells** (revised from 10; see history below); Pentomino 5×4 LPWY (4 pieces); Hidato 5×4 (existing puzzle bank, 8 puzzles — see NEAR-6 in future_steps for expansion) | n_empty=6 matches rl_b5's actual eval distribution (SudokuEnv default difficulty="easy"). Diagnostic 2026-05-03: at n_empty=10 rl_b5 gets 0/5 Pass@1; at n_empty=6 it gets 4/5. Earlier "10 to match B-5" was a misread of the B-5 evaluation setup. |
| 3 | Sampling library | HuggingFace `model.generate(num_return_sequences=K, output_scores=True)` (one batched call per state) | vLLM speedup not worth setup time on toy-run scale |
| 4 | Runtime machines | Sudoku → autodl2; Pentomino → autodl3; **Hidato → autodl1** (revised — `sft_hidato_no_leak/final` is in place there; archive done locally but not yet deleted from cloud) | All three games run in parallel on different machines. autodl1 cleanup deferred until after Hidato data-gen completes. |
| 5 | Provenance `env_version` | `sudoku4_env_v1_6empty` (revised), `pentomino5x4_env_v1_LPWY`, `hidato5x4_env_v1_8puzzles` | Distinguish per-env configs |
| 6 | Schema version | `save_sibling_set_v1.2` (verbatim from spec) | No deviations |
| 7 | Sudoku prompt for `π_θ` | **`sudoku_minimal_4x4_corrected_v1`** (corrected 4×4 prompt; user override 2026-05-03) | Phase 0 sanity check at n_empty=6: buggy 40% Pass@1 vs fixed 33.3% (gap 6.7pp). The verdict logic said SHIP_BUGGY but the user directed: any prompt difference due to existing model training prior should be ignored. We use the corrected prompt for SAVE end-to-end (lt/ht sampling, policy_eval_logprob, downstream f_φ training/inference). The 6.7pp degradation is accepted as the cost of a clean paper story; no prompt-wart footnote needed. |
| 7b | Pentomino prompt for `π_θ` | `polyomino_minimal` from `SFTFormatter.SYSTEM_PROMPTS` verbatim | Already correct (no 1-9/3x3-style wart); confirmed via training data inspection |
| 7c | Hidato prompt for `π_θ` | `hidato_minimal` from `SFTFormatter.SYSTEM_PROMPTS` verbatim | Already correct |
| 8 | State rendering format per env | Sudoku: B-5 `|`/`---` format (`sudoku_text_b5_compat_v1`). Pentomino: existing B-8 board rendering. Hidato: existing grid rendering with `..` for empty. | Each `π_θ` is in-distribution at its own state format |
| 9 | SAVE model `f_φ` prompt | Free choice (independent of `π_θ`); decide during Phase 1 | `f_φ` doesn't exist yet — we train it with whatever input format we choose |
| 10 | Leak avoidance | All three games use no-leak versions exclusively. The leaked Hidato (`rl_b_h1_v8_anchor`, doom-suffix in next_state) is archived but not used as `π_θ`. | User directive 2026-05-03; preserves paper's "clean experiment" framing |

### The B-5 prompt wart (resolved 2026-05-03 → SHIP_FIXED, user override)

The system prompt rl_b5 was trained with — `sudoku_minimal` in [src/data/sft_formatter.py:144-153](../src/data/sft_formatter.py#L144-L153) — says "numbers 1-9, 3×3 box" but rl_b5 was trained on 4×4 data with values 1-4 and 2×2 boxes.

**Phase 0 sanity-check measurement** (script: `scripts/sanity_check_rl_b5_under_corrected_prompt.py`, n_puzzles=30, n_empty=6):
- Buggy prompt: **40.0% Pass@1** (12/30)
- Fixed prompt: **33.3% Pass@1** (10/30)
- Gap = 6.7pp

**User directive 2026-05-03**: any prompt difference due to existing model training prior should be ignored. Override the threshold logic and ship the **corrected 4×4 prompt** end-to-end. The 6.7pp performance hit is accepted as the cost of a cleaner paper story.

**Implication:** the SAVE pipeline uses `sudoku_minimal_4x4_corrected_v1` everywhere — `lt`/`ht` sampling, `policy_eval_logprob` computation, and downstream `f_φ` training/inference. rl_b5 is queried slightly OOD at the system-prompt level, but its actions (which respond to the user-message grid) remain valid. No prompt-wart footnote needed in paper.

Raw verdict + both Pass@1 numbers stored in `data/sudoku4/_phase0_prompt_decision.json` for provenance.

---

## Open questions still to resolve

- **Q1 — `f_φ` prompt design:** decide during Phase 1, when we know `π_θ`'s prompt is settled. Likely: same B-5 format for state, but with a SAVE-specific system prompt explaining the (s, a) → ⟨next_state, viability, state_viable⟩ task. Independent of Q7 above.
- **Q2 — Multiple solutions for puzzles:** existing `generate_sudoku_puzzle()` doesn't enforce solution uniqueness. SAVE's `state_solver.num_solutions` should reflect this honestly; downstream metrics may need to handle `num_solutions > 1` cases (e.g., a state may be "trivially viable" via multiple paths).

---

## Phased work plan

### Phase 0 — Skeleton + checkpoint sanity (1 day)

**Local-only work (no GPU, no checkpoint):**

- [ ] `scripts/save_schema.py` — Pydantic models from spec §9 (paste, runs on import-test).
- [ ] `scripts/sudoku4_solver.py` — fresh instrumented backtracker. Returns `SolverResult{solvable, num_solutions, nodes, backtracks, solution_depth, solve_time_ms, solution_path}`. Runs exhaustively even on unsolvable states. Does NOT touch `src/environments/sudoku*.py`.
- [ ] `scripts/sudoku4_env.py` — wraps existing `SudokuEnv` for state struct + B-5 format renderer (no separators? with separators? — match B-5's actual training data format byte-for-byte). Action struct + transition wrapper.
- [ ] `scripts/progress_sudoku4.py` — formula from spec §5.2.

**Cloud-side (gated on rl_b5 checkpoint arriving on autodl3):**

- [ ] `scripts/sanity_check_rl_b5_under_corrected_prompt.py` — **gates Q7.** Loads rl_b5, runs Pass@1 on 30 fresh easy 4×4 puzzles under each prompt:
  - `prompt_buggy`: verbatim from `SFTFormatter.SYSTEM_PROMPTS["sudoku_minimal"]`
  - `prompt_fixed`: corrected text ("numbers 1-4, 2×2 box, etc.")
  
  Outputs: one-line decision string (`SHIP_FIXED` if `|Pass@1_fixed - Pass@1_buggy| ≤ 0.05`, else `SHIP_BUGGY`). Records both Pass@1 numbers in `data/sudoku4/_phase0_prompt_decision.json` for provenance. ~10 min on H800.

**Phase 0 exit criterion:** schema validates on a hand-built sample record; solver returns expected structure on a known state; prompt-decision file exists with a clear `SHIP_*` verdict.

### Phase 1 — Single-state sibling-set generator (1–2 days)

- [ ] `scripts/policy_sampler.py` — HF `model.generate(num_return_sequences=K, output_scores=True)` for `lt`/`ht`; logprob extraction. `policy_eval_logprob` computation is a separate forward pass (reusable from `compute_response_logprobs` in `rl_trainer_v6.py` — import, don't copy).
- [ ] In `scripts/generate_sudoku4_data.py`:
  - 5 candidate sources (`lt`, `ht`, `rand` — off, `sol`, `prt`)
  - dedup by canonical action hash
  - shuffle, validate, transition, solver-label
  - `policy_eval_logprob` per candidate
  - `set_stats`, `deceptive_pairs`, `selection_criteria`, `provenance`
  - Pydantic-validate before emit
- [ ] Smoke test: emit 30 sibling sets to `data/sudoku4/smoke_30.jsonl`, manually inspect per spec §10 checklist.

**Phase 1 exit criterion:** spec §10 manual-inspection items 1-7 pass.

### Phase 2 — Trajectory + boundary pipeline (0.5 day)

- [ ] 500 root puzzles, hashed split 70/15/15 by `state_hash` of initial grid.
- [ ] Solver-derived viable solution path per train-split puzzle.
- [ ] Boundary-state detection (spec §4.3) at intermediate states; oversample at rate 0.5.
- [ ] Loop to target sizes (1500 / 500 / 500). Val/test get the no-oracle protocol (no `sol` or `prt`).
- [ ] Records per spec.

**Phase 2 exit criterion:** all three role files reach target size; sample records pass validate-on-emit.

### Phase 3 — Sanity + metadata (0.5 day)

- [ ] `scripts/validate_dataset.py` — schema, field-consistency, role-specific, leakage, distributional checks per spec §7.
- [ ] `data/sudoku4/metadata.json` per spec §8.
- [ ] `data/sudoku4/sanity_check_report.txt` from `validate_dataset.py` stdout.
- [ ] Hand-off checklist (spec §12) all green.

**Phase 3 exit criterion:** all 9 hand-off-checklist items checked.

**Sudoku estimate: ~3–4 days** of focused work.

### Phase 4 — Multi-env extension (Pentomino + Hidato, ~2–3 days after Sudoku Phase 1 verifies)

The env-agnostic core (`scripts/save_schema.py`, `scripts/generate_save_data.py`, `scripts/validate_dataset.py`) is shared. Per-env adapters per game:

| Module | Sudoku (built) | Pentomino (TBW) | Hidato (TBW) | What it wraps |
|---|---|---|---|---|
| `*_solver.py` | `sudoku4_solver.py` ✅ | `pentomino5x4_solver.py` | `hidato5x4_solver.py` | Fresh adapter per env, instrumented (nodes/backtracks/solution_path); wraps existing checker (`PolyominoSolvabilityChecker`, `is_solvable`) without modification |
| `*_env.py` | `sudoku4_env.py` ✅ | `pentomino5x4_env.py` | `hidato5x4_env.py` | State render in env-native format, action parse, legality, transition, state_hash |
| `progress_*.py` | `progress_sudoku4.py` ✅ | `progress_pentomino5x4.py` | `progress_hidato5x4.py` | Handcrafted formula. **Pentomino:** `filled_normalized − 0.1 × n_holes` (counts isolated empty regions of size 1–4 that no remaining piece can fit). **Hidato:** `path_length / N − 0.1 × n_unreachable_givens` (counts pre-filled cells now adjacency-isolated from the live path) |

**Sequencing:**

```
                                                  Pentomino (autodl3): adapters + Phase 0 + full gen
Phase 0 (Sudoku)  ─→  Phase 1 (Sudoku)  ─┬─→  ───────────────────────────────
                      generator design   │    
                      verified working   │
                                         └─→  Hidato (autodl3 after Pentomino): adapters + Phase 0 + full gen
                                              ───────────────────────────────
Phase 2 (Sudoku, full gen, autodl2)
```

**Per-env Phase 0 sanity check:** parameterize `sanity_check_rl_b5_under_corrected_prompt.py` to take `--env` and the env's prompt-version pair. For Pentomino + Hidato, expected verdict is `SHIP_FIXED` (their existing prompts are env-correct already), but we run the check anyway as cheap insurance.

**Hidato note:** since `π_θ` is SFT-only (no RL), `lt`/`ht` sampling will produce more invalid/doomed candidates than for the other two games. Mitigation: the spec's `K_lt`+`K_ht`+`K_sol`+`K_prt` mixture already has `K_sol=3` from oracle, which guarantees at least 3 viable candidates per train_balanced sibling set. Boundary oversampling threshold may need tuning down to 0.2.

**Multi-env total: ~5–6 days** of focused work across all three games.

---

## Risks (ranked by likelihood × impact)

1. **`prt` source starves on some states** (spec §11). Many states won't have a locally-legal-but-doomed alternative at the perturbation cell. Mitigation: try multiple cells along the solution path, not just `sol[0]`'s. Records actual skip rate. Worst case: deceptive-pair count in `train_balanced` falls below 100 and the §3.4 deceptive benchmark is data-starved — fix by raising boundary oversample rate and rerunning.
2. **Boundary-state hit rate falls below 60%** in `train_balanced`. Mitigation: lower `boundary_threshold` from 0.3 to 0.2 per spec §11.
3. **`rl_b5` behavior drifts under corrected prompt.** Caught by Phase 0 sanity check; falls back to buggy prompt with footnote.
4. **Solver `nodes` counter unreliable.** Mitigation: write the solver clean from scratch with tracking built in (Phase 0). 4×4 is small enough that exhaustive search is cheap.
5. **JSONL records get large** (~70KB × 1500 ≈ 100MB train file). Acceptable; gzip if needed.

---

## Ownership / hand-off

This plan + its decisions log is the durable artifact. Future sessions:

1. Read this doc first, then [data_generation_sudoku.md](data_generation_sudoku.md), then [paper_SAVE.pdf](paper_SAVE.pdf).
2. Check `data/sudoku4/_phase0_prompt_decision.json` for the prompt verdict.
3. Continue from the highest-numbered unchecked item above.

---
