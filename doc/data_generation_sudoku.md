# SAVE Data Generation Spec — Sudoku 4×4

**Schema version**: save_sibling_set_v1.2
**Target**: Generate sibling-set training/eval data for the SAVE paper, Sudoku 4×4 environment.
**Last updated**: 2026-05-04

## Three-stage scaling roadmap

| Stage | train / val / test (per env) | Purpose | Promotion gate |
|---|---|---|---|
| **Toy** | 1500 / 500 / 500 | Pipeline validation: schema, solver interface, generator runs end-to-end, sanity checks pass | SFT loss decreasing + sanity checks all pass |
| **Pilot** | 3000 / 1000 / 1000 | Paper trend validation: capacity dissociation + deceptive benchmark show real signal (not necessarily polished numbers) | Key trends emerge: viability AUC differentiates by model size + SAVE ≥ score baseline on deceptive subset |
| **Paper-final** | 8000 / 1500 / 1500 | Stable numbers for submission; reviewer-acceptable variance | Paper claims hold; numbers stable across reseeded runs |

Each stage gates the next. **Don't scale up if the current stage's gate fails — fix method or code first.**

For Sudoku 4×4 specifically, all three stages use the same `--n-target` flag mechanism on `scripts/generate_save_data.py`; only the integer changes. No piece-set or board-size axis to vary, so scaling is straightforward (unlike Pentomino).

### State coverage and scaling — Sudoku has no bank

Sudoku does **not** use a fixed puzzle bank. `generate_root_puzzle(seed=puzzle_seed, n_empty=6)` procedurally constructs a fresh 4×4 puzzle for each `puzzle_seed`. The space of (filled solution × 6-empty pattern) is large enough (~10⁶ combinations) that collisions across seeds are negligible.

Empirical (toy stage, 2026-05-04 generation):
- Toy 2500 records → **2498 unique states** (validated)
- 0 schema violations
- Records-per-state ratio ~1.0 — every record is a near-unique anchor

Predicted at higher scales:
- Pilot 5000 records → ~5000 unique states (no scaling concern)
- Paper-final 11000 records → ~11000 unique states (no scaling concern)

**No bank expansion is needed at any stage.** Train/val/test split is by trajectory `seed` range (70/15/15) and there is no shared-puzzle leakage because each seed gets its own procedurally generated puzzle.

This contrasts with Hidato (bounded 5×4 puzzle bank, requires algorithmic bank expansion for pilot+) and Pentomino (mathematically capped at 172 valid 6-piece subsets at 5×6, requires per-trajectory expansion to scale records).

---

## 1. Goal & Context

This document specifies the data generation pipeline for the SAVE (Sibling-Action Viability Estimation) project. SAVE trains an LLM agent to predict, per candidate action, whether the resulting successor state remains recoverable (`viability`). Training requires a structured dataset where each sample is a *sibling set*: one state s_t together with K candidate actions, each labeled by the environment's deterministic transition and an oracle solver.

The generated data feeds:
- SFT training of the SAVE viability head and auxiliary state-conditioned head
- A learned progress-score baseline (same architecture, different supervision target)
- Threshold calibration for the inference-time CVCP operator
- Evaluation tables in the paper (capacity dissociation, viability quality, deceptive benchmark, online rollout, mid-rollout truncation, ablations)

Generate data into THREE independent dataset roles (see §3); reusing one role for another purpose violates paper-level guarantees.

### Paper-relevant constraints (do not violate)

1. Calibration set must NOT contain solver-injected candidates. The threshold tau in the paper's Proposition 4.1 is calibrated under the base-policy occupancy. Mixing in solver-derived candidates breaks the assumption.
2. Deceptive subset must contain only `valid_doomed` candidates as the doomed action. Otherwise the paper's "surface-valid but unrecoverable" claim collapses to "we filter illegal actions."
3. Score labels must NOT be a relabeling of viability. The progress score serves a baseline that competes with viability; if the score's information content equals viability, the baseline is fake.

---

## 2. Output Schema (`save_sibling_set_v1.2`)

Each generated record is one JSON object representing a single sibling set. Output as JSONL (one record per line).

### 2.1 Top-level structure

A record looks like this (all fields detailed in §2.2):

    {
      "schema": "save_sibling_set_v1.2",
      "env": "sudoku4",
      "dataset_role": "train_balanced",
      "split": "train",
      "root_id": "sudoku4_train_000001",
      "trajectory_id": "sudoku4_train_000001_solver_00",
      "sibling_set_id": "sudoku4_train_000001_solver_00_t003_set00",
      "t": 3,
      "state": { ... },
      "sampling_protocol": { ... },
      "candidates": [ ... ],
      "set_stats": { ... },
      "deceptive_pairs": [ ... ],
      "selection_criteria": { ... },
      "provenance": { ... }
    }

### 2.2 Field reference

#### Identity fields

| Field            | Type   | Description                                                  |
| ---------------- | ------ | ------------------------------------------------------------ |
| `schema`         | string | Always `save_sibling_set_v1.2`.                              |
| `env`            | string | `sudoku4` for this run.                                      |
| `dataset_role`   | string | One of: `train_balanced`, `val_balanced_diagnostic`, `test_balanced_diagnostic`, `val_natural_calibration`, `test_natural_policy`, `test_deceptive_curated`. See §3. |
| `split`          | string | `train`, `val`, or `test`. Coarse-grained split; `dataset_role` is the authoritative discriminator. |
| `root_id`        | string | Format: `{env}_{split}_{root_idx:06d}`. Identifies the initial puzzle. |
| `trajectory_id`  | string | Format: `{root_id}_solver_{solver_idx:02d}`. A root may have multiple solver paths. |
| `sibling_set_id` | string | Format: `{trajectory_id}_t{t:03d}_set{set_idx:02d}`. Globally unique. |
| `t`              | int    | Trajectory step (0-indexed).                                 |

#### `state` block

Example:

    "state": {
      "state_hash": "sha1:7a3f...",
      "state_struct": {"grid": [[1,2,3,4],[3,4,0,0],[0,0,4,3],[4,3,0,0]]},
      "state_text": "1 2 | 3 4\n3 4 | . .\n---------\n. . | 4 3\n4 3 | . .",
      "state_text_version": "sudoku_text_b5_compat_v1",
      "state_viable": true,
      "state_is_goal": false,
      "state_solver": {
        "num_solutions": 1,
        "nodes": 13,
        "backtracks": 2,
        "solve_time_ms": 0.7
      },
      "action_space_stats": {
        "num_legal_actions": 7,
        "num_legal_viable_actions": 3,
        "num_legal_doomed_actions": 4
      }
    }

| Field                  | Type            | Description                                                  |
| ---------------------- | --------------- | ------------------------------------------------------------ |
| `state_hash`           | string          | SHA-1 hash of `json.dumps(state_struct.grid)`, prefixed `sha1:`. Used for dedup and leakage check. |
| `state_struct.grid`    | list[list[int]] | 4x4 matrix; integers 1-4 for filled, 0 for empty.            |
| `state_text`           | string          | LLM-facing rendering. Format: B-5-compat — cells space-separated, ` | ` between left/right 2×2 boxes per row, line of 9 dashes between row 2 and row 3. Empty cells shown as `.`. Matches the format `rl_b5_phase3_v8_anchor` saw at training time. |
| `state_text_version`   | string          | `sudoku_text_b5_compat_v1` (locked 2026-05-03 per plan_2026-05-03 decision #8). Was `sudoku_text_v1` in earlier draft; changed to match B-5 training distribution. |
| `state_viable`         | bool            | v(s_t) ∈ {0, 1} from the oracle solver.                      |
| `state_is_goal`        | bool            | True iff grid is fully filled and satisfies all constraints. |
| `state_solver.*`       | object          | Solver statistics on s_t. Used for score label and diagnostics. |
| `action_space_stats.*` | object          | Counts over the LEGAL ACTION SPACE at s_t (NOT over the sampled candidates). Used for paper §3.6 candidate-budget vs state-level distinction. |

`action_space_stats` is computed by enumerating all legal actions A_legal(s_t) (here, all (row, col, value) triples that don't immediately violate Sudoku constraints), running solver on T(s_t, a) to get viability, and summing.

#### `sampling_protocol` block

Example:

    "sampling_protocol": {
      "K_total": 12,
      "K_lt": 3,
      "K_ht": 3,
      "K_rand": 0,
      "K_sol": 3,
      "K_prt": 3,
      "lt_temperature": 0.3,
      "ht_temperature": 1.0,
      "top_p": 0.95,
      "policy_model": "{policy_model_path}",
      "policy_checkpoint": "{checkpoint_id}",
      "training_phase": "sft",
      "candidate_order": "random_shuffled",
      "shuffle_seed": 12345,
      "set_seed": 94231,
      "dedup_policy": "canonical_action_hash",
      "candidate_pool_size_before_dedup": 14,
      "candidate_pool_size_after_dedup": 12
    }

| Field                                      | Type   | Description                                                  |
| ------------------------------------------ | ------ | ------------------------------------------------------------ |
| `K_total`                                  | int    | 12 by default for `train_balanced`. See §4 for per-role config. |
| `K_lt`, `K_ht`, `K_rand`, `K_sol`, `K_prt` | int    | Source breakdown. Sum equals `K_total`.                      |
| `lt_temperature`, `ht_temperature`         | float  | Sampling temperatures for low- and high-temp policy candidates. |
| `top_p`                                    | float  | Nucleus sampling parameter (applies to lt and ht).           |
| `policy_model`                             | string | Placeholder; replaced at runtime by your model path.         |
| `policy_checkpoint`                        | string | E.g. `sft_final`, `step_200`.                                |
| `training_phase`                           | string | `sft`, `rl_phase_1`, etc.                                    |
| `candidate_order`                          | string | Always `random_shuffled`. After grouping candidates by source, shuffle the final order. |
| `shuffle_seed`                             | int    | Seed for the shuffle (per sibling set).                      |
| `set_seed`                                 | int    | Master seed for this sibling set's stochastic ops.           |
| `dedup_policy`                             | string | `canonical_action_hash`: dedup candidates by their canonical action representation (for Sudoku, the (row, col, value) triple). |
| `candidate_pool_size_before_dedup`         | int    | How many raw samples were drawn before dedup (for diagnostics). |
| `candidate_pool_size_after_dedup`          | int    | Equals `K_total`.                                            |

#### `candidates` (list, length = K_total)

Each entry has the following structure:

    {
      "candidate_id": "c000",
      "action_hash": "sha1:...",
      "display_rank": 0,
      "source": "lt",
      "source_meta": {
        "temperature": 0.3,
        "top_p": 0.95,
        "is_oracle_injected": false
      },
      "action_text": "place 1 at row 2 column 3",
      "action_text_canonical": "R2C3=1",
      "action_struct": {"row": 2, "col": 3, "value": 1},
      "logprobs": {
        "generation_logprob": -1.34,
        "policy_eval_logprob": -1.42,
        "policy_eval_model": "{policy_model_path}",
        "policy_eval_prompt_version": "policy_prompt_v3"
      },
      "parse_valid": true,
      "local_valid": true,
      "transition_valid": true,
      "candidate_class": "valid_viable",
      "eligible_for_viability_eval": true,
      "next_state": {
        "next_state_hash": "sha1:...",
        "next_state_struct": {"grid": [[1,2,3,4],[3,4,1,0],[0,0,4,3],[4,3,0,0]]},
        "next_state_text": "row1: 1 2 3 4\nrow2: 3 4 1 .\nrow3: . . 4 3\nrow4: 4 3 . .",
        "next_state_text_version": "sudoku_text_v1",
        "next_is_goal": false,
        "next_viable": true
      },
      "solver": {
        "solvable": true,
        "num_solutions": 1,
        "nodes": 5,
        "backtracks": 0,
        "solution_depth": 5,
        "solve_time_ms": 0.41,
        "solver_version": "sudoku_solver_v1"
      },
      "progress": {
        "formula_id": "sudoku_local_progress_v1",
        "formula_spec": "filled_normalized - 0.1 * (rows_violated + cols_violated + boxes_violated)",
        "local_progress_score": 0.5625,
        "features": {
          "filled_cells": 9,
          "filled_normalized": 0.5625,
          "constraint_violations": 0,
          "rows_violated": 0,
          "cols_violated": 0,
          "boxes_violated": 0
        }
      },
      "score_labels": {
        "local_progress": {
          "value": 0.5625,
          "use_for_main_score_baseline": true
        },
        "solver_residual_difficulty": {
          "value": -1.791759,
          "formula": "-log(1 + solver_nodes)",
          "use_for_main_score_baseline": false
        }
      }
    }

| Field                                           | Type          | Description                                                  |
| ----------------------------------------------- | ------------- | ------------------------------------------------------------ |
| `candidate_id`                                  | string        | `c000` to `c{K_total-1:03d}`. Order is post-shuffle.         |
| `action_hash`                                   | string        | SHA-1 hash of canonical action representation. Used for dedup. |
| `display_rank`                                  | int           | Same as `candidate_id` index. Reserved for future ordering.  |
| `source`                                        | string        | One of `lt`, `ht`, `rand`, `sol`, `prt`. CRITICAL field for downstream ablations. |
| `source_meta.temperature`                       | float or null | Per-candidate sampling temperature (matches lt/ht). For sol/prt: null. |
| `source_meta.is_oracle_injected`                | bool          | True for `sol` and `prt`. False for `lt`, `ht`, `rand`.      |
| `action_text`                                   | string        | LLM-style natural language form.                             |
| `action_text_canonical`                         | string        | Compact canonical form for dedup (e.g., `R2C3=1`).           |
| `action_struct.{row, col, value}`               | int           | Structured action; 1-indexed for display. value ∈ {1,2,3,4}. |
| `logprobs.generation_logprob`                   | float or null | Log P(action \| state) under the policy at generation time. NULL for `sol`, `prt`, `rand`. |
| `logprobs.policy_eval_logprob`                  | float         | ALWAYS POPULATED. Score every candidate (regardless of source) under one canonical policy + prompt. Used by CVCP and best-of-K baselines. |
| `logprobs.policy_eval_model`                    | string        | Model used for `policy_eval_logprob`. Same across all candidates in a sibling set. |
| `logprobs.policy_eval_prompt_version`           | string        | E.g. `policy_prompt_v3`. Documents the prompt template used. |
| `parse_valid`                                   | bool          | Did `action_text` parse to `action_struct` successfully?     |
| `local_valid`                                   | bool          | Is `action_struct` a legal action at s_t?                    |
| `transition_valid`                              | bool          | Did T(s_t, a) execute without error?                         |
| `candidate_class`                               | string        | One of: `parse_invalid`, `local_invalid`, `valid_doomed`, `valid_viable`, `goal_reaching`. |
| `eligible_for_viability_eval`                   | bool          | True iff `local_valid=true`. Filter for downstream eval tasks. |
| `next_state.*`                                  | object        | Same structure as top-level `state`, but for s_{t+1}. NULL if `transition_valid=false`. |
| `next_viable`                                   | bool          | v(s_{t+1}) from solver. Equivalent to `solver.solvable`.     |
| `solver.nodes`                                  | int           | Search tree nodes expanded by the solver on s_{t+1}. Used for `solver_residual_difficulty` score label. |
| `solver.solver_version`                         | string        | E.g. `sudoku_solver_v1`.                                     |
| `progress.formula_id`                           | string        | E.g. `sudoku_local_progress_v1`.                             |
| `progress.formula_spec`                         | string        | Human-readable formula.                                      |
| `progress.local_progress_score`                 | float         | The handcrafted progress score. MUST NOT depend on solver. See §5.2 for Sudoku formula. |
| `progress.features.*`                           | object        | Decomposition for transparency / debug.                      |
| `score_labels.local_progress.value`             | float         | Equals `progress.local_progress_score`. Duplicated here for ease of training-time access. |
| `score_labels.solver_residual_difficulty.value` | float         | -log(1 + solver.nodes).                                      |
| `score_labels.*.use_for_main_score_baseline`    | bool          | Mark exactly ONE entry as `true`. For Sudoku, default to `local_progress`. |

`candidate_class` decision tree:

    if not parse_valid:    candidate_class = "parse_invalid"
    elif not local_valid:  candidate_class = "local_invalid"
    elif next_is_goal:     candidate_class = "goal_reaching"
    elif next_viable:      candidate_class = "valid_viable"
    else:                  candidate_class = "valid_doomed"

#### `set_stats` block

    "set_stats": {
      "num_candidates": 12,
      "num_parse_invalid": 0,
      "num_local_invalid": 0,
      "num_valid_viable": 5,
      "num_valid_doomed": 7,
      "num_goal_reaching": 0,
      "mixed": true,
      "all_viable": false,
      "all_doomed": false,
      "source_breakdown": {
        "lt": {"selected": 3, "valid_viable": 1, "valid_doomed": 2, "invalid": 0, "goal_reaching": 0},
        "ht": {"selected": 3, "valid_viable": 1, "valid_doomed": 2, "invalid": 0, "goal_reaching": 0},
        "sol": {"selected": 3, "valid_viable": 3, "valid_doomed": 0, "invalid": 0, "goal_reaching": 0},
        "prt": {"selected": 3, "valid_viable": 0, "valid_doomed": 3, "invalid": 0, "goal_reaching": 0}
      }
    }

`mixed = (num_valid_viable > 0 AND num_valid_doomed > 0)`. THIS FLAG CONTROLS WHETHER RANKING LOSS L_rank ACTIVATES DURING TRAINING.

#### `deceptive_pairs` (list, may be empty)

    "deceptive_pairs": [
      {
        "pair_id": "pair_000",
        "a_plus_candidate_id": "c002",
        "a_minus_candidate_id": "c007",
        "condition": {
          "plus_next_viable": true,
          "minus_next_viable": false,
          "both_local_valid": true,
          "minus_progress_ge_plus": true,
          "progress_gap": 0.14
        }
      }
    ]

Mined post-hoc from this sibling set. For every (a+, a-) pair where:
- `a_plus.candidate_class == "valid_viable"`
- `a_minus.candidate_class == "valid_doomed"`
- `a_minus.progress.local_progress_score >= a_plus.progress.local_progress_score`

Emit one entry. `progress_gap = a_minus.progress - a_plus.progress`. List can be empty.

#### `selection_criteria` block

    "selection_criteria": {
      "mix_score": 0.42,
      "mix_score_formula": "min(num_valid_viable, num_valid_doomed) / num_candidates",
      "is_boundary": true,
      "boundary_threshold": 0.3,
      "selection_reason": ["mixed_sibling_set", "boundary_state"]
    }

#### `provenance` block

    "provenance": {
      "env_version": "sudoku4_env_v1",
      "solver_version": "sudoku_solver_v1",
      "generator_version": "save_data_gen_v1",
      "git_commit": "{git_commit_hash}",
      "created_at": "2026-05-03T00:00:00Z"
    }

`git_commit`: read at runtime via `git rev-parse HEAD`.

---

## 3. Three Dataset Roles

Generate three independent files. Each role differs in (a) which sources are allowed in `sampling_protocol`, (b) target size, (c) downstream usage.

### 3.1 train_balanced

- File: `data/sudoku4/train_balanced.jsonl`
- Size: 1500 sibling sets
- Sampling protocol: K_total=12, breakdown lt:ht:rand:sol:prt = 3:3:0:3:3
- Used for: SAVE SFT, learned progress-score baseline SFT, transition-only baseline SFT
- Coverage requirement: at least 60% of sibling sets must have `mixed=true` (else regenerate boundary states more aggressively; see §4.3)

### 3.2 val_natural_calibration

- File: `data/sudoku4/val_natural_calibration.jsonl`
- Size: 500 sibling sets
- Sampling protocol: K_total=8, breakdown lt:ht:rand:sol:prt = 4:4:0:0:0
- Used for: tau_keep and tau_fb threshold calibration (paper §3.4); reliability diagram; empirical epsilon measurement
- CRITICAL: must NOT contain `sol` or `prt` candidates. Rationale: paper Proposition 4.1's calibration assumption holds under base-policy occupancy; oracle-injected candidates violate this distribution.

### 3.3 test_natural_policy

- File: `data/sudoku4/test_natural_policy.jsonl`
- Size: 500 sibling sets
- Sampling protocol: same as `val_natural_calibration`
- Used for: Online rollout starting states (paper §4.5), termination evaluation (paper §4.6), held-out test of viability prediction quality (paper §4.3)
- CRITICAL: trajectories must not overlap with `train_balanced` or `val_natural_calibration` (see §6.4 leakage check).

### 3.4 (Optional, generate after the above three)

`val_balanced_diagnostic` and `test_balanced_diagnostic` use the same protocol as `train_balanced` but on held-out trajectories. Useful for diagnostic AUC/Brier/ECE that benefit from balanced class coverage. Skip for the toy run; add if reviewers request.

---

## 4. Sampling Protocol

### 4.1 Five candidate sources

For each sibling set, sample K_lt + K_ht + K_rand + K_sol + K_prt = K_total candidates.

#### `lt` — low-temperature policy

    prompt = build_prompt(state_text, env="sudoku4")
    outputs = vllm.generate(
        prompt=prompt,
        sampling_params=SamplingParams(
            temperature=0.3, top_p=0.95, n=K_lt,
            max_tokens=64, logprobs=1
        )
    )
    for output in outputs:
        candidates.append({
            "source": "lt",
            "action_text": output.text,
            "generation_logprob": sum(output.logprobs),
            ...
        })

#### `ht` — high-temperature policy

Identical to `lt` but with `temperature=1.0`.

#### `rand` — uniform random over legal action space

Skip for the default toy-run config (`K_rand=0`). Reserved for ablation.

#### `sol` — solver-derived viable continuation

    solution_path = solver.find_one_solution(state)
    if solution_path is None:
        raise ValueError("Cannot generate sol candidates from a dead-end state")
    sol_actions = solution_path[:K_sol]

If only one unique viable next action exists at s_t, request fewer `sol` candidates and adjust mixture (record actual counts in `sampling_protocol`).

#### `prt` — perturbed solver-path action

Generate one `sol` action, then locally edit to introduce constraint cascade:

    sol_action = solution_path[0]  # e.g. {"row": 2, "col": 3, "value": 1}
    # Find a different value v' such that:
    #   1. v' is locally legal at (row, col)
    #   2. v' is NOT viable from s_t (solver says doomed)
    candidates_perturbed = []
    for v_prime in legal_values_at(state, row, col):
        if v_prime == sol_action["value"]:
            continue
        candidate_state = apply_action(state, {"row": row, "col": col, "value": v_prime})
        if not solver.is_viable(candidate_state):
            candidates_perturbed.append({"row": row, "col": col, "value": v_prime})
            if len(candidates_perturbed) >= K_prt:
                break

If you cannot find K_prt valid perturbations, record actual count and proceed.

### 4.2 Per-candidate `policy_eval_logprob` computation

After all candidates are collected and validated, compute `policy_eval_logprob` for EVERY candidate (including sol/prt) using the same base policy + same prompt + the SAME RESPONSE FORMAT (just the bare `action_text`):

    for c in candidates:
        prompt = build_prompt(state_text, env="sudoku4",
                              prompt_version="sudoku_minimal_4x4_corrected_v1")
        # CRITICAL: response is just the bare action text, NOT the full LLM XML wrapper.
        # Using bare action_text uniformly across all sources keeps logprobs comparable
        # for CVCP's `arg max log π_θ(a | s)` tie-break at inference.
        response = c["action_text"]   # e.g., "place 4 at row 2 col 2"
        logprob = compute_logprob_under_policy(
            model=policy_model,
            prompt=prompt,
            response=response,
        )
        c["logprobs"]["policy_eval_logprob"] = logprob
        c["logprobs"]["policy_eval_model"] = policy_model_path
        c["logprobs"]["policy_eval_prompt_version"] = "sudoku_minimal_4x4_corrected_v1"

This is a SEPARATE forward pass from generation (necessary for sol/prt which were not sampled by the policy). **Do NOT use the LLM's full raw response (with `<think>` reasoning) for lt/ht** — that would make lt/ht logprobs incomparable to sol/prt logprobs. Bare `action_text` for everyone.

**Prompt template**: `sudoku_minimal_4x4_corrected_v1` (locked 2026-05-03 per plan_2026-05-03 decision #7). The corrected 4×4 prompt; supersedes the buggy `sudoku_minimal_b5_legacy` ("1-9 / 3×3" wart).

### 4.3 Boundary state preference

Many random Sudoku states yield all-viable or all-doomed sibling sets, providing no contrastive signal. To bias toward boundary states:

    def is_boundary_state(state, threshold=0.3):
        legal = enumerate_legal_actions(state)
        if len(legal) == 0:
            return False
        viable_count = sum(1 for a in legal if solver.is_viable(apply(state, a)))
        doomed_count = len(legal) - viable_count
        mix = min(viable_count, doomed_count) / len(legal)
        return mix >= threshold

When sampling root puzzles for `train_balanced`:

1. Generate a random Sudoku puzzle.
2. **Walk a STOCHASTIC trajectory under `π_θ`** (low-temperature sampling, T=0.3) instead of the deterministic solver path. For each step:
   - Sample one action from `π_θ(·|s_t)` at T=0.3
   - If parseable + locally-valid: apply it, advance to s_{t+1}
   - Else: terminate the walk (record it as ending early — that's fine)
3. For each s_t along the walk (skipping initial state and final two steps): check `is_boundary_state(s_t)`. If true, include this s_t as a sibling-set anchor.
4. Aim for at least 60% boundary states in `train_balanced`. For `val_natural_calibration` and `test_natural_policy`, do not bias—use natural distribution.

**Why stochastic walks instead of deterministic solver path** (locked 2026-05-04): the solver path always reaches the same canonical states from the same root, which limits state diversity. Stochastic walks under `π_θ` give the "natural occupancy" distribution that paper Proposition 4.1 calibrates against, and produce different trajectories per `traj_seed` — enabling state diversity even from the same root puzzle.

### 4.4 Trajectory-level split for leakage prevention

1. Generate N total Sudoku root puzzles (e.g., N=500 for the toy run)
2. Hash each by initial-state-grid SHA-1
3. Split root_indices into 70% train / 15% val / 15% test
4. Sibling sets in `train_balanced` come ONLY from train root puzzles
5. Sibling sets in `val_natural_calibration` come ONLY from val root puzzles
6. Sibling sets in `test_natural_policy` come ONLY from test root puzzles

This prevents the same root puzzle (and its derived states) from appearing across splits.

---

## 5. Sudoku 4×4 Environment Spec

### 5.1 State and action representation

State (`state_struct.grid`) is a 4x4 list of lists. Each cell ∈ {0, 1, 2, 3, 4}, where 0 means empty.

Constraints:
- Each row contains each of {1,2,3,4} exactly once when complete.
- Each column contains each of {1,2,3,4} exactly once.
- Each of the four 2x2 boxes contains each of {1,2,3,4} exactly once.

Box layout:

    Box 0: rows 0-1, cols 0-1
    Box 1: rows 0-1, cols 2-3
    Box 2: rows 2-3, cols 0-1
    Box 3: rows 2-3, cols 2-3

State rendering format (`state_text_version=sudoku_text_v1`):

    row1: 1 2 3 4
    row2: 3 4 . .
    row3: . . 4 3
    row4: 4 3 . .

1-indexed row labels. Cells space-separated. Empty cells shown as `.`.

Action structure (`action_struct`):

    {"row": 2, "col": 3, "value": 1}

Use 1-indexed for `row` and `col` to match `state_text` rendering. `value` ∈ {1, 2, 3, 4}. The agent places `value` at cell (row, col).

Action text default (`action_text`): `place 1 at row 2 column 3`. Match the prompt template in your existing SFT pipeline. If your training-time format differs, override this and bump `state_text_version`.

Action canonical form (`action_text_canonical`): `R2C3=1`. Used for dedup hash.

### 5.2 Local progress formula (`sudoku_local_progress_v1`)

    def local_progress(state):
        grid = state["grid"]
        filled_cells = sum(1 for row in grid for cell in row if cell != 0)
        filled_normalized = filled_cells / 16.0
        rows_violated = count_rows_with_duplicates(grid)
        cols_violated = count_cols_with_duplicates(grid)
        boxes_violated = count_boxes_with_duplicates(grid)
        score = filled_normalized - 0.1 * (rows_violated + cols_violated + boxes_violated)
        return score, {
            "filled_cells": filled_cells,
            "filled_normalized": filled_normalized,
            "rows_violated": rows_violated,
            "cols_violated": cols_violated,
            "boxes_violated": boxes_violated,
            "constraint_violations": rows_violated + cols_violated + boxes_violated
        }

`count_rows_with_duplicates(grid)`: number of rows where some non-zero value appears twice. Same for cols and boxes. Empty cells (value 0) are NOT counted as duplicates.

This formula:
- Rewards filled cells (progress)
- Penalizes constraint violations (only when an explicit violation already occurred; does NOT predict future violations — that's viability's job)

CRITICALLY: this formula depends ONLY on surface features of the grid. It does NOT consult the solver. This is what makes it a valid "progress" signal distinct from viability.

### 5.3 Solver interface

Assume an existing Sudoku 4×4 solver in your codebase. Required interface:

    class Sudoku4Solver:
        version = "sudoku_solver_v1"
    
        def solve(self, state) -> SolverResult:
            """
            Returns a SolverResult with:
              solvable: bool
              num_solutions: int (0 if unsolvable)
              nodes: int (total search nodes expanded)
              backtracks: int
              solution_depth: Optional[int] (length of solution path; None if unsolvable)
              solve_time_ms: float
              solution_path: Optional[List[Action]] (one example viable continuation)
            """
            ...
    
        def is_viable(self, state) -> bool:
            return self.solve(state).solvable
    
        def find_one_solution(self, state) -> Optional[List[Action]]:
            return self.solve(state).solution_path

If your existing solver has a different API, write a thin adapter rather than rewriting it.

IMPORTANT: when computing `solver.nodes` for the score label `solver_residual_difficulty`, the solver should expand the FULL search tree even for unsolvable states (i.e., do not early-terminate on the first proof of unsolvability). This ensures unsolvable states have a finite, comparable `nodes` value rather than a sentinel. If your existing solver early-terminates, add a flag `exhaustive=True` and use it during data generation.

### 5.4 Initial puzzle generation

For the toy run, generate 500 Sudoku 4×4 puzzles with 6 initially empty cells (i.e., 10 cells pre-filled). This gives a reasonable trajectory length (~6 steps) while keeping solver fast.

    def generate_root_puzzle(rng_seed):
        full_grid = generate_random_complete_sudoku(rng_seed)
        cells_to_remove = rng.sample(range(16), 6)
        for idx in cells_to_remove:
            r, c = idx // 4, idx % 4
            full_grid[r][c] = 0
        result = solver.solve({"grid": full_grid})
        if result.num_solutions == 0:
            return None  # regenerate
        return {"grid": full_grid}

Filter out puzzles with 0 solutions. Allow puzzles with multiple solutions (Sudoku 4×4 frequently has multiple).

---

## 6. Generation Pipeline

### 6.1 High-level flow

1. Generate 500 root puzzles, split 70/15/15 by puzzle index
2. For each root puzzle in train split:
   - Run solver to get a viable solution path
   - For each intermediate state s_t along the path (excluding first step and last 2 steps): check is_boundary_state(s_t); skip non-boundary states with probability 0.5
   - Sample one sibling set from s_t (12 candidates: 3 lt + 3 ht + 3 sol + 3 prt)
   - Validate, label, deduplicate, and emit JSONL line
   - Continue until either ~3 sibling sets per root, or path exhausted
3. Repeat step 2 for val and test splits with sampling_protocol = lt:ht=4:4 (no oracle injection)
4. Stop when each role file has reached its target size (1500 / 500 / 500)

### 6.2 Per-sibling-set generation pseudocode

    def generate_sibling_set(state, dataset_role, root_idx, solver_idx, t, set_idx):
        sampling_protocol = SAMPLING_PROTOCOLS[dataset_role]
        K_total = sampling_protocol["K_total"]
    
        # Step 1: gather candidates from each source
        candidates_raw = []
        candidates_raw += sample_lt(state, K=sampling_protocol["K_lt"], temp=0.3)
        candidates_raw += sample_ht(state, K=sampling_protocol["K_ht"], temp=1.0)
        candidates_raw += sample_rand(state, K=sampling_protocol["K_rand"])
        candidates_raw += sample_sol(state, K=sampling_protocol["K_sol"])
        candidates_raw += sample_prt(state, K=sampling_protocol["K_prt"])
    
        # Step 2: dedup by canonical action hash
        seen_hashes = set()
        candidates = []
        for c in candidates_raw:
            h = canonical_action_hash(c["action_struct"])
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            candidates.append(c)
    
        # Step 3: shuffle
        rng = Random(set_seed)
        rng.shuffle(candidates)
        for i, c in enumerate(candidates):
            c["candidate_id"] = f"c{i:03d}"
    
        # Step 4: validate, transition, solver-label
        for c in candidates:
            c["parse_valid"] = (c["action_struct"] is not None)
            c["local_valid"] = is_local_valid(state, c["action_struct"])
            if c["local_valid"]:
                next_state = apply_transition(state, c["action_struct"])
                c["transition_valid"] = True
                c["next_state"] = render_state(next_state)
                solver_result = solver.solve(next_state)
                c["next_viable"] = solver_result.solvable
                c["solver"] = serialize(solver_result)
                c["next_is_goal"] = is_goal(next_state)
                c["progress"] = compute_progress(next_state)
                c["score_labels"] = {
                    "local_progress": {
                        "value": c["progress"]["local_progress_score"],
                        "use_for_main_score_baseline": True
                    },
                    "solver_residual_difficulty": {
                        "value": -math.log(1 + solver_result.nodes),
                        "formula": "-log(1 + solver_nodes)",
                        "use_for_main_score_baseline": False
                    }
                }
            else:
                c["transition_valid"] = False
                c["next_state"] = None
            c["candidate_class"] = classify(c)
            c["eligible_for_viability_eval"] = c["local_valid"]
    
        # Step 5: compute policy_eval_logprob for ALL candidates
        compute_policy_eval_logprobs(candidates, state, policy_model)
    
        # Step 6: compute set stats
        set_stats = compute_set_stats(candidates)
    
        # Step 7: mine deceptive pairs
        deceptive_pairs = mine_deceptive_pairs(candidates)
    
        # Step 8: assemble and emit
        return assemble_record(state, candidates, set_stats, deceptive_pairs, ...)

### 6.3 Hyperparameters (toy run defaults)

    TOY_RUN_CONFIG = {
        "n_root_puzzles": 500,
        "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
        "sibling_sets_per_root": {
            "train_balanced": 4,
            "val_natural_calibration": 7,
            "test_natural_policy": 7
        },
        "target_size": {
            "train_balanced": 1500,
            "val_natural_calibration": 500,
            "test_natural_policy": 500
        },
        "boundary_threshold": 0.3,
        "boundary_oversampling_rate": 0.5,
        "max_attempts_per_set": 3,
        "vllm_batch_size": 32
    }
    
    SAMPLING_PROTOCOLS = {
        "train_balanced": {
            "K_total": 12, "K_lt": 3, "K_ht": 3, "K_rand": 0, "K_sol": 3, "K_prt": 3,
            "lt_temperature": 0.3, "ht_temperature": 1.0, "top_p": 0.95
        },
        "val_natural_calibration": {
            "K_total": 8, "K_lt": 4, "K_ht": 4, "K_rand": 0, "K_sol": 0, "K_prt": 0,
            "lt_temperature": 0.3, "ht_temperature": 1.0, "top_p": 0.95
        },
        "test_natural_policy": {
            "K_total": 8, "K_lt": 4, "K_ht": 4, "K_rand": 0, "K_sol": 0, "K_prt": 0,
            "lt_temperature": 0.3, "ht_temperature": 1.0, "top_p": 0.95
        }
    }

### 6.4 Performance hints

- Use vLLM with `n=K_lt` (or `n=K_ht`) per request to batch sample multiple candidates per state in one call.
- Solver calls are CPU-bound; parallelize with multiprocessing.Pool if Sudoku 4×4 solver is slow.
- For 1500 sibling sets x 12 candidates x ~3 solver calls each = ~54k solver calls. Sudoku 4×4 should solve in <10ms each, so ~10 minutes total CPU time.
- LLM forward passes are the bottleneck. Estimate: 1500 sets x (12 generation + 12 logprob eval) = ~36k forward passes. With vLLM batched, expect 30-60 minutes on a single GPU.

---

## 7. Sanity Checks (run after each role completes)

Write a `validate_dataset.py` script that loads a JSONL file and runs:

### 7.1 Schema validation

    from pydantic import ValidationError
    errors = []
    for line_idx, line in enumerate(open(path)):
        try:
            record = SiblingSetRecord.model_validate_json(line)
        except ValidationError as e:
            errors.append((line_idx, e))
    assert not errors, f"{len(errors)} schema violations"

### 7.2 Field-consistency checks

For each record:
- `set_stats.num_candidates` equals `len(candidates)` equals `sampling_protocol.K_total - skipped`
- `set_stats.num_valid_viable` equals `sum(c.candidate_class == "valid_viable" for c in candidates)`
- `set_stats.mixed` equals `(num_valid_viable > 0 and num_valid_doomed > 0)`
- For every deceptive pair:
  - `a_plus.candidate_class == "valid_viable"`
  - `a_minus.candidate_class == "valid_doomed"`
  - `a_minus.progress.local_progress_score >= a_plus.progress.local_progress_score`
- For every candidate with `local_valid=true`: `next_state` is not None and `solver` is not None

### 7.3 Role-specific checks

For `val_natural_calibration` and `test_natural_policy`:
- No candidate has source in {"sol", "prt", "rand"}
- No candidate has `source_meta.is_oracle_injected == true`

For `train_balanced`:
- At least 60% of records have `set_stats.mixed == true`
- Source breakdown roughly matches sampling protocol (within +/- 20%)

### 7.4 Leakage check

    train_state_hashes = {r.state.state_hash for r in load("train_balanced.jsonl")}
    val_state_hashes = {r.state.state_hash for r in load("val_natural_calibration.jsonl")}
    test_state_hashes = {r.state.state_hash for r in load("test_natural_policy.jsonl")}
    
    assert not (train_state_hashes & val_state_hashes), "Train/val state leak"
    assert not (train_state_hashes & test_state_hashes), "Train/test state leak"
    assert not (val_state_hashes & test_state_hashes), "Val/test state leak"

Also check (state_hash, action_hash) pair leakage for finer guarantee.

### 7.5 Distributional sanity

Print to stdout:
- Distribution of `set_stats.mixed` (true/false ratio per role)
- Distribution of `next_viable` across all candidates per role
- Distribution of `candidate_class`
- Histogram of `solver.nodes` (sanity check: unsolvable states should have finite, non-zero nodes)
- Histogram of `progress.local_progress_score` for viable vs doomed candidates (visual sanity check that they overlap—if doomed candidates uniformly have lower progress, you have no deceptive pairs)
- Number of mined deceptive pairs per role

If `train_balanced` has fewer than 100 deceptive pairs total, the deceptive benchmark for paper §4.4 will be data-starved — increase boundary oversampling and rerun.

---

## 8. File Layout

    data/
      sudoku4/
        train_balanced.jsonl              # 1500 records
        val_natural_calibration.jsonl     # 500 records
        test_natural_policy.jsonl         # 500 records
        metadata.json                     # generation metadata
        sanity_check_report.txt           # output of validate_dataset.py
    
    scripts/
      generate_sudoku4_data.py            # main generation script
      validate_dataset.py                 # sanity checker
      sudoku4_solver.py                   # solver wrapper
      sudoku4_env.py                      # state, action, transition, render
      policy_sampler.py                   # vLLM wrapper for lt/ht sampling
      progress_sudoku4.py                 # local progress formula

`data/sudoku4/metadata.json` contents:

    {
      "schema_version": "save_sibling_set_v1.2",
      "env": "sudoku4",
      "generator_version": "save_data_gen_v1",
      "git_commit": "<hash>",
      "generated_at": "2026-05-03T...",
      "config": {
        "n_root_puzzles": 500,
        "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
        "sampling_protocols": {...},
        "boundary_threshold": 0.3
      },
      "policy_model": "{policy_model_path}",
      "policy_checkpoint": "{checkpoint_id}",
      "files": {
        "train_balanced.jsonl": {"n_records": 1500, "size_mb": 0},
        "val_natural_calibration.jsonl": {"n_records": 500, "size_mb": 0},
        "test_natural_policy.jsonl": {"n_records": 500, "size_mb": 0}
      }
    }

---

## 9. Pydantic Schema (paste-ready Python)

Use this for runtime validation. The script should `pip install pydantic>=2.0`.

    from pydantic import BaseModel, Field
    from typing import Optional, List, Dict, Literal
    
    class StateStruct(BaseModel):
        grid: List[List[int]]
    
    class StateSolver(BaseModel):
        num_solutions: int
        nodes: int
        backtracks: int
        solve_time_ms: float
    
    class ActionSpaceStats(BaseModel):
        num_legal_actions: int
        num_legal_viable_actions: int
        num_legal_doomed_actions: int
    
    class State(BaseModel):
        state_hash: str
        state_struct: StateStruct
        state_text: str
        state_text_version: str
        state_viable: bool
        state_is_goal: bool
        state_solver: StateSolver
        action_space_stats: ActionSpaceStats
    
    class SamplingProtocol(BaseModel):
        K_total: int
        K_lt: int
        K_ht: int
        K_rand: int
        K_sol: int
        K_prt: int
        lt_temperature: float
        ht_temperature: float
        top_p: float
        policy_model: str
        policy_checkpoint: str
        training_phase: str
        candidate_order: Literal["random_shuffled"]
        shuffle_seed: int
        set_seed: int
        dedup_policy: Literal["canonical_action_hash"]
        candidate_pool_size_before_dedup: int
        candidate_pool_size_after_dedup: int
    
    class SourceMeta(BaseModel):
        temperature: Optional[float] = None
        top_p: Optional[float] = None
        is_oracle_injected: bool
    
    class ActionStruct(BaseModel):
        row: int
        col: int
        value: int
    
    class Logprobs(BaseModel):
        generation_logprob: Optional[float]
        policy_eval_logprob: float
        policy_eval_model: str
        policy_eval_prompt_version: str
    
    class NextState(BaseModel):
        next_state_hash: str
        next_state_struct: StateStruct
        next_state_text: str
        next_state_text_version: str
        next_is_goal: bool
        next_viable: bool
    
    class CandidateSolver(BaseModel):
        solvable: bool
        num_solutions: int
        nodes: int
        backtracks: int
        solution_depth: Optional[int]
        solve_time_ms: float
        solver_version: str
    
    class ProgressFeatures(BaseModel):
        filled_cells: int
        filled_normalized: float
        constraint_violations: int
        rows_violated: int
        cols_violated: int
        boxes_violated: int
    
    class Progress(BaseModel):
        formula_id: str
        formula_spec: str
        local_progress_score: float
        features: ProgressFeatures
    
    class ScoreLabelEntry(BaseModel):
        value: float
        use_for_main_score_baseline: bool
        formula: Optional[str] = None
    
    class ScoreLabels(BaseModel):
        local_progress: ScoreLabelEntry
        solver_residual_difficulty: ScoreLabelEntry
    
    class Candidate(BaseModel):
        candidate_id: str
        action_hash: str
        display_rank: int
        source: Literal["lt", "ht", "rand", "sol", "prt"]
        source_meta: SourceMeta
        action_text: str
        action_text_canonical: str
        action_struct: Optional[ActionStruct]
        logprobs: Logprobs
        parse_valid: bool
        local_valid: bool
        transition_valid: bool
        candidate_class: Literal["parse_invalid", "local_invalid", "valid_doomed", "valid_viable", "goal_reaching"]
        eligible_for_viability_eval: bool
        next_state: Optional[NextState]
        solver: Optional[CandidateSolver]
        progress: Optional[Progress]
        score_labels: Optional[ScoreLabels]
    
    class SourceCount(BaseModel):
        selected: int
        valid_viable: int
        valid_doomed: int
        invalid: int
        goal_reaching: int
    
    class SetStats(BaseModel):
        num_candidates: int
        num_parse_invalid: int
        num_local_invalid: int
        num_valid_viable: int
        num_valid_doomed: int
        num_goal_reaching: int
        mixed: bool
        all_viable: bool
        all_doomed: bool
        source_breakdown: Dict[str, SourceCount]
    
    class DeceptivePairCondition(BaseModel):
        plus_next_viable: Literal[True]
        minus_next_viable: Literal[False]
        both_local_valid: Literal[True]
        minus_progress_ge_plus: Literal[True]
        progress_gap: float
    
    class DeceptivePair(BaseModel):
        pair_id: str
        a_plus_candidate_id: str
        a_minus_candidate_id: str
        condition: DeceptivePairCondition
    
    class SelectionCriteria(BaseModel):
        mix_score: float
        mix_score_formula: str
        is_boundary: bool
        boundary_threshold: float
        selection_reason: List[str]
    
    class Provenance(BaseModel):
        env_version: str
        solver_version: str
        generator_version: str
        git_commit: str
        created_at: str
    
    class SiblingSetRecord(BaseModel):
        schema_: Literal["save_sibling_set_v1.2"] = Field(alias="schema")
        env: Literal["sudoku4"]
        dataset_role: Literal[
            "train_balanced",
            "val_balanced_diagnostic",
            "test_balanced_diagnostic",
            "val_natural_calibration",
            "test_natural_policy",
            "test_deceptive_curated"
        ]
        split: Literal["train", "val", "test"]
        root_id: str
        trajectory_id: str
        sibling_set_id: str
        t: int
        state: State
        sampling_protocol: SamplingProtocol
        candidates: List[Candidate]
        set_stats: SetStats
        deceptive_pairs: List[DeceptivePair]
        selection_criteria: SelectionCriteria
        provenance: Provenance

---

## 10. Test Run Protocol (do this first, before scaling)

Before running the full 1500-set generation, do a 30-set smoke test:

    python scripts/generate_sudoku4_data.py \
        --role train_balanced \
        --n_target 30 \
        --output data/sudoku4/smoke_test.jsonl \
        --policy_model {policy_model_path} \
        --seed 42
    python scripts/validate_dataset.py data/sudoku4/smoke_test.jsonl

Inspect manually:
1. Open the first record. Does every field match the schema?
2. Are there any null values where there shouldn't be?
3. Look at one mixed sibling set. Does the viable/doomed split match the solver labels?
4. Look at one deceptive pair (if any exists). Manually verify: a_plus is viable, a_minus is doomed, AND a_minus has higher progress.
5. Are state_text and next_state_text correctly rendered?
6. For sol candidates: does `is_oracle_injected=true` and `generation_logprob=null`?
7. For lt candidates: does `generation_logprob` look reasonable (negative, finite)?

Only after the smoke test passes all manual checks, scale to 1500.

---

## 11. Common Failures and Recovery

| Symptom                                                 | Likely cause                                                 | Fix                                                          |
| ------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `set_stats.mixed=false` for >60% of train_balanced      | Boundary oversampling not aggressive enough; or too few prt candidates | Lower `boundary_threshold` to 0.2; force-include at least 1 sol + 1 prt per set |
| prt candidates failing (cannot find valid perturbation) | State has very few legal but doomed alternatives             | Skip this state, sample another; record skip rate            |
| `solver.nodes` near zero for unsolvable states          | Solver early-terminates on unsolvability                     | Pass `exhaustive=True` to solver; ensure full search tree expansion |
| `policy_eval_logprob` = null for some candidates        | Logprob computation failed (long sequence, OOM)              | Increase max_tokens for logprob eval; if persistent, skip candidate and record |
| Schema validation errors                                | Bug in serializer                                            | Run on first 10 records before scaling; use Pydantic's `.model_dump_json()` |
| Leakage check fails                                     | Trajectory split is wrong                                    | Recheck §4.4; ensure split is at root_idx level not state level |
| Many records have `num_legal_actions=0`                 | State s_t is already goal or dead-end                        | Filter these out before generating sibling sets              |

---

## 12. Hand-off Checklist

Before considering data generation done:

- [ ] All three role files exist and have target sizes
- [ ] `validate_dataset.py` passes for all three files
- [ ] No leakage between any two roles
- [ ] `train_balanced` has at least 60% mixed sibling sets
- [ ] `train_balanced` has at least 100 deceptive pairs total
- [ ] `val_natural_calibration` and `test_natural_policy` contain zero sol/prt candidates
- [ ] `metadata.json` is up to date
- [ ] First 5 records of each file have been visually inspected
- [ ] A backup of all generated data exists

---

## End of spec

Questions or ambiguities — flag them as comments in code rather than inventing default behavior for missing spec details.