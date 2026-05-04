"""SAVE sibling-set v1.2 schema (Pydantic models).

Direct port of doc/data_generation_sudoku.md §9. Use SiblingSetRecord for
runtime validation: SiblingSetRecord.model_validate_json(line) on each
JSONL line during generation and during validate_dataset.py.

Dependency: pydantic >= 2.0 (uses v2 syntax).
"""
from __future__ import annotations

from typing import Optional, List, Dict, Literal
from pydantic import BaseModel, Field


# --- State-side models ---

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


# --- Sampling protocol ---

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


# --- Candidate-side models ---

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
    candidate_class: Literal[
        "parse_invalid", "local_invalid", "valid_doomed", "valid_viable", "goal_reaching"
    ]
    eligible_for_viability_eval: bool
    next_state: Optional[NextState]
    solver: Optional[CandidateSolver]
    progress: Optional[Progress]
    score_labels: Optional[ScoreLabels]


# --- Set-level summaries ---

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


# --- Top-level record ---

class SiblingSetRecord(BaseModel):
    schema_: Literal["save_sibling_set_v1.2"] = Field(alias="schema")
    env: Literal["sudoku4"]
    dataset_role: Literal[
        "train_balanced",
        "val_balanced_diagnostic",
        "test_balanced_diagnostic",
        "val_natural_calibration",
        "test_natural_policy",
        "test_deceptive_curated",
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


# --- Smoke test ---

if __name__ == "__main__":
    sample = {
        "schema": "save_sibling_set_v1.2",
        "env": "sudoku4",
        "dataset_role": "train_balanced",
        "split": "train",
        "root_id": "sudoku4_train_000001",
        "trajectory_id": "sudoku4_train_000001_solver_00",
        "sibling_set_id": "sudoku4_train_000001_solver_00_t003_set00",
        "t": 3,
        "state": {
            "state_hash": "sha1:7a3f0000000000000000000000000000000000aa",
            "state_struct": {"grid": [[1,2,3,4],[3,4,0,0],[0,0,4,3],[4,3,0,0]]},
            "state_text": "3 2 | 1 4\n1 . | 3 .\n---------\n. . | 2 1\n. 1 | . 3",
            "state_text_version": "sudoku_text_b5_compat_v1",
            "state_viable": True,
            "state_is_goal": False,
            "state_solver": {"num_solutions": 1, "nodes": 13, "backtracks": 2, "solve_time_ms": 0.7},
            "action_space_stats": {"num_legal_actions": 7, "num_legal_viable_actions": 3, "num_legal_doomed_actions": 4},
        },
        "sampling_protocol": {
            "K_total": 12, "K_lt": 3, "K_ht": 3, "K_rand": 0, "K_sol": 3, "K_prt": 3,
            "lt_temperature": 0.3, "ht_temperature": 1.0, "top_p": 0.95,
            "policy_model": "outputs/rl_b5_phase3_v8_anchor/final",
            "policy_checkpoint": "rl_b5_phase3_v8_anchor_final",
            "training_phase": "rl",
            "candidate_order": "random_shuffled",
            "shuffle_seed": 12345, "set_seed": 94231,
            "dedup_policy": "canonical_action_hash",
            "candidate_pool_size_before_dedup": 14,
            "candidate_pool_size_after_dedup": 12,
        },
        "candidates": [],
        "set_stats": {
            "num_candidates": 0, "num_parse_invalid": 0, "num_local_invalid": 0,
            "num_valid_viable": 0, "num_valid_doomed": 0, "num_goal_reaching": 0,
            "mixed": False, "all_viable": False, "all_doomed": True,
            "source_breakdown": {},
        },
        "deceptive_pairs": [],
        "selection_criteria": {
            "mix_score": 0.42, "mix_score_formula": "min(num_valid_viable, num_valid_doomed) / num_candidates",
            "is_boundary": True, "boundary_threshold": 0.3,
            "selection_reason": ["mixed_sibling_set", "boundary_state"],
        },
        "provenance": {
            "env_version": "sudoku4_env_v1_10empty",
            "solver_version": "sudoku4_solver_v1",
            "generator_version": "save_data_gen_v1",
            "git_commit": "test",
            "created_at": "2026-05-03T00:00:00Z",
        },
    }
    rec = SiblingSetRecord.model_validate(sample)
    print(f"OK: schema validates. record id={rec.sibling_set_id}")
    print(f"   round-trip ok: {len(rec.model_dump_json()) > 0}")
