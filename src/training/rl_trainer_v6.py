"""GRPO RL trainer (v6/v7) for termination prediction (Sudoku + Polyomino).

Phase 1 of the RL approach (per doc/plan_2026-04-29_rl_approach.md): no rollout
truncation, multi-step rollouts.

v6 reward (per rollout = full puzzle attempt):
  per-step:
    - format compliance:   +0.05 per of 4 required tags (max +0.20)
    - <solvable> correctness:
        TP (pred F, GT F):  +1.0  (caught doom)
        FN (pred T, GT F):  -0.7  (missed doom)
        FP (pred F, GT T):  -0.5  (spurious doom)
        TN (pred T, GT T):  +0.3  (correct salvation)
  end-of-trajectory:
    +3.0 if env.is_solved else 0.0  (v6.1: down from +10)

v7 reward — first attempt to fix the B-7 Pentomino collapse (see
doc/eval_2026-04-30_b7_rl_phase1.md). v7 changes vs v6:

  1. Symmetric magnitudes — TP=+1.0, FN=-1.0, FP=-1.0, TN=+1.0
  2. Per-batch class balancing — inverse GT-frequency, capped to [floor, cap]
  3. Per-step progress bonus on valid actions

v7 partially helped (delayed collapse from step ~50 to step ~75) but didn't
prevent it: the sanity test (sanity_2026-04-30_b7_rollout_stats.json) showed
the static reward landscape favors oracle (oracle +1.00 vs always_false +0.46
under v7), so the collapse must be a *dynamic* drift off the SFT optimum
under noisy GRPO advantages, not a static reward-landscape problem.

v8 reward = v7 + auxiliary KL anchor on <viability>/<solvable> tag tokens
against the frozen ref_policy. The KL coefficient is large (default 0.5)
and concentrated only on the tag content tokens (typically the "true" or
"false" inside the tag), so calibration stays locked at SFT quality while
action tokens still optimize freely. This directly addresses the dynamic
calibration drift identified in the sanity test.

Uses pure transformers (no TRL, no vLLM).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Repo path for src.* imports
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.environments.sudoku import SudokuEnv  # noqa: E402
from src.data.sft_formatter import SFTFormatter  # noqa: E402


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

@dataclass
class RLConfig:
    sft_checkpoint: str
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    output_dir: str = "outputs/rl_b5_phase1"

    # Env
    grid_size: int = 4
    difficulty: str = "easy"
    max_rollout_steps: int = 12

    # Rollout
    n_puzzles_per_batch: int = 4   # number of distinct puzzles per RL step
    group_size: int = 8            # K rollouts per puzzle
    temperature: float = 0.7
    max_response_tokens: int = 256

    # Optimization
    learning_rate: float = 1e-6
    n_total_steps: int = 200
    ppo_epochs: int = 2
    clip_eps: float = 0.2
    kl_coef: float = 0.05
    max_grad_norm: float = 1.0

    # Reward — v6 (default) preserves Sudoku tuning; v7 fixes the B-7 Pentomino
    # collapse caused by short rollouts + per-class asymmetry (see module docstring).
    # v8 = v7 + auxiliary KL anchor on <viability>/<solvable> tag tokens against
    # the SFT reference. Fixes the dynamic calibration drift observed when v7 alone
    # was insufficient (sanity test 2026-04-30 showed oracle is global max but the
    # local gradient drifts the policy off-manifold). The tag-specific anchor is a
    # tighter leash on just the calibration tokens, leaving action tokens free.
    reward_version: str = "v6"            # "v6" | "v7" | "v8"
    tp_reward: float = 1.0
    fn_reward: float = -0.7
    fp_reward: float = -0.5
    tn_reward: float = 0.3
    format_per_tag: float = 0.05
    success_bonus: float = 3.0
    fail_bonus: float = 0.0
    # v7 only — inverse-frequency class balancing on per-step calibration reward
    class_balance: bool = False
    class_balance_floor: float = 0.5      # min weight (avoid suppressing common class)
    class_balance_cap: float = 5.0        # max weight (avoid rare-class instability)
    # v7 only — per-step progress bonus on valid actions that advance the trajectory
    progress_bonus_per_step: float = 0.0
    # v8 only — extra KL penalty applied to <viability>/<solvable> tag content tokens
    # against ref_policy. Coefficient is per-step (not per-token), so a single
    # anchor applies the same total pressure regardless of how the tag tokenizes.
    # Set to 0.0 to disable.
    viability_kl_coef: float = 0.0

    # Eval / save
    eval_every: int = 50
    save_every: int = 100
    eval_n_puzzles: int = 30  # cheap eval during training

    # Misc
    seed: int = 42
    bf16: bool = True
    truncation_mode: str = "off"      # 'off' | 'conservative' (Phase 2)
    truncation_threshold: float = 0.95


# ----------------------------------------------------------------------------
# Reward helpers
# ----------------------------------------------------------------------------

REQUIRED_TAGS = ("<observation>", "<prediction>", "<solvable>", "<answer>")

_re_solvable = re.compile(r"<solvable>\s*(true|false)\s*</solvable>", re.IGNORECASE)
# Polyomino + future MKD use `<viability>` instead of `<solvable>`.
_re_viability = re.compile(r"<viability>\s*(true|false)\s*</viability>", re.IGNORECASE)
_re_action = re.compile(
    r"<answer>.*?place\s+(\d+)\s+at\s+row\s+(\d+)\s+col\s+(\d+)",
    re.IGNORECASE | re.DOTALL,
)


def parse_solvable(text: str) -> Optional[bool]:
    """Parse <solvable> (Sudoku) OR <viability> (Polyomino, MKD) — same semantics, different tag name."""
    m = _re_solvable.search(text)
    if not m:
        m = _re_viability.search(text)
    if not m:
        return None
    return m.group(1).lower() == "true"


_re_viability_or_solvable = re.compile(
    r"<(?:viability|solvable)>\s*(true|false)\s*</(?:viability|solvable)>",
    re.IGNORECASE,
)


def find_viability_token_positions(tokenizer, response_text: str, response_ids: list) -> list:
    """Return token indices in `response_ids` that fall inside <viability>...</viability>
    (or <solvable>...</solvable>) — specifically, the positions of the "true"/"false"
    content. Used by the v8 viability-tag KL anchor.

    Implementation: find the char span of the tag content via regex, then re-tokenize
    `response_text` with offset_mapping (fast tokenizer) to map char span → token span.
    Returns [] if no tag is present, the regex doesn't match, or the re-tokenization
    length disagrees with `response_ids` (rare tokenization roundtrip mismatch).

    The tradeoff with using offset_mapping: it requires a HF "fast" tokenizer. Qwen2.5
    is fast, but if a future model isn't we'd need a fallback. For now we just return []
    on any inconsistency rather than risk masking the wrong tokens.
    """
    m = _re_viability_or_solvable.search(response_text)
    if not m:
        return []
    char_start, char_end = m.start(1), m.end(1)
    try:
        enc = tokenizer(response_text, return_offsets_mapping=True, add_special_tokens=False)
    except (TypeError, NotImplementedError):
        return []  # slow tokenizer, no offset mapping
    re_ids = enc["input_ids"]
    re_offsets = enc["offset_mapping"]
    if len(re_ids) != len(response_ids):
        # Tokenization roundtrip mismatch (extremely rare with bf16 sampled responses).
        # Skip the anchor on this step rather than risk masking the wrong tokens.
        return []
    return [i for i, (s, e) in enumerate(re_offsets) if s < char_end and e > char_start]


def parse_action(text: str) -> Optional[tuple]:
    """Returns (row_0idx, col_0idx, num) or None if unparseable."""
    m = _re_action.search(text)
    if not m:
        return None
    n, r, c = map(int, m.groups())
    return r - 1, c - 1, n


def format_reward(text: str, per_tag: float) -> float:
    return sum(per_tag for t in REQUIRED_TAGS if t.lower() in text.lower())


def solvable_reward(pred: Optional[bool], gt: bool, cfg: RLConfig) -> float:
    if pred is None:
        # Treat unparseable <solvable> as the worst case (FN if GT is False, FP if GT is True)
        return cfg.fn_reward if not gt else cfg.fp_reward
    if not pred and not gt: return cfg.tp_reward
    if pred and not gt:     return cfg.fn_reward
    if not pred and gt:     return cfg.fp_reward
    return cfg.tn_reward


# ----------------------------------------------------------------------------
# Rollout
# ----------------------------------------------------------------------------

@dataclass
class StepRecord:
    prompt_text: str
    response_text: str
    response_ids: list  # token ids of just the response (without prompt)
    action: Optional[tuple]
    pred_solvable: Optional[bool]
    gt_solvable: bool
    is_breaking_point: bool
    step_reward: float
    # PPO bug fix: cache rollout-time logprobs so PPO ratio is computed correctly.
    # Without this, old_logp == new_logp under the same policy → ratio always 1, no clipping.
    old_logps: Optional[list] = None  # per-token logprobs under the policy AT ROLLOUT TIME
    # Reward components — kept separately so v7 class-balancing can rescale only
    # the calibration term and recompute step_reward from the parts.
    calib_reward: float = 0.0
    fmt_reward: float = 0.0
    progress_reward: float = 0.0
    action_was_valid: bool = False


@dataclass
class Rollout:
    puzzle_seed: int
    steps: list  # list of StepRecord
    is_solved: bool
    final_reward: float

    def num_steps(self) -> int:
        return len(self.steps)


def build_prompt(tokenizer, system_prompt: str, history: list, current_user_msg: str) -> str:
    """Render the chat prompt up to and including <|im_start|>assistant\\n."""
    msgs = [{"role": "system", "content": system_prompt}]
    for u, a in history:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": current_user_msg})
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def sample_response(model, tokenizer, prompt_text: str, cfg: RLConfig, device: str):
    """Generate one assistant response AND compute its per-token logprob under the
    current (rollout-time) policy. Returns (text, response_ids, old_logps).

    Computing old_logps here at rollout time is the PPO bug fix — without it, recomputing
    old_logps inside the PPO loop (under the same model) yields ratio=1 always.
    """
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]
    out = model.generate(
        **inputs,
        max_new_tokens=cfg.max_response_tokens,
        temperature=cfg.temperature,
        do_sample=cfg.temperature > 0,
        top_p=0.95,
        top_k=50,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )
    sequences = out.sequences  # [1, prompt_len + n_new]
    response_ids = sequences[0, prompt_len:].tolist()
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

    # Compute per-token logprobs of the sampled tokens.
    # out.scores is a tuple of length n_new; scores[i] is logits[1, V] for the (i+1)-th generated token.
    if out.scores is not None and len(out.scores) > 0:
        old_logps = []
        for i, score in enumerate(out.scores):
            log_probs = F.log_softmax(score[0], dim=-1)
            tok_id = sequences[0, prompt_len + i].item()
            old_logps.append(log_probs[tok_id].item())
    else:
        old_logps = []
    return response_text, response_ids, old_logps


def do_rollout(
    model, tokenizer, env: SudokuEnv, system_prompt: str,
    puzzle_seed: int, cfg: RLConfig, device: str,
) -> Rollout:
    """One full puzzle rollout from s_0 to solved/deadlock/timeout."""
    obs = env.reset(seed=puzzle_seed)
    steps = []
    history = []
    is_solved = False

    for t in range(cfg.max_rollout_steps):
        prompt_text = build_prompt(tokenizer, system_prompt, history, obs)
        response_text, response_ids, old_logps = sample_response(model, tokenizer, prompt_text, cfg, device)

        pred = parse_solvable(response_text)
        action = parse_action(response_text)
        fmt_r = format_reward(response_text, cfg.format_per_tag)

        if action is None:
            # Unparseable action — give worst-case step reward and stop.
            calib_r = cfg.fn_reward
            steps.append(StepRecord(
                prompt_text=prompt_text, response_text=response_text, response_ids=response_ids,
                action=None, pred_solvable=pred, gt_solvable=False, is_breaking_point=False,
                step_reward=calib_r + fmt_r, old_logps=old_logps,
                calib_reward=calib_r, fmt_reward=fmt_r, progress_reward=0.0,
                action_was_valid=False,
            ))
            break

        r, c, n = action
        action_str = f"place {n} at row {r + 1} col {c + 1}"
        next_obs, _env_reward, done, info = env.step(action_str)
        gt_solvable = bool(info.get("is_solvable", True))
        is_bp = bool(info.get("is_breaking_point", False))
        is_solved = bool(info.get("success", False))
        calib_r = solvable_reward(pred, gt_solvable, cfg)
        progress_r = cfg.progress_bonus_per_step  # valid action that advanced trajectory

        steps.append(StepRecord(
            prompt_text=prompt_text, response_text=response_text, response_ids=response_ids,
            action=action, pred_solvable=pred, gt_solvable=gt_solvable, is_breaking_point=is_bp,
            step_reward=calib_r + fmt_r + progress_r, old_logps=old_logps,
            calib_reward=calib_r, fmt_reward=fmt_r, progress_reward=progress_r,
            action_was_valid=True,
        ))
        history.append((obs, response_text))
        obs = next_obs

        # Phase 2 (truncation_mode=conservative): would terminate here on high-confidence False.
        # Phase 1 leaves truncation_mode='off' so this never fires.
        if cfg.truncation_mode == "conservative" and pred is False:
            # NOTE: needs P(false) > truncation_threshold from a teacher-forced pass; not wired up yet.
            pass

        if done:
            break

    final_reward = sum(s.step_reward for s in steps)
    final_reward += cfg.success_bonus if is_solved else cfg.fail_bonus
    return Rollout(
        puzzle_seed=puzzle_seed, steps=steps,
        is_solved=is_solved, final_reward=final_reward,
    )


# ----------------------------------------------------------------------------
# Batched rollout — parallelize K rollouts (and N puzzles × K) through one
# `model.generate()` call per turn. ~5x speedup over the single-rollout path.
# Used for full Phase 1; the single-rollout path stays as a fallback for
# debugging.
# ----------------------------------------------------------------------------

def _build_env_factory(env_template):
    """Make a function that produces fresh env instances of the same type/config as env_template."""
    cls = type(env_template)
    if cls.__name__ == "SudokuEnv":
        return lambda: cls(
            grid_size=env_template.grid_size,
            difficulty=env_template.difficulty,
            max_steps=env_template.max_steps,
        )
    if cls.__name__ == "PolyominoEnv":
        return lambda: cls(
            board_h=env_template.board_h,
            board_w=env_template.board_w,
            piece_set=env_template.initial_pieces,
            max_steps=env_template.max_steps,
        )
    raise ValueError(f"unknown env class: {cls.__name__}")


@torch.no_grad()
def sample_responses_batched(model, tokenizer, prompts: list, cfg: RLConfig, device: str):
    """Generate K responses in a single batched forward pass.

    Returns a list of (text, response_ids, old_logps) tuples, one per prompt.
    Token-level old_logps cached at rollout time (PPO bug fix).
    """
    if not prompts:
        return []

    # Left-pad for batched causal-LM generation
    saved_pad_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(prompts, padding=True, return_tensors="pt", truncation=False).to(device)
    finally:
        tokenizer.padding_side = saved_pad_side
    prompt_len = inputs.input_ids.shape[1]

    out = model.generate(
        **inputs,
        max_new_tokens=cfg.max_response_tokens,
        temperature=cfg.temperature,
        do_sample=cfg.temperature > 0,
        top_p=0.95,
        top_k=50,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )
    sequences = out.sequences  # [K, prompt_len + n_new]
    response_block = sequences[:, prompt_len:]  # [K, n_new]
    n_new = response_block.shape[1]

    # Per-token logprobs of the SAMPLED tokens.
    # out.scores is a tuple of n_new tensors, each [K, V].
    # For each rollout i and step t, logprob = log_softmax(scores[t][i])[response_block[i, t]]
    eos_token_id = tokenizer.eos_token_id

    results = []
    for i in range(len(prompts)):
        # Find first EOS in response (truncate at it)
        row_ids = response_block[i].tolist()
        # Truncate at first EOS or pad token
        cut = n_new
        for t, tid in enumerate(row_ids):
            if tid == eos_token_id or (tokenizer.pad_token_id is not None and tid == tokenizer.pad_token_id and t > 0):
                cut = t + 1  # include the EOS
                break
        response_ids_i = row_ids[:cut]
        text_i = tokenizer.decode(response_ids_i, skip_special_tokens=True)

        # Extract logprobs for the kept tokens
        old_logps_i = []
        for t in range(cut):
            log_probs = F.log_softmax(out.scores[t][i], dim=-1)
            old_logps_i.append(log_probs[response_block[i, t]].item())
        results.append((text_i, response_ids_i, old_logps_i))
    return results


def do_rollouts_batched(
    model, tokenizer, env_template, system_prompt: str,
    puzzle_seeds: list, cfg: RLConfig, device: str, group_size: int,
) -> list:
    """Run K rollouts per puzzle in parallel, batching generation across all alive rollouts.

    Args:
        env_template: an env instance whose class/config we replicate K×N_puzzles times
        puzzle_seeds: list of N_puzzles seeds; each gets `group_size` rollouts (K)
        group_size: K rollouts per puzzle

    Returns:
        list of Rollout objects, len = group_size × len(puzzle_seeds), grouped by puzzle.
    """
    factory = _build_env_factory(env_template)

    # Initialize one env per (puzzle, k) slot.
    rollouts_state = []  # list of dicts: {env, history, alive, is_solved, steps, puzzle_seed}
    for seed in puzzle_seeds:
        for _k in range(group_size):
            env_i = factory()
            obs = env_i.reset(seed=seed)
            rollouts_state.append({
                "env": env_i,
                "obs": obs,
                "history": [],
                "alive": True,
                "is_solved": False,
                "steps": [],
                "puzzle_seed": seed,
            })

    for t in range(cfg.max_rollout_steps):
        alive_indices = [i for i, r in enumerate(rollouts_state) if r["alive"]]
        if not alive_indices:
            break

        # Build prompts for all alive rollouts
        prompts = [
            build_prompt(tokenizer, system_prompt, rollouts_state[i]["history"], rollouts_state[i]["obs"])
            for i in alive_indices
        ]

        # Batched generation
        gen_results = sample_responses_batched(model, tokenizer, prompts, cfg, device)

        for slot_idx, idx in enumerate(alive_indices):
            text, response_ids, old_logps = gen_results[slot_idx]
            r = rollouts_state[idx]
            prompt_text = prompts[slot_idx]

            pred = parse_solvable(text)
            fmt_r = format_reward(text, cfg.format_per_tag)
            action_str = _extract_answer(text)
            env_i = r["env"]

            # Pass the raw <answer> string to env.step — each env parses its own action format.
            # If extraction failed (no <answer> tag), fall back to a sentinel that the env will reject.
            if not action_str:
                calib_r = cfg.fn_reward
                r["steps"].append(StepRecord(
                    prompt_text=prompt_text, response_text=text, response_ids=response_ids,
                    action=None, pred_solvable=pred, gt_solvable=False, is_breaking_point=False,
                    step_reward=calib_r + fmt_r, old_logps=old_logps,
                    calib_reward=calib_r, fmt_reward=fmt_r, progress_reward=0.0,
                    action_was_valid=False,
                ))
                r["alive"] = False
                continue

            next_obs, _env_reward, done, info = env_i.step(action_str)
            action_was_valid = info.get("action_is_valid", True)
            gt_solvable = bool(info.get("is_solvable", True))
            is_bp = bool(info.get("is_breaking_point", False))
            is_solved = bool(info.get("success", False))
            calib_r = solvable_reward(pred, gt_solvable, cfg) if action_was_valid else cfg.fn_reward
            progress_r = cfg.progress_bonus_per_step if action_was_valid else 0.0

            r["steps"].append(StepRecord(
                prompt_text=prompt_text, response_text=text, response_ids=response_ids,
                action=action_str if action_was_valid else None,
                pred_solvable=pred, gt_solvable=gt_solvable, is_breaking_point=is_bp,
                step_reward=calib_r + fmt_r + progress_r,
                old_logps=old_logps,
                calib_reward=calib_r, fmt_reward=fmt_r, progress_reward=progress_r,
                action_was_valid=bool(action_was_valid),
            ))
            r["history"].append((r["obs"], text))
            r["obs"] = next_obs
            r["is_solved"] = r["is_solved"] or is_solved

            if done or not action_was_valid:
                r["alive"] = False

    # Collect Rollout objects
    rollouts = []
    for r in rollouts_state:
        final_reward = sum(s.step_reward for s in r["steps"])
        final_reward += cfg.success_bonus if r["is_solved"] else cfg.fail_bonus
        rollouts.append(Rollout(
            puzzle_seed=r["puzzle_seed"], steps=r["steps"],
            is_solved=r["is_solved"], final_reward=final_reward,
        ))
    return rollouts


def rebalance_rewards(rollouts: list, cfg: RLConfig) -> dict:
    """v7: rescale per-step calibration rewards by inverse GT-class frequency,
    then recompute step_reward and final_reward in place.

    Counts GT=True (solvable) and GT=False (doom) steps across all rollouts in the
    batch. Action-invalid steps (no env transition observed) are excluded — their
    `calib_reward` is the fixed worst-case `fn_reward` and not a true class signal.

    The weight formula `total / (2 * max(n, 1))` makes each class contribute
    equally in expectation; floor/cap prevent extreme weights when one class is
    very rare (which would otherwise let a single sample dominate the gradient).

    Returns a small metrics dict for logging.
    """
    if not cfg.class_balance:
        return {"class_balance_applied": False}

    n_solv = 0
    n_doom = 0
    for ro in rollouts:
        for s in ro.steps:
            if not s.action_was_valid:
                continue
            if s.gt_solvable:
                n_solv += 1
            else:
                n_doom += 1
    total = n_solv + n_doom
    if total == 0:
        return {"class_balance_applied": False, "n_solv": 0, "n_doom": 0}

    w_solv = total / (2.0 * max(n_solv, 1))
    w_doom = total / (2.0 * max(n_doom, 1))
    floor, cap = cfg.class_balance_floor, cfg.class_balance_cap
    w_solv = max(floor, min(cap, w_solv))
    w_doom = max(floor, min(cap, w_doom))

    for ro in rollouts:
        for s in ro.steps:
            if not s.action_was_valid:
                # Keep worst-case calib reward unscaled — these are not class signals.
                continue
            w = w_solv if s.gt_solvable else w_doom
            s.calib_reward = float(s.calib_reward) * w
            s.step_reward = s.calib_reward + s.fmt_reward + s.progress_reward
        ro.final_reward = sum(s.step_reward for s in ro.steps)
        ro.final_reward += cfg.success_bonus if ro.is_solved else cfg.fail_bonus

    return {
        "class_balance_applied": True,
        "n_solv": n_solv, "n_doom": n_doom,
        "w_solv": round(w_solv, 3), "w_doom": round(w_doom, 3),
    }


def _extract_answer(text: str) -> str:
    """Pull the contents of <answer>...</answer> for env-agnostic action passing."""
    import re as _re
    m = _re.search(r"<answer>\s*(.*?)\s*</answer>", text, _re.IGNORECASE | _re.DOTALL)
    return m.group(1) if m else text


# ----------------------------------------------------------------------------
# PPO update
# ----------------------------------------------------------------------------

def compute_response_logprobs(model, tokenizer, prompt_text: str, response_ids: list, device: str) -> torch.Tensor:
    """Forward pass; return per-token logprob for each token in response (length = len(response_ids))."""
    full_text = prompt_text + tokenizer.decode(response_ids, skip_special_tokens=False)
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    prompt_len = prompt_ids.shape[1]

    # Use full_ids for the forward pass; logits at position i predict token i+1.
    out = model(full_ids)
    logits = out.logits  # [1, T, V]

    # response tokens occupy positions [prompt_len, T)
    # to compute logprob for response token at position j, we need logits at position j-1
    # (since logits[i] predicts token at i+1 → token at j is predicted by logits[j-1])
    response_positions = list(range(prompt_len, full_ids.shape[1]))
    logp_per_token = []
    for j in response_positions:
        if j - 1 < 0 or j >= full_ids.shape[1]:
            continue
        log_probs = F.log_softmax(logits[0, j - 1], dim=-1)
        logp_per_token.append(log_probs[full_ids[0, j]])
    if not logp_per_token:
        return torch.zeros(0, device=device)
    return torch.stack(logp_per_token)


def grpo_advantages(rollouts: list, group_size: int) -> list:
    """Group-relative advantage: (reward_i - mean(group)) / (std(group) + eps).

    Rollouts are assumed grouped consecutively in groups of `group_size`.
    """
    advs = []
    for g_start in range(0, len(rollouts), group_size):
        group = rollouts[g_start:g_start + group_size]
        rewards = np.array([r.final_reward for r in group])
        baseline = rewards.mean()
        std = rewards.std() + 1e-8
        for r in rewards:
            advs.append(float((r - baseline) / std))
    return advs


def ppo_update(
    policy, ref_policy, optimizer, tokenizer,
    rollouts: list, advantages: list, cfg: RLConfig, device: str,
) -> dict:
    """One PPO update across all (rollout, step) tokens.

    For the smoke-test version, each step's response tokens get the rollout-level advantage broadcast.
    PPO clipped objective + KL-to-ref penalty.
    """
    policy.train()
    optimizer.zero_grad()

    total_pg_loss = 0.0
    total_kl_loss = 0.0
    total_clipfrac = 0.0
    total_tokens = 0
    total_via_kl = 0.0    # v8: weighted sum of viability-tag KL across all steps
    total_via_tokens = 0  # v8: total number of viability tokens seen

    for adv, ro in zip(advantages, rollouts):
        for step in ro.steps:
            if not step.response_ids:
                continue

            # Old logprobs: use cached values from rollout time (PPO bug fix).
            # Without this, old_logp recomputed under the same policy = new_logp, ratio=1 always.
            if step.old_logps:
                old_logp = torch.tensor(step.old_logps, device=device, dtype=torch.float32)
            else:
                # Fallback for rollouts without cached logps (e.g., legacy data)
                with torch.no_grad():
                    old_logp = compute_response_logprobs(policy, tokenizer, step.prompt_text, step.response_ids, device)

            with torch.no_grad():
                ref_logp = compute_response_logprobs(ref_policy, tokenizer, step.prompt_text, step.response_ids, device)

            # New logprobs (with grad) — recomputed each PPO epoch as the policy updates
            new_logp = compute_response_logprobs(policy, tokenizer, step.prompt_text, step.response_ids, device)

            if new_logp.numel() == 0:
                continue

            # Length-mismatch guard: cached old_logps and freshly computed new_logp/ref_logp
            # should align by token count. If tokenization round-trips lose tokens, truncate.
            min_len = min(old_logp.numel(), new_logp.numel(), ref_logp.numel())
            if min_len == 0:
                continue
            old_logp = old_logp[:min_len]
            new_logp = new_logp[:min_len]
            ref_logp = ref_logp[:min_len]

            ratio = torch.exp(new_logp - old_logp)
            adv_t = torch.full_like(new_logp, adv)
            unclipped = ratio * adv_t
            clipped = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_t
            pg_loss = -torch.min(unclipped, clipped).mean()

            kl = (new_logp - ref_logp).pow(2).mean()  # squared logprob deviation as proxy KL
            loss = pg_loss + cfg.kl_coef * kl

            # v8: auxiliary KL anchor on <viability>/<solvable> tag content tokens
            # against the ref_policy. Targets the dynamic calibration drift observed
            # in the v6/v7 collapse — keeps the calibration tag locked at SFT quality
            # while leaving action tokens free to optimize.
            via_kl_step = 0.0
            n_via = 0
            if cfg.viability_kl_coef > 0.0:
                via_idx = find_viability_token_positions(
                    tokenizer, step.response_text, step.response_ids[:min_len]
                )
                if via_idx:
                    idx_t = torch.tensor(via_idx, device=device, dtype=torch.long)
                    new_via = new_logp.index_select(0, idx_t)
                    ref_via = ref_logp.index_select(0, idx_t)
                    via_kl = (new_via - ref_via).pow(2).mean()
                    loss = loss + cfg.viability_kl_coef * via_kl
                    via_kl_step = float(via_kl.item())
                    n_via = len(via_idx)

            (loss / max(1, len(rollouts))).backward()

            with torch.no_grad():
                clipfrac = ((ratio - 1.0).abs() > cfg.clip_eps).float().mean().item()
                total_pg_loss += pg_loss.item() * new_logp.numel()
                total_kl_loss += kl.item() * new_logp.numel()
                total_clipfrac += clipfrac * new_logp.numel()
                total_tokens += new_logp.numel()
                total_via_kl += via_kl_step * max(1, n_via)
                total_via_tokens += n_via

    if total_tokens == 0:
        return {"pg_loss": 0.0, "kl": 0.0, "clipfrac": 0.0, "n_tokens": 0,
                "via_kl": 0.0, "n_via_tokens": 0}

    torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()

    return {
        "pg_loss": total_pg_loss / total_tokens,
        "kl": total_kl_loss / total_tokens,
        "clipfrac": total_clipfrac / total_tokens,
        "n_tokens": total_tokens,
        "via_kl": (total_via_kl / total_via_tokens) if total_via_tokens > 0 else 0.0,
        "n_via_tokens": total_via_tokens,
    }


# ----------------------------------------------------------------------------
# Eval (cheap, in-loop)
# ----------------------------------------------------------------------------

@torch.no_grad()
def quick_pass1(policy, tokenizer, env, system_prompt: str, cfg: RLConfig, device: str, n_puzzles: int) -> dict:
    """Quick Pass@1 check: greedy rollouts on `n_puzzles` distinct puzzles."""
    n_solved = 0
    n_total = 0
    n_bp_caught = 0
    n_bp_total = 0
    sol_correct = 0
    sol_total = 0
    for i in range(n_puzzles):
        seed = 100000 + i  # distinct from training seeds
        old_temp = cfg.temperature
        cfg.temperature = 0.0  # greedy
        ro = do_rollout(policy, tokenizer, env, system_prompt, seed, cfg, device)
        cfg.temperature = old_temp
        n_total += 1
        if ro.is_solved:
            n_solved += 1
        for s in ro.steps:
            if s.pred_solvable is not None:
                sol_total += 1
                if s.pred_solvable == s.gt_solvable:
                    sol_correct += 1
            if s.is_breaking_point:
                n_bp_total += 1
                if s.pred_solvable is False:
                    n_bp_caught += 1
    return {
        "pass@1": n_solved / max(1, n_total),
        "n_solved": n_solved,
        "n_total": n_total,
        "solvable_acc": sol_correct / max(1, sol_total),
        "bp_recall": n_bp_caught / max(1, n_bp_total) if n_bp_total > 0 else 0.0,
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="sudoku", choices=["sudoku", "polyomino"],
                   help="Which env to train against")
    p.add_argument("--sft-checkpoint", required=True)
    p.add_argument("--output-dir", default="outputs/rl_b5_phase1")
    p.add_argument("--n-total-steps", type=int, default=50)  # smoke-test default
    # Sudoku-specific
    p.add_argument("--grid-size", type=int, default=4)
    p.add_argument("--difficulty", default="easy")
    # Polyomino-specific
    p.add_argument("--board-h", type=int, default=5)
    p.add_argument("--board-w", type=int, default=4)
    p.add_argument("--piece-set", type=str, default="L,P,W,Y")
    # Common
    p.add_argument("--n-puzzles-per-batch", type=int, default=4)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--kl-coef", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=25)
    # v7/v8 reward shapes
    p.add_argument("--reward-version", default="v6", choices=["v6", "v7", "v8"],
                   help="v6 = per-class asymmetric (Sudoku-tuned); "
                        "v7 = symmetric magnitudes + class-balanced + progress bonus "
                        "(intermediate fix for B-7 collapse); "
                        "v8 = v7 + auxiliary KL anchor on <viability>/<solvable> tag tokens "
                        "against ref policy (fixes the dynamic calibration drift).")
    p.add_argument("--progress-bonus", type=float, default=None,
                   help="Override progress bonus per valid step. Defaults: v6=0.0, v7/v8=0.1.")
    p.add_argument("--class-balance-cap", type=float, default=5.0,
                   help="Cap on inverse-frequency class weight (v7/v8).")
    p.add_argument("--viability-kl-coef", type=float, default=None,
                   help="Coefficient on the per-step viability-tag KL anchor. "
                        "Defaults: v6/v7=0.0, v8=0.5.")
    args = p.parse_args()

    cfg = RLConfig(
        sft_checkpoint=args.sft_checkpoint,
        output_dir=args.output_dir,
        n_total_steps=args.n_total_steps,
        grid_size=args.grid_size,
        difficulty=args.difficulty,
        n_puzzles_per_batch=args.n_puzzles_per_batch,
        group_size=args.group_size,
        learning_rate=args.learning_rate,
        kl_coef=args.kl_coef,
        seed=args.seed,
        eval_every=args.eval_every,
        reward_version=args.reward_version,
        class_balance_cap=args.class_balance_cap,
    )

    # Apply v7/v8 reward defaults (per module docstring).
    # v7: symmetric magnitudes + class balance + progress bonus.
    # v8: v7 + viability-tag KL anchor (default coef 0.5).
    if cfg.reward_version in ("v7", "v8"):
        cfg.tp_reward = 1.0
        cfg.fn_reward = -1.0
        cfg.fp_reward = -1.0
        cfg.tn_reward = 1.0
        cfg.class_balance = True
        cfg.progress_bonus_per_step = 0.1 if args.progress_bonus is None else args.progress_bonus
    elif args.progress_bonus is not None:
        cfg.progress_bonus_per_step = args.progress_bonus

    if cfg.reward_version == "v8":
        cfg.viability_kl_coef = 0.5 if args.viability_kl_coef is None else args.viability_kl_coef
    elif args.viability_kl_coef is not None:
        cfg.viability_kl_coef = args.viability_kl_coef

    # Setup
    os.makedirs(cfg.output_dir, exist_ok=True)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"=== rl_trainer_v6 — Phase 1 (truncation={cfg.truncation_mode}) ===")
    print(f"  config: {json.dumps(asdict(cfg), indent=2)}")

    # Tokenizer
    print(f"\nLoading tokenizer: {cfg.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Models — policy (trainable) + ref (frozen B-5)
    print(f"\nLoading policy from: {cfg.sft_checkpoint}")
    policy = AutoModelForCausalLM.from_pretrained(
        cfg.sft_checkpoint,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    policy.train()

    print(f"Loading frozen reference from: {cfg.sft_checkpoint}")
    ref_policy = AutoModelForCausalLM.from_pretrained(
        cfg.sft_checkpoint,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    ref_policy.eval()
    for p_ in ref_policy.parameters():
        p_.requires_grad_(False)

    # Env + system prompt — env-specific
    if args.env == "polyomino":
        from src.environments.polyomino import PolyominoEnv
        piece_set = tuple(p_.strip().upper() for p_ in args.piece_set.split(","))
        env = PolyominoEnv(
            board_h=args.board_h, board_w=args.board_w,
            piece_set=piece_set, max_steps=cfg.max_rollout_steps,
        )
        formatter = SFTFormatter(variant="polyomino_minimal")
    else:
        env = SudokuEnv(grid_size=cfg.grid_size, difficulty=cfg.difficulty, max_steps=cfg.max_rollout_steps)
        formatter = SFTFormatter(variant="sudoku_minimal")
    system_prompt = formatter.system_prompt

    # Optimizer
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.learning_rate)

    # Initial eval
    print(f"\n=== initial Pass@1 (greedy, n={cfg.eval_n_puzzles}) ===")
    init_eval = quick_pass1(policy, tokenizer, env, system_prompt, cfg, device, cfg.eval_n_puzzles)
    print(f"  {init_eval}")
    log_path = os.path.join(cfg.output_dir, "rl_log.jsonl")
    with open(log_path, "w") as f:
        f.write(json.dumps({"step": 0, "phase": "init_eval", **init_eval}) + "\n")

    # Training loop
    for step in range(1, cfg.n_total_steps + 1):
        t0 = time.time()
        # Sample N_puzzles puzzles, K rollouts each, all batched through model.generate.
        # do_rollouts_batched returns rollouts grouped by puzzle (len = N × K, contiguous groups).
        puzzle_seeds = [random.randint(0, 2**31 - 1) for _ in range(cfg.n_puzzles_per_batch)]
        rollouts = do_rollouts_batched(
            policy, tokenizer, env, system_prompt,
            puzzle_seeds, cfg, device, group_size=cfg.group_size,
        )

        rollout_time = time.time() - t0

        # v7: rescale per-step calibration rewards by inverse GT-class frequency
        # before computing GRPO advantages. No-op when cfg.class_balance is False.
        cb_metrics = rebalance_rewards(rollouts, cfg)

        rewards = np.array([r.final_reward for r in rollouts])
        solved_rate = sum(1 for r in rollouts if r.is_solved) / len(rollouts)

        advs = grpo_advantages(rollouts, cfg.group_size)

        # PPO update (multiple epochs over the same rollouts)
        ppo_metrics = {}
        for _ in range(cfg.ppo_epochs):
            ppo_metrics = ppo_update(policy, ref_policy, optimizer, tokenizer, rollouts, advs, cfg, device)

        elapsed = time.time() - t0
        log = {
            "step": step,
            "rollout_time_s": round(rollout_time, 1),
            "step_time_s": round(elapsed, 1),
            "n_rollouts": len(rollouts),
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std()),
            "solved_rate": float(solved_rate),
            "adv_min": float(min(advs)),
            "adv_max": float(max(advs)),
            **cb_metrics,
            **ppo_metrics,
        }
        via_kl_str = (f" | via_kl {log.get('via_kl', 0):.4f}"
                      if cfg.viability_kl_coef > 0 else "")
        print(f"step {step:4d} | reward {log['reward_mean']:+.2f}±{log['reward_std']:.2f} | "
              f"solved {solved_rate*100:.0f}% | pg_loss {log.get('pg_loss', 0):+.3f} | "
              f"kl {log.get('kl', 0):.4f} | clipfrac {log.get('clipfrac', 0):.2f}"
              f"{via_kl_str} | step_t {elapsed:.0f}s")
        with open(log_path, "a") as f:
            f.write(json.dumps(log) + "\n")

        # Periodic eval
        if step % cfg.eval_every == 0:
            print(f"\n=== eval at step {step} (Pass@1 greedy, n={cfg.eval_n_puzzles}) ===")
            ev = quick_pass1(policy, tokenizer, env, system_prompt, cfg, device, cfg.eval_n_puzzles)
            print(f"  {ev}")
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step, "phase": "eval", **ev}) + "\n")

        # Periodic save
        if step % cfg.save_every == 0:
            ckpt = os.path.join(cfg.output_dir, f"checkpoint-{step}")
            print(f"\n=== saving checkpoint to {ckpt} ===")
            policy.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)

    # Final save
    final_dir = os.path.join(cfg.output_dir, "final")
    print(f"\n=== final save to {final_dir} ===")
    policy.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("done.")


if __name__ == "__main__":
    main()
