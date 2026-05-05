"""Q4 Hidato candidate-selection methods (paper §3.5 / Table 4).

Each Method exposes a single API:

    method.choose_action(grid: List[List[int]]) -> ChoiceResult

ChoiceResult contains the chosen action (or terminate=True), plus token
accounting for NetCompute. Methods share π_θ K-sampling via PolicyClient
(below) so the candidate budget is matched across methods that need it.

MVP set (this file):
  - PolicyTop1Method     : K=1 greedy from π_θ
  - BestOfKMethod        : K samples, argmax generation_logprob
  - LocalProgressMethod  : K samples, argmax compute_progress(next_state)
  - SAVEMethod           : K samples, filter by f_φ(s,a) ≥ τ_keep,
                           tie-break by π_θ logprob; fallback to argmax(p);
                           TERMINATE if max(p) < τ_fb
  - OracleMethod         : K samples, filter by oracle v(T(s,a))=1;
                           tie-break by π_θ logprob

Deferred to v2 (still in this file, easy to add):
  - PromptedScoreOnlyMethod
  - LearnedProgressScoreMethod
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from hidato5x4_env import (  # noqa: E402
    render_state_hidato, parse_action_text, is_local_valid, apply_action,
    enumerate_legal_actions, ActionStruct,
)
from hidato5x4_solver import Hidato5x4Solver  # noqa: E402
from progress_hidato5x4 import compute_progress  # noqa: E402

from q4_episode import ChoiceResult  # noqa: E402


# π_θ system prompt — must match the prompt rl_b5 was trained under.
# Verified against scripts/sanity_check_rl_b5_under_corrected_prompt.py
POLICY_SYSTEM_PROMPT = """You are solving a Hidato (number-path) puzzle. The board is a rectangular grid where you must fill in numbers from 1 to N (where N = rows × cols) so that consecutive numbers (k and k+1) are placed in cells that share an edge (orthogonally adjacent — up, down, left, or right).

Grid format: each cell shows its placed number, or '.' for empty.

You place numbers in sequential order (1, then 2, then 3, ...). Some cells are pre-filled (givens) and don't need to be placed; the env will skip past them. Each step you place the next required number into an empty cell that's adjacent to the previous number's cell.

In your reasoning:
1. Describe the current state in <observation>
2. Predict the next state after your move in <prediction>
3. Assess whether the resulting state will still be solvable (all remaining numbers can be placed legally) in <solvable>: true/false

Then provide your action in <answer> using format: place {N} at row {R} col {C}
where {N} is the next sequential number to place and (R, C) are 1-indexed cell coordinates."""


import re
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
VIAB_RE = re.compile(r"<viability>\s*(true|false)\s*</viability>", re.IGNORECASE)


# -----------------------------------------------------------------------------
# Policy client — K-sampling from π_θ
# -----------------------------------------------------------------------------


@dataclass
class Candidate:
    action: ActionStruct
    raw_text: str
    generation_logprob: float
    n_tokens: int           # generated tokens for this single candidate


class PolicyClient:
    """Wraps π_θ with K-sample + greedy interfaces."""

    def __init__(self, model, tokenizer, device="cuda",
                 temperature: float = 1.0, top_p: float = 0.95,
                 max_new_tokens: int = 256):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

    def _build_prompt(self, grid):
        from policy_sampler import build_chat_prompt  # type: ignore
        return build_chat_prompt(
            self.tokenizer,
            POLICY_SYSTEM_PROMPT,
            f"Current state:\n{render_state_hidato(grid)}",
        )

    def _parse_answer(self, raw_text: str) -> Optional[ActionStruct]:
        m = ANSWER_RE.search(raw_text)
        if not m:
            return None
        return parse_action_text(m.group(1).strip())

    @torch.no_grad()
    def sample_k(self, grid, K: int, dedup: bool = True) -> Tuple[List[Candidate], int]:
        """Sample K candidates from π_θ. Returns (candidates, total_policy_tokens).

        Keeps only candidates whose <answer> parses AND is locally valid at `grid`.
        If dedup=True, dedupes by canonical action string (R{r}C{c}={v}).
        """
        prompt = self._build_prompt(grid)
        from policy_sampler import batched_sample  # type: ignore
        samples = batched_sample(
            self.model, self.tokenizer, prompt, K=K,
            temperature=self.temperature, top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
        )
        cands: List[Candidate] = []
        seen_canon = set()
        total_tokens = 0
        for s in samples:
            n_tok = len(self.tokenizer.encode(s.text, add_special_tokens=False))
            total_tokens += n_tok
            action = self._parse_answer(s.text)
            if action is None or not is_local_valid(grid, action):
                continue
            if dedup:
                canon = f"R{action.row}C{action.col}={action.value}"
                if canon in seen_canon:
                    continue
                seen_canon.add(canon)
            cands.append(Candidate(
                action=action, raw_text=s.text,
                generation_logprob=s.generation_logprob, n_tokens=n_tok,
            ))
        return cands, total_tokens

    @torch.no_grad()
    def greedy_one(self, grid) -> Tuple[Optional[Candidate], int]:
        """Greedy single decode. Returns (candidate_or_None, policy_tokens)."""
        prompt = self._build_prompt(grid)
        from policy_sampler import batched_sample  # type: ignore
        samples = batched_sample(
            self.model, self.tokenizer, prompt, K=1,
            temperature=0.0, top_p=1.0, max_new_tokens=self.max_new_tokens,
        )
        if not samples:
            return None, 0
        s = samples[0]
        n_tok = len(self.tokenizer.encode(s.text, add_special_tokens=False))
        action = self._parse_answer(s.text)
        if action is None or not is_local_valid(grid, action):
            return None, n_tok
        return Candidate(
            action=action, raw_text=s.text,
            generation_logprob=s.generation_logprob, n_tokens=n_tok,
        ), n_tok

    @torch.no_grad()
    def eval_logprob(self, grid, action: ActionStruct) -> float:
        """log π_θ(answer | state) for a specific candidate. Used for CVCP tie-break.

        Constructs minimal answer text "<answer>place V at row R col C</answer>".
        We do NOT include <think>...</think> here — just the answer tokens —
        which is a slight simplification of paper Algorithm 2 line 5. For
        Sudoku 4×4 this is fine because the answer is the only step-relevant
        token.
        """
        from policy_sampler import policy_eval_logprob  # type: ignore
        prompt = self._build_prompt(grid)
        # Minimal response: the model's actual output would have <observation>
        # etc. — for tie-breaking purposes the answer string alone is what
        # determines which action is chosen. Higher = more preferred by π_θ.
        from hidato5x4_env import action_text as _action_text
        response = f"<answer>{_action_text(action)}</answer>"
        return policy_eval_logprob(self.model, self.tokenizer, prompt, response)


# -----------------------------------------------------------------------------
# SAVE scorer — generate-then-read v̂_φ(s, a)
# -----------------------------------------------------------------------------


SAVE_SYSTEM_PROMPT = (
    "You are a viability scorer for Hidato (number-path) puzzles on a 5×4 grid. "
    "Given a current state and a proposed action (a number placement), predict "
    "the next state, whether the resulting puzzle still has a valid completion "
    "(viable), and whether the current state itself is viable."
)

SAVE_USER_TEMPLATE = (
    "Current state:\n"
    "{state_text}\n"
    "\n"
    "Proposed action: {action_text}\n"
    "\n"
    "Predict the next state, whether the next state is viable, and whether "
    "the current state is viable. Respond in the following format:\n"
    "<next_state>...</next_state>\n"
    "<viability>true|false</viability>\n"
    "<state_viable>true|false</state_viable>"
)


class SaveScorer:
    """Compute v̂_φ(s, a) on a trained f_φ checkpoint.

    Implementation: env-computed-prefix scoring (option B' in our design notes).
    Since the Sudoku transition is deterministic and computable, at deployment
    we render the FULL response prefix with env-computed next_state,
    teacher-force it through f_φ in a single forward pass, and read the
    logit at the <viability> value slot:
        v̂_φ = σ((logit_true - logit_false) / T)

    This matches what was measured in Q2 (AUC = 0.91) and what calibration was
    fit against. The alternative — generate-then-read — suffers from
    autoregressive cascade through the model's self-generated (often wrong)
    next_state, collapsing into overconfident scores ≈ 1.0 regardless of
    oracle viability (verified empirically on toy data).
    """

    def __init__(self, model, tokenizer, temperature: float = 1.0,
                 device: str = "cuda", max_new_tokens: int = 200):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.device = device
        self.max_new_tokens = max_new_tokens  # kept for API compatibility; unused

        true_ids = tokenizer.encode("true", add_special_tokens=False)
        false_ids = tokenizer.encode("false", add_special_tokens=False)
        if len(true_ids) != 1 or len(false_ids) != 1:
            raise RuntimeError(f"Need single-token true/false; got {true_ids}/{false_ids}")
        self.true_id = true_ids[0]
        self.false_id = false_ids[0]

    def _build_prompt(self, grid, action: ActionStruct):
        from hidato5x4_env import action_text as _action_text
        state_text = render_state_hidato(grid)
        msgs = [
            {"role": "system", "content": SAVE_SYSTEM_PROMPT},
            {"role": "user", "content": SAVE_USER_TEMPLATE.format(
                state_text=state_text,
                action_text=_action_text(action),
            )},
        ]
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    @torch.no_grad()
    def score(self, grid, action: ActionStruct) -> Tuple[float, int]:
        """Returns (calibrated_prob_viable, tokens_consumed). prob ∈ [0, 1].

        Single-forward teacher-forced scoring with env-computed next_state.
        """
        next_grid = apply_action(grid, action)
        next_state_text = render_state_hidato(next_grid)

        # Prompt: same as training
        prompt = self._build_prompt(grid, action)
        # Response prefix: full <next_state>...</next_state>\n<viability>
        # We stop right at the position where the value token is predicted.
        prefix = (
            f"<next_state>\n{next_state_text}\n</next_state>\n<viability>"
        )

        full_text = prompt + prefix
        full_ids = self.tokenizer(
            full_text, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.device)

        # Forward once
        out = self.model(full_ids)
        logits = out.logits  # [1, T, V]

        # The model's prediction for the next token (the viab value) is at
        # logits[0, -1, :] — last position predicts the next token after the prefix.
        last_logits = logits[0, -1, :]

        ell = (last_logits[self.true_id].item()
               - last_logits[self.false_id].item())
        prob = 1.0 / (1.0 + math.exp(-ell / self.temperature))
        # "tokens consumed" = the prefix length we paid for
        n_tokens = full_ids.shape[1]
        return prob, n_tokens


# -----------------------------------------------------------------------------
# Method classes
# -----------------------------------------------------------------------------


class Method:
    name: str = "Method"
    needs_save_scorer: bool = False

    def choose_action(self, grid) -> ChoiceResult:
        raise NotImplementedError


class PolicyTop1Method(Method):
    name = "policy_top1"

    def __init__(self, policy: PolicyClient):
        self.policy = policy

    def choose_action(self, grid):
        cand, n_tok = self.policy.greedy_one(grid)
        if cand is None:
            return ChoiceResult(action=None, policy_tokens=n_tok, info={"reason": "parse_or_illegal"})
        return ChoiceResult(action=cand.action, policy_tokens=n_tok)


class BestOfKMethod(Method):
    name = "best_of_k"

    def __init__(self, policy: PolicyClient, K: int = 8):
        self.policy = policy
        self.K = K

    def choose_action(self, grid):
        cands, total_tokens = self.policy.sample_k(grid, K=self.K)
        if not cands:
            return ChoiceResult(action=None, policy_tokens=total_tokens, info={"reason": "no_valid_candidates"})
        best = max(cands, key=lambda c: c.generation_logprob)
        return ChoiceResult(action=best.action, policy_tokens=total_tokens)


class LocalProgressMethod(Method):
    name = "local_progress"

    def __init__(self, policy: PolicyClient, K: int = 8):
        self.policy = policy
        self.K = K

    def choose_action(self, grid):
        cands, total_tokens = self.policy.sample_k(grid, K=self.K)
        if not cands:
            return ChoiceResult(action=None, policy_tokens=total_tokens, info={"reason": "no_valid_candidates"})
        scored = []
        for c in cands:
            next_grid = apply_action(grid, c.action)
            score = compute_progress(next_grid)["local_progress_score"]
            scored.append((c, score))
        # Tie-break: highest progress; if tied, highest gen_logprob
        best = max(scored, key=lambda cs: (cs[1], cs[0].generation_logprob))
        return ChoiceResult(action=best[0].action, policy_tokens=total_tokens)


class SAVEMethod(Method):
    """CVCP — paper Algorithm 2."""
    name = "save"
    needs_save_scorer = True

    def __init__(self, policy: PolicyClient, scorer: SaveScorer,
                 tau_keep: float, tau_fb: float = 0.0, K: int = 8):
        self.policy = policy
        self.scorer = scorer
        self.tau_keep = tau_keep
        self.tau_fb = tau_fb
        self.K = K

    def choose_action(self, grid):
        cands, policy_tokens = self.policy.sample_k(grid, K=self.K)
        if not cands:
            return ChoiceResult(action=None, policy_tokens=policy_tokens, info={"reason": "no_valid_candidates"})

        # Score each candidate
        scored = []
        eval_tokens_total = 0
        for c in cands:
            p, n_tok = self.scorer.score(grid, c.action)
            eval_tokens_total += n_tok
            scored.append((c, p))

        # Filter
        kept = [(c, p) for c, p in scored if p >= self.tau_keep]
        if kept:
            # tie-break by π_θ logprob (using sampling logprob; eval_logprob would
            # require another forward pass per candidate — skipping for cost)
            best = max(kept, key=lambda cp: cp[0].generation_logprob)
            return ChoiceResult(
                action=best[0].action, policy_tokens=policy_tokens,
                eval_tokens=eval_tokens_total,
                info={"path": "main", "n_kept": len(kept), "best_p": best[1]},
            )

        # Empty kept set → fallback or terminate
        max_p = max(p for _, p in scored)
        if max_p >= self.tau_fb:
            best = max(scored, key=lambda cp: cp[1])
            return ChoiceResult(
                action=best[0].action, policy_tokens=policy_tokens,
                eval_tokens=eval_tokens_total,
                info={"path": "fallback", "max_p": max_p},
            )
        return ChoiceResult(
            action=None, terminate=True, policy_tokens=policy_tokens,
            eval_tokens=eval_tokens_total,
            info={"path": "terminate", "max_p": max_p},
        )


# -----------------------------------------------------------------------------
# Prompted score-only (zero-shot base model utility scorer)
# -----------------------------------------------------------------------------


PROMPTED_SYSTEM = (
    "You are evaluating Sudoku 4×4 actions. Each row, column, and 2×2 box "
    "must contain digits 1–4 exactly once. Given a current state and a "
    "proposed placement, output a probability between 0 and 1 that this "
    "specific action leads to eventually solving the puzzle. Higher = "
    "more likely to succeed. Respond with ONLY a numeric probability."
)
PROMPTED_USER = (
    "Current state:\n{state_text}\n\n"
    "Proposed action: {action_text}\n\n"
    "Probability of solving:"
)
SCORE_RE = re.compile(r"\b(0(?:\.\d+)?|1(?:\.0+)?|\.\d+)\b")


class PromptedScorer:
    """Zero-shot prompted utility scorer using a base LLM (untrained)."""

    def __init__(self, model, tokenizer, device="cuda", max_new_tokens=12):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def score(self, grid, action):
        from hidato5x4_env import action_text as _at
        msgs = [
            {"role": "system", "content": PROMPTED_SYSTEM},
            {"role": "user", "content": PROMPTED_USER.format(
                state_text=render_state_hidato(grid), action_text=_at(action))},
        ]
        prompt = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = ids.input_ids.shape[1]
        eos = self.tokenizer.eos_token_id
        im_end = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        eos_used = im_end[0] if len(im_end) == 1 else eos
        out = self.model.generate(
            **ids, max_new_tokens=self.max_new_tokens, do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id or eos, eos_token_id=eos_used,
        )
        gen_ids = out[0, prompt_len:]
        n_tokens = gen_ids.shape[0]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        m = SCORE_RE.search(text)
        if not m:
            return 0.5, n_tokens
        try:
            v = float(m.group(1))
            if 0.0 <= v <= 1.0:
                return v, n_tokens
        except ValueError:
            pass
        return 0.5, n_tokens


class PromptedScoreOnlyMethod(Method):
    """Score K policy candidates with a zero-shot prompted base LLM."""
    name = "prompted_score_only"

    def __init__(self, policy: PolicyClient, scorer: PromptedScorer, K: int = 8):
        self.policy = policy
        self.scorer = scorer
        self.K = K

    def choose_action(self, grid):
        cands, policy_tokens = self.policy.sample_k(grid, K=self.K)
        if not cands:
            return ChoiceResult(action=None, policy_tokens=policy_tokens,
                                info={"reason": "no_valid_candidates"})
        eval_tokens_total = 0
        scored = []
        for c in cands:
            s, n_tok = self.scorer.score(grid, c.action)
            eval_tokens_total += int(n_tok)
            scored.append((c, s))
        # Tie-break: highest score; tied → highest gen_logprob
        best = max(scored, key=lambda cs: (cs[1], cs[0].generation_logprob))
        return ChoiceResult(action=best[0].action, policy_tokens=policy_tokens,
                            eval_tokens=eval_tokens_total)


# -----------------------------------------------------------------------------
# Learned progress-score (g_ψ trained to output <progress>X.XXXX</progress>)
# -----------------------------------------------------------------------------


PROGRESS_SYSTEM = (
    "You are a Sudoku 4×4 progress scorer. Given a current state and a "
    "proposed action, predict the local progress score of the resulting "
    "state. Higher progress means a more advanced board (more cells filled, "
    "fewer constraint violations). Respond with the score only."
)
PROGRESS_USER = (
    "Current state:\n{state_text}\n\nProposed action: {action_text}\n\n"
    "Predict the local progress score of the next state. Respond in the "
    "following format:\n<progress>NUMBER</progress>"
)
PROGRESS_PARSE_RE = re.compile(r"<progress>\s*(?P<v>-?\d+\.?\d*)\s*</progress>")


class LearnedProgressScorer:
    """Wraps a trained g_ψ that outputs <progress>X.XXXX</progress>."""

    def __init__(self, model, tokenizer, device="cuda", max_new_tokens=24):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def score(self, grid, action):
        from hidato5x4_env import action_text as _at
        msgs = [
            {"role": "system", "content": PROGRESS_SYSTEM},
            {"role": "user", "content": PROGRESS_USER.format(
                state_text=render_state_hidato(grid), action_text=_at(action))},
        ]
        prompt = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = ids.input_ids.shape[1]
        eos = self.tokenizer.eos_token_id
        im_end = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        eos_used = im_end[0] if len(im_end) == 1 else eos
        out = self.model.generate(
            **ids, max_new_tokens=self.max_new_tokens, do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id or eos, eos_token_id=eos_used,
        )
        gen_ids = out[0, prompt_len:]
        n_tokens = gen_ids.shape[0]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        m = PROGRESS_PARSE_RE.search(text)
        if not m:
            return 0.0, n_tokens
        try:
            return float(m.group("v")), n_tokens
        except ValueError:
            return 0.0, n_tokens


class LearnedProgressScoreMethod(Method):
    name = "learned_progress_score"

    def __init__(self, policy: PolicyClient, scorer: LearnedProgressScorer, K: int = 8):
        self.policy = policy
        self.scorer = scorer
        self.K = K

    def choose_action(self, grid):
        cands, policy_tokens = self.policy.sample_k(grid, K=self.K)
        if not cands:
            return ChoiceResult(action=None, policy_tokens=policy_tokens,
                                info={"reason": "no_valid_candidates"})
        eval_tokens_total = 0
        scored = []
        for c in cands:
            s, n_tok = self.scorer.score(grid, c.action)
            eval_tokens_total += int(n_tok)
            scored.append((c, s))
        best = max(scored, key=lambda cs: (cs[1], cs[0].generation_logprob))
        return ChoiceResult(action=best[0].action, policy_tokens=policy_tokens,
                            eval_tokens=eval_tokens_total)


# -----------------------------------------------------------------------------
# Q5 termination methods (paper Table 5)
# -----------------------------------------------------------------------------


class NoTerminationMethod(Method):
    """SAVE-style scoring + filter, but NEVER terminates. Equivalent to
    SAVEMethod with τ_fb=0 (always falls back to argmax viability).
    Used as the Q5 'No termination' baseline."""
    name = "no_termination"
    needs_save_scorer = True

    def __init__(self, policy: PolicyClient, scorer: SaveScorer,
                 tau_keep: float, K: int = 8):
        self.policy = policy
        self.scorer = scorer
        self.tau_keep = tau_keep
        self.K = K

    def choose_action(self, grid):
        cands, policy_tokens = self.policy.sample_k(grid, K=self.K)
        if not cands:
            return ChoiceResult(action=None, policy_tokens=policy_tokens, info={"reason": "no_valid_candidates"})
        scored = []
        eval_tokens_total = 0
        for c in cands:
            p, n_tok = self.scorer.score(grid, c.action)
            eval_tokens_total += n_tok
            scored.append((c, p))
        kept = [(c, p) for c, p in scored if p >= self.tau_keep]
        if kept:
            best = max(kept, key=lambda cp: cp[0].generation_logprob)
            return ChoiceResult(action=best[0].action, policy_tokens=policy_tokens,
                                eval_tokens=eval_tokens_total, info={"path": "main"})
        # No-term: always fallback to argmax viability
        best = max(scored, key=lambda cp: cp[1])
        return ChoiceResult(action=best[0].action, policy_tokens=policy_tokens,
                            eval_tokens=eval_tokens_total, info={"path": "fallback_no_term"})


class GreedyTerminationMethod(Method):
    """Aggressive termination: τ_fb_greedy is set close to τ_keep so
    we terminate often even if some candidates have moderate viability.
    Implementation = SAVEMethod with τ_fb = τ_keep_fraction × τ_keep
    (default 0.9 × τ_keep)."""
    name = "greedy_termination"
    needs_save_scorer = True

    def __init__(self, policy: PolicyClient, scorer: SaveScorer,
                 tau_keep: float, K: int = 8, fraction: float = 0.9):
        self.policy = policy
        self.scorer = scorer
        self.tau_keep = tau_keep
        self.tau_fb = fraction * tau_keep
        self.K = K

    def choose_action(self, grid):
        cands, policy_tokens = self.policy.sample_k(grid, K=self.K)
        if not cands:
            return ChoiceResult(action=None, policy_tokens=policy_tokens, info={"reason": "no_valid_candidates"})
        scored = []
        eval_tokens_total = 0
        for c in cands:
            p, n_tok = self.scorer.score(grid, c.action)
            eval_tokens_total += n_tok
            scored.append((c, p))
        kept = [(c, p) for c, p in scored if p >= self.tau_keep]
        if kept:
            best = max(kept, key=lambda cp: cp[0].generation_logprob)
            return ChoiceResult(action=best[0].action, policy_tokens=policy_tokens,
                                eval_tokens=eval_tokens_total, info={"path": "main"})
        max_p = max(p for _, p in scored)
        if max_p >= self.tau_fb:
            best = max(scored, key=lambda cp: cp[1])
            return ChoiceResult(action=best[0].action, policy_tokens=policy_tokens,
                                eval_tokens=eval_tokens_total, info={"path": "fallback", "max_p": max_p})
        return ChoiceResult(action=None, terminate=True, policy_tokens=policy_tokens,
                            eval_tokens=eval_tokens_total, info={"path": "terminate_greedy", "max_p": max_p})


class SAVERetryMethod(Method):
    """Like SAVEMethod, but if the kept set is empty AND max(p) < τ_fb,
    sample K more candidates and retry once. Only terminate if both
    candidate sets fail."""
    name = "save_retry"
    needs_save_scorer = True

    def __init__(self, policy: PolicyClient, scorer: SaveScorer,
                 tau_keep: float, tau_fb: float, K: int = 8):
        self.policy = policy
        self.scorer = scorer
        self.tau_keep = tau_keep
        self.tau_fb = tau_fb
        self.K = K

    def _score_one_round(self, grid):
        cands, policy_tokens = self.policy.sample_k(grid, K=self.K)
        if not cands:
            return None, None, policy_tokens, 0
        scored = []
        eval_tokens_total = 0
        for c in cands:
            p, n_tok = self.scorer.score(grid, c.action)
            eval_tokens_total += n_tok
            scored.append((c, p))
        return cands, scored, policy_tokens, eval_tokens_total

    def choose_action(self, grid):
        cands, scored, policy_tokens_1, eval_tokens_1 = self._score_one_round(grid)
        if cands is None:
            return ChoiceResult(action=None, policy_tokens=policy_tokens_1,
                                eval_tokens=eval_tokens_1, info={"reason": "no_valid_candidates"})
        kept = [(c, p) for c, p in scored if p >= self.tau_keep]
        if kept:
            best = max(kept, key=lambda cp: cp[0].generation_logprob)
            return ChoiceResult(action=best[0].action, policy_tokens=policy_tokens_1,
                                eval_tokens=eval_tokens_1, info={"path": "main"})
        max_p_1 = max(p for _, p in scored)
        if max_p_1 >= self.tau_fb:
            best = max(scored, key=lambda cp: cp[1])
            return ChoiceResult(action=best[0].action, policy_tokens=policy_tokens_1,
                                eval_tokens=eval_tokens_1, info={"path": "fallback"})
        # Retry: sample another K candidates
        cands_2, scored_2, policy_tokens_2, eval_tokens_2 = self._score_one_round(grid)
        total_pol = policy_tokens_1 + (policy_tokens_2 or 0)
        total_eval = eval_tokens_1 + (eval_tokens_2 or 0)
        if cands_2 is None:
            return ChoiceResult(action=None, terminate=True,
                                policy_tokens=total_pol, eval_tokens=total_eval,
                                info={"path": "terminate_after_retry"})
        kept_2 = [(c, p) for c, p in scored_2 if p >= self.tau_keep]
        if kept_2:
            best = max(kept_2, key=lambda cp: cp[0].generation_logprob)
            return ChoiceResult(action=best[0].action, policy_tokens=total_pol,
                                eval_tokens=total_eval, info={"path": "main_retry"})
        max_p_2 = max(p for _, p in scored_2)
        if max_p_2 >= self.tau_fb:
            best = max(scored_2, key=lambda cp: cp[1])
            return ChoiceResult(action=best[0].action, policy_tokens=total_pol,
                                eval_tokens=total_eval, info={"path": "fallback_retry"})
        return ChoiceResult(action=None, terminate=True,
                            policy_tokens=total_pol, eval_tokens=total_eval,
                            info={"path": "terminate_after_retry", "max_p_1": max_p_1, "max_p_2": max_p_2})


class RandomMatchedRateMethod(Method):
    """Terminate at random with given probability. Used to control for raw
    termination rate when comparing against SAVE termination."""
    name = "random_matched"

    def __init__(self, policy: PolicyClient, term_rate: float, K: int = 8, seed: int = 0):
        self.policy = policy
        self.term_rate = term_rate
        self.K = K
        import random as _random
        self.rng = _random.Random(seed)

    def choose_action(self, grid):
        if self.rng.random() < self.term_rate:
            return ChoiceResult(action=None, terminate=True, info={"path": "random_terminate"})
        cands, policy_tokens = self.policy.sample_k(grid, K=self.K)
        if not cands:
            return ChoiceResult(action=None, policy_tokens=policy_tokens, info={"reason": "no_valid_candidates"})
        best = max(cands, key=lambda c: c.generation_logprob)
        return ChoiceResult(action=best.action, policy_tokens=policy_tokens)


class OracleTerminationMethod(Method):
    """Oracle terminates iff current state is doomed (per solver).
    Otherwise picks BoK action. Establishes upper bound on termination
    safety (false term = 0)."""
    name = "oracle_termination"

    def __init__(self, policy: PolicyClient, solver: Hidato5x4Solver, K: int = 8):
        self.policy = policy
        self.solver = solver
        self.K = K

    def choose_action(self, grid):
        if not self.solver.is_viable(grid):
            return ChoiceResult(action=None, terminate=True, info={"path": "oracle_terminate"})
        cands, policy_tokens = self.policy.sample_k(grid, K=self.K)
        if not cands:
            return ChoiceResult(action=None, policy_tokens=policy_tokens, info={"reason": "no_valid_candidates"})
        best = max(cands, key=lambda c: c.generation_logprob)
        return ChoiceResult(action=best.action, policy_tokens=policy_tokens)


class OracleMethod(Method):
    """Same K candidates, oracle solver decides which are viable."""
    name = "oracle"

    def __init__(self, policy: PolicyClient, solver: Hidato5x4Solver, K: int = 8):
        self.policy = policy
        self.solver = solver
        self.K = K

    def choose_action(self, grid):
        cands, policy_tokens = self.policy.sample_k(grid, K=self.K)
        if not cands:
            return ChoiceResult(action=None, policy_tokens=policy_tokens, info={"reason": "no_valid_candidates"})
        viable = []
        for c in cands:
            next_g = apply_action(grid, c.action)
            if self.solver.is_viable(next_g):
                viable.append(c)
        if not viable:
            # No viable candidate — paper oracle "filters by v=1"; if none, fallback to argmax logprob
            best = max(cands, key=lambda c: c.generation_logprob)
            return ChoiceResult(
                action=best.action, policy_tokens=policy_tokens,
                info={"path": "fallback_no_viable"},
            )
        best = max(viable, key=lambda c: c.generation_logprob)
        return ChoiceResult(
            action=best.action, policy_tokens=policy_tokens,
            info={"path": "main", "n_viable": len(viable)},
        )
