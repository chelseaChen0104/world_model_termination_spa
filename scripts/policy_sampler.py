"""HuggingFace-based policy sampler for SAVE data generation.

Provides two operations needed by `generate_save_data.py`:

1. `batched_sample(model, tok, prompt, K, temperature)` — sample K candidate
   action strings in one forward pass via num_return_sequences=K. Returns
   per-candidate text + generation_logprob (sum over generated tokens).

2. `policy_eval_logprob(model, tok, prompt, response_text)` — compute
   log π_θ(response | prompt) for an arbitrary response (e.g. sol/prt
   candidates the policy didn't sample). Single forward pass over the
   concatenated prompt+response tokens.

Both work for any HF AutoModelForCausalLM checkpoint. No env coupling —
the caller is responsible for action parsing + legality checking.

Usage:
    from scripts.policy_sampler import (
        load_model, build_chat_prompt, batched_sample, policy_eval_logprob,
    )
    model, tok, device = load_model("outputs/rl_b5_phase3_v8_anchor/final")
    prompt = build_chat_prompt(tok, system_prompt, user_message)
    samples = batched_sample(model, tok, prompt, K=3, temperature=0.3)
    for text, gen_logprob in samples:
        ...
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn.functional as F


@dataclass
class Sample:
    text: str                   # generated text (no prompt prefix)
    generation_logprob: float   # sum of per-token logprobs under the sampling distribution


def load_model(checkpoint_path: str, device: Optional[str] = None,
                dtype: torch.dtype = torch.bfloat16):
    """Load tokenizer + model from a HF checkpoint dir, place on device, set eval mode."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, torch_dtype=dtype)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer, device


def build_chat_prompt(tokenizer, system_prompt: str, user_message: str) -> str:
    """Apply the tokenizer's chat template; return string prompt ready to tokenize.

    Standard structure: [system, user]. SAVE data-gen wraps the user message as
    "Current state:\\n{state_text}" before passing here.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def batched_sample(
    model,
    tokenizer,
    prompt: str,
    K: int,
    temperature: float,
    top_p: float = 0.95,
    max_new_tokens: int = 512,
) -> List[Sample]:
    """Sample K candidates from the model in a single batched call.

    Uses `num_return_sequences=K`. Returns per-candidate (text, generation_logprob).
    `generation_logprob` is the sum of per-token logprobs under the sampling
    distribution (post-temperature, post-top_p) — matches what the policy
    actually produced.

    For deterministic greedy sampling (rare for SAVE — only used internally for
    debugging), pass temperature=0.0; this routes to do_sample=False and K is
    effectively forced to 1.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    prompt_len = input_ids.shape[1]

    do_sample = temperature > 0.0
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        do_sample=do_sample,
    )
    if do_sample:
        gen_kwargs.update(
            num_return_sequences=K,
            temperature=temperature,
            top_p=top_p,
        )
    else:
        # Greedy: K=1 effectively
        gen_kwargs["num_return_sequences"] = 1

    with torch.no_grad():
        out = model.generate(input_ids, **gen_kwargs)

    # out.sequences: [K, prompt_len + new_len]
    # out.scores: tuple of T_new tensors, each [K, vocab] (post-softmax-warp logits;
    #     these are the SCORES the model assigned at each generation step, AFTER
    #     temperature scaling and top_p masking — we softmax them to get the
    #     sampling distribution from which the chosen token was drawn)
    sequences = out.sequences  # [K, total_len]
    scores = out.scores         # tuple of T tensors of shape [K, V]
    K_actual = sequences.shape[0]

    samples: List[Sample] = []
    for k in range(K_actual):
        # Extract just the generated tokens
        gen_ids = sequences[k, prompt_len:]
        # Find effective end (stop at first eos / pad)
        eos_id = tokenizer.eos_token_id
        eff_len = gen_ids.shape[0]
        for j in range(gen_ids.shape[0]):
            if gen_ids[j].item() == eos_id:
                eff_len = j
                break
        gen_ids_trimmed = gen_ids[:eff_len]
        text = tokenizer.decode(gen_ids_trimmed, skip_special_tokens=True)

        # Sum logprob over generated tokens
        # scores[t][k] is the score distribution at step t for sequence k
        gen_logprob = 0.0
        for t in range(eff_len):
            step_scores = scores[t][k]  # [V]
            log_probs = F.log_softmax(step_scores.float(), dim=-1)
            tok_id = gen_ids[t].item()
            gen_logprob += log_probs[tok_id].item()

        samples.append(Sample(text=text, generation_logprob=gen_logprob))

    return samples


def policy_eval_logprob(
    model,
    tokenizer,
    prompt: str,
    response_text: str,
) -> float:
    """Compute log π_θ(response | prompt) under the model's NATIVE distribution
    (no temperature, no top_p — raw softmax over logits).

    This is the value used by CVCP at inference (Algorithm 2 line 5):
    `arg max_{a in C_τ} log π_θ(a | s)`. Must match how the model is queried at
    deployment for the math to hold.

    Implementation: tokenize (prompt + response), forward pass once, sum
    log_softmax over the response-token positions.
    """
    device = next(model.parameters()).device

    full_text = prompt + response_text
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = prompt_ids.shape[1]
    full_len = full_ids.shape[1]

    if full_len <= prompt_len:
        # Empty response
        return 0.0

    with torch.no_grad():
        out = model(full_ids)
    logits = out.logits  # [1, T, V]

    # logits[i] predicts token i+1. Response token at position j (>= prompt_len)
    # is predicted by logits[j-1].
    total_logprob = 0.0
    for j in range(prompt_len, full_len):
        if j - 1 < 0:
            continue
        log_probs = F.log_softmax(logits[0, j - 1].float(), dim=-1)
        tok_id = full_ids[0, j].item()
        total_logprob += log_probs[tok_id].item()
    return total_logprob


# --- Smoke / self-test (requires a real checkpoint) ---

if __name__ == "__main__":
    # Skipped in CI; manual smoke test:
    #   python scripts/policy_sampler.py outputs/rl_b5_phase3_v8_anchor/final
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    args = p.parse_args()
    print(f"Loading {args.checkpoint} ...")
    model, tok, dev = load_model(args.checkpoint)
    print(f"  loaded; device={dev}")
    sys_prompt = "You are a helpful assistant. Reply with a single short sentence."
    user_msg = "What is 2 + 2?"
    prompt = build_chat_prompt(tok, sys_prompt, user_msg)
    samples = batched_sample(model, tok, prompt, K=3, temperature=0.7)
    for i, s in enumerate(samples):
        print(f"  [{i}] gen_logprob={s.generation_logprob:.2f}  text={s.text[:120]!r}")
    # Eval logprob of one of the samples
    eval_lp = policy_eval_logprob(model, tok, prompt, samples[0].text)
    print(f"  policy_eval_logprob of sample 0: {eval_lp:.2f}")
