"""Single-rollout debug for B-H1: load SFT, run greedy on puzzle 0,
print full prompt + response + outcome at each step. Lets us see exactly
what the model emits and whether it parses.
"""
from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.environments.hidato import HidatoEnv
from src.data.sft_formatter import SFTFormatter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sft-path", required=True)
    p.add_argument("--seed", type=int, default=100000)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--prepend-current-state", action="store_true",
                   help="If set, wrap obs with 'Current state:\\n{obs}' to match SFT prompt format.")
    p.add_argument("--reset-history-per-step", action="store_true",
                   help="If set, each step is a fresh single-turn prompt (no multi-turn history). "
                        "Matches single-turn SFT training distribution.")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== B-H1 single-rollout debug ===")
    print(f"  SFT path: {args.sft_path}")
    print(f"  seed: {args.seed}, temperature: {args.temperature}")
    print(f"  prepend 'Current state:': {args.prepend_current_state}")

    tokenizer = AutoTokenizer.from_pretrained(args.sft_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.sft_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()

    env = HidatoEnv()
    formatter = SFTFormatter(variant="hidato_minimal")
    sys_prompt = formatter.system_prompt

    obs = env.reset(seed=args.seed)
    print(f"\n--- system prompt (truncated) ---\n{sys_prompt[:300]}\n...\n")

    history = []
    for step_idx in range(args.max_steps):
        # Match SFT format if requested
        if args.prepend_current_state:
            if step_idx == 0:
                user_msg = f"Current state:\n{obs}"
            else:
                user_msg = f"Action executed. Current state:\n{obs}"
        else:
            user_msg = obs

        msgs = [{"role": "system", "content": sys_prompt}]
        if not args.reset_history_per_step:
            for u, a in history:
                msgs.append({"role": "user", "content": u})
                msgs.append({"role": "assistant", "content": a})
        msgs.append({"role": "user", "content": user_msg})
        prompt_text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else None,
                top_p=0.95 if args.temperature > 0 else None,
                pad_token_id=tokenizer.eos_token_id,
            )
        resp_ids = out_ids[0][inputs["input_ids"].shape[1]:]
        resp_text = tokenizer.decode(resp_ids, skip_special_tokens=True)

        print(f"\n========== STEP {step_idx} ==========")
        print(f"--- USER MSG (last 400 chars) ---")
        print(user_msg[-400:])
        print(f"\n--- RESPONSE ---")
        print(resp_text)

        # Extract <answer> and step env
        import re
        m = re.search(r"<answer>(.*?)</answer>", resp_text, re.DOTALL)
        if not m:
            print("\n[STOP] No <answer> tag found — action unparseable")
            break
        action_str = m.group(1).strip()
        next_obs, reward, done, info = env.step(action_str)
        print(f"\n--- ENV STEP ---")
        print(f"action_str: {action_str!r}")
        print(f"action_is_valid: {info.get('action_is_valid')}")
        print(f"is_solvable: {info.get('is_solvable')}")
        print(f"success: {info.get('success')}")
        print(f"done: {done}")

        history.append((user_msg, resp_text))
        obs = next_obs
        if done:
            print(f"\n[STOP] done at step {step_idx}")
            break


if __name__ == "__main__":
    main()
