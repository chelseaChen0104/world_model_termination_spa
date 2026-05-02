"""
LLM-Policy Trajectory Generator

Generates trajectories by having a base LLM play the game, rather than
using random actions. This produces in-distribution training data that
matches the states the model will encounter during RL and inference.

The SPA paper's approach:
1. LLM sees current state → generates <think>...</think><answer>action</answer>
2. Action is parsed from <answer> tag and executed in the environment
3. Real next state (not LLM prediction) is shown back
4. Full trajectory with ground-truth annotations becomes SFT data

This module implements that pipeline for Sudoku (and can be extended to other envs).
"""

import os
import re
import random
import time
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

from src.data.trajectory_generator import TrajectoryStep, TrajectoryMetadata, bucket_steps
from src.data.sft_formatter import SFTFormatter


@dataclass
class LLMTrajectoryConfig:
    """Configuration for LLM trajectory generation."""
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    batch_size: int = 1  # Number of parallel envs (1 for simplicity)
    random_fallback: bool = True  # Fall back to random action if LLM output unparseable
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"
    use_minimal_data_gen_prompt: bool = True
    # When True (default), the base model is prompted with a minimal "output only <answer>"
    # system prompt during data gen — base Qwen-1.5B can follow this reliably (>90% parse
    # rate) and decode ~30 tokens instead of ~700, giving 3–5x wall-clock speedup AND
    # higher-quality LLM-policy data. The full <observation>/<prediction>/<solvable>/...
    # XML format is still used in the SFT TARGET (rebuilt from env ground truth in
    # SFTFormatter.format_step), so the trained model still learns the full format.
    # See doc/SPEC.md §7.5.
    # When True, llm_raw_response is NOT stored on TrajectoryStep (it would just be
    # <answer>...</answer>, useless as multi-turn prior context); SFTFormatter falls back
    # to template-generated ground-truth XML for prior turns, keeping prior/target shape
    # consistent.


class LLMTrajectoryGenerator:
    """Generates trajectories using an LLM to play the game.

    The LLM receives the current state and generates a response with
    <think>...</think><answer>action</answer>. The action is parsed
    and executed in the environment. Ground-truth annotations come
    from the environment's solvability checker.
    """

    DATA_GEN_SYSTEM_PROMPT_SUDOKU = (
        "You are playing Sudoku on a 9x9 grid. The user will show you the current grid; "
        "some cells are filled with digits 1-9 and some are empty (shown as `.`).\n\n"
        "Your job: choose ONE empty cell and place a digit. Output your move EXACTLY in this format:\n"
        "<answer>place N at row R col C</answer>\n"
        "where N is the digit (1-9), R is the row number (1-9), C is the column number (1-9).\n\n"
        "Output ONLY the <answer>...</answer> tag. No reasoning, no extra text, no other tags."
    )

    DATA_GEN_SYSTEM_PROMPT_POLYOMINO = (
        "You are solving a pentomino tiling puzzle. The user will show you the current board, "
        "where '.' = empty cell and a letter = a cell occupied by that pentomino. Below the board "
        "is the list of remaining pieces.\n\n"
        "Your job: choose ONE remaining piece, an orientation, and an anchor cell to place it. "
        "Output your move EXACTLY in this format:\n"
        "<answer>place {P} ori={K} at row {R} col {C}</answer>\n"
        "where {P} is the piece letter (one of the remaining pieces), {K} is the orientation id "
        "(an integer >= 0), and (R, C) are 1-indexed cell coordinates for the anchor cell "
        "(top-most leftmost cell of the placed piece's footprint).\n\n"
        "Output ONLY the <answer>...</answer> tag. No reasoning, no extra text, no other tags."
    )

    DATA_GEN_SYSTEM_PROMPT_HIDATO = (
        "You are solving a Hidato (number-path) puzzle. The user will show you a grid where "
        "filled cells contain integers and empty cells are shown as `.`. The goal is to fill "
        "every empty cell with an integer from 1 to N (= rows * cols) such that consecutive "
        "numbers (k and k+1) are placed in cells that share an edge (orthogonal adjacency: "
        "up/down/left/right).\n\n"
        "On each turn you place ONE number — specifically the next sequential number not yet "
        "on the board — into an empty cell adjacent to the previous number's cell. "
        "Output your move EXACTLY in this format:\n"
        "<answer>place {N} at row {R} col {C}</answer>\n"
        "where {N} is the next sequential number to place and (R, C) are 1-indexed cell "
        "coordinates.\n\n"
        "Output ONLY the <answer>...</answer> tag. No reasoning, no extra text, no other tags."
    )

    def __init__(self, env, config: Optional[LLMTrajectoryConfig] = None,
                 variant: str = "sudoku_full"):
        """
        Args:
            env: BaseTerminationEnv instance (e.g., SudokuEnv, PolyominoEnv)
            config: LLM configuration
            variant: SFT formatter variant (determines the SFT-target system prompt;
                     the data-gen system prompt is independent — see config.use_minimal_data_gen_prompt)
        """
        self.env = env
        self.config = config or LLMTrajectoryConfig()
        self.variant = variant

        # Choose data-gen system prompt:
        # - minimal: only ask for <answer> tag — high parse rate, fast decode
        # - full: same prompt as the SFT target (with full reasoning tags) — base Qwen
        #   struggles to comply, parse rate ~30-50%, decode is slow
        if self.config.use_minimal_data_gen_prompt:
            if variant.startswith("polyomino"):
                self.system_prompt = self.DATA_GEN_SYSTEM_PROMPT_POLYOMINO
            elif variant.startswith("hidato"):
                self.system_prompt = self.DATA_GEN_SYSTEM_PROMPT_HIDATO
            else:
                # default: sudoku (also covers legacy unknown variants)
                self.system_prompt = self.DATA_GEN_SYSTEM_PROMPT_SUDOKU
        else:
            self.system_prompt = SFTFormatter.SYSTEM_PROMPTS[variant]

        # Model/tokenizer loaded lazily
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy-load the model and tokenizer."""
        if self._model is not None:
            return

        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"Loading model: {self.config.model_name_or_path}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Detect device
        device = self.config.device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                # MPS has issues with cumsum int64 in transformers generate()
                # Fall back to CPU for reliable generation
                device = "cpu"

        # Select dtype
        if device == "cuda":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device)
        self._model.eval()
        self._device = device
        print(f"Model loaded on {device} ({dtype})")

    def _generate_response(self, state: str, step_num: int) -> str:
        """Generate LLM response for a given state.

        Args:
            state: Current game state text
            step_num: Current step number (for context)

        Returns:
            Raw LLM response text
        """
        self._load_model()

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Current state:\n{state}"},
        ]

        # Apply chat template
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._tokenizer(text, return_tensors="pt").to(self._device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        # Decode only the new tokens
        response_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response = self._tokenizer.decode(response_ids, skip_special_tokens=True)
        return response.strip()

    def _parse_action_from_response(self, response: str) -> Optional[str]:
        """Extract action string from LLM response.

        Looks for <answer>...</answer> tags.

        Returns:
            Action string or None if parsing fails
        """
        # Try to find <answer>...</answer>
        match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if match:
            action = match.group(1).strip()
            if action:
                return action

        # Fallback: look for "place N at row R col C" pattern anywhere
        match = re.search(r'place\s+\d+\s+at\s+row\s+\d+\s+col\s+\d+', response, re.IGNORECASE)
        if match:
            return match.group(0)

        return None

    def generate_trajectory(
        self,
        max_steps: int = 30,
        seed: Optional[int] = None,
    ) -> Tuple[List[TrajectoryStep], TrajectoryMetadata]:
        """Generate a single trajectory using LLM policy.

        Args:
            max_steps: Maximum steps per episode
            seed: Random seed for environment reset

        Returns:
            Tuple of (trajectory steps, metadata)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        trajectory = []
        state = self.env.reset(seed=seed)
        done = False
        t = 0
        prev_solvable = True
        breaking_point_step = None
        parse_failures = 0

        while not done and t < max_steps:
            # Get LLM response
            response = self._generate_response(state, t)

            # Parse action from response
            action_str = self._parse_action_from_response(response)

            if action_str is None:
                parse_failures += 1
                if self.config.random_fallback:
                    # Fall back to random valid action
                    all_actions = self.env.get_all_actions()
                    if not all_actions:
                        break
                    action_str = random.choice(all_actions)
                else:
                    # Skip this step
                    t += 1
                    continue

            # Execute action in environment
            next_state, reward, done, info = self.env.step(action_str)

            # Check if action was actually valid
            if not info.get('action_is_valid', True):
                # Invalid action - try random fallback
                if self.config.random_fallback:
                    all_actions = self.env.get_all_actions()
                    if not all_actions:
                        break
                    action_str = random.choice(all_actions)
                    next_state, reward, done, info = self.env.step(action_str)
                else:
                    t += 1
                    state = next_state
                    continue

            # Get solvability info
            is_solvable = info.get('is_solvable', True)
            deadlock_type = info.get('deadlock_type', None)

            # Detect breaking point
            is_breaking = prev_solvable and not is_solvable
            if is_breaking and breaking_point_step is None:
                breaking_point_step = t

            # Get action name from info or use the parsed string
            action_name = info.get('action_name', action_str)

            step = TrajectoryStep(
                state=state,
                action=action_str,
                action_name=action_name,
                next_state=next_state,
                reward=reward,
                step=t,
                done_label=0,
                steps_left=0,
                steps_left_bucket='',
                is_solvable=is_solvable,
                is_breaking_point=is_breaking,
                deadlock_type=deadlock_type,
                steps_since_break=None,
                success=info.get('success', False),
                # Only store the raw LLM response for multi-turn priors when we used the
                # full SFT-shaped prompt; with the minimal prompt the response is just
                # <answer>...</answer>, useless as prior context — let SFTFormatter fall
                # back to template-generated ground-truth XML for those priors.
                llm_raw_response=response if not self.config.use_minimal_data_gen_prompt else None,
            )

            trajectory.append(step)
            prev_solvable = is_solvable
            state = next_state
            t += 1

        # Post-hoc annotation
        T = len(trajectory)
        B = breaking_point_step

        for i, step in enumerate(trajectory):
            step.done_label = 1 if i == T - 1 else 0
            step.steps_left = T - i
            step.steps_left_bucket = bucket_steps(T - i)
            if B is not None and i >= B:
                step.steps_since_break = i - B

        # Determine termination reason
        if trajectory and trajectory[-1].success:
            termination_reason = 'success'
        elif B is not None:
            termination_reason = 'deadlock'
        else:
            termination_reason = 'timeout'

        metadata = TrajectoryMetadata(
            total_steps=T,
            success=trajectory[-1].success if trajectory else False,
            has_breaking_point=B is not None,
            breaking_point_step=B,
            steps_wasted=(T - 1 - B) if B is not None else 0,
            final_reward=trajectory[-1].reward if trajectory else 0,
            termination_reason=termination_reason,
        )

        if parse_failures > 0:
            print(f"  [Traj seed={seed}] Parse failures: {parse_failures}/{t} "
                  f"({100*parse_failures/max(t,1):.0f}% random fallback)")

        return trajectory, metadata

    def generate_balanced_dataset(
        self,
        target_size: int = 1280,
        success_ratio: float = 0.0,
        failure_ratio: float = 1.0,
        timeout_ratio: float = 0.0,
        max_steps: int = 30,
        seed: Optional[int] = None,
    ) -> List[Tuple[List[TrajectoryStep], TrajectoryMetadata]]:
        """Generate a balanced dataset using LLM policy.

        Args:
            target_size: Total trajectories to generate
            success_ratio: Target ratio of successful trajectories
            failure_ratio: Target ratio of failure (deadlock) trajectories
            timeout_ratio: Target ratio of timeout trajectories
            max_steps: Maximum steps per episode
            seed: Base random seed

        Returns:
            List of (trajectory, metadata) tuples
        """
        target_success = int(target_size * success_ratio)
        target_failure = int(target_size * failure_ratio)
        target_timeout = target_size - target_success - target_failure

        success_trajs = []
        failure_trajs = []
        timeout_trajs = []

        attempt = 0
        max_attempts = target_size * 5
        t0 = time.time()

        while (len(success_trajs) < target_success or
               len(failure_trajs) < target_failure or
               len(timeout_trajs) < target_timeout) and attempt < max_attempts:

            traj_seed = (seed + attempt) if seed is not None else attempt
            traj, meta = self.generate_trajectory(max_steps, seed=traj_seed)
            attempt += 1

            decision = "empty"
            if traj:
                if meta.success and len(success_trajs) < target_success:
                    success_trajs.append((traj, meta))
                    decision = "keep_success"
                elif meta.has_breaking_point and len(failure_trajs) < target_failure:
                    failure_trajs.append((traj, meta))
                    decision = "keep_BP"
                elif not meta.success and not meta.has_breaking_point and len(timeout_trajs) < target_timeout:
                    timeout_trajs.append((traj, meta))
                    decision = "keep_timeout"
                elif meta.success:
                    decision = "drop_success_full"
                elif meta.has_breaking_point:
                    decision = "drop_BP_full"
                else:
                    decision = "drop_timeout_full"

            total = len(success_trajs) + len(failure_trajs) + len(timeout_trajs)
            elapsed = time.time() - t0
            bp_at = meta.breaking_point_step if (traj and meta.has_breaking_point) else None
            print(
                f"[attempt {attempt} t={elapsed:.0f}s] {decision} "
                f"len={len(traj) if traj else 0} bp@{bp_at} "
                f"acc={total}/{target_size} (S:{len(success_trajs)} F:{len(failure_trajs)} T:{len(timeout_trajs)})",
                flush=True,
            )

        all_trajs = success_trajs + failure_trajs + timeout_trajs
        random.shuffle(all_trajs)

        print(f"\nGenerated balanced dataset:", flush=True)
        print(f"  Success: {len(success_trajs)} (target: {target_success})", flush=True)
        print(f"  Failure: {len(failure_trajs)} (target: {target_failure})", flush=True)
        print(f"  Timeout: {len(timeout_trajs)} (target: {target_timeout})", flush=True)
        print(f"  Total: {len(all_trajs)} / {target_size}", flush=True)

        return all_trajs


def main():
    """Generate Sudoku SFT data using LLM policy."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate SFT data using LLM policy")
    parser.add_argument("--env", type=str, default="sudoku", choices=["sudoku", "polyomino", "hidato"],
                        help="Which environment to generate data for")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--num-trajectories", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=30)
    # Sudoku-specific args
    parser.add_argument("--grid-size", type=int, default=9, help="(sudoku) grid size — 4 or 9")
    parser.add_argument("--difficulty", type=str, default="easy", help="(sudoku) difficulty")
    # Polyomino-specific args
    parser.add_argument("--board-h", type=int, default=5, help="(polyomino) board height")
    parser.add_argument("--board-w", type=int, default=4, help="(polyomino) board width")
    parser.add_argument("--piece-set", type=str, default="L,P,W,Y",
                        help="(polyomino) comma-separated piece letters; locked easy variant: L,P,W,Y")
    parser.add_argument("--output-dir", type=str, default="data/sudoku_llm_policy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--multi-turn", action="store_true", default=False,
                        help="Use multi-turn conversation format for SFT data")
    parser.add_argument("--max-context-turns", type=int, default=None,
                        help="Max prior turns in multi-turn context (None=all)")
    parser.add_argument("--variant", type=str, default=None,
                        help="SFT formatter variant. Default = 'sudoku_full' for sudoku, 'polyomino_minimal' for polyomino.")
    args = parser.parse_args()

    from src.data.sft_formatter import SFTFormatter

    # Build env + formatter variant per --env
    if args.env == "sudoku":
        from src.environments.sudoku import SudokuEnv
        env = SudokuEnv(
            grid_size=args.grid_size,
            difficulty=args.difficulty,
            max_steps=args.max_steps,
        )
        default_variant = "sudoku_full"
        env_descr = f"Sudoku grid={args.grid_size}x{args.grid_size}, difficulty={args.difficulty}"
    elif args.env == "polyomino":
        from src.environments.polyomino import PolyominoEnv
        piece_set = tuple(p.strip().upper() for p in args.piece_set.split(","))
        env = PolyominoEnv(
            board_h=args.board_h,
            board_w=args.board_w,
            piece_set=piece_set,
            max_steps=args.max_steps,
        )
        default_variant = "polyomino_minimal"
        env_descr = f"Polyomino board={args.board_h}x{args.board_w}, pieces={{{','.join(piece_set)}}}"
    elif args.env == "hidato":
        from src.environments.hidato import HidatoEnv
        env = HidatoEnv(max_steps=args.max_steps)
        default_variant = "hidato_minimal"
        env_descr = f"Hidato (puzzle bank from src/environments/hidato_puzzle_bank.py)"
    else:
        raise ValueError(f"unknown --env: {args.env}")

    variant = args.variant or default_variant

    print("=" * 60)
    print(f"LLM-Policy {args.env.title()} Data Generation")
    print("=" * 60)
    print(f"Model:         {args.model}")
    print(f"Trajectories:  {args.num_trajectories}")
    print(f"Max steps:     {args.max_steps}")
    print(f"Env:           {env_descr}")
    print(f"Variant:       {variant}")
    print(f"Output:        {args.output_dir}")
    print(f"Multi-turn:    {args.multi_turn}")
    if args.multi_turn:
        print(f"  max ctx:     {args.max_context_turns or 'all'}")
    print("=" * 60)

    # Create LLM trajectory generator
    config = LLMTrajectoryConfig(
        model_name_or_path=args.model,
        temperature=args.temperature,
        device=args.device,
    )
    generator = LLMTrajectoryGenerator(env, config, variant=variant)

    # Generate trajectories
    trajectories = generator.generate_balanced_dataset(
        target_size=args.num_trajectories,
        success_ratio=0.0,
        failure_ratio=1.0,
        timeout_ratio=0.0,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    # Format for SFT
    print("\nFormatting for SFT...")
    formatter = SFTFormatter(variant=variant)
    df = formatter.create_sft_dataset(
        trajectories,
        multi_turn=args.multi_turn,
        max_context_turns=args.max_context_turns,
    )

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    formatter.save_dataset(df, args.output_dir, split_ratio=args.val_split)

    # Statistics
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(df)}")

    # Class distribution
    import json
    solvable_count = sum(
        1 for _, row in df.iterrows()
        if json.loads(row['extra_info']).get('is_solvable', False)
    )
    bp_count = sum(
        1 for _, row in df.iterrows()
        if json.loads(row['extra_info']).get('is_breaking_point', False)
    )
    print(f"  Solvable: {solvable_count}/{len(df)} ({100*solvable_count/len(df):.1f}%)")
    print(f"  Breaking points: {bp_count}/{len(df)} ({100*bp_count/len(df):.1f}%)")

    # Trajectory statistics
    success_count = sum(1 for _, m in trajectories if m.success)
    bp_traj_count = sum(1 for _, m in trajectories if m.has_breaking_point)
    avg_steps = sum(m.total_steps for _, m in trajectories) / len(trajectories)
    print(f"\nTrajectory Statistics:")
    print(f"  Trajectories: {len(trajectories)}")
    print(f"  Success: {success_count}")
    print(f"  Deadlock: {bp_traj_count}")
    print(f"  Avg steps: {avg_steps:.1f}")

    if bp_traj_count > 0:
        avg_bp_step = sum(m.breaking_point_step for _, m in trajectories if m.has_breaking_point) / bp_traj_count
        avg_wasted = sum(m.steps_wasted for _, m in trajectories if m.has_breaking_point) / bp_traj_count
        print(f"  Avg breaking point step: {avg_bp_step:.1f}")
        print(f"  Avg steps wasted: {avg_wasted:.1f}")


if __name__ == "__main__":
    main()
