"""
SFT Data Formatting

This module formats trajectories into the SFT training format
with observation, prediction, and termination annotations.
"""

import json
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np


def format_state_with_coordinates(state: str, entities: Dict = None) -> str:
    """
    Format state with coordinate annotations.

    Args:
        state: Grid string representation
        entities: Dictionary with entity positions

    Returns:
        Formatted state string with coordinates
    """
    lines = state.strip().split('\n')

    # Parse entities from the grid if not provided
    if entities is None:
        entities = parse_entities_from_grid(state)

    # Format coordinate description
    coord_parts = []

    if entities.get('player'):
        r, c = entities['player']
        coord_parts.append(f"Player (P) is at ({r},{c})")

    if entities.get('boxes'):
        for i, (r, c) in enumerate(entities['boxes']):
            coord_parts.append(f"box (X) is at ({r},{c})")

    if entities.get('goals'):
        for i, (r, c) in enumerate(entities['goals']):
            coord_parts.append(f"target (O) is at ({r},{c})")

    coord_str = "; ".join(coord_parts) + "."

    return state + "\n" + coord_str


def parse_entities_from_grid(grid_str: str) -> Dict:
    """Parse entity positions from grid string."""
    lines = grid_str.strip().split('\n')
    entities = {
        'player': None,
        'boxes': [],
        'goals': [],
    }

    for i, line in enumerate(lines):
        for j, char in enumerate(line):
            if char == 'P':
                entities['player'] = (i, j)
            elif char == 'X':
                entities['boxes'].append((i, j))
            elif char == 'O':
                entities['goals'].append((i, j))
            elif char == 'V':  # Box on target
                entities['boxes'].append((i, j))
                entities['goals'].append((i, j))
            elif char == '@':  # Player on target
                entities['player'] = (i, j)
                entities['goals'].append((i, j))

    return entities


class SFTFormatter:
    """
    Formats trajectories into SFT training data.

    Supports multiple format variants:
    - baseline: Original SPA format (observation + prediction)
    - termination: Add termination prediction tags
    - full: Add termination + breaking point detection
    """

    # System prompts for different variants
    SYSTEM_PROMPTS = {
        'baseline': """You are solving the Sokoban puzzle. You are the player and you need to push all boxes to targets You are provided with a symbol grid and the zero-indexed coordinates of the player, each box, and each target.

Symbol meanings: #=wall, _=empty, O=target, X=box, P=player, V=box on target

In your reasoning:
1. Describe the current state in <observation>
2. Predict the next state in <prediction>

Then provide your action in <answer>.""",

        'termination': """You are solving the Sokoban puzzle. You are the player and you need to push all boxes to targets You are provided with a symbol grid and the zero-indexed coordinates.

Symbol meanings: #=wall, _=empty, O=target, X=box, P=player, V=box on target

In your reasoning:
1. Describe the current state in <observation>
2. Predict the next state in <prediction>
3. Estimate termination probability in <terminate_prob>
4. Estimate steps remaining in <steps_left>: immediate/near/medium/far

Then provide your action in <answer>.""",

        'full': """You are solving the Sokoban puzzle. You are the player and you need to push all boxes to targets You are provided with a symbol grid and the zero-indexed coordinates.

Symbol meanings: #=wall, _=empty, O=target, X=box, P=player, V=box on target

In your reasoning:
1. Describe the current state in <observation>
2. Predict the next state in <prediction>
3. Estimate termination probability in <terminate_prob>
4. Estimate steps remaining in <steps_left>: immediate/near/medium/far
5. Assess if puzzle is still solvable in <solvable>: true/false
6. Identify if this action creates a deadlock in <breaking_point>: true/false

Then provide your action in <answer>.""",

        # Sudoku variants
        'sudoku_full': """You are solving a Sudoku puzzle. Fill in empty cells (shown as .) with numbers 1-9 so that each row, column, and 3x3 box contains each number exactly once.

Grid format: Numbers separated by spaces, | separates 3x3 boxes, - separates rows of boxes.

In your reasoning:
1. Describe the current state in <observation>
2. Predict the next state after your move in <prediction>
3. Estimate termination probability in <terminate_prob>
4. Estimate steps remaining in <steps_left>: immediate/near/medium/far
5. Assess if puzzle is still solvable in <solvable>: true/false
6. Identify if the last move created an unsolvable state in <breaking_point>: true/false

Then provide your action in <answer> using format: place N at row R col C""",

        # Minimal variant for the post-pivot single-step training (action-conditional <solvable>).
        # Drops <terminate_prob>, <steps_left>, <breaking_point> — see doc/spec_project.md §7.5 v4.
        'sudoku_minimal': """You are solving a Sudoku puzzle. Fill in empty cells (shown as .) with numbers 1-9 so that each row, column, and 3x3 box contains each number exactly once.

Grid format: Numbers separated by spaces, | separates 3x3 boxes, - separates rows of boxes.

In your reasoning:
1. Describe the current state in <observation>
2. Predict the next state after your move in <prediction>
3. Assess whether the resulting state will still be solvable in <solvable>: true/false

Then provide your action in <answer> using format: place N at row R col C""",

        # Pentomino tiling — uses the new tag set (per doc/spec_pentomino.md §4):
        # <observation> + <next_state> + <viability> + <answer>. <prediction> renamed to <next_state>;
        # <solvable> renamed to <viability>. Sudoku variants keep their original tag names for
        # backwards compatibility with B-0..B-5.
        'polyomino_minimal': """You are solving a pentomino tiling puzzle. The board is a rectangular grid; you must place the given pentomino pieces so that every cell is covered exactly once, with no overlaps and no piece extending outside the board.

Pieces use the standard letters: F, I, L, N, P, T, U, V, W, X, Y, Z. Each piece is 5 unit squares. Pieces can be rotated and reflected, giving multiple orientations per piece (orientation IDs 0..N-1, deterministic per piece).

Board format: each cell shows '.' for empty or the piece-letter that occupies it. Remaining pieces are listed below the board.

In your reasoning:
1. Describe the current state in <observation>
2. Predict the board after your placement in <next_state>
3. Assess whether the resulting board is still tileable with the remaining pieces in <viability>: true/false

Then provide your action in <answer> using format: place {piece} ori={K} at row {R} col {C}
where {piece} is one of the remaining pieces, {K} is the orientation id, and (R, C) are 1-indexed anchor coordinates (the anchor is the top-most leftmost cell of the piece's footprint at orientation K).""",

        # Hidato (Numbrix variant) — sequential number-fill on a grid.
        # Uses the Sudoku-style tag set (<observation> + <prediction> + <solvable> + <answer>)
        # since it's a constraint-satisfaction puzzle.
        'hidato_minimal': """You are solving a Hidato (number-path) puzzle. The board is a rectangular grid where you must fill in numbers from 1 to N (where N = rows × cols) so that consecutive numbers (k and k+1) are placed in cells that share an edge (orthogonally adjacent — up, down, left, or right).

Grid format: each cell shows its placed number, or '.' for empty.

You place numbers in sequential order (1, then 2, then 3, ...). Some cells are pre-filled (givens) and don't need to be placed; the env will skip past them. Each step you place the next required number into an empty cell that's adjacent to the previous number's cell.

In your reasoning:
1. Describe the current state in <observation>
2. Predict the next state after your move in <prediction>
3. Assess whether the resulting state will still be solvable (all remaining numbers can be placed legally) in <solvable>: true/false

Then provide your action in <answer> using format: place {N} at row {R} col {C}
where {N} is the next sequential number to place and (R, C) are 1-indexed cell coordinates.""",
    }

    def __init__(self, variant: str = 'full', include_coordinates: bool = True):
        """
        Initialize the formatter.

        Args:
            variant: Format variant ('baseline', 'termination', 'full')
            include_coordinates: Whether to include coordinate descriptions
        """
        assert variant in self.SYSTEM_PROMPTS, f"Unknown variant: {variant}"
        self.variant = variant
        self.include_coordinates = include_coordinates
        self.system_prompt = self.SYSTEM_PROMPTS[variant]

    def format_step(self, step, include_breaking_point: bool = True) -> str:
        """
        Format a single step for SFT training.

        Args:
            step: TrajectoryStep object
            include_breaking_point: Whether to include breaking point tags

        Returns:
            Formatted XML string
        """
        # Format observation (coordinates only apply to Sokoban grid format)
        is_sudoku = self.variant.startswith('sudoku')
        is_polyomino = self.variant.startswith('polyomino')
        is_hidato = self.variant.startswith('hidato')
        if self.include_coordinates and not is_sudoku and not is_polyomino and not is_hidato:
            observation = format_state_with_coordinates(step.state)
            prediction = format_state_with_coordinates(step.next_state)
        else:
            observation = step.state
            prediction = step.next_state

        # Build XML parts.
        # The state-prediction tag name differs by env family:
        #   - Sudoku / Sokoban variants:   <prediction>...</prediction>
        #   - Polyomino (and future MKD):  <next_state>...</next_state>
        # See doc/spec_pentomino.md §4 for the rename rationale.
        next_tag = 'next_state' if is_polyomino else 'prediction'
        xml_parts = [
            "<think>",
            f"<observation>\n{observation}\n</observation>",
            f"<{next_tag}>\n{prediction}\n</{next_tag}>",
        ]

        # Add termination tags if not baseline (legacy variants only)
        if self.variant in ['termination', 'full', 'sudoku_full']:
            # Calculate terminate probability (1.0 if done_label, else based on steps_left)
            if step.done_label == 1:
                term_prob = 0.9
            elif step.steps_left <= 3:
                term_prob = 0.5
            elif step.steps_left <= 7:
                term_prob = 0.2
            else:
                term_prob = 0.1

            xml_parts.append(f"<terminate_prob>{term_prob:.2f}</terminate_prob>")
            xml_parts.append(f"<steps_left>{step.steps_left_bucket}</steps_left>")

        # Solvability / viability tag — same semantic content, different tag name per env family:
        #   - Sudoku-family variants use <solvable>; polyomino-family use <viability>.
        # Action-conditional in both cases: is_solvable(s_{t+1}) given the action in <answer>.
        viab_value = 'true' if step.is_solvable else 'false'
        if self.variant in ['full', 'sudoku_full', 'sudoku_minimal', 'hidato_minimal'] and include_breaking_point:
            xml_parts.append(f"<solvable>{viab_value}</solvable>")
        if self.variant == 'polyomino_minimal' and include_breaking_point:
            xml_parts.append(f"<viability>{viab_value}</viability>")

        # <breaking_point> tag — legacy full variants only. Dropped in sudoku_minimal /
        # polyomino_minimal: the same information is recoverable from the time-series at eval.
        if self.variant in ['full', 'sudoku_full'] and include_breaking_point:
            xml_parts.append(f"<breaking_point>{'true' if step.is_breaking_point else 'false'}</breaking_point>")

        xml_parts.extend([
            "</think>",
            f"<answer>{step.action_name}</answer>"
        ])

        return "\n".join(xml_parts)

    def format_trajectory(self, trajectory: List, metadata=None,
                          multi_turn: bool = False,
                          max_context_turns: Optional[int] = None) -> List[Dict]:
        """
        Format a full trajectory into SFT samples.

        Args:
            trajectory: List of TrajectoryStep objects
            metadata: Optional trajectory metadata
            multi_turn: If True, each sample includes prior turn history.
                        If False (default), single-turn format (backward-compatible).
            max_context_turns: Max prior turns to include in multi-turn context.
                               None = include all prior turns. Ignored when multi_turn=False.

        Returns:
            List of SFT sample dictionaries
        """
        if not multi_turn:
            return self._format_single_turn(trajectory, metadata)
        else:
            return self._format_multi_turn(trajectory, metadata, max_context_turns)

    def _format_single_turn(self, trajectory: List, metadata=None) -> List[Dict]:
        """Format trajectory as independent single-turn samples (original behavior)."""
        samples = []

        for step in trajectory:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Current state:\n{step.state}"},
            ]

            response = self.format_step(step)

            sample = {
                'messages': messages,
                'response': response,
                'metadata': {
                    'step': step.step,
                    'is_solvable': step.is_solvable,
                    'is_breaking_point': step.is_breaking_point,
                    'steps_left': step.steps_left,
                    'done_label': step.done_label,
                    'success': step.success,
                }
            }
            samples.append(sample)

        return samples

    def _format_multi_turn(self, trajectory: List, metadata=None,
                           max_context_turns: Optional[int] = None) -> List[Dict]:
        """Format trajectory as multi-turn conversation samples.

        For step K (0-indexed), the training sample contains:
          prompt messages:
            [system]
            [user: "Current state:\\n{grid_0}"]                         # Turn 0
            [assistant: response_0]                                     # Prior (masked)
            [user: "Action executed. Current state:\\n{grid_1}"]        # Turn 1
            [assistant: response_1]                                     # Prior (masked)
            ...
            [user: "Action executed. Current state:\\n{grid_K}"]        # Current turn
          response (trained on):
            response_K  <- ground-truth annotated XML

        Prior assistant content uses step.llm_raw_response if available (LLM-policy),
        otherwise falls back to template-generated ground truth (random-play).
        """
        samples = []

        for k, current_step in enumerate(trajectory):
            # Determine history window
            if max_context_turns is None:
                history_start = 0
            else:
                history_start = max(0, k - max_context_turns)

            # Build message list
            messages = [{"role": "system", "content": self.system_prompt}]

            # Add prior turns as alternating user/assistant pairs
            for h_idx in range(history_start, k):
                prior_step = trajectory[h_idx]

                # User message for this prior turn
                if h_idx == 0:
                    user_content = f"Current state:\n{prior_step.state}"
                else:
                    user_content = f"Action executed. Current state:\n{prior_step.state}"
                messages.append({"role": "user", "content": user_content})

                # Assistant message: prefer LLM raw response, fall back to template
                if getattr(prior_step, 'llm_raw_response', None) is not None:
                    assistant_content = prior_step.llm_raw_response
                else:
                    assistant_content = self.format_step(prior_step)
                messages.append({"role": "assistant", "content": assistant_content})

            # Add current turn's user message
            if k == 0:
                current_user_content = f"Current state:\n{current_step.state}"
            else:
                current_user_content = f"Action executed. Current state:\n{current_step.state}"
            messages.append({"role": "user", "content": current_user_content})

            # Response for current turn (ground-truth annotated)
            current_response = self.format_step(current_step)

            sample = {
                'messages': messages,
                'response': current_response,
                'metadata': {
                    'step': current_step.step,
                    'is_solvable': current_step.is_solvable,
                    'is_breaking_point': current_step.is_breaking_point,
                    'steps_left': current_step.steps_left,
                    'done_label': current_step.done_label,
                    'success': current_step.success,
                    'num_context_turns': k - history_start,
                }
            }
            samples.append(sample)

        return samples

    def create_sft_dataset(self, trajectories: List, output_format: str = 'parquet',
                           multi_turn: bool = False,
                           max_context_turns: Optional[int] = None) -> pd.DataFrame:
        """
        Create an SFT dataset from trajectories.

        Args:
            trajectories: List of (trajectory, metadata) tuples
            output_format: Output format ('parquet', 'csv', 'json')
            multi_turn: If True, use multi-turn conversation format
            max_context_turns: Max prior turns in multi-turn context (None=all)

        Returns:
            pandas DataFrame with formatted data
        """
        rows = []

        for traj, meta in trajectories:
            samples = self.format_trajectory(
                traj, meta,
                multi_turn=multi_turn,
                max_context_turns=max_context_turns,
            )

            for sample in samples:
                # Format prompt as array of message dicts for verl compatibility
                prompt = np.array(sample['messages'])

                if self.variant.startswith('sudoku'):
                    data_source = 'sudoku'
                elif self.variant.startswith('polyomino'):
                    data_source = 'polyomino'
                elif self.variant.startswith('hidato'):
                    data_source = 'hidato'
                else:
                    data_source = 'sokoban'
                extra = {
                    'step': sample['metadata']['step'],
                    'is_solvable': sample['metadata']['is_solvable'],
                    'is_breaking_point': sample['metadata']['is_breaking_point'],
                }
                if 'num_context_turns' in sample['metadata']:
                    extra['num_context_turns'] = sample['metadata']['num_context_turns']

                rows.append({
                    'data_source': data_source,
                    'prompt': prompt,
                    'response': sample['response'],
                    'ability': 'world_model',
                    'reward_model': "{'style': 'rule'}",
                    'extra_info': json.dumps(extra),
                })

        df = pd.DataFrame(rows)
        return df

    def save_dataset(self, df: pd.DataFrame, output_dir: str, split_ratio: float = 0.2):
        """
        Save dataset to files with train/val split.

        Args:
            df: DataFrame with formatted data
            output_dir: Output directory
            split_ratio: Validation split ratio
        """
        import os
        from sklearn.model_selection import train_test_split

        os.makedirs(output_dir, exist_ok=True)

        # Split data
        train_df, val_df = train_test_split(df, test_size=split_ratio, random_state=42, shuffle=True)

        # Save files
        train_df.to_parquet(os.path.join(output_dir, 'wm_train.parquet'), index=False)
        val_df.to_parquet(os.path.join(output_dir, 'wm_val.parquet'), index=False)

        # Also save as CSV for inspection
        train_df.to_csv(os.path.join(output_dir, 'wm_train.csv'), index=False)
        val_df.to_csv(os.path.join(output_dir, 'wm_val.csv'), index=False)

        print(f"Saved dataset to {output_dir}")
        print(f"  Train: {len(train_df)} samples")
        print(f"  Val: {len(val_df)} samples")


def create_sample_output() -> str:
    """Create a sample formatted output for documentation."""
    return """<think>
<observation>
######
#___O#
#__X_#
###P_#
###__#
######
Player (P) is at (3,3); box (X) is at (2,3); target (O) is at (1,4).
</observation>
<prediction>
######
#___O#
#____#
###X_#
###P_#
######
</prediction>
<terminate_prob>0.0</terminate_prob>
<steps_left>near</steps_left>
<solvable>true</solvable>
<breaking_point>false</breaking_point>
</think>
<answer>Up</answer>"""


if __name__ == '__main__':
    # Test the formatter
    print("Sample SFT output format:")
    print(create_sample_output())

    print("\n" + "="*50)
    print("System prompts by variant:")
    for variant, prompt in SFTFormatter.SYSTEM_PROMPTS.items():
        print(f"\n{variant.upper()}:")
        print(prompt[:200] + "...")
