"""
Sokoban Environment with Deadlock Detection and Random Puzzle Generation

This module provides:
1. Random puzzle generation via reverse-playing (adapted from gym_sokoban)
2. Deadlock detection for identifying unsolvable states
3. Utilities for state parsing and rendering
"""

import numpy as np
import random as _random
import marshal
import copy
from typing import Tuple, Set, List, Dict, Optional, Any
from collections import deque

from .base import BaseTerminationEnv


# ============================================================
# Puzzle Generation (adapted from gym_sokoban / SPA repo)
# ============================================================

CHANGE_COORDINATES = {
    0: np.array((-1, 0)),
    1: np.array((1, 0)),
    2: np.array((0, -1)),
    3: np.array((0, 1)),
}


def _room_topology_generation(dim=(6, 6), p_change_directions=0.35, num_steps=15):
    """Generate room topology using a random walk to carve out floor."""
    dim_x, dim_y = dim
    masks = [
        [[0, 0, 0], [1, 1, 1], [0, 0, 0]],
        [[0, 1, 0], [0, 1, 0], [0, 1, 0]],
        [[0, 0, 0], [1, 1, 0], [0, 1, 0]],
        [[0, 0, 0], [1, 1, 0], [1, 1, 0]],
        [[0, 0, 0], [0, 1, 1], [0, 1, 0]],
    ]
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    direction = _random.choice(directions)
    position = np.array([
        _random.randint(1, dim_x - 2),
        _random.randint(1, dim_y - 2),
    ])
    level = np.zeros(dim, dtype=int)
    for _ in range(num_steps):
        if _random.random() < p_change_directions:
            direction = _random.choice(directions)
        position = position + np.array(direction)
        position[0] = max(min(position[0], dim_x - 2), 1)
        position[1] = max(min(position[1], dim_y - 2), 1)
        mask = _random.choice(masks)
        ms = position - 1
        level[ms[0]:ms[0] + 3, ms[1]:ms[1] + 3] += mask
    level[level > 0] = 1
    level[:, [0, dim_y - 1]] = 0
    level[[0, dim_x - 1], :] = 0
    return level


def _place_boxes_and_player(room, num_boxes):
    """Place player and box targets randomly on empty floor."""
    possible = np.where(room == 1)
    n_possible = possible[0].shape[0]
    if n_possible <= num_boxes + 1:
        raise RuntimeError(f'Not enough floor ({n_possible}) for {num_boxes} boxes + player')
    # Place player
    idx = np.random.randint(n_possible)
    room[possible[0][idx], possible[1][idx]] = 5
    # Place box targets
    for _ in range(num_boxes):
        possible = np.where(room == 1)
        n_possible = possible[0].shape[0]
        idx = np.random.randint(n_possible)
        room[possible[0][idx], possible[1][idx]] = 2
    return room


def _reverse_move(room_state, room_structure, box_mapping, last_pull, action):
    """Perform a reverse action (player moves + optionally pulls a box)."""
    player_pos = np.where(room_state == 5)
    player_pos = np.array([player_pos[0][0], player_pos[1][0]])
    change = CHANGE_COORDINATES[action % 4]
    next_pos = player_pos + change
    if room_state[next_pos[0], next_pos[1]] in [1, 2]:
        room_state[player_pos[0], player_pos[1]] = room_structure[player_pos[0], player_pos[1]]
        room_state[next_pos[0], next_pos[1]] = 5
        if action < 4:
            pull_from = (int(player_pos[0] - change[0]), int(player_pos[1] - change[1]))
            if room_state[pull_from[0], pull_from[1]] in [3, 4]:
                room_state[player_pos[0], player_pos[1]] = 3
                room_state[pull_from[0], pull_from[1]] = room_structure[pull_from[0], pull_from[1]]
                for k in box_mapping:
                    if box_mapping[k] == (pull_from[0], pull_from[1]):
                        box_mapping[k] = (int(player_pos[0]), int(player_pos[1]))
                        last_pull = k
    return room_state, box_mapping, last_pull


def _box_displacement_score(box_mapping):
    """Sum of Manhattan distances between boxes and their target origins."""
    score = 0
    for target, location in box_mapping.items():
        score += abs(target[0] - location[0]) + abs(target[1] - location[1])
    return score


# Global state for DFS (matches gym_sokoban pattern)
_explored_states = set()
_num_boxes = 0
_best_room_score = -1
_best_room = None
_best_box_mapping = None
_best_action_sequence = []


def _depth_first_search(room_state, room_structure, box_mapping,
                        box_swaps=0, last_pull=(-1, -1), ttl=100, action_sequence=None):
    """DFS to find good reverse-play positions for boxes."""
    global _explored_states, _num_boxes, _best_room_score, _best_room, _best_box_mapping, _best_action_sequence
    if action_sequence is None:
        action_sequence = []
    ttl -= 1
    if ttl <= 0 or len(_explored_states) >= 300000:
        return
    state_hash = marshal.dumps(room_state)
    if state_hash in _explored_states:
        return
    room_score = box_swaps * _box_displacement_score(box_mapping)
    if np.where(room_state == 2)[0].shape[0] != _num_boxes:
        room_score = 0
    if room_score > _best_room_score:
        _best_room = room_state.copy()
        _best_room_score = room_score
        _best_box_mapping = box_mapping.copy()
        _best_action_sequence = action_sequence.copy()
    _explored_states.add(state_hash)
    for action in range(4):
        rs_next = room_state.copy()
        bm_next = box_mapping.copy()
        rs_next, bm_next, lp_next = _reverse_move(rs_next, room_structure, bm_next, last_pull, action)
        bs_next = box_swaps + (1 if lp_next != last_pull else 0)
        _depth_first_search(rs_next, room_structure, bm_next, bs_next, lp_next, ttl, action_sequence + [action])


def _reverse_playing(room_state, room_structure, search_depth=100):
    """Play Sokoban in reverse to create solvable puzzles."""
    global _explored_states, _num_boxes, _best_room_score, _best_room, _best_box_mapping, _best_action_sequence
    box_mapping = {}
    box_locs = np.where(room_structure == 2)
    _num_boxes = len(box_locs[0])
    for i in range(_num_boxes):
        box = (int(box_locs[0][i]), int(box_locs[1][i]))
        box_mapping[box] = box
    _explored_states = set()
    _best_room_score = -1
    _best_room = None
    _best_box_mapping = box_mapping
    _best_action_sequence = []
    _depth_first_search(room_state, room_structure, box_mapping, ttl=search_depth)
    return _best_room, _best_box_mapping, _best_action_sequence


def generate_room(dim=(6, 6), p_change_directions=0.35, num_steps=15,
                  num_boxes=1, tries=4, search_depth=100):
    """Generate a random solvable Sokoban puzzle via reverse-playing.

    Returns:
        (room_structure, room_state, box_mapping, action_sequence)
    """
    for _ in range(tries):
        room = _room_topology_generation(dim, p_change_directions, num_steps)
        room = _place_boxes_and_player(room, num_boxes)
        room_structure = room.copy()
        room_structure[room_structure == 5] = 1
        room_state = room.copy()
        room_state[room_state == 2] = 4
        room_state, box_mapping, action_sequence = _reverse_playing(room_state, room_structure, search_depth)
        if room_state is not None:
            room_state[room_state == 3] = 4
        if box_mapping is not None and _box_displacement_score(box_mapping) > 0:
            return room_structure, room_state, box_mapping, action_sequence
    raise RuntimeWarning('Could not generate puzzle with score > 0')


# ============================================================
# Deadlock Detection
# ============================================================

class SokobanDeadlockDetector:
    """
    Detects common Sokoban deadlock patterns.

    Deadlock types:
    - Corner deadlock: Box pushed into corner (not on goal)
    - Freeze deadlock: Multiple boxes blocking each other
    - Dead square: Box on position from which no goal is reachable
    """

    def __init__(self, walls: Set[Tuple[int, int]], goals: Set[Tuple[int, int]],
                 width: int, height: int):
        self.walls = set(walls)
        self.goals = set(goals)
        self.width = width
        self.height = height
        self.corners = self._compute_corners()
        self.dead_squares = self._compute_dead_squares()

    def _compute_corners(self) -> Set[Tuple[int, int]]:
        corners = set()
        for x in range(self.height):
            for y in range(self.width):
                if self._is_corner(x, y):
                    corners.add((x, y))
        return corners

    def _is_corner(self, x: int, y: int) -> bool:
        if (x, y) in self.walls:
            return False
        wall_up = (x - 1, y) in self.walls or x - 1 < 0
        wall_down = (x + 1, y) in self.walls or x + 1 >= self.height
        wall_left = (x, y - 1) in self.walls or y - 1 < 0
        wall_right = (x, y + 1) in self.walls or y + 1 >= self.width
        return (wall_up or wall_down) and (wall_left or wall_right)

    def _compute_dead_squares(self) -> Set[Tuple[int, int]]:
        alive = set(self.goals)
        changed = True
        while changed:
            changed = False
            for x in range(self.height):
                for y in range(self.width):
                    if (x, y) in alive or (x, y) in self.walls:
                        continue
                    if self._can_reach_alive((x, y), alive):
                        alive.add((x, y))
                        changed = True
        dead = set()
        for x in range(self.height):
            for y in range(self.width):
                if (x, y) not in self.walls and (x, y) not in alive:
                    dead.add((x, y))
        return dead

    def _can_reach_alive(self, pos: Tuple[int, int], alive: Set[Tuple[int, int]]) -> bool:
        x, y = pos
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            push_from = (x - dx, y - dy)
            push_to = (x + dx, y + dy)
            if (push_from not in self.walls and
                0 <= push_from[0] < self.height and 0 <= push_from[1] < self.width and
                push_to in alive):
                return True
        return False

    def detect_deadlock(self, boxes: Set[Tuple[int, int]]) -> Tuple[bool, Optional[str]]:
        boxes = set(boxes)
        for box in boxes:
            if box in self.dead_squares:
                return True, 'dead_square'
        for box in boxes:
            if box in self.corners and box not in self.goals:
                return True, 'corner_deadlock'
        if self._check_freeze_deadlock(boxes):
            return True, 'freeze_deadlock'
        return False, None

    def _check_freeze_deadlock(self, boxes: Set[Tuple[int, int]]) -> bool:
        for box in boxes:
            x, y = box
            frozen_h = ((x - 1, y) in self.walls or (x - 1, y) in boxes) and \
                       ((x + 1, y) in self.walls or (x + 1, y) in boxes)
            frozen_v = ((x, y - 1) in self.walls or (x, y - 1) in boxes) and \
                       ((x, y + 1) in self.walls or (x, y + 1) in boxes)
            if frozen_h and frozen_v and box not in self.goals:
                return True
        return False

    @classmethod
    def from_grid(cls, grid: np.ndarray) -> 'SokobanDeadlockDetector':
        height, width = grid.shape
        walls = set()
        goals = set()
        for i in range(height):
            for j in range(width):
                if grid[i, j] == 0:
                    walls.add((i, j))
                elif grid[i, j] in [2, 3, 6]:
                    goals.add((i, j))
        return cls(walls, goals, width, height)


# ============================================================
# Utility Functions
# ============================================================

def parse_sokoban_grid(grid_str: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Parse text grid to numpy array and entity positions."""
    SYMBOL_TO_CODE = {
        '#': 0, '_': 1, 'O': 2, 'V': 3, 'X': 4, 'P': 5,
    }
    lines = [line for line in grid_str.strip().split('\n') if line]
    height = len(lines)
    width = max(len(line) for line in lines)
    grid = np.zeros((height, width), dtype=np.int32)
    entities = {'player': None, 'boxes': [], 'goals': [], 'walls': []}
    for i, line in enumerate(lines):
        for j, char in enumerate(line):
            code = SYMBOL_TO_CODE.get(char, 1)
            grid[i, j] = code
            if char == 'P':
                entities['player'] = (i, j)
            elif char == 'X':
                entities['boxes'].append((i, j))
            elif char == 'O':
                entities['goals'].append((i, j))
            elif char == '#':
                entities['walls'].append((i, j))
            elif char == 'V':
                entities['boxes'].append((i, j))
                entities['goals'].append((i, j))
    return grid, entities


def get_boxes_from_grid(grid: np.ndarray) -> Set[Tuple[int, int]]:
    """Extract box positions from grid."""
    boxes = set()
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] in [3, 4]:
                boxes.add((i, j))
    return boxes


def check_sokoban_solvability(grid: np.ndarray, detector: SokobanDeadlockDetector) -> Tuple[bool, Optional[str]]:
    """Check if a Sokoban state is solvable."""
    boxes = get_boxes_from_grid(grid)
    is_deadlock, deadlock_type = detector.detect_deadlock(boxes)
    return not is_deadlock, deadlock_type


# ============================================================
# Sokoban Environment
# ============================================================

class SokobanEnv(BaseTerminationEnv):
    """
    Sokoban Environment with random puzzle generation and deadlock detection.
    """

    GRID_LOOKUP = {
        0: '#', 1: '_', 2: 'O', 3: 'V', 4: 'X', 5: 'P', 6: '@',
    }

    ACTION_LOOKUP = {1: 'Up', 2: 'Down', 3: 'Left', 4: 'Right'}

    ACTION_TO_DELTA = {
        1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1),
    }

    def __init__(self, dim_room: Tuple[int, int] = (6, 6), num_boxes: int = 1,
                 max_steps: int = 100, search_depth: int = 100):
        self.dim_room = dim_room
        self.num_boxes = num_boxes
        self.max_steps = max_steps
        self.search_depth = search_depth

        self.room_state = None
        self.room_fixed = None
        self.player_position = None
        self.num_steps = 0
        self.boxes_on_target = 0
        self.detector = None

    def reset(self, seed: Optional[int] = None) -> str:
        """Reset with a new randomly generated puzzle."""
        if seed is not None:
            np.random.seed(seed)
            _random.seed(seed)

        try:
            self.room_fixed, self.room_state, _, _ = generate_room(
                dim=self.dim_room,
                num_steps=max(10, self.dim_room[0] * 2),
                num_boxes=self.num_boxes,
                search_depth=self.search_depth,
            )
        except (RuntimeError, RuntimeWarning):
            # Retry with different seed
            next_seed = abs(hash(str(seed))) % (2**32) if seed is not None else _random.randint(0, 2**32 - 1)
            return self.reset(next_seed)

        self.player_position = np.array(np.where(self.room_state == 5)).flatten()
        if self.player_position.size == 0:
            # Fallback: find player
            for i in range(self.dim_room[0]):
                for j in range(self.dim_room[1]):
                    if self.room_state[i, j] == 5:
                        self.player_position = np.array([i, j])
                        break

        self.num_steps = 0
        self.boxes_on_target = int(np.sum(self.room_state == 3))

        # Initialize deadlock detector
        walls = set()
        goals = set()
        for i in range(self.dim_room[0]):
            for j in range(self.dim_room[1]):
                if self.room_fixed[i, j] == 0:
                    walls.add((i, j))
                elif self.room_fixed[i, j] == 2:
                    goals.add((i, j))

        self.detector = SokobanDeadlockDetector(
            walls, goals, self.dim_room[1], self.dim_room[0]
        )

        return self.render()

    def step(self, action: int) -> Tuple[str, float, bool, Dict]:
        """Take a step in the environment."""
        self.num_steps += 1

        if action not in self.ACTION_TO_DELTA:
            return self.render(), 0, False, {
                'action_is_valid': False, 'success': False,
                'is_solvable': True, 'deadlock_type': None, 'steps': self.num_steps,
            }

        dx, dy = self.ACTION_TO_DELTA[action]
        new_pos = self.player_position + np.array([dx, dy])

        # Check bounds
        if (new_pos[0] < 0 or new_pos[0] >= self.dim_room[0] or
            new_pos[1] < 0 or new_pos[1] >= self.dim_room[1]):
            return self.render(), 0, False, {
                'action_is_valid': False, 'success': False,
                'is_solvable': True, 'deadlock_type': None, 'steps': self.num_steps,
            }

        # Check wall
        if self.room_fixed[new_pos[0], new_pos[1]] == 0:
            return self.render(), 0, False, {
                'action_is_valid': False, 'success': False,
                'is_solvable': True, 'deadlock_type': None, 'steps': self.num_steps,
            }

        # Check box push
        if self.room_state[new_pos[0], new_pos[1]] in [3, 4]:
            box_new_pos = new_pos + np.array([dx, dy])

            if (box_new_pos[0] < 0 or box_new_pos[0] >= self.dim_room[0] or
                box_new_pos[1] < 0 or box_new_pos[1] >= self.dim_room[1] or
                self.room_fixed[box_new_pos[0], box_new_pos[1]] == 0 or
                self.room_state[box_new_pos[0], box_new_pos[1]] in [3, 4]):
                return self.render(), 0, False, {
                    'action_is_valid': False, 'success': False,
                    'is_solvable': True, 'deadlock_type': None, 'steps': self.num_steps,
                }

            # Move box
            old_on_target = self.room_state[new_pos[0], new_pos[1]] == 3
            if old_on_target:
                self.boxes_on_target -= 1

            self.room_state[new_pos[0], new_pos[1]] = self.room_fixed[new_pos[0], new_pos[1]]

            if self.room_fixed[box_new_pos[0], box_new_pos[1]] == 2:
                self.room_state[box_new_pos[0], box_new_pos[1]] = 3
                self.boxes_on_target += 1
            else:
                self.room_state[box_new_pos[0], box_new_pos[1]] = 4

        # Move player
        self.room_state[self.player_position[0], self.player_position[1]] = \
            self.room_fixed[self.player_position[0], self.player_position[1]]
        self.room_state[new_pos[0], new_pos[1]] = 5
        self.player_position = new_pos

        # Check solvability
        is_solvable, deadlock_type = self.check_solvability()

        # Check win/lose
        done = False
        reward = 0
        success = False

        if self.boxes_on_target == self.num_boxes:
            done = True
            reward = 1.0
            success = True
        elif not is_solvable:
            done = True
            reward = -1.0
        elif self.num_steps >= self.max_steps:
            done = True
            reward = -0.5

        info = {
            'action_is_valid': True,
            'success': success,
            'is_solvable': is_solvable,
            'deadlock_type': deadlock_type,
            'steps': self.num_steps,
        }

        return self.render(), reward, done, info

    def render(self) -> str:
        """Render the current state as a string."""
        room = np.where(
            (self.room_state == 5) & (self.room_fixed == 2),
            6, self.room_state
        )
        lines = []
        for row in room:
            line = ''.join(self.GRID_LOOKUP.get(int(cell), '?') for cell in row)
            lines.append(line)
        return '\n'.join(lines)

    def check_solvability(self) -> Tuple[bool, Optional[str]]:
        """Check if current state is solvable."""
        if self.detector is None:
            return True, None
        boxes = set()
        for i in range(self.dim_room[0]):
            for j in range(self.dim_room[1]):
                if self.room_state[i, j] in [3, 4]:
                    boxes.add((i, j))
        is_deadlock, deadlock_type = self.detector.detect_deadlock(boxes)
        return not is_deadlock, deadlock_type

    def get_all_actions(self) -> list:
        """Return list of valid action keys."""
        return list(self.ACTION_LOOKUP.keys())

    def get_state_info(self) -> Dict:
        """Get detailed state information for annotation."""
        is_solvable, deadlock_type = self.check_solvability()
        player_pos = tuple(self.player_position)
        boxes = []
        goals = []
        for i in range(self.dim_room[0]):
            for j in range(self.dim_room[1]):
                if self.room_state[i, j] in [3, 4]:
                    boxes.append((i, j))
                if self.room_fixed[i, j] == 2:
                    goals.append((i, j))
        return {
            'player': player_pos,
            'boxes': boxes,
            'goals': goals,
            'is_solvable': is_solvable,
            'deadlock_type': deadlock_type,
            'boxes_on_target': self.boxes_on_target,
            'steps': self.num_steps,
        }


if __name__ == '__main__':
    env = SokobanEnv(dim_room=(6, 6), num_boxes=1)

    # Test random puzzle generation with different seeds
    for seed in [42, 123, 456, 789, 1000]:
        obs = env.reset(seed=seed)
        print(f"=== Seed {seed} ===")
        print(obs)
        print(f"State info: {env.get_state_info()}")
        print()

    # Test gameplay
    print("=== Gameplay Test ===")
    obs = env.reset(seed=42)
    print(obs)
    for _ in range(10):
        action = _random.choice(env.get_all_actions())
        obs, reward, done, info = env.step(action)
        print(f"Action: {env.ACTION_LOOKUP[action]}, Reward: {reward}, Done: {done}")
        print(obs)
        if done:
            print(f"  Terminated: {info}")
            break
