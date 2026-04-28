"""
Base Environment Interface for Termination Prediction

All environments used in termination prediction training must implement
this interface so that trajectory generation, RL training, and evaluation
work uniformly across different games (Sokoban, Sudoku, etc.).
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, List, Optional, Any


class BaseTerminationEnv(ABC):
    """Abstract base class for environments with termination prediction."""

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> str:
        """Reset the environment and return initial observation string."""

    @abstractmethod
    def step(self, action) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Execute action and return (observation, reward, done, info).

        info must contain at minimum:
        - is_solvable: bool
        - deadlock_type: Optional[str]
        - success: bool
        - action_is_valid: bool
        """

    @abstractmethod
    def render(self) -> str:
        """Return current state as a string."""

    @abstractmethod
    def check_solvability(self) -> Tuple[bool, Optional[str]]:
        """Return (is_solvable, reason_if_not)."""

    @abstractmethod
    def get_all_actions(self) -> list:
        """Return list of valid action keys for random play."""

    @abstractmethod
    def get_state_info(self) -> Dict[str, Any]:
        """Return structured state metadata for annotation."""
