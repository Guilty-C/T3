"""Common algorithm interface used by all baselines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from src.types import Action, Obs, Transition


class AlgorithmBase(ABC):
    """Abstract baseline interface with a consistent five-method API."""

    @abstractmethod
    def __init__(self, config: Dict[str, Any]) -> None:
        """Create the algorithm from a config dictionary."""

    @abstractmethod
    def reset(self, seed: int) -> None:
        """Reset internal state and RNG using a seed."""

    @abstractmethod
    def select_action(self, obs: Obs) -> Action:
        """Return one action for the provided observation."""

    @abstractmethod
    def update(self, transition: Transition) -> None:
        """Update internal state from one transition."""

    @abstractmethod
    def get_debug_state(self) -> Dict[str, Any]:
        """Return debug metrics as a serializable dictionary."""
