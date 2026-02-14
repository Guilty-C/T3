"""Greedy oracle baseline maximizing immediate utility under constraints."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from src.algos.base import AlgorithmBase
from src.types import Action, Obs, Transition


class LyapunovGreedyOracle(AlgorithmBase):
    """Non-learning greedy policy using environment oracle utility and cost models."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.actions: List[Action] = config["actions"]
        self.oracle_predict: Callable[[Obs, Action], tuple[float, float]] = config["oracle_predict"]
        self.safety_threshold_fn: Callable[[Obs], float] = config["safety_threshold_fn"]
        self.penalty: float = float(config.get("penalty", 2.0))

    def reset(self, seed: int) -> None:
        """No internal state to reset beyond API compatibility."""
        del seed

    def select_action(self, obs: Obs) -> Action:
        """Pick action with best immediate constrained utility score."""
        h = self.safety_threshold_fn(obs)
        best_score = float("-inf")
        best_action = self.actions[0]
        for action in self.actions:
            utility, cost = self.oracle_predict(obs, action)
            violation = max(0.0, cost - h)
            score = utility - self.penalty * violation
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def update(self, transition: Transition) -> None:
        """No learning update required for oracle baseline."""
        del transition

    def get_debug_state(self) -> Dict[str, Any]:
        """Return baseline debug fields."""
        return {
            "active_set_size": None,
            "eliminated_count": None,
            "window_stats": None,
            "lambda": None,
        }
