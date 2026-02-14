"""Sliding-Window UCB for non-stationary K-armed bandits."""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Tuple

import numpy as np

from src.algos.base import AlgorithmBase
from src.types import Action, Obs, Transition


class SWUCB(AlgorithmBase):
    """Minimal SW-UCB over a discrete action set."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.actions: List[Action] = config["actions"]
        self.window_size: int = int(config.get("window_size", 50))
        self.alpha: float = float(config.get("alpha", 2.0))
        self._rng = np.random.default_rng(0)
        self._window: Deque[Tuple[int, float]] = deque(maxlen=self.window_size)
        self._arm_index = {(a.P, a.B): i for i, a in enumerate(self.actions)}

    def reset(self, seed: int) -> None:
        """Clear statistics and reset RNG."""
        self._rng = np.random.default_rng(seed)
        self._window.clear()

    def _stats(self) -> Tuple[np.ndarray, np.ndarray]:
        counts = np.zeros(len(self.actions), dtype=float)
        sums = np.zeros(len(self.actions), dtype=float)
        for arm, reward in self._window:
            counts[arm] += 1.0
            sums[arm] += reward
        means = np.divide(sums, np.maximum(counts, 1.0))
        return counts, means

    def select_action(self, obs: Obs) -> Action:
        """Select the arm maximizing sliding-window UCB index."""
        del obs
        counts, means = self._stats()
        n = max(len(self._window), 1)
        bonuses = np.sqrt(self.alpha * np.log(n + 1.0) / np.maximum(counts, 1.0))
        ucb = means + bonuses
        for i, c in enumerate(counts):
            if c == 0:
                return self.actions[i]
        return self.actions[int(np.argmax(ucb))]

    def update(self, transition: Transition) -> None:
        """Add one reward sample to the sliding window."""
        idx = self._arm_index[(transition.action.P, transition.action.B)]
        self._window.append((idx, transition.reward))

    def get_debug_state(self) -> Dict[str, Any]:
        """Return generic debug fields."""
        return {
            "active_set_size": None,
            "eliminated_count": None,
            "window_size": len(self._window),
            "lambda": None,
        }
