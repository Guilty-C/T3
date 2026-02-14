"""RA-UCB++ style discrete non-stationary optimistic elimination bandit."""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Tuple

import numpy as np

from src.algos.base import AlgorithmBase
from src.types import Action, Obs, Transition


class RAUCBPP(AlgorithmBase):
    """UCB-like active-set algorithm with optional sliding window and elimination."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.actions: List[Action] = config["actions"]
        self.window_size: int | None = config.get("window_size")
        self.alpha: float = float(config.get("alpha", 1.8))
        self.elim_margin: float = float(config.get("elim_margin", 0.25))
        self.eps0: float = float(config.get("eps0", 0.2))
        self.eps_decay: float = float(config.get("eps_decay", 0.997))
        self.min_active: int = int(config.get("min_active", 2))

        self._rng = np.random.default_rng(0)
        self._arm_index = {(a.P, a.B): i for i, a in enumerate(self.actions)}
        self._active = set(range(len(self.actions)))
        self._history: Deque[Tuple[int, float]] = deque()
        self._t = 0
        self._eliminated = 0

    def reset(self, seed: int) -> None:
        """Reset active set and statistics."""
        self._rng = np.random.default_rng(seed)
        self._active = set(range(len(self.actions)))
        self._history.clear()
        self._t = 0
        self._eliminated = 0

    def _window_records(self) -> List[Tuple[int, float]]:
        if self.window_size is None or len(self._history) <= self.window_size:
            return list(self._history)
        return list(self._history)[-self.window_size :]

    def _stats(self) -> Tuple[np.ndarray, np.ndarray]:
        counts = np.zeros(len(self.actions), dtype=float)
        sums = np.zeros(len(self.actions), dtype=float)
        for arm, reward in self._window_records():
            counts[arm] += 1.0
            sums[arm] += reward
        means = np.divide(sums, np.maximum(counts, 1.0))
        return counts, means

    def select_action(self, obs: Obs) -> Action:
        """Select active arm with epsilon-randomized optimistic strategy."""
        del obs
        self._t += 1
        eps = max(0.01, self.eps0 * (self.eps_decay**self._t))
        active_list = sorted(self._active)
        if self._rng.uniform() < eps:
            return self.actions[int(self._rng.choice(active_list))]

        counts, means = self._stats()
        n = max(len(self._window_records()), 1)
        ucb = means + np.sqrt(
            self.alpha * np.log(n + 1.0) / np.maximum(counts, 1.0)
        )

        for idx in active_list:
            if counts[idx] == 0:
                return self.actions[idx]

        best_idx = max(active_list, key=lambda i: ucb[i])
        return self.actions[best_idx]

    def update(self, transition: Transition) -> None:
        """Update statistics and perform simple elimination."""
        idx = self._arm_index[(transition.action.P, transition.action.B)]
        self._history.append((idx, transition.reward))
        counts, means = self._stats()
        n = max(len(self._window_records()), 1)
        bonus = np.sqrt(self.alpha * np.log(n + 1.0) / np.maximum(counts, 1.0))
        lcb = means - bonus
        ucb = means + bonus

        current_active = sorted(self._active)
        best_lcb = max(lcb[i] for i in current_active)
        to_remove = [i for i in current_active if ucb[i] + self.elim_margin < best_lcb]
        max_removable = max(0, len(self._active) - self.min_active)
        for i in to_remove[:max_removable]:
            self._active.remove(i)
            self._eliminated += 1

    def get_debug_state(self) -> Dict[str, Any]:
        """Expose active-set and window diagnostics."""
        window_records = self._window_records()
        return {
            "active_set_size": len(self._active),
            "eliminated_count": self._eliminated,
            "window_stats": {
                "effective_window": len(window_records),
                "configured_window": self.window_size,
            },
            "lambda": None,
        }
