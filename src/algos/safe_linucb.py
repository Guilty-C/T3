"""Safe-LinUCB with reward and safety confidence sets."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np

from src.algos.base import AlgorithmBase
from src.types import Action, Obs, Transition


class SafeLinUCB(AlgorithmBase):
    """Minimal safe linear contextual bandit with confidence filtering."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.actions: List[Action] = config["actions"]
        self.feature_fn: Callable[[Obs, Action], np.ndarray] = config["feature_fn"]
        self.safety_threshold_fn: Callable[[Obs], float] = config["safety_threshold_fn"]
        self.d: int = int(config.get("feature_dim", 6))
        self.alpha_r: float = float(config.get("alpha_r", 0.7))
        self.alpha_c: float = float(config.get("alpha_c", 0.7))
        self.reg: float = float(config.get("reg", 1.0))
        self._rng = np.random.default_rng(0)

        self.A_r = np.eye(self.d) * self.reg
        self.b_r = np.zeros(self.d)
        self.A_c = np.eye(self.d) * self.reg
        self.b_c = np.zeros(self.d)

    def reset(self, seed: int) -> None:
        """Reset model statistics and RNG."""
        self._rng = np.random.default_rng(seed)
        self.A_r = np.eye(self.d) * self.reg
        self.b_r = np.zeros(self.d)
        self.A_c = np.eye(self.d) * self.reg
        self.b_c = np.zeros(self.d)

    def _theta(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.linalg.solve(A, b)

    def _radius(self, A: np.ndarray, x: np.ndarray, alpha: float) -> float:
        inv = np.linalg.inv(A)
        return float(alpha * np.sqrt(x.T @ inv @ x))

    def select_action(self, obs: Obs) -> Action:
        """Pick optimistic reward action among safety-feasible arms."""
        th_r = self._theta(self.A_r, self.b_r)
        th_c = self._theta(self.A_c, self.b_c)
        h = self.safety_threshold_fn(obs)

        safe_candidates: List[tuple[float, Action]] = []
        fallback: List[tuple[float, Action]] = []
        for action in self.actions:
            x = self.feature_fn(obs, action)
            mu_r = float(th_r @ x)
            bonus_r = self._radius(self.A_r, x, self.alpha_r)
            ucb_r = mu_r + bonus_r

            mu_c = float(th_c @ x)
            bonus_c = self._radius(self.A_c, x, self.alpha_c)
            lcb_safe_margin = h - (mu_c + bonus_c)
            if lcb_safe_margin >= 0.0:
                safe_candidates.append((ucb_r, action))
            fallback.append((lcb_safe_margin, action))

        if safe_candidates:
            return max(safe_candidates, key=lambda x: x[0])[1]
        return max(fallback, key=lambda x: x[0])[1]

    def update(self, transition: Transition) -> None:
        """Update both linear models with observed reward and cost."""
        x = self.feature_fn(transition.obs, transition.action)
        self.A_r = self.A_r + np.outer(x, x)
        self.b_r = self.b_r + transition.reward * x
        self.A_c = self.A_c + np.outer(x, x)
        self.b_c = self.b_c + transition.cost * x

    def get_debug_state(self) -> Dict[str, Any]:
        """Return safe linear model diagnostics."""
        return {
            "active_set_size": None,
            "eliminated_count": None,
            "window_stats": None,
            "lambda": None,
        }
