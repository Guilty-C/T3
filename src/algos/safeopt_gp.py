"""Minimal SafeOpt-like Gaussian process safe Bayesian optimization."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from src.algos.base import AlgorithmBase
from src.types import Action, Obs, Transition


class SafeOptGP(AlgorithmBase):
    """Discrete SafeOpt with RBF GP and safe set expansion heuristic."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.actions: List[Action] = config["actions"]
        self.safety_threshold: float = float(config.get("safety_threshold", 0.0))
        self.beta: float = float(config.get("beta", 2.0))
        self.lengthscale: float = float(config.get("lengthscale", 0.5))
        self.signal_var: float = float(config.get("signal_var", 1.0))
        self.noise: float = float(config.get("noise", 1e-3))
        self._rng = np.random.default_rng(0)

        self._X: List[np.ndarray] = []
        self._y_reward: List[float] = []
        self._y_cost: List[float] = []

    def reset(self, seed: int) -> None:
        """Clear GP dataset and reset RNG."""
        self._rng = np.random.default_rng(seed)
        self._X.clear()
        self._y_reward.clear()
        self._y_cost.clear()

    def _vec(self, action: Action) -> np.ndarray:
        return np.array([action.P, action.B], dtype=float)

    def _kernel(self, Xa: np.ndarray, Xb: np.ndarray) -> np.ndarray:
        diff = Xa[:, None, :] - Xb[None, :, :]
        sq = np.sum(diff * diff, axis=2)
        return self.signal_var * np.exp(-0.5 * sq / (self.lengthscale**2))

    def _gp_predict(self, y: np.ndarray, Xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(self._X) == 0:
            mu = np.zeros(len(Xs), dtype=float)
            std = np.full(len(Xs), np.sqrt(self.signal_var), dtype=float)
            return mu, std

        X = np.vstack(self._X)
        K = self._kernel(X, X) + self.noise * np.eye(len(X))
        Ks = self._kernel(X, Xs)
        Kss_diag = np.diag(self._kernel(Xs, Xs))
        c, low = cho_factor(K, lower=True, check_finite=False)
        alpha = cho_solve((c, low), y, check_finite=False)
        mu = Ks.T @ alpha

        v = cho_solve((c, low), Ks, check_finite=False)
        var = np.maximum(Kss_diag - np.sum(Ks * v, axis=0), 1e-9)
        return mu, np.sqrt(var)

    def select_action(self, obs: Obs) -> Action:
        """Select safe maximizer or boundary expander candidate."""
        del obs
        Xs = np.vstack([self._vec(a) for a in self.actions])
        mu_r, std_r = self._gp_predict(np.array(self._y_reward, dtype=float), Xs)
        mu_c, std_c = self._gp_predict(np.array(self._y_cost, dtype=float), Xs)

        lcb_safe = self.safety_threshold - (mu_c + self.beta * std_c)
        safe_idx = [i for i, v in enumerate(lcb_safe) if v >= 0.0]

        if not safe_idx:
            return self.actions[int(np.argmin(mu_c))]

        ucb_r = mu_r + self.beta * std_r
        maximizer_idx = max(safe_idx, key=lambda i: ucb_r[i])

        uncertainty = std_r + std_c
        expander_idx = max(safe_idx, key=lambda i: uncertainty[i])

        if uncertainty[expander_idx] > uncertainty[maximizer_idx] * 1.05:
            return self.actions[expander_idx]
        return self.actions[maximizer_idx]

    def update(self, transition: Transition) -> None:
        """Append one observation to GP datasets."""
        self._X.append(self._vec(transition.action))
        self._y_reward.append(float(transition.reward))
        self._y_cost.append(float(transition.cost))

    def get_debug_state(self) -> Dict[str, Any]:
        """Return GP dataset and safeopt debug fields."""
        return {
            "active_set_size": None,
            "eliminated_count": None,
            "window_stats": {"n_gp_points": len(self._X)},
            "lambda": None,
        }
