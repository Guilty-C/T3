"""Toy environment supporting both discrete-grid and continuous 2D actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from src.types import Action, Obs, StepResult


@dataclass(frozen=True)
class ToyEnvConfig:
    """Configuration for the toy constrained wireless-like environment."""

    P_bounds: Tuple[float, float] = (0.1, 2.0)
    B_bounds: Tuple[float, float] = (0.1, 2.0)
    queue_limit: float = 8.0
    base_safety_threshold: float = 0.0


class ToyEnv:
    """Small non-stationary environment with oracle utility and cost prediction."""

    def __init__(self, config: ToyEnvConfig | None = None) -> None:
        self.config = config or ToyEnvConfig()
        self._rng = np.random.default_rng(0)
        self.t = 0
        self._obs = Obs(snr_db=10.0, semantic_weight=1.0, queue=2.0)

    def reset(self, seed: int) -> Obs:
        """Reset state and return initial observation."""
        self._rng = np.random.default_rng(seed)
        self.t = 0
        self._obs = Obs(
            snr_db=float(self._rng.uniform(6.0, 14.0)),
            semantic_weight=float(self._rng.uniform(0.5, 1.5)),
            queue=float(self._rng.uniform(1.0, 4.0)),
        )
        return self._obs

    def step(self, action: Action) -> StepResult:
        """Advance environment by one step using clipped action values."""
        clipped = self.clip_action(action)
        utility, cost = self.oracle_predict(self._obs, clipped)
        reward = utility - 0.5 * max(0.0, cost)
        violated = int(cost > self.config.base_safety_threshold)

        drift = 0.12 * np.sin(self.t / 15.0)
        next_queue = np.clip(self._obs.queue + 0.2 * clipped.B - 0.1 * clipped.P + drift, 0.0, self.config.queue_limit)
        next_obs = Obs(
            snr_db=float(np.clip(self._obs.snr_db + self._rng.normal(0.0, 0.4), 0.0, 20.0)),
            semantic_weight=float(np.clip(self._obs.semantic_weight + self._rng.normal(0.0, 0.05), 0.3, 2.0)),
            queue=float(next_queue),
        )
        self.t += 1
        self._obs = next_obs
        return StepResult(
            obs=next_obs,
            reward=float(reward),
            utility=float(utility),
            cost=float(cost),
            violated=violated,
            done=False,
            info={},
        )

    def clip_action(self, action: Action) -> Action:
        """Clip action to configured bounds."""
        return Action(
            P=float(np.clip(action.P, *self.config.P_bounds)),
            B=float(np.clip(action.B, *self.config.B_bounds)),
        )

    def discrete_action_grid(self, p_points: int = 5, b_points: int = 5) -> List[Action]:
        """Return a discrete action grid over P and B."""
        ps = np.linspace(self.config.P_bounds[0], self.config.P_bounds[1], p_points)
        bs = np.linspace(self.config.B_bounds[0], self.config.B_bounds[1], b_points)
        return [Action(P=float(p), B=float(b)) for p in ps for b in bs]

    def oracle_predict(self, obs: Obs, action: Action) -> Tuple[float, float]:
        """Return utility and cost estimates used by oracle-style baselines."""
        snr_gain = np.log1p(max(obs.snr_db, 0.0))
        qoe = obs.semantic_weight * (1.4 * snr_gain + 0.9 * action.B - 0.6 * action.P)
        queue_penalty = 0.15 * obs.queue
        utility = qoe - queue_penalty

        cost = 0.8 * action.P + 0.55 * action.B + 0.25 * obs.queue - 2.2
        return float(utility), float(cost)

    def safety_threshold(self, obs: Obs) -> float:
        """Return step-dependent safety threshold."""
        return self.config.base_safety_threshold + 0.05 * (obs.queue - 2.0)

    def action_to_features(self, obs: Obs, action: Action) -> np.ndarray:
        """Map observation-action pair to linear features."""
        return np.array(
            [
                1.0,
                action.P,
                action.B,
                obs.snr_db / 20.0,
                obs.semantic_weight,
                obs.queue / max(self.config.queue_limit, 1.0),
            ],
            dtype=float,
        )

    def action_space_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Return action bounds for continuous algorithms."""
        return {"P": self.config.P_bounds, "B": self.config.B_bounds}
