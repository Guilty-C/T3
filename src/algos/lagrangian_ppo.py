"""Minimal PPO with a Lagrange multiplier for constrained optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.algos.base import AlgorithmBase
from src.types import Action, Obs, Transition


@dataclass
class _RolloutItem:
    """One transition item stored for PPO updates."""

    obs_vec: np.ndarray
    action_vec: np.ndarray
    reward: float
    cost: float
    logp: float
    value: float


class _ActorCritic(nn.Module):
    """Tiny shared MLP with Gaussian actor and scalar critic."""

    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
        )
        self.mu_head = nn.Linear(32, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.v_head = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.mu_head(h), self.log_std.expand_as(self.mu_head(h)), self.v_head(h).squeeze(-1)


class LagrangianPPO(AlgorithmBase):
    """Batch PPO updates on-policy with lambda ascent for safety costs."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.P_bounds = config["P_bounds"]
        self.B_bounds = config["B_bounds"]
        self.cost_limit: float = float(config.get("cost_limit", 0.0))
        self.gamma: float = float(config.get("gamma", 0.99))
        self.clip_eps: float = float(config.get("clip_eps", 0.2))
        self.batch_size: int = int(config.get("batch_size", 64))
        self.update_epochs: int = int(config.get("update_epochs", 3))
        self.lr: float = float(config.get("lr", 3e-4))
        self.lambda_lr: float = float(config.get("lambda_lr", 0.02))

        self.device = torch.device("cpu")
        self.net = _ActorCritic(obs_dim=3, act_dim=2).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=self.lr)
        self._rng = np.random.default_rng(0)
        self._rollout: List[_RolloutItem] = []
        self._lambda = 0.0

    def reset(self, seed: int) -> None:
        """Reset rollout buffer and seed all RNGs."""
        self._rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self._rollout.clear()
        self._lambda = 0.0

    def _obs_to_vec(self, obs: Obs) -> np.ndarray:
        return np.array([obs.snr_db / 20.0, obs.semantic_weight, obs.queue / 10.0], dtype=np.float32)

    def _scale_action(self, raw: np.ndarray) -> Action:
        p = 0.5 * (raw[0] + 1.0) * (self.P_bounds[1] - self.P_bounds[0]) + self.P_bounds[0]
        b = 0.5 * (raw[1] + 1.0) * (self.B_bounds[1] - self.B_bounds[0]) + self.B_bounds[0]
        return Action(P=float(np.clip(p, *self.P_bounds)), B=float(np.clip(b, *self.B_bounds)))

    def select_action(self, obs: Obs) -> Action:
        """Sample action from current Gaussian policy."""
        obs_vec = self._obs_to_vec(obs)
        x = torch.as_tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mu, log_std, value = self.net(x)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        raw_action = torch.tanh(dist.rsample())
        logp = dist.log_prob(raw_action).sum(dim=1).item()

        action = self._scale_action(raw_action.squeeze(0).cpu().numpy())
        self._rollout.append(
            _RolloutItem(
                obs_vec=obs_vec,
                action_vec=raw_action.squeeze(0).cpu().numpy(),
                reward=0.0,
                cost=0.0,
                logp=float(logp),
                value=float(value.item()),
            )
        )
        return action

    def update(self, transition: Transition) -> None:
        """Record outcomes and run PPO update when rollout batch is full."""
        if not self._rollout:
            return
        self._rollout[-1].reward = transition.reward
        self._rollout[-1].cost = transition.cost
        if len(self._rollout) >= self.batch_size:
            self._ppo_update()
            self._rollout.clear()

    def _ppo_update(self) -> None:
        rewards = np.array([x.reward for x in self._rollout], dtype=np.float32)
        costs = np.array([x.cost for x in self._rollout], dtype=np.float32)
        adj_rewards = rewards - self._lambda * (costs - self.cost_limit)

        returns = []
        g = 0.0
        for r in adj_rewards[::-1]:
            g = r + self.gamma * g
            returns.append(g)
        returns = np.array(returns[::-1], dtype=np.float32)

        obs = torch.as_tensor(np.stack([x.obs_vec for x in self._rollout]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(
            np.stack([x.action_vec for x in self._rollout]),
            dtype=torch.float32,
            device=self.device,
        )
        old_logp = torch.as_tensor(np.array([x.logp for x in self._rollout]), dtype=torch.float32, device=self.device)
        old_values = torch.as_tensor(np.array([x.value for x in self._rollout]), dtype=torch.float32, device=self.device)
        ret = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        adv = ret - old_values
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)

        for _ in range(self.update_epochs):
            mu, log_std, values = self.net(obs)
            std = log_std.exp()
            dist = torch.distributions.Normal(mu, std)
            new_logp = dist.log_prob(actions).sum(dim=1)
            ratio = torch.exp(new_logp - old_logp)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = ((values - ret) ** 2).mean()
            loss = actor_loss + 0.5 * critic_loss

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        mean_violation = float(np.mean(costs - self.cost_limit))
        self._lambda = max(0.0, self._lambda + self.lambda_lr * mean_violation)

    def get_debug_state(self) -> Dict[str, Any]:
        """Return PPO and constraint multiplier debug values."""
        return {
            "active_set_size": None,
            "eliminated_count": None,
            "window_stats": {"rollout_size": len(self._rollout)},
            "lambda": self._lambda,
        }
