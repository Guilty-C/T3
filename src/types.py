"""Shared typed data structures for the toy safe optimization project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class Obs:
    """Environment observation at one step."""

    snr_db: float
    semantic_weight: float
    queue: float


@dataclass(frozen=True)
class Action:
    """Action with transmission power P and bitrate B."""

    P: float
    B: float


@dataclass(frozen=True)
class StepResult:
    """Output of one environment step."""

    obs: Obs
    reward: float
    utility: float
    cost: float
    violated: int
    done: bool
    info: Dict[str, Any]


@dataclass(frozen=True)
class Transition:
    """Single transition passed to algorithm updates."""

    t: int
    obs: Obs
    action: Action
    reward: float
    utility: float
    cost: float
    violated: int
    next_obs: Obs
    done: bool


@dataclass(frozen=True)
class AlgorithmConfig:
    """Generic lightweight configuration container."""

    params: Dict[str, Any]
