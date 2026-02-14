"""Single runner for six constrained optimization baselines on a toy environment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.algos.base import AlgorithmBase
from src.algos.lagrangian_ppo import LagrangianPPO
from src.algos.lyapunov_greedy_oracle import LyapunovGreedyOracle
from src.algos.ra_ucbpp import RAUCBPP
from src.algos.safe_linucb import SafeLinUCB
from src.algos.safeopt_gp import SafeOptGP
from src.algos.sw_ucb import SWUCB
from src.envs.toy_env import ToyEnv
from src.types import Transition


def make_algorithm(name: str, env: ToyEnv) -> AlgorithmBase:
    """Instantiate one of the six supported algorithms."""
    actions = env.discrete_action_grid(p_points=5, b_points=5)
    if name == "ra_ucbpp":
        return RAUCBPP({"actions": actions, "window_size": 100})
    if name == "safeopt_gp":
        return SafeOptGP({"actions": actions, "safety_threshold": 0.0})
    if name == "safe_linucb":
        return SafeLinUCB(
            {
                "actions": actions,
                "feature_fn": env.action_to_features,
                "safety_threshold_fn": env.safety_threshold,
                "feature_dim": 6,
            }
        )
    if name == "sw_ucb":
        return SWUCB({"actions": actions, "window_size": 60})
    if name == "lagrangian_ppo":
        bounds = env.action_space_bounds()
        return LagrangianPPO({"P_bounds": bounds["P"], "B_bounds": bounds["B"], "batch_size": 32})
    if name == "lyapunov_greedy_oracle":
        return LyapunovGreedyOracle(
            {
                "actions": actions,
                "oracle_predict": env.oracle_predict,
                "safety_threshold_fn": env.safety_threshold,
            }
        )
    raise ValueError(f"Unsupported algo: {name}")


def run(name: str, T: int, seed: int) -> pd.DataFrame:
    """Run algorithm for T steps and return a stable-schema dataframe."""
    env = ToyEnv()
    algo = make_algorithm(name, env)
    obs = env.reset(seed)
    algo.reset(seed)

    rows: List[Dict[str, float | int | str | None]] = []
    for t in range(T):
        action = algo.select_action(obs)
        result = env.step(action)
        transition = Transition(
            t=t,
            obs=obs,
            action=action,
            reward=result.reward,
            utility=result.utility,
            cost=result.cost,
            violated=result.violated,
            next_obs=result.obs,
            done=result.done,
        )
        algo.update(transition)
        debug = algo.get_debug_state()
        rows.append(
            {
                "t": t,
                "seed": seed,
                "algo": name,
                "snr_db": obs.snr_db,
                "semantic_weight": obs.semantic_weight,
                "queue": obs.queue,
                "P": action.P,
                "B": action.B,
                "reward": result.reward,
                "utility": result.utility,
                "cost": result.cost,
                "violated": result.violated,
                "debug_active_set_size": debug.get("active_set_size", np.nan),
                "debug_eliminated_count": debug.get("eliminated_count", np.nan),
                "debug_lambda": debug.get("lambda", np.nan),
            }
        )
        obs = result.obs

    cols = [
        "t",
        "seed",
        "algo",
        "snr_db",
        "semantic_weight",
        "queue",
        "P",
        "B",
        "reward",
        "utility",
        "cost",
        "violated",
        "debug_active_set_size",
        "debug_eliminated_count",
        "debug_lambda",
    ]
    return pd.DataFrame(rows, columns=cols)


def main() -> None:
    """Parse args, run one algorithm, and save stable-schema CSV."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        required=True,
        choices=[
            "ra_ucbpp",
            "safeopt_gp",
            "safe_linucb",
            "sw_ucb",
            "lagrangian_ppo",
            "lyapunov_greedy_oracle",
        ],
    )
    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    df = run(args.algo, args.T, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
