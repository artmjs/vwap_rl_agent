from __future__ import annotations
import numpy as np
from typing import Callable, Dict, Any, Optional

from src.sim.env import VWAPEnv
from src.sim.costs import slippage_normalized
from src.sim.metrics import price_variance_penalty

# helpers 

def simulate_open_loop(
    u_schedule: np.ndarray,
    prices: np.ndarray,
    volumes: np.ndarray,
    spreads: np.ndarray,
    C: float,
    alpha: float = 90.0,
    lam: float = 0.0,
    sigma2: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Execute a fixed per-minute schedule u (open loop) and compute metrics.
    Assumes u sums to C and is non-negative.
    """
    u = np.asarray(u_schedule, float)
    p, m, s = map(lambda x: np.asarray(x, float), (prices, volumes, spreads))
    # Metrics
    S = slippage_normalized(u, p, m, s, C, alpha)
    if lam and sigma2 is not None:
        R = price_variance_penalty(u, m, sigma2)
    else:
        R = 0.0
    J = float(S) + lam * float(R)
    G = _tracking_rmse(u_schedule, volumes, C)
    return {
        "u": u,
        "S": float(S),
        "R": float(R),
        "J": float(J),
        "G": float(G),
        "logs": {"mode": "open_loop"}
    }


def simulate_closed_loop(
    policy_fn: Callable[[np.ndarray], float],
    prices: np.ndarray,
    volumes: np.ndarray,
    spreads: np.ndarray,
    C: float,
    alpha: float,
    lam: float,
    sigma2: Optional[np.ndarray],
    forecaster=None,
    allow_early_finish: bool = False,
    end_on_fill: bool = True,
    return_traj: bool = False,
) -> Dict[str, Any]:
    """
    Closed-loop simulation: env emits obs, policy returns action each step
    """
    env = VWAPEnv(
        prices=prices, volumes=volumes, spreads=spreads,
        C=C, alpha=alpha, forecaster=forecaster,
        allow_early_finish=allow_early_finish, end_on_fill=end_on_fill,
    )
    obs = env.reset()
    traj = {"obs": [], "acts": []} if return_traj else None

    done = False
    while not done:
        a = float(policy_fn(obs))
        obs, _, done, _ = env.step(a)
        if return_traj:
            traj["obs"].append(obs)
            traj["acts"].append(a)

    # fetch realized schedule from env
    u = env.u.copy()
    p, m, s = env.p, env.m, env.s

    S = slippage_normalized(u, p, m, s, C, alpha)
    if lam and sigma2 is not None:
        R = price_variance_penalty(u, m, sigma2)
    else:
        R = 0.0
    J = float(S) + lam * float(R)
    G = _tracking_rmse(u, volumes, C)
    out = {
        "u": u,
        "S": float(S),
        "R": float(R),
        "J": float(J),
        "G": float(G),
        "logs": {"mode": "closed_loop"}
    }
    if return_traj:
        out["traj"] = traj
    return out

def _tracking_rmse(u, m, C):
    V = float(m.sum())
    exec_frac = np.cumsum(u)/max(C,1e-12)
    mkt_frac  = np.cumsum(m)/max(V,1e-12)
    return float(np.sqrt(np.mean((mkt_frac-exec_frac)**2)))

# main  

def evaluate_day_open_loop(
    u_schedule: np.ndarray,
    prices_day: np.ndarray,
    volumes_day: np.ndarray,
    spreads_day: np.ndarray,
    C: float,
    *,
    alpha: float = 90.0,
    lam: float = 0.0,
    sigma2: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    
    """One-day evaluation for a fixed schedule (e.g. static VWAP)."""

    return simulate_open_loop(u_schedule, prices_day, volumes_day, spreads_day, C, alpha, lam, sigma2)


def evaluate_day_closed_loop(
    policy_fn: Callable[[np.ndarray], float],
    prices_day: np.ndarray,
    volumes_day: np.ndarray,
    spreads_day: np.ndarray,
    C: float,
    *,
    forecaster=None,
    alpha: float = 90.0,
    lam: float = 0.0,
    sigma2: Optional[np.ndarray] = None,
    allow_early_finish: bool = False,
    end_on_fill: bool = True,
    return_traj: bool = False,
) -> Dict[str, Any]:
    
    """One-day evaluation for a policy that acts on observations (controller/RL)."""

    return simulate_closed_loop(
        policy_fn, prices_day, volumes_day, spreads_day, C,
        alpha, lam, sigma2, forecaster, allow_early_finish, end_on_fill, return_traj
    )
