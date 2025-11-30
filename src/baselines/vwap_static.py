# src/baselines/vwap_static.py
from __future__ import annotations
import numpy as np

def estimate_intraday_profile(
    vol_history: np.ndarray,
    *,
    eps: float = 1e-12,
    smooth: bool | int = False,
) -> np.ndarray:
    """
    Estimate the expected intraday volume share profile E[m_t / V] from past days

    Parameters
    ----------
    vol_history : np.ndarray
        Shape (D, T). D = number of past days, T = intervals per day (eg 390 minutes)
        Each row is raw per-interval market volume for that past day
    eps : float
        Small constant to avoid division by zero.
    smooth : bool | int
        If False: return simple mean profile over days.
        If an int k>1: apply a centered moving-average of width k to the final profile
        
    Returns
    -------
    profile : np.ndarray
        Shape (T,). Nonnegative, sums to ~1. This is E[m_t / V].
    """
    vol_history = np.asarray(vol_history, dtype=float)
    assert vol_history.ndim == 2, "vol_history (D, T)"
    
    # Convert each day's volumes to shares and average across days
    daily_totals = np.maximum(vol_history.sum(axis=1, keepdims=True), eps)
    shares = vol_history / daily_totals
    profile = shares.mean(axis=0)

    # Ensure nonnegative and renormalize
    profile = np.clip(profile, 0.0, None)
    s = profile.sum()
    profile = profile / (s if s > eps else 1.0)

    # Optional smoothing
    if isinstance(smooth, int) and smooth > 1:
        k = smooth
        pad = k // 2
        x = np.pad(profile, (pad, pad), mode="edge")
        kernel = np.ones(k) / k
        smoothed = np.convolve(x, kernel, mode="valid")
        # Renormalize after smoothing
        smoothed = np.clip(smoothed, 0.0, None)
        smoothed /= max(smoothed.sum(), eps)
        profile = smoothed

    return profile


def vwap_static_schedule(C: float, expected_share: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """
    Static VWAP slicer (open-loop):
        u_t = C * E[m_t / V]

    Parameters
    ----------
    C : float
        Total shares to execute across the day.
    expected_share : np.ndarray
        Shape (T,). Expected intraday volume share E[m_t / V] summing to 1
    eps : float
        Numerical safety for division/renormalization.

    Returns
    -------
    u : np.ndarray
        Shape (T,); per-interval shares that sum exactly to C with last element adjusted for rounding 
    """
    w = np.asarray(expected_share, dtype=float)
    # Renormalize just in case (robust to small drift)
    w = np.clip(w, 0.0, None)
    s = w.sum()
    w = w / (s if s > eps else 1.0)

    u = C * w

    # Enforce non-negativity and exact completion
    u = np.clip(u, 0.0, None)
    drift = C - u.sum()
    if abs(drift) > 10 * eps:
        u[-1] += drift  # adjust last bucket to hit exaclty C 
        if u[-1] < 0:   # edge case if rounding caused negative => re-project
            # set negatives to 0 and renormalize remaining proportionally
            neg = u < 0
            u[neg] = 0.0
            s_pos = u.sum()
            u = (C * u) / max(s_pos, eps)
    return u
