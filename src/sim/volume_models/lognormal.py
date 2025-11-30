from __future__ import annotations
import numpy as np
from dataclasses import dataclass

Array = np.ndarray

@dataclass
class LogNormalParams:
    mu: Array       # (T,)
    Sigma: Array    # (T, T)
    b: float

# helpers

def to_log_volumes(vol_window_days: Array, eps: float = 1e-12) -> Array:
    """
    Parameters
    ----------
    vol_window_days: (D, T) raw volumes for the last D days

    Returns
    -------
    X = log(max(vol, eps)) of shape (D, T)
    """
    V = np.asarray(vol_window_days, float)
    return np.log(np.maximum(V, eps))


def band_matrix(S: Array, band: int) -> Array:
    """
    Keep entries within |i-j| <= band, zero out the rest
    """
    if band is None or band < 0:
        return S
    S = np.asarray(S, float)
    T = S.shape[0]
    M = S.copy()
    for i in range(T):
        # zero out left block
        if i - band - 1 >= 0:
            M[i, :i - band] = 0.0
        # zero out right block
        if i + band + 1 <= T - 1:
            M[i, i + band + 1:] = 0.0
    # symmetrize to avoid tiny asymmetries
    return 0.5 * (M + M.T)


def leading_rank1_component(S: Array) -> Array:
    """
    Return a vector f such that ff^T approximates the leading component of S.
    Uses leading eigenvector (PCA-1). If S is not PSD due to noise, we add a floor.
    """
    S = 0.5 * (S + S.T)
    # eigh is for symmetric; sort descending by eigenvalue
    w, V = np.linalg.eigh(S)
    idx = np.argmax(w)
    lam1 = max(float(w[idx]), 0.0)
    v1 = V[:, idx]
    # scale so that ff^T has approximately the leading variance lam1 on that direction
    f = np.sqrt(max(lam1, 0.0)) * v1
    return f


def estimate_b_from_daily_means(X: Array, mu: Array) -> float:
    """
    Daily mean of log-volumes vs cross-minute mean mu gives level offset b.
    """
    daily_means = X.mean(axis=1, keepdims=True)         # (D, 1)
    return float(np.mean(daily_means) - np.mean(mu))


def project_rank1_banded(S_emp: Array, band: int | None, add_rank1: bool) -> Array:
    """
    Stabilize an empirical covariance: optional rank-1 + banded residual
    """
    S_emp = 0.5 * (S_emp + S_emp.T)
    if add_rank1:
        f = leading_rank1_component(S_emp)
        # residual after removing rank-1
        R = S_emp - np.outer(f, f)
    else:
        f = None
        R = S_emp

    Rb = band_matrix(R, band) if band is not None else R
    S_stable = Rb if f is None else (np.outer(f, f) + Rb)
    # symmetrize
    return 0.5 * (S_stable + S_stable.T)


# main fitters

def fit_lognormal_params(
    vol_window_days: Array,
    *,
    band: int | None = 3,
    add_rank1: bool = True,
    ridge: float = 1e-8,
) -> LogNormalParams:
    """
    Fit log-normal model parameters on a rolling window of days.

    Model: log m ~ N(mu + 1*b, Sigma), with Sigma ≈ ff^T + banded residual.

    Steps:
      1) X = log volumes
      2) mu = mean across days (per minute)
      3) center each day by its daily mean in log space to estimate shape covariance
      4) Sigma_emp = cov(centered.T)
      5) Sigma = project_rank1_banded(Sigma_emp, band, add_rank1) + ridge*I
      6) b = average (daily_mean - mean(mu))

    Returns: LogNormalParams(mu, Sigma, b)
    """
    X = to_log_volumes(vol_window_days)     # (D, T)
    D, T = X.shape

    # 1) mean minute profile in log space
    mu = X.mean(axis=0)                     # (T,)

    # 2) remove each day's log-level so Sigma captures intraday shape dependence
    daily_means = X.mean(axis=1, keepdims=True)   # (D,1)
    Xc = X - daily_means                          # (D,T)

    # 3) empirical covariance of shape
    # cov on columns -> (T,T)
    Sigma_emp = np.cov(Xc.T, bias=False)

    # 4) stabilize covariance
    Sigma_stable = project_rank1_banded(Sigma_emp, band=band, add_rank1=add_rank1)

    # 5) ridge to ensure PSD and numerical stability
    Sigma = 0.5 * (Sigma_stable + Sigma_stable.T) + ridge * np.eye(T)

    # 6) stock-level b (average daily level minus mean of mu)
    b = estimate_b_from_daily_means(X, mu)

    return LogNormalParams(mu=mu, Sigma=Sigma, b=b)


def build_forecaster_from_window(
    vol_window_days: Array,
    *,
    band: int | None = 3,
    add_rank1: bool = True,
    ridge: float = 1e-8,
):
    """
    Convenience: fit params and construct MVLogNormalForecaster.
    """
    from src.sim.volume_models.forecaster import MVLogNormalForecaster
    params = fit_lognormal_params(vol_window_days, band=band, add_rank1=add_rank1, ridge=ridge)
    return MVLogNormalForecaster(params.mu, params.Sigma, b_scalar=params.b), params
