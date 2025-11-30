from __future__ import annotations
import numpy as np
from typing import Tuple
from src.sim.volume_models.lognormal import fit_lognormal_params
from src.sim.volume_models.forecaster import MVLogNormalForecaster

def rolling_fit_lognormal(
    vol_window_days: np.ndarray,
    *,
    band: int | None = 3,
    add_rank1: bool = True,
    ridge: float = 1e-8,
) -> Tuple[MVLogNormalForecaster, np.ndarray, np.ndarray, float]:
    """
    Fit log-normal params on the rolling window and build a forecaster.

    Parameters
    ----------
    vol_window_days : (D, T) raw per-minute volumes for the last D days.
    band            : keep |i-j|<=band in the residual covariance (stability).
    add_rank1       : include a rank-1 factor ff^T (day-strength) + banded residual.
    ridge           : small diagonal to ensure PD.

    Returns
    -------
    forecaster : MVLogNormalForecaster
    mu         : (T,) mean of log-volumes
    Sigma      : (T,T) covariance of log-volumes (stabilized)
    b          : float level offset
    """
    params = fit_lognormal_params(
        vol_window_days, band=band, add_rank1=add_rank1, ridge=ridge
    )
    fore = MVLogNormalForecaster(params.mu, params.Sigma, b_scalar=params.b)
    return fore, params.mu, params.Sigma, params.b