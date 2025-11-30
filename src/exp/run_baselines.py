from __future__ import annotations
import numpy as np
from dataclasses import dataclass, asdict

from src.baselines.vwap_static import estimate_intraday_profile, vwap_static_schedule
from src.baselines.shdp_controller import SHDPController, SHDPParams
from src.sim.volume_models.lognormal import build_forecaster_from_window
from src.exp.evaluate import evaluate_day_open_loop, evaluate_day_closed_loop
from src.sim.robust_volume import sigma2_schedule_t
from src.data.loaders import load_intraday_csv

# helpers
def minute_returns_from_prices(price_days):
    rets = []
    for p in price_days:
        p = np.asarray(p, float)
        r = np.zeros_like(p)
        r[1:] = (p[1:] - p[:-1]) / np.maximum(p[:-1], 1e-12)
        rets.append(r)
    return np.vstack(rets)

def _as_forecaster(x):
    return x[0] if isinstance(x, tuple) else x


# small synthetic dataset
def make_synth(N_days=40, T=120, seed=0):
    """
    Generates synthetic data

    returns prices, volumes, spreads arrays
    """
    rng = np.random.default_rng(seed)
    # U-shape volumes in expectation
    x = np.linspace(0, 1, T)
    shape = 0.6 + 1.4*( (x-0.5)**2 * 4 ) 
    vols = []
    prices = []
    spreads = []
    for d in range(N_days):
        day_strength = rng.lognormal(mean=0.0, sigma=0.4)
        vol_day = day_strength * shape * 1e6 * np.exp(rng.normal(0, 0.25, size=T))  # lognormal noise
        vols.append(vol_day.clip(1.0, None))
        # random walk prices with a bit higher vol at ends
        sig = 0.001*(0.7 + 1.3*( (x-0.5)**2 * 4 ))
        ret = rng.normal(0, sig)
        p = 100.0 * np.cumprod(1.0 + ret)
        prices.append(p)
        # spreads in bps
        s = 0.0002 + 0.0006*( (x-0.5)**2 * 4 )
        spreads.append(s)
    return np.array(prices), np.array(vols), np.array(spreads)

# ---------- Main ----------
@dataclass
class Config:
    T: int = 120
    roll: int = 20      # rolling window size
    lam: float = 20     # risk aversion
    alpha: float = 90.0
    C: float = 1_000_000 # shares to execute

def run_once(prices, volumes, spreads, cfg: Config):
    N, T = volumes.shape
    assert T == cfg.T
    # choose a test day with enough history
    test_day = cfg.roll + 2
    price_window = prices[test_day - cfg.roll:test_day]
    vol_window   = volumes[test_day - cfg.roll:test_day]
    # today
    p_day = prices[test_day]
    m_day = volumes[test_day]
    s_day = spreads[test_day]

    # (optional) build sigma2 for price-risk penalty
    sigma2 = None
    if cfg.lam > 0:
        returns_matrix = minute_returns_from_prices(price_window)
        sigma2 = sigma2_schedule_t(returns_matrix, dof_nu=5)

    # Static VWAP baseline
    profile = estimate_intraday_profile(vol_window, smooth=5)
    u_static = vwap_static_schedule(cfg.C, profile)
    res_static = evaluate_day_open_loop(
        u_schedule=u_static, prices_day=p_day, volumes_day=m_day, spreads_day=s_day,
        C=cfg.C, alpha=cfg.alpha, lam=cfg.lam, sigma2=sigma2
    )

    # SHDP baseline
    fore, _params = build_forecaster_from_window(vol_window), None
    params = SHDPParams(lam=cfg.lam, alpha=cfg.alpha, tc_scale=0.5, u_max_frac=0.3)
    ctrl = SHDPController(C=cfg.C, sigma2=(sigma2 if sigma2 is not None else np.zeros(T)), spreads=s_day, params=params)

    # policy wrapper that calls SHDP each minute
    t_state = {"t": 0, "exec": 0.0}
    def policy_fn(obs: np.ndarray) -> float:
        t = t_state["t"]
        # prefix = observed volumes up to t
        m_prefix = m_day[:t]
        u = ctrl.act(t, m_prefix, t_state["exec"], fore[0] if isinstance(fore, tuple) else fore)
        t_state["exec"] += u
        t_state["t"] += 1
        return u

    res_shdp = evaluate_day_closed_loop(
        policy_fn=policy_fn,
        prices_day=p_day, volumes_day=m_day, spreads_day=s_day, C=cfg.C,
        forecaster=fore[0] if isinstance(fore, tuple) else fore,
        alpha=cfg.alpha, lam=cfg.lam, sigma2=sigma2,
        allow_early_finish=False, end_on_fill=True, return_traj=False
    )

    return res_static, res_shdp, test_day

def sweep_tc_scale(prices, volumes, spreads, cfg, tc_list=(0.5, 1.0, 1.5, 2.0), n_days = 10):
    rows = []
    for tcs in tc_list:
        params = SHDPParams(cfg.lam, alpha=cfg.alpha, tc_scale=tcs, u_max_frac=0.3)
        summary = run_many_with_params(prices, volumes, spreads, cfg, params, n_days=n_days)
        rows.append((tcs, summary["static"], summary["shdp"]))

    # summary printout
    print("tc_scale |  Static J     SHDP J      Static S     SHDP S      SHDP R")
    for tcs, st, sh in rows:
        print(f"{tcs:7.2f} | {st['J']:.6e}  {sh['J']:.6e}  {st['S']:.6e}  {sh['S']:.6e}  {sh['R']:.6e}")
    return rows

def run_many_with_params(prices, volumes, spreads, cfg, shdp_params, n_days=10):
    out = []
    for i in range(cfg.roll+2, cfg.roll+2+n_days):
        price_win = prices[i-cfg.roll:i]; vol_win = volumes[i-cfg.roll:i]
        p_day, m_day, s_day = prices[i], volumes[i], spreads[i]

        # rebuild sigma2
        if cfg.lam > 0:
            Rmat = minute_returns_from_prices(price_win)
            sigma2 = sigma2_schedule_t(Rmat, dof_nu=3)
        else:
            sigma2 = np.zeros(cfg.T)

        # Static
        prof = estimate_intraday_profile(vol_win, smooth=5)
        u_stat = vwap_static_schedule(cfg.C, prof)
        res_stat = evaluate_day_open_loop(u_stat, p_day, m_day, s_day, cfg.C,
                                          alpha=cfg.alpha, lam=cfg.lam, sigma2=sigma2)
        # SHDP
        fore, _ = build_forecaster_from_window(vol_win)
        fore = _as_forecaster(fore)
        ctrl = SHDPController(C=cfg.C, sigma2=sigma2, spreads=s_day, params=shdp_params, profile=prof)
        t_state = {"t": 0, "exec": 0.0}
        def pol(_obs):
            t = t_state["t"]
            u = ctrl.act(t, m_day[:t], t_state["exec"], fore)
            t_state["exec"] += u; t_state["t"] += 1
            return u
        res_shdp = evaluate_day_closed_loop(
            policy_fn=pol,
            prices_day=p_day, volumes_day=m_day, spreads_day=s_day, C=cfg.C,
            forecaster=fore, alpha=cfg.alpha, lam=cfg.lam, sigma2=sigma2,
            allow_early_finish=False, end_on_fill=True, return_traj=False
        )
        out.append((res_stat, res_shdp))
    import numpy as np
    return {
        "first_test_day": cfg.roll + 2,
        "static": {k: float(np.mean([r[0][k] for r in out])) for k in ("S","R","J", "G")},
        "shdp":   {k: float(np.mean([r[1][k] for r in out])) for k in ("S","R","J", "G")},
    }

if __name__ == "__main__":
    cfg = Config()
    n_days = 10

    prices, volumes, spreads = make_synth(N_days=50, T=cfg.T, seed=7)

    params = SHDPParams(lam=cfg.lam, alpha=cfg.alpha, tc_scale=1.0, u_max_frac=None)
    summary = run_many_with_params(prices, volumes, spreads, cfg, params, n_days=n_days)

    fmt = lambda d: {k: f"{v:.6e}" for k, v in d.items()}
    print(f"First test day #{summary['first_test_day']} over {n_days} days")
    print("Static (mean):", fmt(summary["static"]))
    print("SHDP   (mean):", fmt(summary["shdp"]))