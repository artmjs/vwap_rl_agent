from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class SHDPParams:
    lam: float = 1.0               # risk aversion lambda
    alpha: float = 90.0            # trade-impact coefficient from the slippage model

    # If you model spread cost per step ~ (alpha * s_t / (2 * C * m_t)) u_t^2,
    # then R_t ~ (alpha * s_t) / (2 * C) * E_t[1/m_t]. You can scale it here:
    tc_scale: float = 0.5          # corresponds to the "1/2" above
    u_max_frac: float = 0.3        # safety cap per step (fraction of remaining)
    epsilon: float = 1e-12         # numerical guard

class SHDPController:
    """
    Shrinking-Horizon Dynamic Programming controller.
    Each step:
      1) Get conditional moments for all remaining minutes from forecaster.
      2) Build Q_t (risk) from sigma2 and R_t (cost) from E_t[1/m_t] and spreads.
      3) Run a short scalar Riccati backward recursion to get k_t for *current* step.
      4) Output u_t = expected_next_slice + k_t * current_gap, clipped to [0, remaining].
    """

    def __init__(self, C: float, sigma2: np.ndarray, spreads: np.ndarray, params: SHDPParams, profile=None):
        self.C = float(C)
        self.sigma2 = np.asarray(sigma2, float)    # (T,)
        self.spreads = np.asarray(spreads, float)  # (T,)
        self.params = params

        if profile is not None:
            prof = np.asarray(profile, float)
            s = prof.sum()
            if s <= 0:
                raise ValueError("Profile must have positive sum.")
            prof = prof / s                     # ensure it sums to 1
            if prof.shape[0] != self.sigma2.shape[0]:
                raise ValueError("Profile length must equal T.")
            self.profile = prof
        else:
            self.profile = None

    def _build_QR_vectors(self, t: int, E_inv_m_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Build Q and R for remaining steps at decision time t.
        We weight gap at 'after-minute' times, hence sigma2[t+1:].
        """
        C = self.C
        eps = self.params.epsilon
        n = self.sigma2.shape[0] - t
        # Q on pre-action gap x_k: use sigma2[t : t+n]
        Q = self.params.lam * self.sigma2[t : t + n]          # (n,)
        # R on action u_k: from spreads and E[1/m_k], align to same n
        s_tail = self.spreads[t : t + n]
        R = (self.params.alpha * s_tail / max(C, eps)) * (0.5 * E_inv_m_vec[:n])
        R *= self.params.tc_scale 
        # safety
        R = np.maximum(R, 1e-12)
        Q = np.maximum(Q, 0.0)
        return Q, R

    def _riccati_scalar_backward(self, Q: np.ndarray, R: np.ndarray) -> float:
        """
        Run Riccati on the remaining horizon and return k_t for the current step
        State x = gap, A=1, B=-1/C
        """
        C = self.C
        B2 = 1.0 / (C*C)
        P_next = 0.0
        n = R.shape[0]
        # check len(Q) = len(R)
        assert len(Q) == n, f"Q and R must have same length, got {len(Q)} vs {len(R)}"
        for k in range(n-1, -1, -1):
            denom = R[k] + B2 * P_next
            P = Q[k] + P_next - ((P_next / C)**2) / max(denom, 1e-20)
            P_next = P
        # feedback gain for current step (k=0); K = (P_{1} * B) / (R_0 + B^2 P_{1})
        denom0 = R[0] + B2 * P_next
        K = ( -P_next / C ) / max(denom0, 1e-20)  # K is negative (since B<0)
        k_fb = -K                                  # so u = u_part + k_fb * x
        return float(max(k_fb, 0.0))

    def act(
        self,
        t: int,
        m_prefix: np.ndarray,       # observed m_1 up to m_t
        x_exec_shares: float,       # executed shares so far
        forecaster,                 # MVLogNormalForecaster
    ) -> float:
        """
        Compute u_t (shares for minute t+1 in 0-based indexing) given info up to t.
        """
        eps = self.params.epsilon
        C = self.C

        # helper per step debug line
        def _dbg_line(t, g, Em_next, EinvV, k, u_part, u_fb, u):
            return (f"t={t:3d}  g={g:+.3e}  E[m_next]={Em_next:.3e}  E[1/V]={EinvV:.3e}  "
                f"k={k:.3e}  u_part={u_part:.3e}  u_fb={u_fb:+.3e}  u={u:.3e}")

        log_prefix = np.log(np.maximum(m_prefix, eps)) if m_prefix.size else np.empty((0,), float)
        nu, Sig, _ = forecaster._conditional_params(log_prefix)  # (remaining,), (remaining,remaining)

        # Moment vectors for remaining minutes
        diag = np.diag(Sig)
        E_m_vec = np.exp(nu + 0.5 * diag)          # E_t[m_k] for k = t+1..T
        E_inv_m_vec = np.exp(-nu + 0.5 * diag)     # E_t[1/m_k]
        
        # First element corresponds to the next decision minute
        E_m_next = float(E_m_vec[0])

        # E_t[V], E_t[1/V]
        moms = forecaster.moments(m_prefix)
        E_V = moms["E_V"]; E_inv_V = moms["E_inv_V"]

        # Build Q and R for remaining horizon at time t
        Q, R = self._build_QR_vectors(t, E_inv_m_vec)

        # Scalar Riccati to get k_t for current step
        k_t = self._riccati_scalar_backward(Q, R)

        # lam = 0 case
        if self.params.lam <= 0.0:
            k_t = 0.0
        else: 
            self._riccati_scalar_backward(Q,R)

        # Current tracking gap x_t 
        M_t = float(np.sum(m_prefix))
        x_frac = x_exec_shares / max(C, eps)
        mkt_frac = M_t / max(E_V, eps)
        x_t_gap = mkt_frac - x_frac

        # Particular action
        u_part = C * E_inv_V * E_m_next

        if self.profile is not None:
            # deterministic target: same VWAP curve as Static baseline
            u_part = self.C * self.profile[t]
        else:
            # fallback: stochastic target from forecaster
            u_part = self.C * E_inv_V * E_m_next

        # Total action (shares)
        u = u_part + k_t * x_t_gap

        # Safety: clip to [0, remaining] and to per-step max fraction
        remaining = max(C - x_exec_shares, 0.0)
        u = max(float(u), 0.0)
        
        if self.params.u_max_frac is not None:
            u = float(np.clip(u, 0.0, self.params.u_max_frac * remaining))
        else:
            u = float(min(u, remaining))

        # diagnostic
        # if t in (0, 10, 30, 60, 90, 110):
        #     print(
        #         f"t={t:3d}  gap={x_t_gap:+.3e}  E[m_next]={E_m_next:.3e}  "
        #         f"E[1/V]={E_inv_V:.3e}  k={k_t:.3e}  u_part={u_part:.3e}  "
        #         f"u_fb={(k_t*x_t_gap):+.3e}  u={u:.3e}"
        #     )
        return u
