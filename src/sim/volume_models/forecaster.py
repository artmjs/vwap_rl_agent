import numpy as np 

class MVLogNormalForecaster:
    """
    Conditional multi-var lognormal forecaster.
    Parameters (mu, sigma) are for log(volume) of the whole day (unconditional).
    At each t, condition on observed log m_{1:t-1} using Schur complement,
    then expose E_t[m_t], E_t[1/m_t], and E_t[1/V].
    """

    def __init__(self, mu, Sigma, b_scalar=0.0):
    # mu: (T,), Sigma: (T,T) for log m 
        self.mu = np.asarray(mu, float)
        self.Sigma = np.asarray(Sigma, float)
        self.b = float(b_scalar)
        self.T = self.mu.shape[0]
        assert self.Sigma.shape == (self.T, self.T)

    def _conditional_params(self, log_prefix):
        t = len(log_prefix) + 1 
        if t <= 1:
            nu_t = self.mu + self.b
            Sig_t = self.Sigma
            return nu_t, Sig_t, 1
        idx1 = slice(0, t-1)  # observed
        idx2 = slice(t-1, self.T)  # remaining (includes current t)
        S11 = self.Sigma[idx1, idx1] 
        S12 = self.Sigma[idx1, idx2]
        S21 = self.Sigma[idx2, idx1]
        S22 = self.Sigma[idx2, idx2]

        # conditional mean/covariance of remaining logs
        nu_t = (self.mu[idx2] + self.b
                + S21 @ np.linalg.solve(S11, (log_prefix - (self.mu[idx1] + self.b))))
        Sig_t = S22 - S21 @ np.linalg.solve(S11, S12)
        return nu_t, Sig_t, t
    
    def moments(self, volumes_prefix):
        """
        volumes_prefix: array of observed m_1..m_{t-1}
        Returns dict with:
          E_m_next     = E_t[m_t]
          E_inv_m_next = E_t[1/m_t]
          E_V          = E_t[V] (useful for logging)
          E_inv_V      = E_t[1/V] (approx; see note)
        """

        obs = np.asarray(volumes_prefix, float)
        obs_sum = float(obs.sum())
        log_prefix = np.log(obs) if obs.size > 0 else np.empty((0,), float)

        nu_t, Sig_t, t = self._conditional_params(log_prefix)

        if t > self.T:
            EV = float(obs.sum())
            return {
                "E_m_next": 0.0,
                "E_inv_m_next": 0.0,
                "E_V": EV,
                "E_inv_V": 1.0 / max(EV, 1e-12),
            }
        # next-step index in the conditional block
        k0 = 0
        # E_t[m_t] and E_t[1/m_t] under lognormal
        E_m_next = np.exp(nu_t[k0] + 0.5 * Sig_t[k0, k0])
        E_inv_m_next = np.exp(-nu_t[k0] + 0.5 * Sig_t[k0, k0])

        # E_t[V] = observed sum + expected remaining sum
        remaining_mean = np.sum(np.exp(nu_t + 0.5 * np.diag(Sig_t)))
        E_V = obs_sum + remaining_mean

        # E_t[1/V]: exact expression for inverse of a sum of correlated lognormals
        # has no simple closed form; the paper derives a usable expression in B.3.
        # As a practical baseline, we use a second-moment (delta) approximation:
        #   E[1/V] ≈ 1/E[V] * (1 + Var(V)/E[V]^2)^(-1)
        # SWAP OUT FOR B.3 FORMULA LATER

        var_remaining = 0.0
        diag = np.diag(Sig_t)
        means = np.exp(nu_t + 0.5 * diag)

        # Var of sum of lognormals with covariance Sig_t:
        # Cov(m_i, m_j) = exp(nu_i+nu_j + 0.5*(sii + sjj))*(exp(sij)-1)
        expSig = np.exp(Sig_t) - 1.0
        cov_mat = np.outer(means, means) * expSig
        var_remaining = float(cov_mat.sum())
        Var_V = var_remaining  # observed past is constant
        E_inv_V = (1.0 / max(E_V, 1e-12)) * (1.0 + Var_V / max(E_V**2, 1e-24))**(-1.0)

        return {
            "E_m_next": float(E_m_next),
            "E_inv_m_next": float(E_inv_m_next),
            "E_V": float(E_V),
            "E_inv_V": float(E_inv_V),
        }