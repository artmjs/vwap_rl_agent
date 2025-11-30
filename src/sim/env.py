import numpy as np 
from .costs import slippage_normalized

class VWAPEnv: 
    """
    Offline replay env with conditional moments for observation. Mirrors the paper's 
    LQSC/SHDP information set: at time t, use E_t[m_t], E_t[1/m_t], and E_t[1/V] 
    from the conditional volume model. 
    """
    def __init__(self, prices, volumes, spreads, C, alpha=90.0, forecaster=None,
                 allow_early_finish=False, end_on_fill=True):
        self.p = np.asarray(prices, float)
        self.m = np.asarray(volumes, float)
        self.s = np.asarray(spreads, float)
        self.T = len(self.p)
        self.C = float(C)
        self.alpha = float(alpha)
        self.forecaster = forecaster  # MVLogNormalForecaster or None
        self.allow_early_finish = allow_early_finish
        self.end_on_fill = end_on_fill
        self.reset()

    def reset(self):
        self.t = 0
        self.u = np.zeros(self.T, float)
        self.exec_so_far = 0.0
        self.mkt_so_far = 0.0
        self.done_trading = False
        return self._obs()
    
    def _base_obs(self):
        C = self.C if self.C > 0 else 1.0 
        Vexp = self.m.sum() if self.m.sum() > 0 else 1.0
        return np.array([
            self.exec_so_far / C,
            self.mkt_so_far / Vexp,
            self.t / max(self.T - 1, 1)
        ], float)

    def _conditional_features(self):
        if self.forecaster is None:

            # fallback: use current-day empirical share as a crude proxy
            m_remain = self.m[self.t:].sum()
            E_m_next = float(self.m[self.t]) if self.t < self.T else 0.0
            E_inv_m_next = 1.0 / max(E_m_next, 1e-12)
            E_V = self.mkt_so_far + m_remain
            E_inv_V = 1.0 / max(E_V, 1e-12)
            return np.array([E_m_next, E_inv_m_next, E_inv_V], float)
        
        # if we are at or past the last minute
        if self.t >= self.T:
            EV = float(np.sum(self.m[:self.T]))  # total day volume
            return np.array([0.0, 0.0, 1.0 / max(EV, 1e-12)], dtype=float)

        obs_prefix = self.m[:self.t]  # volumes observed so far
        mom = self.forecaster.moments(obs_prefix)
        return np.array([mom["E_m_next"], mom["E_inv_m_next"], mom["E_inv_V"]], float)
    
    def _obs(self):
        return np.concatenate([self._base_obs(), self._conditional_features()])
    
    def step(self, a):

        if self.done_trading:
            self._advance_time()
            done = (self.t == self.T)
            if done:
                r = -slippage_normalized(self.u, self.p, self.m, self.s, self.C, self.alpha)
                return self._obs(), float(r), True, {}
            return self._obs(), 0.0, False, {}
            
        a = max(float(a), 0.0)
        remain = self.C - self.exec_so_far
        if self.t == self.T - 1:
            a = remain
        else:
            a = min(a, max(remain, 0.0))

        # apply trade
        self.u[self.t] = a
        self.exec_so_far += a
        self.mkt_so_far += self.m[self.t]

        # optimal early finish 
        if self.allow_early_finish and self.exec_so_far >= self.C and self.t < self.T - 1: 
            self.done_trading = True
            if self.end_on_fill:
                r = -slippage_normalized(self.u, self.p, self.m, self.s, self.C, self.alpha)
                self.t = self.T
                return self._obs(), float(r), True, {}
            
        # advance time
        self._advance_time()
        done = (self.t == self.T)
        if done:
            r = -slippage_normalized(self.u, self.p, self.m, self.s, self.C, self.alpha)
            return self._obs(), float(r), True, {}
        return self._obs(), 0.0, False, {}

    def _advance_time(self):
        self.t += 1
