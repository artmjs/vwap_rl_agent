import numpy as np
EPS = 1e-12

def step_cost(u_t, C, m_t, s_t, alpha, sigma_next, U_t, M_t, lam):
    '''
    Implements the per-step term from the paper's objective (11):
      (s_t/2) * ( alpha * u_t^2 / (C m_t) - u_t / C ) + lam * sigma_{t+1}^2 * (U_t - M_t)^2
    '''
    m_t = max(float(m_t), EPS)
    tx = (s_t/2.0) * (alpha * (u_t**2)/(C*m_t) - (u_t/C))
    risk = lam * (sigma_next**2) * ((U_t - M_t)**2)
    return float(tx + risk)

def price_variance_penalty(u, m, sigma2):
    """
    u: trades per minute (T,),
    m: market volume per minute (T,),
    sigma2: per minute variance (T,) where sigma[t+1] weights the gap after minute t
    """
    C = max(float(np.sum(u)), 1e-12)
    V = max(float(np.sum(m)), 1e-12)
    exec_frac = np.cumsum(u)[:-1] / C
    market_frac = np.cumsum(m)[:-1] / V
    gap = market_frac - exec_frac
    return float(np.sum(sigma2[1:] * gap**2))

