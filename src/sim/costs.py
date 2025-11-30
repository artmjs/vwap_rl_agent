import numpy as np

EPS = 1e-12

def effective_price(p_t: float, s_t: float, u_t, m_t, alpha: float) -> float:
    '''
    p̂_t = p_t * ( 1 - s_t/2 + alpha * (s_t/2) * (u_t/m_t) )
    '''

    m_t_safe = max(float(m_t), EPS)
    return float(p_t) * (1.0 - s_t/2.0 + alpha * (s_t/2.0) * (float(u_t) / m_t_safe))

def vwap(price: np.ndarray, volume: np.ndarray) -> float:
    '''
    p_VWAP = sum over t m_t * p_t / V, V being total market volume for the day
    '''

    price = np.asarray(price, float)
    volume = np.asarray(volume, float)
    V = max(volume.sum(), EPS)
    return float((price*volume).sum() / V)

def slippage_value(u, p, m, s, alpha) -> float:
    '''
    Unnormalized slippage in currency units
    => ∑ u_t p_hat_t - C * p_VWAP   
    '''
    u = np.asarray(u, float)
    p = np.asarray(p, float)
    m = np.asarray(m, float)
    s = np.asarray(s, float)
    C = float(u.sum())
    p_hat = np.array([effective_price(p[t], s[t], u[t], m[t], alpha) for t in range(len(u))])
    return float((p_hat * u).sum() - C * vwap(p, m))

def slippage_normalized(u, p, m, s, C: float, alpha: float) -> float:
    """
    Slippage normalised to VWAP total price
    S = (∑ u_t p̂_t - C * p_VWAP) / (C * p_VWAP)
    """
    p = np.asarray(p, float); m = np.asarray(m, float); s = np.asarray(s, float)
    num = slippage_value(u, p, m, s, alpha)
    den = max(float(C) * vwap(p, m), EPS)
    return float(num / den)