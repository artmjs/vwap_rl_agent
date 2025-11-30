import numpy as np 

def winsorize(x, q=0.995):
    lo, hi = np.quantile(x, [1-q, q])
    return np.clip(x, lo, hi)


def mad_scale(x):
    m = np.median(x)
    mad = np.median(np.abs(x-m)) + 1e-12
    return 1.4826 * mad

def huber_scale(x, c=1.345, iters=20):
    m = np.median(x)
    s = mad_scale(x) 
    for _ in range(iters):
        z = (x-m) / (s+1e-12)
        w = np.minimum(1.0, c/np.maximum(np.abs(z), 1e-12)) 
        s = np.sqrt(np.sum((w*(x-m))**2) / max(np.sum(w**2), 1e-12))
    return s 

def t_var(scale_s, dof_nu: float):
    assert dof_nu > 2, "Student-t var finite only for nu>2"
    return (scale_s**2) * (dof_nu / (dof_nu - 2.0)) 

def sigma2_schedule_t(returns_matrix, dof_nu=5, use_huber=False, winsor_q=0.995): 
    """
    returns_matrix: shape (D, T) — D past days, T minutes/day, per-minute simple returns
    Returns: sigma2 array length T giving per-minute return variance under Student-t.
    """
    D, T = returns_matrix.shape
    sigma2 = np.zeros(T, float)
    for t in range(T):
        r = returns_matrix[:, t]
        s = huber_scale(r) if use_huber else mad_scale(winsorize(r, q=winsor_q))
        sigma2[t] = t_var(s, dof_nu)
    return sigma2