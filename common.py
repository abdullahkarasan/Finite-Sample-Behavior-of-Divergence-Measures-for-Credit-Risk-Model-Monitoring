"""Shared definitions: divergence statistics with explicit zero conventions,
and chi-square critical values."""
import numpy as np
from scipy import stats

ALPHA = 0.05
NAMES = ["PSI", "KLD", "JSD"]
CONST = np.array([1.0, 0.5, 0.125])          # c_phi = phi''(1)/2


def divergences(cx, cy, n, m):
    """PSI, KLD(p||q) and JSD from count arrays of shape (R, B).

    Conventions (0 log 0 = 0):
      - a bin empty in both samples contributes zero to all three statistics;
      - PSI is +inf if some bin is empty in exactly one sample;
      - KLD(p||q) is +inf if some bin has p > 0 = q, and a bin with p = 0 < q
        contributes zero;
      - JSD is always finite.
    Returns an array of shape (R, 3)."""
    P = cx / n
    Q = cy / m
    pp = P > 0
    qq = Q > 0
    both = pp & qq
    lp = np.log(np.where(pp, P, 1.0))
    lq = np.log(np.where(qq, Q, 1.0))
    lr = np.where(both, lp - lq, 0.0)
    psi = np.sum(np.where(both, (P - Q) * lr, 0.0), axis=1)
    psi = np.where((pp != qq).any(axis=1), np.inf, psi)
    kld = np.sum(np.where(both, P * lr, 0.0), axis=1)
    kld = np.where((pp & ~qq).any(axis=1), np.inf, kld)
    M = 0.5 * (P + Q)
    lm = np.log(np.where(M > 0, M, 1.0))
    jsd = 0.5 * np.sum(np.where(pp, P * (lp - lm), 0.0) + np.where(qq, Q * (lq - lm), 0.0), axis=1)
    return np.stack([psi, kld, jsd], axis=1)


def critical_values(B, n, m, alpha=ALPHA):
    """c_phi (1/n + 1/m) chi2_{B-1, 1-alpha} for PSI, KLD, JSD."""
    return CONST * (1.0 / n + 1.0 / m) * stats.chi2.ppf(1 - alpha, B - 1)
