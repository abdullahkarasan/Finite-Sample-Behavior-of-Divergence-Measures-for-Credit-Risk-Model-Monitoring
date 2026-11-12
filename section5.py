import numpy as np
from scipy.stats import chi2, norm
from common import *

PAIRS = [("standard", "jump"), ("standard", "sv_jump"), ("jump", "sv_jump")]
LAB = {"standard": "Merton", "jump": "Merton with Jump", "sv_jump": "SVJ"}

def row_counts(x, edges):
    """Counts per simulation (row) and bin, shape (n_sim, B); edges are the B+1 bin boundaries."""
    B = len(edges) - 1
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, B - 1)
    R = x.shape[0]
    return np.bincount((np.arange(R)[:, None] * B + idx).ravel(), minlength=R * B).reshape(R, B)

def equal_width_edges(allp, nb=50):
    pad = (allp.max() - allp.min()) * 0.05
    return np.linspace(allp.min() - pad, allp.max() + pad, nb + 1)

def decile_edges(allp):
    q = np.quantile(allp, np.linspace(0, 1, 11)); q[0], q[-1] = -np.inf, np.inf
    return q

def block_perm_pvalues(Ra, Rb, obs, nperm, rng):
    pool = np.vstack([Ra, Rb]); tot = pool.sum(0); ns = Ra.shape[0]; n = Ra.sum(); m = Rb.sum()
    ge = np.zeros(3)
    for _ in range(nperm):
        cx = pool[rng.permutation(pool.shape[0])[:ns]].sum(0)
        ge += divergences(cx[None], (tot - cx)[None], n, m)[0] >= obs
    return (1 + ge) / (1 + nperm)

def application(M, nperm=9999):
    allp = np.concatenate([v.ravel() for v in M.values()])
    rng = np.random.default_rng([2026, 5])
    for label, edges in (("(a) equal-width bins", equal_width_edges(allp)), ("(b) deciles of the pooled data", decile_edges(allp))):
        R = {k: row_counts(v, edges) for k, v in M.items()}
        keep = sum(R[k].sum(0) for k in R) > 0          # drop bins empty in all three samples
        B = int(keep.sum()); n = m = M["standard"].size
        cv = critical_values(B, n, m)
        print(f"\n=== {label}: B = {B}, min count {min(R[k][:, keep].sum(0).min() for k in R)}, critical values x1e3 {np.round(cv*1e3, 3)} ===")
        for a, b in PAIRS:
            Ra, Rb = R[a][:, keep], R[b][:, keep]
            obs = divergences(Ra.sum(0)[None], Rb.sum(0)[None], n, m)[0]
            pchi = chi2.sf(obs / (CONST * (1 / n + 1 / m)), B - 1)
            pperm = block_perm_pvalues(Ra, Rb, obs, nperm, rng)
            print(f"{LAB[a]} vs {LAB[b]}: x1e3 {np.round(obs*1e3, 3)} | /crit {np.round(obs/cv, 2)} | KLD/PSI {obs[1]/obs[0]:.4f} JSD/PSI {obs[2]/obs[0]:.4f} | p_chi2 {np.round(pchi, 4)} | p_perm {np.round(pperm, 4)}", flush=True)

def merton_pd_sample(S0, mu, sigma, corr, nsim, rng, r=0.05, L=100.0, T=1.0, N=252):
    """Default probabilities under the Merton specification, drawn exactly:
    the simulated log terminal value is Gaussian with 251 daily increments."""
    dt = T / N; k = N - 1
    A = np.linalg.cholesky(np.outer(sigma, sigma) * corr)
    Z = rng.standard_normal((nsim, len(S0)))
    logA = np.log(S0) + (mu - 0.5 * sigma**2) * k * dt + np.sqrt(k * dt) * Z @ A.T
    d = (logA - np.log(L) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(-d)

def null_calibration(S0, mu, sigma, rho, R, nperm, seed, nsim=5000):
    """Rejection rates when both samples come from the Merton specification."""
    rng = np.random.default_rng([2026, 55, seed])
    d = len(S0); corr = np.full((d, d), rho); np.fill_diagonal(corr, 1.0)
    rej_chi = np.zeros((2, 3)); rej_perm = np.zeros((2, 3)); nper = 0
    for rep in range(R):
        X = merton_pd_sample(S0, mu, sigma, corr, nsim, rng); Y = merton_pd_sample(S0, mu, sigma, corr, nsim, rng)
        allp = np.concatenate([X.ravel(), Y.ravel()])
        for b, edges in enumerate((equal_width_edges(allp), decile_edges(allp))):
            Ra, Rb = row_counts(X, edges), row_counts(Y, edges)
            keep = (Ra.sum(0) + Rb.sum(0)) > 0
            Ra, Rb = Ra[:, keep], Rb[:, keep]; B = int(keep.sum()); n = m = X.size
            obs = divergences(Ra.sum(0)[None], Rb.sum(0)[None], n, m)[0]
            rej_chi[b] += obs > critical_values(B, n, m)
            if rep < nperm[0]:
                rej_perm[b] += block_perm_pvalues(Ra, Rb, obs, nperm[1], rng) <= ALPHA
        nper += rep < nperm[0]
    print(f"rho = {rho}: chi-square rejection rates (R={R})  equal-width {np.round(rej_chi[0]/R, 4)}  deciles {np.round(rej_chi[1]/R, 4)}", flush=True)
    if nper:
        print(f"           block-permutation rejection rates (R={nper}, {nperm[1]} perms)  equal-width {np.round(rej_perm[0]/nper, 4)}  deciles {np.round(rej_perm[1]/nper, 4)}", flush=True)

if __name__ == "__main__":
    d = np.load("pd.npz"); M = {k: d[k] for k in ("standard", "jump", "sv_jump")}
    # check that the exact sampler matches the path simulation for the Merton specification
    rng = np.random.default_rng(1); corr = np.full((10, 10), 0.3); np.fill_diagonal(corr, 1.0)
    X = merton_pd_sample(d["S0"], d["mu"], d["sigma"], corr, 200000, rng)
    print("check exact sampler vs path simulation: mean %.4f vs %.4f, P(PD<0.05) %.4f vs %.4f, P(PD>0.95) %.4f vs %.4f" % (
        X.mean(), M["standard"].mean(), (X < .05).mean(), (M["standard"] < .05).mean(), (X > .95).mean(), (M["standard"] > .95).mean()))
    print("\n##### null calibration under the Merton specification #####")
    null_calibration(d["S0"], d["mu"], d["sigma"], 0.3, R=2000, nperm=(1000, 199), seed=1)
    null_calibration(d["S0"], d["mu"], d["sigma"], 0.0, R=2000, nperm=(0, 0), seed=2)
    print("\n##### application #####")
    application(M)
