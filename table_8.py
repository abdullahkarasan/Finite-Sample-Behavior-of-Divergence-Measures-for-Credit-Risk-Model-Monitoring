import numpy as np
from common import *

def run(B, n, R=10000, nperm=199):
    rng = np.random.default_rng([2026, 8, B, n])
    p = np.full(B, 1 / B); cv = critical_values(B, n, n)
    rc = np.zeros(3); rp = np.zeros(3); inf = np.zeros(3)
    for _ in range(R):
        cx = rng.multinomial(n, p); cy = rng.multinomial(n, p)
        obs = divergences(cx[None], cy[None], n, n)[0]
        pooled = cx + cy
        d = rng.multivariate_hypergeometric(pooled, n, size=nperm)
        null = divergences(d, pooled[None] - d, n, n)
        pval = (1 + (null >= obs).sum(axis=0)) / (nperm + 1)
        rc += obs > cv; rp += pval <= ALPHA; inf += np.isinf(obs)
    return rc / R, rp / R, 100 * inf / R

if __name__ == "__main__":
    print("=== Table 8: B n | chi2 PSI KLD JSD | perm PSI KLD JSD | %inf PSI KLD ===")
    for B, n in [(10, 1000), (50, 1000), (100, 1000), (20, 200)]:
        c, p_, i = run(B, n)
        print(f"{B} & {n} & " + " & ".join(f"{x:.4f}" for x in c) + " & " + " & ".join(f"{x:.4f}" for x in p_) + f" \\\\   %inf PSI {i[0]:.1f} KLD {i[1]:.1f}", flush=True)
