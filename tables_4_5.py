import numpy as np
from scipy import stats
from common import *

B = 10
EDGES = stats.norm.ppf(np.linspace(0, 1, B + 1))[1:-1]

def counts(x):
    R = x.shape[0]
    idx = np.searchsorted(EDGES, x)
    return np.bincount((np.arange(R)[:, None] * B + idx).ravel(), minlength=R * B).reshape(R, B)

def cell(n, m, mu2, R, rng, chunk=10000):
    out = []
    for r in range(0, R, chunk):
        k = min(chunk, R - r)
        out.append(divergences(counts(rng.normal(0, 1, (k, n))), counts(rng.normal(mu2, 1, (k, m))), n, m))
    return np.vstack(out)

if __name__ == "__main__":
    R = 50000; sizes = [200, 400, 600, 800, 1000]
    rng = np.random.default_rng([2026, 45])
    size = np.zeros((3, 5, 5)); power = np.zeros((3, 5, 5)); adj = np.zeros((3, 5, 5)); ninf = 0
    for i, n in enumerate(sizes):
        for j, m in enumerate(sizes):
            D0 = cell(n, m, 0.0, R, rng); D1 = cell(n, m, 0.25, R, rng)
            ninf += np.isinf(D0).sum() + np.isinf(D1).sum()
            cv = critical_values(B, n, m)
            size[:, i, j] = (D0 > cv).mean(axis=0); power[:, i, j] = (D1 > cv).mean(axis=0)
            emp = np.quantile(D0, 0.95, axis=0)
            adj[:, i, j] = (D1 > emp).mean(axis=0)
        print("row done", n, flush=True)
    np.savez("t45.npz", size=size, power=power, adj=adj)
    print("infinite statistics:", ninf)
    for label, A in (("Table 4: size", size), ("Table 5: power", power), ("size-adjusted power", adj)):
        print(f"\n=== {label} ===")
        for k, name in enumerate(NAMES):
            print(name)
            for i, n in enumerate(sizes):
                print(f"& {n}  & " + " & ".join(f"{A[k,i,j]:.4f}" for j in range(5)) + " \\\\")
    print("\nmax spread across measures, asymptotic critical values:", np.round((power.max(0) - power.min(0)).max() * 100, 2), "pp")
    print("max spread across measures, size-adjusted:", np.round((adj.max(0) - adj.min(0)).max() * 100, 2), "pp")
    print("spread by cell (size-adjusted, pp):\n", np.round((adj.max(0) - adj.min(0)) * 100, 2))
    print("spread by cell (asymptotic, pp):\n", np.round((power.max(0) - power.min(0)) * 100, 2))
