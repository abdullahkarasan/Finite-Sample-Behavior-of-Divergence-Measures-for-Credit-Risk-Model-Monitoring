import numpy as np
from scipy import stats
from common import *   # divergences(), critical_values(), CONST, NAMES, ALPHA

def multinomial_stats(p, n, m, R, rng, chunk=20000):
    out = []
    for r in range(0, R, chunk):
        k = min(chunk, R - r)
        out.append(divergences(rng.multinomial(n, p, size=k), rng.multinomial(m, p, size=k), n, m))
    return np.vstack(out)

def table2(B=10, n=1000, m=1000, R=10000):
    rng = np.random.default_rng([2026, 2])
    D = multinomial_stats(np.full(B, 1 / B), n, m, R, rng)
    np.save("validation_stats.npy", D)
    s, df = 1 / n + 1 / m, B - 1
    print(f"\n=== Table 2 (B={B}, n=m={n}, R={R}); infinite: {np.isinf(D).sum()} ===")
    for j, name in enumerate(NAMES):
        v = D[:, j]; c = CONST[j]
        ks = stats.kstest(v / (c * s), "chi2", args=(df,))
        q = [np.quantile(v, a) / (c * s * stats.chi2.ppf(a, df)) for a in (0.90, 0.95, 0.99)]
        print(f"{name} & {v.mean():.4f} & {c*df*s:.4f} & {v.std():.4f} & {c*np.sqrt(2*df)*s:.4f} & {ks.statistic:.4f} & {ks.pvalue:.3f} \\\\   quantile ratios 90/95/99: {np.round(q,3)}")

def table3(B=10, R=40000):
    rng = np.random.default_rng([2026, 3])
    print(f"\n=== Table 3 (B={B}, R={R}) ===")
    for n in [100, 500, 2000, 10000]:
        D = multinomial_stats(np.full(B, 1 / B), n, n, R, rng)
        ok = np.isfinite(D).all(axis=1) & (D[:, 0] > 0)
        r1, r2 = D[ok, 2] / D[ok, 0], D[ok, 1] / D[ok, 0]
        print(f"{n} & {r1.mean():.5f} & {r1.std():.5f} & {r2.mean():.5f} & {r2.std():.5f} \\\\   (excluded {(~ok).sum()})")

def table6(B=10, R=100000):
    rng = np.random.default_rng([2026, 6])
    print(f"\n=== Table 6 (B={B}, R={R}): n ratio Neff*pmin | uncond PSI KLD JSD | %inf PSI KLD | cond PSI KLD ===")
    for n in [200, 500, 1000, 5000]:
        for ratio in [1, 5, 20, 100]:
            w = np.geomspace(1.0, 1.0 / ratio, B); p = w / w.sum()
            D = multinomial_stats(p, n, n, R, rng)
            rej = (D > critical_values(B, n, n)).mean(axis=0)
            inf = np.isinf(D).mean(axis=0) * 100
            cond = [(D[np.isfinite(D[:, j]), j] > critical_values(B, n, n)[j]).mean() for j in (0, 1)]
            print(f"{n} & {ratio} & {n/2*p.min():.1f} & {rej[0]:.4f} & {rej[1]:.4f} & {rej[2]:.4f} & {inf[0]:.1f} & {inf[1]:.1f} \\\\   cond {cond[0]:.4f} {cond[1]:.4f}")

def table7(n=1000, R=100000):
    rng = np.random.default_rng([2026, 7])
    print(f"\n=== Table 7 (n=m={n}, R={R}): B Neff/B | uncond PSI KLD JSD | %inf PSI KLD | cond PSI KLD ===")
    for B in [5, 10, 20, 50, 100, 200, 500]:
        D = multinomial_stats(np.full(B, 1 / B), n, n, R, rng, chunk=5000)
        cv = critical_values(B, n, n)
        rej = (D > cv).mean(axis=0); inf = np.isinf(D).mean(axis=0) * 100
        cond = [((D[np.isfinite(D[:, j]), j] > cv[j]).mean() if np.isfinite(D[:, j]).any() else np.nan) for j in (0, 1)]
        print(f"{B} & {n/2/B:.1f} & {rej[0]:.4f} & {rej[1]:.4f} & {rej[2]:.4f} & {inf[0]:.1f} & {inf[1]:.1f} \\\\   cond {cond[0]:.4f} {cond[1]:.4f}")

def ks_repeat(B=10, n=1000, R=10000, runs=40):
    """Run-to-run variability of the Kolmogorov-Smirnov p-value for the PSI (Section 4.1)."""
    ps = []
    for k in range(runs):
        D = multinomial_stats(np.full(B, 1 / B), n, n, R, np.random.default_rng([99, k]))
        ps.append(stats.kstest(D[:, 0] / (2 / n), "chi2", args=(B - 1,)).pvalue)
    ps = np.array(ps)
    print(f"\nKS p-value of the PSI over {runs} runs: share below 0.05 = {(ps < 0.05).mean():.2f}, median = {np.median(ps):.3f}")

if __name__ == "__main__":
    table2(); ks_repeat(); table3(); table6(); table7()
