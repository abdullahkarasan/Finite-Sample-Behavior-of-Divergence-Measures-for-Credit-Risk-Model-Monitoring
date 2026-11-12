import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from common import *

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
D = np.load("validation_stats.npy"); R = D.shape[0]; B, n, m = 10, 1000, 1000; df = B - 1; s = 1 / n + 1 / m
COL = {"PSI": "#0072B2", "KLD": "#D55E00", "JSD": "#009E73"}     # colour-blind-safe; line styles also differ
STY = {"PSI": dict(color=COL["PSI"], ls="-"), "KLD": dict(color=COL["KLD"], ls="--"), "JSD": dict(color=COL["JSD"], ls=":")}

# ---- Figure 1: histograms and Q-Q plots ----
fig, ax = plt.subplots(3, 2, figsize=(6.6, 7.4))
pp = (np.arange(1, R + 1) - 0.5) / R                    # plotting positions (i - 1/2)/R
for j, name in enumerate(NAMES):
    c = CONST[j] * s; v = np.sort(D[:, j])
    a = ax[j, 0]; x = np.linspace(0, v.max(), 400)
    a.hist(v, bins=50, density=True, color=COL[name], alpha=0.45, edgecolor="white", linewidth=0.4)
    a.plot(x, stats.chi2.pdf(x / c, df) / c, color="black", lw=1.2)
    a.set_xlabel(name); a.set_ylabel("Density")
    a.text(0.97, 0.93, f"({'ace'[j]})", transform=a.transAxes, ha="right", va="top")
    b = ax[j, 1]; tq = c * stats.chi2.ppf(pp, df)
    b.plot(tq, v, ".", color=COL[name], ms=2.5, rasterized=True)
    lim = [0, max(tq.max(), v.max()) * 1.03]; b.plot(lim, lim, color="black", lw=0.8)
    b.set_xlim(lim); b.set_ylim(lim); b.set_xlabel(f"Asymptotic quantile of {name}"); b.set_ylabel("Empirical quantile")
    b.text(0.05, 0.93, f"({'bdf'[j]})", transform=b.transAxes, ha="left", va="top")
fig.tight_layout()
fig.savefig("divergence_validation_plots.png", dpi=600); fig.savefig("Fig1.pdf", dpi=600); plt.close(fig)

# ---- Figure 2: CDF differences and normalised quantile ratios ----
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.9))
x = np.linspace(0, 30, 600)
for j, name in enumerate(NAMES):
    z = np.sort(D[:, j] / (CONST[j] * s))
    ax[0].plot(x, np.searchsorted(z, x, side="right") / R - stats.chi2.cdf(x, df), lw=1.3, label=name, **STY[name])
ax[0].axhline(0, color="0.6", lw=0.6); ax[0].set_xlabel(r"Statistic in $\chi^2_9$ units"); ax[0].set_ylabel("Empirical minus asymptotic CDF")
ax[0].legend(frameon=False); ax[0].text(0.03, 0.95, "(a)", transform=ax[0].transAxes, va="top")
pr = np.linspace(0.01, 0.99, 99); q = np.quantile(D, pr, axis=0)
ax[1].plot(pr, q[:, 1] / q[:, 0] / 0.5, lw=1.3, label="KLD/PSI relative to 1/2", **STY["KLD"])
ax[1].plot(pr, q[:, 2] / q[:, 0] / 0.125, lw=1.3, label="JSD/PSI relative to 1/8", **STY["JSD"])
ax[1].axhline(1, color="0.6", lw=0.6); ax[1].set_ylim(0.97, 1.03)
ax[1].set_xlabel("Probability level"); ax[1].set_ylabel("Quantile ratio / theoretical value")
ax[1].legend(frameon=False, loc="lower left"); ax[1].text(0.03, 0.95, "(b)", transform=ax[1].transAxes, va="top")
fig.tight_layout()
fig.savefig("cdf_convergence_analysis.png", dpi=600); fig.savefig("Fig2.pdf"); plt.close(fig)
for j, name in enumerate(NAMES):
    z = D[:, j] / (CONST[j] * s)
    print(name, "max |CDF diff| on grid:", np.abs(np.searchsorted(np.sort(z), x, side="right") / R - stats.chi2.cdf(x, df)).max().round(4),
          " emp/theory quantile at 0.5,0.9,0.99,0.999:", np.round([np.quantile(z, a) / stats.chi2.ppf(a, df) for a in (0.5, 0.9, 0.99, 0.999)], 3))
print("quantile ratio ranges: KLD/PSI/0.5", (q[:,1]/q[:,0]/0.5).min().round(4), (q[:,1]/q[:,0]/0.5).max().round(4), " JSD/PSI/0.125", (q[:,2]/q[:,0]/0.125).min().round(4), (q[:,2]/q[:,0]/0.125).max().round(4))
