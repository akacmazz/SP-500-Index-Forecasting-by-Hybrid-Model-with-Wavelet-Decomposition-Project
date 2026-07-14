"""Build the three summary figures for the README.

Every number plotted here is recomputed from the data, not copied from a notebook.

    python scripts/make_figures.py
"""

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.tsa.arima.model import ARIMA

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.25, "font.size": 9})

RED, GREEN, BLUE, GREY, ORANGE = "#c1121f", "#2a9d5c", "#1d4e89", "0.55", "#e07a1f"
SEED, ANN = 0, np.sqrt(252)
rng = np.random.default_rng(SEED)

df = pd.read_csv("data/gspc.csv", index_col="Date", parse_dates=True)
O, H, L, C = (np.array(df[k].to_numpy(), float) for k in ("Open", "High", "Low", "Close"))
gk = 0.5 * np.log(H / L) ** 2 - (2 * np.log(2) - 1) * np.log(C / O) ** 2
r = np.diff(np.log(C))
lrv = 0.5 * np.log(gk)[1:]
overnight = np.log(O[1:] / C[:-1])
dates = df.index[1:]
split = int(len(r) * 0.8)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — the noise ablation: the model contributes nothing to its own result
# ══════════════════════════════════════════════════════════════════════════════
# Reproduce the leaky combiner exactly: fit the Random Forest on the TEST targets,
# then "predict" the same rows. Then feed it different things in place of the LSTM
# outputs and see whether the reported R² cares.
arima = ARIMA(r[:split], order=(2, 0, 0)).fit()
arima_test = arima.forecast(len(r) - split)          # long-horizon AR(2) -> ~constant
resid_test = r[split:] - arima_test
SEQ, N_PRED = 10, 40                                  # what the leaky pipeline actually scored
y_leak = resid_test[SEQ:SEQ + N_PRED]
truth = r[split:][SEQ:SEQ + N_PRED]
base = arima_test[SEQ:SEQ + N_PRED]


def leaky_r2(X):
    """The leak, verbatim: fit on the answer key, predict the same rows."""
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=SEED)
    rf.fit(X, y_leak)
    return r2_score(truth, base + rf.predict(X))


cases = [
    ("A plausible 'LSTM output'\n(noisy copy of the signal)",
     np.column_stack([y_leak * 0.1 + rng.normal(0, .002, N_PRED) for _ in range(5)]), GREY),
    ("PURE GAUSSIAN NOISE", rng.normal(size=(N_PRED, 5)), RED),
    ("PURE NOISE (another seed)", np.random.default_rng(7).normal(size=(N_PRED, 5)), RED),
    ("A meaningless ramp\n0, 1, 2, …, N",
     np.column_stack([np.arange(N_PRED)] + [rng.normal(size=N_PRED) for _ in range(4)]), RED),
    ("Four zero columns\n+ one noise column",
     np.column_stack([np.zeros(N_PRED)] * 4 + [rng.normal(size=N_PRED)]), RED),
]
scores = [(lbl, leaky_r2(X), c) for lbl, X, c in cases]

fig, ax = plt.subplots(figsize=(10.5, 4.6))
ys = np.arange(len(scores))[::-1]
ax.barh(ys, [s for _, s, _ in scores], color=[c for _, _, c in scores], height=0.62, alpha=0.9)
for y, (_, s, _) in zip(ys, scores):
    ax.text(s + 0.012, y, f"R² = {s:.3f}", va="center", fontsize=10, fontweight="bold")
ax.set_yticks(ys)
ax.set_yticklabels([lbl for lbl, _, _ in scores], fontsize=9)
ax.axvline(0, color="k", lw=1)
ax.set_xlim(0, 1.05)
ax.set_xlabel("out-of-sample R² the pipeline reports")
ax.set_title("The pipeline contributes nothing to its own headline result\n"
             "Replace every LSTM output with garbage — the reported R² does not move",
             fontsize=11, fontweight="bold", color=RED)
ax.text(0.99, -0.19, "because the Random Forest combiner was fitted on the test set's own labels",
        transform=ax.transAxes, ha="right", fontsize=8.5, style="italic", color=RED)
plt.tight_layout()
plt.savefig("figures/noise_ablation.png", dpi=145, bbox_inches="tight", facecolor="white")
print("wrote figures/noise_ablation.png")
for lbl, s, _ in scores:
    print(f"   {lbl.splitlines()[0]:<40} R² = {s:+.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — the whole project in one chart
# ══════════════════════════════════════════════════════════════════════════════
FIRST, J = 128, 5


def roll(x, w):
    cs = np.concatenate([[0.0], np.cumsum(x)])
    o = np.full(len(x), np.nan)
    o[w - 1:] = (cs[w:] - cs[:-w]) / w
    return o


def haar(x, levels=J):
    x = np.asarray(x, float)
    c, cols = x.copy(), []
    for j in range(levels):
        lag = 2 ** j
        cn = c.copy()
        cn[lag:] = 0.5 * (c[lag:] + c[:-lag])
        cols.append(c - cn)
        c = cn
    cols.append(c)
    return np.column_stack(cols)


iv = {}
for t in ["^VIX", "^VIX3M", "^VVIX"]:
    d = yf.download(t, start="2010-01-01", end="2024-01-01", progress=False, auto_adjust=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    iv[t] = np.log(d["Close"].reindex(dates).ffill().bfill().to_numpy())

log_vix = iv["^VIX"] - np.log(100 * ANN)
VIXF = np.column_stack([log_vix, roll(log_vix, 5), iv["^VIX3M"] - iv["^VIX"],
                        iv["^VVIX"], log_vix - lrv])
lev, jmp = np.minimum(r, 0.0), np.abs(overnight)
LJ = np.column_stack([lev, roll(lev, 5), jmp, roll(jmp, 5)])
HAR = np.column_stack([lrv, roll(lrv, 5), roll(lrv, 22)])

y_v = np.full(len(lrv), np.nan)
y_v[:-1] = lrv[1:]
origins = np.arange(FIRST, len(lrv) - 1)
tr, te = origins[origins < split], origins[origins >= split]


def rolling_ols(X, window=500):
    p = np.full(len(te), np.nan)
    for i, t in enumerate(te):
        us = origins[(origins <= t - 1) & (origins >= t - window)]
        if len(us) < 250:
            continue
        m = LinearRegression().fit(X[us], y_v[us])
        p[i] = X[t] @ m.coef_ + m.intercept_
    return p


VOL = {
    "HAR-RV\n(field standard, 2009)": HAR,
    "+ leverage & jumps": np.column_stack([HAR, LJ]),
    "+ causal wavelet\n(WaveHAR-LJ)": np.column_stack([haar(lrv), LJ]),
    "+ implied volatility\n(VIX suite)": np.column_stack([haar(lrv), LJ, VIXF]),
}
vol_r2 = {}
for k, X in VOL.items():
    p = rolling_ols(X)
    ok = ~np.isnan(p)
    vol_r2[k] = r2_score(y_v[te][ok], p[ok])

naive = pd.read_csv("data/predictions_naive_leaky.csv")
honest = pd.read_csv("data/predictions_leakfree_returns.csv")
r2_leaky = r2_score(naive["actual"], naive["pred"])
r2_honest = r2_score(honest["actual"], honest["pred"])

fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1, 1.35]})

a1.bar([0, 1], [r2_leaky, r2_honest], color=[RED, BLUE], width=0.55)
for x, v in zip([0, 1], [r2_leaky, r2_honest]):
    a1.text(x, v + 0.03 if v > 0 else 0.03, f"{v:+.3f}", ha="center", fontsize=12,
            fontweight="bold")
a1.set_xticks([0, 1])
a1.set_xticklabels(["leaky pipeline\n(as published)", "leak-free\n(honest)"], fontsize=9)
a1.axhline(0, color="k", lw=1)
a1.set_ylim(-0.08, 1.0)
a1.set_ylabel("out-of-sample R²")
a1.set_title("TARGET: daily returns\nthere is nothing to find — and that is the answer",
             fontsize=10.5, fontweight="bold", color=RED)

ks = list(VOL)
vals = [vol_r2[k] for k in ks]
cols = [GREY, GREY, ORANGE, GREEN]
a2.bar(range(len(ks)), vals, color=cols, width=0.6)
for x, v in enumerate(vals):
    a2.text(x, v + 0.008, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
a2.set_xticks(range(len(ks)))
a2.set_xticklabels(ks, fontsize=8.5)
a2.set_ylim(0, max(vals) * 1.18)
a2.set_ylabel("out-of-sample R²")
a2.set_title("TARGET: realized volatility — the same architecture, pointed correctly\n"
             "the biggest single gain comes from INFORMATION (VIX), not architecture",
             fontsize=10.5, fontweight="bold", color=GREEN)
plt.tight_layout()
plt.savefig("figures/project_summary.png", dpi=145, bbox_inches="tight", facecolor="white")
print("\nwrote figures/project_summary.png")
print(f"   returns  leaky {r2_leaky:+.3f} | honest {r2_honest:+.3f}")
for k, v in vol_r2.items():
    print(f"   volatility  {k.splitlines()[0]:<28} {v:+.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — the pipeline, and where each stage leaked
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 5.6))
ax.set_xlim(0, 100)
ax.set_ylim(0, 46)
ax.axis("off")
ax.grid(False)


def box(x, y, w, h, text, fc, ec, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.5)


def arrow(x1, y1, x2, y2, color="0.35"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15,
                                 lw=1.5, color=color, shrinkA=2, shrinkB=2))


STAGES = [
    (2, "S&P 500\nlog returns"),
    (21, "ARFIMA\n→ residuals"),
    (40, "Wavelet\ndb4, level 4"),
    (59, "one LSTM\nper component"),
    (78, "Random Forest\ncombiner"),
]
for x, t in STAGES:
    box(x, 28, 18, 11, t, "#eef2f7", BLUE, bold=True)
for i in range(len(STAGES) - 1):
    arrow(STAGES[i][0] + 18, 33.5, STAGES[i + 1][0], 33.5)

ax.text(50, 43.5, "The proposed pipeline (Bukhari et al., IEEE Access 2020)",
        ha="center", fontsize=11.5, fontweight="bold")

LEAKS = [
    (40, "❶  decomposed over the\nWHOLE series → the past\nsees the future"),
    (59, "❷  coefficient index used\nas a day index → inputs from\nup to 735 days AHEAD"),
    (78, "❸  .fit() on the TEST labels\ninside predict() → the model\nmemorises the answers"),
]
for x, t in LEAKS:
    arrow(x + 9, 27.5, x + 9, 21.5, RED)
    box(x, 8, 18, 13, t, "#fdecec", RED, fs=7.8)

ax.text(2, 17, "Reported:\nR² = 0.83", ha="left", fontsize=11, fontweight="bold", color=RED)
ax.text(2, 10.5, "Honest:\nR² ≈ 0", ha="left", fontsize=11, fontweight="bold", color=BLUE)
ax.text(50, 2, "Every stage of the pipeline leaks future information. "
               "Notebook 1 measures each one.",
        ha="center", fontsize=9, style="italic", color="0.35")
plt.tight_layout()
plt.savefig("figures/pipeline_leaks.png", dpi=145, bbox_inches="tight", facecolor="white")
print("\nwrote figures/pipeline_leaks.png")
