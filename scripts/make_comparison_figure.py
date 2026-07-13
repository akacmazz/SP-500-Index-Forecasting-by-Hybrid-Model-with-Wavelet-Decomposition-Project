"""Build the old-vs-new comparison figure.

Recreates the two charts from the original project exactly as they were drawn, and puts each one
next to the same quantity measured honestly. Reads only the prediction CSVs exported by the
notebooks — no retraining.

    python scripts/make_comparison_figure.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import r2_score

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.25,
                     "font.size": 9, "axes.titlesize": 10})

RED, GREEN, BLUE, GREY = "#c1121f", "#2a9d5c", "#1d4e89", "0.45"

naive = pd.read_csv("data/predictions_naive_leaky.csv", parse_dates=["date"])
honest = pd.read_csv("data/predictions_leakfree_returns.csv", parse_dates=["date"])
close = pd.read_csv("data/close_prices.csv", parse_dates=["date"]).set_index("date")["close"]
vol = pd.read_csv("data/predictions_spx_h1.csv", parse_dates=["date"]).set_index("date")

r2_naive = r2_score(naive["actual"], naive["pred"])
r2_honest = r2_score(honest["actual"], honest["pred"])

# the same 40-day window the original project reported, but scored with the leak-free model
win = honest[honest["date"].isin(naive["date"])]
r2_honest_win = r2_score(win["actual"], win["pred"]) if len(win) > 2 else np.nan


def to_prices(p0, rets):
    """The original project's reconstruction: start from the TRUE price, compound the returns."""
    return p0 * np.exp(np.cumsum(rets))


fig = plt.figure(figsize=(15.5, 12.2))
gs = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.16, height_ratios=[1, 1, 1])

fig.suptitle("The same project, before and after the leak was found",
             fontsize=15, fontweight="bold", y=0.975)
fig.text(0.5, 0.945, "left: what the original notebook reported          "
                     "right: the identical quantity, measured without leakage",
         ha="center", fontsize=10, color="0.35")

# ─────────── ROW 1 — daily log returns, on the ORIGINAL's own 40-day window ───────────
ax = fig.add_subplot(gs[0, 0])
ax.plot(naive["date"], naive["actual"], lw=1.3, color=GREY, label="Actual")
ax.plot(naive["date"], naive["pred"], lw=1.6, color=GREEN, label="Predicted")
ylim = ax.get_ylim()
ax.set_title(f"BEFORE — 'Hybrid Model Predictions'\n"
             f"R² = {r2_naive:.3f}   on the {len(naive)} days the Random Forest was fitted on",
             color=RED, fontweight="bold")
ax.set_ylabel("daily log return")
ax.legend(fontsize=8, loc="upper left")
ax.xaxis.set_major_locator(MaxNLocator(5))
plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
ax.set_xlabel("the model was shown these answers before it 'predicted' them",
              fontsize=8, color=RED, style="italic", labelpad=8)

ax = fig.add_subplot(gs[0, 1])
ax.plot(win["date"], win["actual"], lw=1.3, color=GREY, label="Actual")
ax.plot(win["date"], win["pred"], lw=1.8, color=BLUE, label="Leak-free prediction")
ax.set_ylim(ylim)                                   # identical axis — this is apples to apples
ax.set_title(f"AFTER — the exact same {len(win)} days, leak-free\n"
             f"R² = {r2_honest_win:+.4f}   (and {r2_honest:+.4f} across all "
             f"{len(honest)} test days)", color=GREEN, fontweight="bold")
ax.set_ylabel("daily log return")
ax.legend(fontsize=8, loc="upper left")
ax.xaxis.set_major_locator(MaxNLocator(5))
plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
ax.set_xlabel("same days, same axis — the flat line IS the right answer",
              fontsize=8, color=GREEN, style="italic", labelpad=8)

# ────────────────────────────── ROW 2 — the price chart ──────────────────────────────
train_end = naive["date"].iloc[0]
hist = close[close.index <= train_end]

ax = fig.add_subplot(gs[1, 0])
ax.plot(hist.index, hist.values, lw=0.8, color=BLUE, label="Model Training Data")
p0 = close.loc[naive["date"].iloc[0]]
ax.plot(naive["date"], to_prices(p0, naive["actual"]), lw=1.6, color=RED, label="Actual Data")
ax.plot(naive["date"], to_prices(p0, naive["pred"]), lw=1.6, color="orange", label="Predicted Data")
ax.set_title("BEFORE — 'Stock Price Prediction by LSTM'\n"
             "11 years of training data next to a 40-day forecast, at the same scale",
             color=RED, fontweight="bold")
ax.set_ylabel("S&P 500 index level")
ax.legend(fontsize=8, loc="upper left")
ax.set_xlabel("the forecast is the ~1% sliver on the right — and both curves start from the "
              "SAME true price", fontsize=8, color=RED, style="italic", labelpad=8)

ax = fig.add_subplot(gs[1, 1])
p0h = close.loc[honest["date"].iloc[0]]
ax.plot(honest["date"], to_prices(p0h, honest["actual"]), lw=1.4, color=GREY, label="Actual")
ax.plot(honest["date"], to_prices(p0h, honest["pred"]), lw=1.8, color=BLUE,
        label="Leak-free prediction")
ax.set_title("AFTER — the identical reconstruction, full test set\n"
             "a leak-free return forecast cannot track the price path at all",
             color=GREEN, fontweight="bold")
ax.set_ylabel("S&P 500 index level")
ax.legend(fontsize=8, loc="upper left")
ax.set_xlabel("zoom out past 40 days and the illusion disappears completely",
              fontsize=8, color=GREEN, style="italic", labelpad=8)

# ────────────────────────────── ROW 3 — what actually works ──────────────────────────────
ax = fig.add_subplot(gs[2, :])
ann = np.sqrt(252) * 100
ax.plot(vol.index, np.exp(vol["y_log_rv"]) * ann, lw=0.8, color=GREY, label="Realized volatility")
ax.plot(vol.index, np.exp(vol["WaveHAR"]) * ann, lw=1.5, color=RED,
        label="WaveHAR forecast (causal wavelet + OLS)")
r2_vol = r2_score(vol["y_log_rv"], vol["WaveHAR"])
ax.set_title(f"WHERE THE SIGNAL ACTUALLY IS — next-day realized volatility: "
             f"same wavelet, same discipline, same test days\n"
             f"leak-free out-of-sample R² = {r2_vol:+.4f}   "
             f"(vs {r2_honest:+.4f} for returns — the architecture was never wrong, "
             f"the target was)", color=GREEN, fontweight="bold")
ax.set_ylabel("annualised volatility (%)")
ax.legend(fontsize=8, loc="upper right")

fig.legend(handles=[
    Patch(facecolor="white", edgecolor=RED, label="reported by the original project — an artifact"),
    Patch(facecolor="white", edgecolor=GREEN, label="measured without leakage — real")],
    loc="lower center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.005))

plt.savefig("figures/old_vs_new.png", dpi=145, bbox_inches="tight", facecolor="white")
print("wrote figures/old_vs_new.png")
print(f"  returns, leaky   R² = {r2_naive:+.4f}  ({len(naive)} days)")
print(f"  returns, honest  R² = {r2_honest:+.4f}  ({len(honest)} days)")
print(f"  returns, honest on the ORIGINAL's own 40-day window: R² = {r2_honest_win:+.4f}")
print(f"  volatility       R² = {r2_vol:+.4f}  ({len(vol)} days)")
