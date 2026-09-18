"""Vector PDF figures at the sn-jnl text width (31 pc = 5.15 in), colour-blind-safe palette."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from fsb.algorithms import METAHEURISTICS, SEARCH
from fsb.data import EXACT

TAB, FIG = Path("results/tables"), Path("figures")
FIG.mkdir(exist_ok=True)
W = 5.15
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.0})
FAMILY = {"GA": "#E69F00", "PSO": "#56B4E9", "GWO": "#009E73", "WOA": "#0072B2",
          "SCA": "#D55E00", "RS": "#CC79A7", "EA": "#000000", "SFS": "#8C8C8C", "ALL": "#F0E442"}


def col(m):
    return FAMILY[m.split("-")[0]]


def style(m):
    return dict(color=col(m), linestyle="--" if m.endswith("-V") else "-",
                marker="o", markersize=3.2, markeredgewidth=0.7,
                markerfacecolor="white" if m.endswith("-V") else col(m))


def label(ax, letter, text):
    ax.set_title(letter, loc="left", fontweight="bold", fontsize=8)
    ax.set_title(text, loc="center")


def log_ticks(axis, values, pct=False):
    axis.set_major_locator(matplotlib.ticker.FixedLocator(values))
    axis.set_minor_locator(matplotlib.ticker.NullLocator())
    axis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: (f"{100 * v:g}%" if pct else f"{v:g}")))


def method_legend(fig, methods, ncol=7):
    handles = [plt.Line2D([], [], label=m, **style(m)) for m in methods]
    fig.legend(handles=handles, loc="outside lower center", ncol=ncol, frameon=False,
               handlelength=2.2, columnspacing=1.0)


def read(name, **kw):
    return pd.read_csv(TAB / f"{name}.csv", **kw)


summary = json.loads(Path("results/summary.json").read_text())

# Figure 2: optimality
hit = read("p1_hit_rate_N", index_col=0).loc[EXACT, SEARCH]
pct = read("p1_percentile_N", index_col=0).loc[EXACT, SEARCH]
t1 = read("table1_datasets", index_col=0)
rows = t1.sort_values("d").index
fig = plt.figure(figsize=(W, 3.3), layout="constrained")
gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1])
ax = fig.add_subplot(gs[0])
im = ax.imshow(hit.loc[rows].values, cmap="cividis", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(SEARCH)), SEARCH, rotation=60, ha="right")
ax.set_yticks(range(len(rows)), [f"{r} ({int(t1.d[r])})" for r in rows])
for i, r in enumerate(rows):
    for j, m in enumerate(SEARCH):
        v = hit.loc[r, m]
        ax.text(j, i, f"{v:.2f}".lstrip("0") if v < 1 else "1", ha="center", va="center",
                fontsize=4.6, color="white" if v < 0.55 else "black")
ax.spines[["left", "bottom"]].set_visible(False)
ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01)
cb.set_label("Hit rate (exact optimum reached)")
label(ax, "a", "Hit rate (features in brackets)")
ax = fig.add_subplot(gs[1])
floor = 1e-6
for i, m in enumerate(SEARCH):
    v = np.maximum(pct[m].values, floor)
    ax.scatter(v, np.full(len(v), i) + np.linspace(-0.25, 0.25, len(v)), s=4, color="#BBBBBB", lw=0)
    ax.scatter([max(pct[m].mean(), floor)], [i], s=22, marker="D" if m.endswith("-V") else "o",
               facecolor="white" if m.endswith("-V") else col(m), edgecolor=col(m), zorder=3)
ax.set_xscale("log")
ax.set_xlim(5e-7, 1.5e-1)
log_ticks(ax.xaxis, [1e-6, 1e-4, 1e-2], pct=True)
ax.set_yticks(range(len(SEARCH)), SEARCH)
ax.invert_yaxis()
ax.set_xlabel("Share of better subsets")
label(ax, "b", "Gap to the optimum")
fig.savefig(FIG / "fig2_optimality.pdf")
plt.close(fig)

# Figure 3: rankings
rk = read("p3_mean_ranks_N", index_col=0).loc[SEARCH]
fig, axes = plt.subplots(1, 2, figsize=(W, 2.9), layout="constrained", width_ratios=[1.25, 1])
ax = axes[0]
cols = ["best_of_30", "mean", "median"]
for m in SEARCH:
    ax.plot(range(3), rk.loc[m, cols].values, **style(m))
ax.set_xticks(range(3), ["Best of 30", "Mean", "Median"])
ax.set_xlim(-0.15, 2.15)
ax.invert_yaxis()
ax.set_ylabel("Mean rank across datasets (1 = best)")
p3 = summary["P3"]
label(ax, "a", "Rank by summary statistic")
ax.text(0.02, 0.02, f"Kendall \u03c4 (best, median) = {p3['tau_best_vs_median']:.2f}",
        transform=ax.transAxes, fontsize=6)
ax = axes[1]
for m in SEARCH:
    s = style(m)
    ax.scatter(rk.loc[m, "validation_fitness"], rk.loc[m, "median"], s=18,
               facecolor=s["markerfacecolor"], edgecolor=s["color"], zorder=3)
lim = [0.5, len(SEARCH) + 0.5]
ax.plot(lim, lim, color="#BBBBBB", lw=0.6, zorder=1)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.invert_xaxis()
ax.invert_yaxis()
ax.set_xlabel("Mean rank by validation fitness")
ax.set_ylabel("Mean rank by test accuracy")
label(ax, "b", "Fitness rank vs test rank")
ax.text(0.98, 0.02, f"Kendall \u03c4 = {p3['tau_fitness_vs_test']:.2f}", transform=ax.transAxes,
        fontsize=6, ha="right")
method_legend(fig, SEARCH, ncol=6)
fig.savefig(FIG / "fig3_rankings.pdf")
plt.close(fig)

# Figure 4: leakage and optimism
p2 = summary["P2"]
labels = SEARCH + ["ALL", "exhaustive_optimum"]
vals = [p2[m]["inflation_pp"] for m in labels]
cis = [p2[m]["ci"] for m in labels]
order = np.argsort(vals)
fig, axes = plt.subplots(1, 2, figsize=(W, 2.8), layout="constrained", width_ratios=[1, 1.1])
ax = axes[0]
for pos, i in enumerate(order):
    m = labels[i]
    c = "#000000" if m == "exhaustive_optimum" else col(m)
    ax.plot(cis[i], [pos, pos], color=c, lw=1.0)
    ax.scatter(vals[i], pos, s=18, facecolor="white" if m.endswith("-V") else c,
               edgecolor=c, zorder=3, marker="s" if m == "exhaustive_optimum" else "o")
ax.axvline(0, color="#BBBBBB", lw=0.6)
ax.set_yticks(range(len(labels)), ["Exhaustive" if labels[i] == "exhaustive_optimum" else labels[i]
                                   for i in order])
ax.set_xlabel("Reported (L) minus leak-free (N), pp")
label(ax, "a", "Inflation of reported accuracy")
ax = axes[1]
hy = read("e6_harder_you_try", index_col=[0, 1])
m_val = hy.val_acc.groupby(level=1).mean() * 100
m_test = hy.test_acc.groupby(level=1).mean() * 100
for ds in EXACT:
    d_ = hy.loc[ds]
    ax.plot(d_.index, (d_.val_acc - d_.test_acc) * 100, color="#CCCCCC", lw=0.5)
ax.plot(m_val.index, m_val - m_test, color="#0072B2", lw=1.6, marker="o", markersize=3,
        label="Mean over datasets")
ax.set_xscale("log")
ax.invert_xaxis()
ax.axhline(0, color="#BBBBBB", lw=0.6)
log_ticks(ax.xaxis, [1.0, 1e-1, 1e-2, 1e-3, 1e-4], pct=True)
ax.set_xlabel("Top share of subsets by validation fitness")
ax.set_ylabel("Validation minus test accuracy, pp")
label(ax, "b", "Optimism of stricter selection")
ax.legend(frameon=False, loc="upper left")
fig.savefig(FIG / "fig4_leakage.pdf")
plt.close(fig)

# Figure 5: budget
bc = read("e2_budget_curve", index_col=0)
fig, axes = plt.subplots(1, 3, figsize=(W, 2.6), layout="constrained")
for ax, (colname, lab, letter) in zip(axes, [("hit_rate", "Hit rate", "a"),
                                             ("test_acc", "Test accuracy (N)", "b"),
                                             ("coverage", "Share of space visited", "c")]):
    for m in SEARCH:
        g = bc[bc.method == m].sort_values("budget")
        ax.plot(g.budget, g[colname], **style(m))
    ax.set_xscale("log")
    ax.set_xticks([100, 300, 1000, 3000], ["100", "300", "1k", "3k"])
    ax.set_xlabel("Fitness calls")
    label(ax, letter, lab)
axes[2].set_yscale("log")
log_ticks(axes[2].yaxis, [0.02, 0.05, 0.1, 0.2, 0.5], pct=True)
method_legend(fig, SEARCH, ncol=6)
fig.savefig(FIG / "fig5_budget.pdf")
plt.close(fig)

# Figure 6: how many datasets a ranking needs (exploratory)
e1 = read("e1_datasets_needed", index_col=0)
fig, axes = plt.subplots(1, 2, figsize=(W, 2.3), layout="constrained")
ax = axes[0]
ax.plot(e1.k, e1.p_same_winner, color="#0072B2", marker="o", markersize=3,
        label=f"Same winner as all {len(EXACT)}")
ax.plot(e1.k, e1.p_friedman_sig, color="#D55E00", marker="s", markersize=3, linestyle="--",
        label="Friedman p < 0.05")
ax.set_ylim(0, 1.02)
ax.set_xlabel("Datasets in the comparison")
ax.set_ylabel("Probability over 1,000 draws")
ax.legend(frameon=False)
label(ax, "a", "Winner stability")
ax = axes[1]
ax.plot(e1.k, e1.mean_tau, color="#009E73", marker="o", markersize=3)
ax.set_ylim(0, 1.02)
ax.set_xlabel("Datasets in the comparison")
ax.set_ylabel("Kendall \u03c4 with the full ranking")
label(ax, "b", "Ranking agreement")
fig.savefig(FIG / "fig6_datasets_needed.pdf")
plt.close(fig)

# Figure S1: structure of the exact tables and tie-rule agreement (supplementary)
st = read("e5_structure", index_col=0)
agg = st.groupby("dataset").agg(d=("d", "first"), lo=("local_optima", "mean"))
agg["lo_frac"] = agg.lo / (2.0 ** agg.d - 1)
agg["meta_hit"] = hit[METAHEURISTICS].mean(axis=1)
tie = read("e8_sklearn_agreement", index_col=0).iloc[:, 0].reindex(EXACT).dropna()
fig, axes = plt.subplots(1, 2, figsize=(W, 2.5), layout="constrained")
ax = axes[0]
ax.scatter(agg.lo_frac, agg.meta_hit, s=16, color="#0072B2")
for ds, r in agg.iterrows():
    ax.annotate(ds, (r.lo_frac, r.meta_hit), fontsize=5, xytext=(2, 2), textcoords="offset points")
ax.set_xscale("log")
ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 3.0)))
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:g}%"))
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel("Local optima under one-bit flips (share of subsets)")
ax.set_ylabel("Mean metaheuristic hit rate")
label(ax, "a", "Search difficulty")
ax = axes[1]
ax.barh(range(len(tie)), tie.values * 100, color="#56B4E9")
ax.set_yticks(range(len(tie)), tie.index)
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel("Agreement with scikit-learn, %")
label(ax, "b", "Effect of tie handling")
fig.savefig(FIG / "figS1_structure_ties.pdf")
plt.close(fig)
print("Wrote", ", ".join(sorted(p.name for p in FIG.glob("*.pdf"))))
