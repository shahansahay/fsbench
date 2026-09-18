"""Exploratory additions made after the frozen analysis (label them as such in the paper).

X1  Friedman test and Kendall's W for optimisation (hit rate, validation fitness) versus
    generalisation (leak-free test accuracy); protocol N, budget 1,000, 12 search methods.
X2  Bayesian signed-rank test (Benavoli et al. 2017) of each metaheuristic against the (1+1) EA
    and against SFS on leak-free test accuracy. Region of practical equivalence +-1 pp
    (sensitivity +-0.5 pp); prior strength 0.5 on a pseudo-observation at zero; 50,000 draws.
Writes results/tables/x1_friedman_contrast.csv, results/tables/x2_bayes_signed_rank.csv,
results/summary_extra.json and figures/figS2_equivalence.pdf.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from fsb.algorithms import METAHEURISTICS, SEARCH
from fsb.data import EXACT

OUT, TAB, FIG = Path("results"), Path("results/tables"), Path("figures")
RNG = np.random.default_rng(20260920)
NSAMPLES, PRIOR = 50000, 0.5

runs = pd.read_csv(OUT / "runs.csv.gz")
N = runs[(runs.experiment == "main") & (runs.protocol == "N") & runs.method.isin(SEARCH)]


def dataset_table(value, stat):
    g = N.groupby(["dataset", "fold", "method"])[value]
    fold = g.median() if stat == "median" else g.mean()
    t = fold.groupby(["dataset", "method"]).mean().unstack("method")
    return t.reindex(index=[d for d in EXACT if d in t.index], columns=SEARCH)


def friedman(table, higher_is_better):
    fr = stats.friedmanchisquare(*[table[m].values for m in SEARCH])
    n, k = table.shape
    ranks = table.rank(axis=1, ascending=not higher_is_better).mean(axis=0)
    return dict(chi2=float(fr.statistic), p=float(fr.pvalue),
                kendall_w=float(fr.statistic / (n * (k - 1))),
                best=ranks.idxmin(), worst=ranks.idxmax(),
                mean_rank_best=float(ranks.min()), mean_rank_worst=float(ranks.max()))


def bayes_signed_rank(diff, rope):
    """Posterior probabilities (left, rope, right) of the Bayesian signed-rank test."""
    z = np.concatenate([[0.0], np.asarray(diff, dtype=float)])
    alpha = np.ones(len(z))
    alpha[0] = PRIOR
    w = RNG.dirichlet(alpha, NSAMPLES)
    s = z[:, None] + z[None, :]

    def mass(region):
        return ((w @ region.astype(float)) * w).sum(axis=1)

    masses = np.stack([mass(s < -2 * rope), mass(np.abs(s) <= 2 * rope), mass(s > 2 * rope)], axis=1)
    winner = masses.argmax(axis=1)
    return [float((winner == k).mean()) for k in range(3)]


hit, fit, acc = dataset_table("hit", "mean"), dataset_table("fitness", "median"), dataset_table("test_acc", "median")
x1 = {"optimisation: hit rate": friedman(hit, True),
      "optimisation: validation fitness": friedman(fit, False),
      "generalisation: leak-free test accuracy": friedman(acc, True)}
pd.DataFrame(x1).T.to_csv(TAB / "x1_friedman_contrast.csv", float_format="%.6g")

rows = []
for c in ("EA", "SFS"):
    for m in METAHEURISTICS:
        diff = (acc[m] - acc[c]).values * 100
        for rope in (1.0, 0.5):
            p_c, p_eq, p_m = bayes_signed_rank(diff, rope)
            rows.append(dict(method=m, control=c, rope_pp=rope, mean_diff_pp=float(diff.mean()),
                             p_control_better=p_c, p_equivalent=p_eq, p_method_better=p_m))
x2 = pd.DataFrame(rows)
x2.to_csv(TAB / "x2_bayes_signed_rank.csv", index=False, float_format="%.6g")

main1 = x2[x2.rope_pp == 1.0]
summary = dict(X1=x1, X2_rope_1pp=dict(
    min_p_equivalent=float(main1.p_equivalent.min()), median_p_equivalent=float(main1.p_equivalent.median()),
    max_p_method_better=float(main1.p_method_better.max()),
    max_p_control_better=float(main1.p_control_better.max())))
(OUT / "summary_extra.json").write_text(json.dumps(summary, indent=2))

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 7, "axes.titlesize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "legend.fontsize": 6, "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6})
fig, axes = plt.subplots(1, 2, figsize=(5.15, 2.6), layout="constrained", sharey=True)
parts = [("p_control_better", "Control better", "#D55E00"), ("p_equivalent", "Equivalent (\u00b11 pp)", "#BBBBBB"),
         ("p_method_better", "Metaheuristic better", "#0072B2")]
for ax, c, letter in zip(axes, ("EA", "SFS"), "ab"):
    d = main1[main1.control == c].set_index("method").loc[METAHEURISTICS]
    left = np.zeros(len(d))
    for colname, lab, colour in parts:
        ax.barh(range(len(d)), d[colname].values, left=left, color=colour, label=lab, height=0.7)
        left += d[colname].values
    ax.set_yticks(range(len(d)), d.index)
    if letter == "a":
        ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Posterior probability")
    ax.set_title(letter, loc="left", fontweight="bold", fontsize=8)
    ax.set_title("Against the (1+1) EA" if c == "EA" else "Against SFS", loc="center")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
fig.savefig(FIG / "figS2_equivalence.pdf")

print(pd.DataFrame(x1).T[["chi2", "p", "kendall_w"]].to_string(float_format=lambda v: f"{v:.3g}"))
print()
print(main1[["method", "control", "mean_diff_pp", "p_control_better", "p_equivalent", "p_method_better"]]
      .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print()
print(json.dumps(summary["X2_rope_1pp"], indent=2))
print("Wrote results/tables/x1_friedman_contrast.csv, x2_bayes_signed_rank.csv, "
      "results/summary_extra.json, figures/figS2_equivalence.pdf")
