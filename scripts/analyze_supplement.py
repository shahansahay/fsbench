"""Tables, summary and Figure S3 for the exploratory larger-dataset replication (live evaluation)."""
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

from fsb.algorithms import SEARCH

DATASETS = ["BreastEW", "IonosphereEW", "SonarEW"]
FEATURES = {"BreastEW": 30, "IonosphereEW": 34, "SonarEW": 60}
ORDER = SEARCH + ["ALL"]
OUT = Path("results/supplement")
(OUT / "tables").mkdir(parents=True, exist_ok=True)

runs = pd.read_csv(OUT / "runs.csv.gz")
N, L = runs[runs.protocol == "N"], runs[runs.protocol == "L"]


def table(df, value, methods=ORDER):
    t = df.groupby(["dataset", "fold", "method"])[value].median().groupby(["dataset", "method"]).mean()
    return t.unstack("method").reindex(index=DATASETS, columns=methods)


test_n, rep_l, fit_n = table(N, "test_acc"), table(L, "fit_acc"), table(N, "fitness")
infl = (rep_l - test_n) * 100
gain_n = test_n[SEARCH].sub(test_n["ALL"], axis=0) * 100
gain_l = rep_l[SEARCH].sub(rep_l["ALL"], axis=0) * 100
best_known = N.groupby(["dataset", "fold"]).fitness.min().rename("best_known")
Nk = N.join(best_known, on=["dataset", "fold"])
Nk = Nk.assign(at_best=(Nk.fitness == Nk.best_known).astype(int))
reach = (Nk.groupby(["dataset", "fold", "method"]).at_best.mean().groupby(["dataset", "method"]).mean()
    .unstack("method").reindex(index=DATASETS, columns=SEARCH))
r_fit = fit_n[SEARCH].rank(axis=1).mean()
r_test = test_n[SEARCH].rank(axis=1, ascending=False).mean()
tau = float(stats.kendalltau(r_fit.values, r_test.reindex(r_fit.index).values).statistic)

for name, t in [("test_acc_N", test_n), ("reported_acc_L", rep_l), ("inflation_pp", infl),
                ("gain_vs_all_N_pp", gain_n), ("gain_vs_all_L_pp", gain_l), ("reach_best_known_N", reach)]:
    t.to_csv(OUT / "tables" / f"{name}.csv", float_format="%.6g")

exact_infl = json.loads(Path("results/summary.json").read_text())["P2"]["all_search_methods"]["inflation_pp"]
summary = dict(
    inflation_pp_mean=float(infl[SEARCH].values.mean()),
    inflation_pp_by_dataset={ds: float(infl.loc[ds, SEARCH].mean()) for ds in DATASETS},
    inflation_pp_exact_suite=exact_infl,
    gain_vs_all_N_pp={ds: float(gain_n.loc[ds].mean()) for ds in DATASETS},
    gain_vs_all_L_pp={ds: float(gain_l.loc[ds].mean()) for ds in DATASETS},
    test_acc_spread_across_methods_pp={ds: float((test_n.loc[ds, SEARCH].max() - test_n.loc[ds, SEARCH].min()) * 100)
                                       for ds in DATASETS},
    tau_fitness_vs_test=tau,
    reach_best_known_N={m: float(reach[m].mean()) for m in SEARCH})
(OUT / "summary_supplement.json").write_text(json.dumps(summary, indent=2))

FAMILY = {"GA": "#E69F00", "PSO": "#56B4E9", "GWO": "#009E73", "WOA": "#0072B2",
          "SCA": "#D55E00", "RS": "#CC79A7", "EA": "#000000", "SFS": "#8C8C8C"}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 7, "axes.titlesize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "legend.fontsize": 5.8, "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6})
fig, axes = plt.subplots(1, 3, figsize=(5.15, 2.9), layout="constrained", width_ratios=[1.1, 0.8, 1])
ax = axes[0]
marks = {"BreastEW": "o", "IonosphereEW": "s", "SonarEW": "^"}
for i, m in enumerate(SEARCH):
    for ds in DATASETS:
        ax.scatter(infl.loc[ds, m], i, s=12, marker=marks[ds], facecolor="white",
                   edgecolor=FAMILY[m.split("-")[0]], lw=0.8, zorder=3)
ax.axvline(exact_infl, color="#888888", lw=0.8, ls="--")
ax.set_yticks(range(len(SEARCH)), SEARCH)
ax.invert_yaxis()
ax.set_xlabel("L minus N, pp")
ax.set_title("a", loc="left", fontweight="bold", fontsize=8)
ax.set_title("Inflation", loc="center")
handles = [plt.Line2D([], [], marker=marks[ds], ls="", markersize=4, markerfacecolor="white",
                      markeredgecolor="#444444", label=f"{ds} ({FEATURES[ds]})") for ds in DATASETS]
handles.append(plt.Line2D([], [], color="#888888", ls="--", label="Mean, 12 exact datasets"))
handles.append(plt.Line2D([], [], marker="s", ls="", markersize=4, markerfacecolor="white",
                          markeredgecolor="#D55E00", label="Reported under L"))
handles.append(plt.Line2D([], [], marker="o", ls="", markersize=4, color="#0072B2", label="Leak-free under N"))
fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
ax = axes[1]
xs = np.arange(len(DATASETS))
gl, gn = gain_l.mean(axis=1).values, gain_n.mean(axis=1).values
for x in xs:
    ax.plot([x, x], [gn[x], gl[x]], color="#BBBBBB", lw=0.9, zorder=1)
ax.scatter(xs, gl, s=20, marker="s", facecolor="white", edgecolor="#D55E00", zorder=3)
ax.scatter(xs, gn, s=20, color="#0072B2", zorder=3)
ax.axhline(0, color="#888888", lw=0.6)
ax.set_xticks(xs, ["Breast\n(30)", "Iono.\n(34)", "Sonar\n(60)"])
ax.set_xlim(-0.5, len(DATASETS) - 0.5)
ax.set_ylabel("Gain over all features, pp")
ax.set_title("b", loc="left", fontweight="bold", fontsize=8)
ax.set_title("Selection gain", loc="center")
ax = axes[2]
for m in SEARCH:
    c = FAMILY[m.split("-")[0]]
    ax.scatter(r_fit[m], r_test[m], s=18, facecolor="white" if m.endswith("-V") else c, edgecolor=c, zorder=3)
lim = [0.5, len(SEARCH) + 0.5]
ax.plot(lim, lim, color="#BBBBBB", lw=0.6, zorder=1)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.invert_xaxis()
ax.invert_yaxis()
ax.set_xlabel("Rank by validation fitness")
ax.set_ylabel("Rank by test accuracy")
ax.text(0.98, 0.02, f"Kendall \u03c4 = {tau:.2f}", transform=ax.transAxes, fontsize=6, ha="right")
ax.set_title("c", loc="left", fontweight="bold", fontsize=8)
ax.set_title("Fitness vs test rank", loc="center")
Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/figS3_larger_datasets.pdf")
print(json.dumps(summary, indent=2))
print("Wrote results/supplement/tables/*.csv, results/supplement/summary_supplement.json, "
      "figures/figS3_larger_datasets.pdf")
