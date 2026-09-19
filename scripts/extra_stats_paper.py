"""Exploratory analyses added after results-v1 was frozen, for the manuscript.
X3  Two-sided Wilcoxon signed-rank test (SciPy defaults) of the per-dataset mean leak-free test
    accuracy of the 12 search methods minus that of all features (ALL), n = 12 datasets.
X4  Spearman rank correlation over the 12 exact datasets between the mean share of subsets that
    are local optima under one-bit flips and the mean hit rate of the nine metaheuristic variants.
Reads existing result tables only and writes results/summary_extra_paper.json; nothing else changes."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

TAB = Path("results/tables")
MH = ["GA", "PSO-S", "PSO-V", "GWO-S", "GWO-V", "WOA-S", "WOA-V", "SCA-S", "SCA-V"]
SEARCH = MH + ["RS", "EA", "SFS"]

acc = pd.read_csv(TAB / "p2_test_acc_N.csv", index_col=0)
gain = acc[SEARCH].mean(axis=1) - acc["ALL"]
w = stats.wilcoxon(gain.values)
ranks = stats.rankdata(np.abs(gain.values))
x3 = dict(n=int(len(gain)), sum_positive_ranks=float(ranks[gain.values > 0].sum()), p=float(w.pvalue),
          datasets_below_all=int((gain < 0).sum()), mean_gain_pp=float(gain.mean() * 100))

e5 = pd.read_csv(TAB / "e5_structure.csv", index_col=0)
lo = (e5.local_optima / (2.0 ** e5.d - 1)).groupby(e5.dataset).mean()
hit = pd.read_csv(TAB / "p1_hit_rate_N.csv", index_col=0)
mh = hit[MH].mean(axis=1)
sp = stats.spearmanr(lo.reindex(mh.index).values, mh.values)
x4 = dict(n=int(len(mh)), spearman=float(sp.statistic), p=float(sp.pvalue))

out = {"X3_selection_vs_all_features": x3, "X4_local_optima_vs_hit_rate": x4}
Path("results/summary_extra_paper.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
