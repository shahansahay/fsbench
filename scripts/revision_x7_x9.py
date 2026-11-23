"""Revision analyses X7 and X9 (REVISION_PLAN.md, tag revision-plan-v1). Exploratory.

X7  Power and equivalence for the 18 comparisons of H2 on dataset-level leak-free accuracy.
X9  The protocol shift set against the variation across datasets and methods.

Reads results/runs.csv.gz (results-v1) only. Writes results/revision/x7_*.csv, x9_*.csv and
updates results/revision/summary_revision.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import itertools
import json

import numpy as np
import pandas as pd
from scipy import stats

from fsb.algorithms import METAHEURISTICS, SEARCH
from fsb.data import EXACT

OUT = Path("results/revision")
OUT.mkdir(parents=True, exist_ok=True)
NBOOT = 10000
NDRAW = 2000
ALPHAS = (0.05, 0.05 / 18)


def dataset_table(df, value, stat="median", methods=SEARCH):
    g = df.groupby(["dataset", "fold", "method"])[value]
    fold = g.median() if stat == "median" else g.mean()
    t = fold.groupby(["dataset", "method"]).mean().unstack("method")
    return t.reindex(index=[d for d in EXACT if d in t.index], columns=methods)


def boot_ci(values, rng, level=95):
    v = np.asarray(values, dtype=float)
    means = v[rng.integers(0, len(v), (NBOOT, len(v)))].mean(axis=1)
    lo = (100 - level) / 2
    return [float(x) for x in np.percentile(means, [lo, 100 - lo])]


# ---------- vectorised two-sided Wilcoxon signed-rank P, as scipy 1.17 computes it for n <= 13 ----------
# No ties or zeros: exact null distribution; otherwise exhaustive enumeration of sign assignments.
# Both equal the share of the 2^n sign assignments whose statistic lies at least as far from its
# mean as the observed one, so one enumeration serves both cases.
_SIGNS = {}


def _signs(n):
    if n not in _SIGNS:
        _SIGNS[n] = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(np.float64)
    return _SIGNS[n]


def wilcoxon_p_batch(D):
    """D: (draws, n) differences. Returns two-sided P per row (zeros discarded, average ranks)."""
    D = np.asarray(D, dtype=float)
    zero = D == 0
    # zeros have the smallest |d|, so ranking all and subtracting the zero count ranks the non-zeros
    r = stats.rankdata(np.abs(D), axis=1) - zero.sum(axis=1, keepdims=True)
    r[zero] = 0.0
    t_obs = (r * (D > 0)).sum(axis=1)
    mu = r.sum(axis=1) / 2
    null = r @ _signs(D.shape[1]).T                    # (draws, 2^n) statistics under sign flips
    p = np.mean(np.abs(null - mu[:, None]) >= (np.abs(t_obs - mu) - 1e-9)[:, None], axis=1)
    p[zero.all(axis=1)] = 1.0
    return np.minimum(1.0, p)


def check_against_scipy(rng):
    worst = 0.0
    for _ in range(300):
        d = np.round(rng.normal(0.3, 1.0, 12), 1)  # rounding forces ties and zeros
        if np.all(d == 0):
            continue
        p_ref = float(stats.wilcoxon(d).pvalue)
        worst = max(worst, abs(p_ref - wilcoxon_p_batch(d[None])[0]))
    for _ in range(100):
        d = rng.normal(0.3, 1.0, 12)
        worst = max(worst, abs(float(stats.wilcoxon(d).pvalue) - wilcoxon_p_batch(d[None])[0]))
    return worst


def mde(diffs, rng, alpha):
    """Smallest shift (0.1 pp grid) with power >= 0.8; the same draws are reused for every shift."""
    c = diffs - diffs.mean()
    idx = rng.integers(0, len(c), (NDRAW, len(c)))
    base = c[idx]
    for delta in np.round(np.arange(0.0, 20.01, 0.1), 1):
        if np.mean(wilcoxon_p_batch(base + delta) < alpha) >= 0.8:
            return float(delta)
    return float("nan")


def main():
    runs = pd.read_csv("results/runs.csv.gz")
    main_runs = runs[runs.experiment == "main"]
    N, L = main_runs[main_runs.protocol == "N"], main_runs[main_runs.protocol == "L"]
    acc_N = dataset_table(N, "test_acc") * 100          # leak-free accuracy, pp
    acc_L = dataset_table(L, "fit_acc") * 100           # reported accuracy under L, pp
    summary_path = OUT / "summary_revision.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    # ------------------------------- X7 -------------------------------
    rng_pow = np.random.default_rng(20260921)
    rng_boot = np.random.default_rng(20260919)
    max_dev = check_against_scipy(np.random.default_rng(1))
    rows = []
    for c in ("EA", "SFS"):
        for m in METAHEURISTICS:
            d = (acc_N[m] - acc_N[c]).values
            row = dict(method=m, control=c, mean_diff_pp=float(d.mean()), median_diff_pp=float(np.median(d)),
                       sd_diff_pp=float(d.std(ddof=1)))
            for a, name in zip(ALPHAS, ("mde_pp_alpha_0.05", "mde_pp_alpha_0.05_over_18")):
                row[name] = mde(d, np.random.default_rng(rng_pow.integers(2**63)), a)
            for margin in (1.0, 0.5):
                p_low = float(stats.wilcoxon(d + margin, alternative="greater").pvalue)
                p_high = float(stats.wilcoxon(d - margin, alternative="less").pvalue)
                row[f"tost_p_lower_{margin}"] = p_low
                row[f"tost_p_upper_{margin}"] = p_high
                row[f"equivalent_{margin}"] = bool(max(p_low, p_high) < 0.05)
            row["ci90_mean_diff_pp"] = boot_ci(d, rng_boot, level=90)
            rows.append(row)
    x7 = pd.DataFrame(rows)
    x7.to_csv(OUT / "x7_power_equivalence.csv", index=False, float_format="%.6g")
    summary["X7"] = dict(
        wilcoxon_implementation_max_abs_dev_from_scipy=max_dev,
        mde_pp_alpha_0p05=dict(min=float(x7["mde_pp_alpha_0.05"].min()), median=float(x7["mde_pp_alpha_0.05"].median()),
                               max=float(x7["mde_pp_alpha_0.05"].max())),
        mde_pp_alpha_holm_first_step=dict(min=float(x7["mde_pp_alpha_0.05_over_18"].min()),
                                          median=float(x7["mde_pp_alpha_0.05_over_18"].median()),
                                          max=float(x7["mde_pp_alpha_0.05_over_18"].max())),
        equivalent_1pp={c: x7[(x7.control == c) & x7["equivalent_1.0"]].method.tolist() for c in ("EA", "SFS")},
        equivalent_0p5pp={c: x7[(x7.control == c) & x7["equivalent_0.5"]].method.tolist() for c in ("EA", "SFS")},
    )

    # ------------------------------- X9 -------------------------------
    Y = np.stack([acc_N[SEARCH].values, acc_L[SEARCH].values], axis=2)  # dataset x method x protocol, pp
    gm = Y.mean()
    a = Y.mean(axis=(1, 2)) - gm                       # dataset
    b = Y.mean(axis=(0, 2)) - gm                       # method
    p = Y.mean(axis=(0, 1)) - gm                       # protocol
    ab = Y.mean(axis=2) - gm - a[:, None] - b[None, :]
    ap = Y.mean(axis=1) - gm - a[:, None] - p[None, :]
    bp = Y.mean(axis=0) - gm - b[:, None] - p[None, :]
    nb, nm, npr = Y.shape
    fitted = gm + a[:, None, None] + b[None, :, None] + p[None, None, :] + ab[:, :, None] + ap[:, None, :] + bp[None, :, :]
    ss = dict(dataset=nm * npr * (a ** 2).sum(), method=nb * npr * (b ** 2).sum(), protocol=nb * nm * (p ** 2).sum(),
              dataset_x_method=npr * (ab ** 2).sum(), dataset_x_protocol=nm * (ap ** 2).sum(),
              method_x_protocol=nb * (bp ** 2).sum(), three_way=((Y - fitted) ** 2).sum())
    total = ((Y - gm) ** 2).sum()
    assert abs(sum(ss.values()) - total) < 1e-6 * total
    x9a = pd.DataFrame(dict(sum_of_squares=ss, share_pct={k: 100 * v / total for k, v in ss.items()}))
    x9a.to_csv(OUT / "x9_variance_decomposition.csv", float_format="%.6g")

    infl = acc_L[SEARCH] - acc_N[SEARCH]
    mean_infl = infl.mean(axis=1)
    spread = acc_N[METAHEURISTICS].max(axis=1) - acc_N[METAHEURISTICS].min(axis=1)
    seed_sd = (N[N.method.isin(SEARCH)].groupby(["dataset", "fold", "method"]).test_acc.std(ddof=1) * 100) \
        .groupby("dataset").mean().reindex(acc_N.index)
    idx = pd.read_csv("tables/index.csv").groupby("dataset").agg(d=("d", "first"), n_test=("n_outer_test", "mean"))
    idx = idx.reindex(acc_N.index)
    scale = np.sqrt(np.log(2.0 ** idx.d - 1) / idx.n_test)
    rho = stats.spearmanr(mean_infl.values, scale.values)
    x9b = pd.DataFrame(dict(mean_inflation_pp=mean_infl, leak_free_range_metaheuristics_pp=spread,
                            ratio=mean_infl / spread, positive_methods=(infl > 0).sum(axis=1),
                            seed_sd_pp=seed_sd, d=idx.d, mean_n_test=idx.n_test, scale=scale))
    x9b.to_csv(OUT / "x9_per_dataset.csv", float_format="%.6g")
    summary["X9"] = dict(
        share_pct={k: float(v) for k, v in x9a.share_pct.items()},
        datasets_with_positive_mean_inflation=int((mean_infl > 0).sum()),
        positive_dataset_counts_per_method={m: int((infl[m] > 0).sum()) for m in SEARCH},
        ratio_inflation_to_range=dict(min=float(x9b.ratio.min()), median=float(x9b.ratio.median()), max=float(x9b.ratio.max())),
        median_seed_sd_pp=float(seed_sd.median()),
        spearman_inflation_vs_scale=dict(rho=float(rho.statistic), p=float(rho.pvalue), n=len(scale)),
    )
    summary_path.write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: summary[k] for k in ("X7", "X9")}, indent=2, default=float))


if __name__ == "__main__":
    main()
