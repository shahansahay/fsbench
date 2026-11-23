"""Helpers shared by the revision scripts (REVISION_PLAN.md): runs on given tables and the frozen statistics."""
import numpy as np
import pandas as pd
from scipy import stats

from fsb.algorithms import CONTROLS, METAHEURISTICS, METHODS, SEARCH
from fsb.problem import Oracle, Problem
from fsb.tables import BASE_SEED, seed_for

REPS, NBOOT = 30, 10000


def run_methods(e_val, e_test, n_val, n_test, ds, fold, protocol, alpha=0.99, budget=1000, seed_protocol=None):
    """All 12 methods (30 seeds) and ALL on one pair of tables; the same records as run_experiments.py."""
    pr = Problem.from_arrays(e_val, e_test, n_val, n_test, protocol, alpha)
    rows = []
    for method, fn in METHODS.items():
        for rep in range(1 if method == "ALL" else REPS):
            seed = seed_for(ds, fold, seed_protocol or protocol, method, alpha, budget, rep, BASE_SEED)
            o = Oracle(pr, budget)
            fn(o, pr.d, np.random.default_rng(seed))
            m = o.best_mask
            rows.append(dict(dataset=ds, fold=fold, protocol=protocol, method=method, rep=rep, mask=m,
                             size=bin(m).count("1"), fitness=o.best_f, f_star=pr.f_star, hit=int(o.best_f == pr.f_star),
                             fit_errors=int(pr.fit_errors[m]), n_fit=pr.n_fit,
                             test_errors=int(pr.test_errors[m]), n_test=pr.n_test, d=pr.d))
    df = pd.DataFrame(rows)
    df["fit_acc"] = 1.0 - df.fit_errors / df.n_fit
    df["test_acc"] = 1.0 - df.test_errors / df.n_test
    return df


def dataset_table(df, value, stat="median", methods=SEARCH, order=None):
    g = df.groupby(["dataset", "fold", "method"])[value]
    fold = g.median() if stat == "median" else g.mean()
    t = fold.groupby(["dataset", "method"]).mean().unstack("method")
    idx = [d for d in (order or list(t.index)) if d in t.index]
    return t.reindex(index=idx, columns=[m for m in methods if m in t.columns])


def boot_ci(v, rng):
    v = np.asarray(v, dtype=float)
    means = v[rng.integers(0, len(v), (NBOOT, len(v)))].mean(axis=1)
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def holm(p):
    p = np.asarray(p, dtype=float)
    adj, running = np.empty(len(p)), 0.0
    for k, i in enumerate(np.argsort(p)):
        running = max(running, (len(p) - k) * p[i])
        adj[i] = min(1.0, running)
    return adj


def wilcoxon_p(x, y):
    diff = np.asarray(x) - np.asarray(y)
    if np.allclose(diff, 0):
        return 1.0
    return float(stats.wilcoxon(x, y).pvalue)


def mean_ranks(table, higher_is_better=True):
    return table.rank(axis=1, ascending=not higher_is_better).mean(axis=0)


def endpoints(N, L, d_of, rng):
    """The frozen decisions (H1-H4, Friedman, P2 inflation) and X3 on a set of runs (protocols N and L)."""
    acc_N = dataset_table(N, "test_acc", methods=SEARCH + ["ALL"])
    rep_L = dataset_table(L, "fit_acc", methods=SEARCH + ["ALL"])
    infl = (rep_L - acc_N) * 100
    med_t, hit = acc_N[SEARCH], dataset_table(N, "hit", "mean")
    fit_t = dataset_table(N, "fitness")
    fr = stats.friedmanchisquare(*[med_t[m].values for m in SEARCH])
    rows = [dict(method=m, control=c, p=wilcoxon_p(med_t[m], med_t[c]),
                 median_diff_pp=float((med_t[m] - med_t[c]).median() * 100))
            for c in ("EA", "SFS") for m in METAHEURISTICS]
    h2 = pd.DataFrame(rows)
    h2["p_holm"] = holm(h2.p)
    small = [ds for ds in hit.index if d_of[ds] <= 13]
    gap = hit.loc[small, METAHEURISTICS].max(axis=1) - hit.loc[small, "RS"]
    top = med_t.mean(axis=0).sort_values(ascending=False)
    top_gap = float((top.iloc[0] - top.iloc[1]) * 100)
    mean_infl = float(infl[SEARCH].values.mean())
    tau = float(stats.kendalltau(mean_ranks(fit_t, False).values,
                                 mean_ranks(med_t).reindex(fit_t.columns).values).statistic)
    gain_N = (med_t.mean(axis=1) - acc_N["ALL"]) * 100
    spread = acc_N[METAHEURISTICS].mean(axis=0)
    return dict(
        n_datasets=int(len(med_t)),
        inflation_pp=mean_infl, inflation_ci=boot_ci(infl[SEARCH].mean(axis=1), rng),
        inflation_positive_datasets=int((infl[SEARCH].mean(axis=1) > 0).sum()),
        leak_free_spread_metaheuristics_pp=float((spread.max() - spread.min()) * 100),
        friedman_stat=float(fr.statistic), friedman_p=float(fr.pvalue),
        H1_mean_gap_points=float(gap.mean() * 100), H1_supported=bool(gap.mean() <= 0.10),
        H2_smallest_p_holm=float(h2.p_holm.min()),
        H2_supported=bool(len(h2[(h2.p_holm < 0.05) & (h2.median_diff_pp > 0)]) == 0),
        H3_top_two=top.index[:2].tolist(), H3_top_two_gap_pp=top_gap, H3_supported=bool(mean_infl > top_gap),
        H4_tau=tau, H4_supported=bool(tau < 0.5),
        X3_below_all_leak_free=int((gain_N < 0).sum()),
        X3_wilcoxon_p=wilcoxon_p(med_t.mean(axis=1).values, acc_N["ALL"].values),
    ), infl
