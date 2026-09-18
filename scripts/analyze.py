"""Statistics for the frozen plan (primary endpoints P1-P3, hypotheses H1-H4) and labelled
exploratory analyses (E1-E8). Writes results/tables/*.csv and results/summary.json."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json

import numpy as np
import pandas as pd
from scipy import stats

from fsb.algorithms import CONTROLS, METAHEURISTICS, SEARCH
from fsb.data import EXACT
from fsb.problem import Problem

OUT = Path("results")
TAB = OUT / "tables"
TAB.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20260919)
NBOOT = 10000
ORDER = SEARCH + ["ALL"]
summary = {}


def fold_level(df, value, stat="median"):
    g = df.groupby(["dataset", "fold", "method"])[value]
    return g.median() if stat == "median" else g.mean()


def dataset_table(df, value, stat="median", methods=ORDER):
    t = fold_level(df, value, stat).groupby(["dataset", "method"]).mean().unstack("method")
    return t.reindex(index=[d for d in EXACT if d in t.index], columns=[m for m in methods if m in t.columns])


def best_of_table(df, value="test_acc", methods=SEARCH):
    b = df.sort_values(["fitness", "rep"]).groupby(["dataset", "fold", "method"]).head(1)
    t = b.groupby(["dataset", "method"])[value].mean().unstack("method")
    return t.reindex(index=[d for d in EXACT if d in t.index], columns=methods)


def mean_ranks(table, higher_is_better=True):
    return table.rank(axis=1, ascending=not higher_is_better).mean(axis=0)


def tau(a, b):
    return float(stats.kendalltau(a.values, b.reindex(a.index).values).statistic)


def boot_ci(values):
    v = np.asarray(values, dtype=float)
    means = v[RNG.integers(0, len(v), (NBOOT, len(v)))].mean(axis=1)
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


def a12(x, y):
    r = stats.rankdata(np.concatenate([x, y]))
    return float((r[: len(x)].sum() - len(x) * (len(x) + 1) / 2) / (len(x) * len(y)))


def save(df, name):
    df.to_csv(TAB / f"{name}.csv", float_format="%.6g")


runs = pd.read_csv(OUT / "runs.csv.gz")
main = runs[runs.experiment == "main"]
N, L = main[main.protocol == "N"], main[main.protocol == "L"]

# ---------------- P1: optimality (protocol N, budget 1,000) ----------------
hit = dataset_table(N, "hit", "mean", SEARCH)
pct = dataset_table(N, "percentile", "median", SEARCH)
save(hit, "p1_hit_rate_N")
save(pct, "p1_percentile_N")
summary["P1"] = {m: dict(hit_rate=float(hit[m].mean()), hit_ci=boot_ci(hit[m]),
                         median_percentile=float(pct[m].mean())) for m in SEARCH}

# ---------------- P2: leakage (reported accuracy under L minus test accuracy under N) ----------------
rep_L = dataset_table(L, "fit_acc", "median", ORDER)
test_N = dataset_table(N, "test_acc", "median", ORDER)
infl = (rep_L - test_N) * 100
save(rep_L, "p2_reported_acc_L")
save(test_N, "p2_test_acc_N")
save(infl, "p2_inflation_pp")
summary["P2"] = {m: dict(inflation_pp=float(infl[m].mean()), ci=boot_ci(infl[m])) for m in ORDER}
summary["P2"]["all_search_methods"] = dict(inflation_pp=float(infl[SEARCH].values.mean()),
                                           ci=boot_ci(infl[SEARCH].mean(axis=1)))

# ---------------- exact tables: exhaustive optimum, structure, optimism, ceiling (E5-E7) ----------------
QS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
rows_opt, rows_struct, rows_try = [], [], []
for ds in EXACT:
    for fold in range(5):
        pn = Problem(ds, fold, "N")
        pl = Problem(ds, fold, "L")
        f, d = pn.fit, pn.d
        n_sub = (1 << d) - 1
        test_acc_all = 1.0 - pn.test_errors.astype(float) / pn.n_test
        opt = np.flatnonzero(f == pn.f_star)
        opt_l = np.flatnonzero(pl.fit == pl.f_star)
        rows_opt.append(dict(dataset=ds, fold=fold,
                             exhaustive_L_reported=float(test_acc_all[opt_l].mean()),
                             leaky_ceiling=float(test_acc_all[1:].max()),
                             exhaustive_N_test=float(test_acc_all[opt].mean()),
                             exhaustive_N_val=float(1.0 - pn.fit_errors[opt[0]] / pn.n_val),
                             n_optimal=int(len(opt))))
        body = f[1:]
        better_nb = np.zeros(n_sub, dtype=bool)
        worse_all = np.ones(n_sub, dtype=bool)
        m = np.arange(1, n_sub + 1, dtype=np.int64)
        for j in range(d):
            nb = f[m ^ (np.int64(1) << j)]
            better_nb |= nb < body
            worse_all &= nb > body
        err = pn.fit_errors[1:].astype(int)
        rows_struct.append(dict(dataset=ds, fold=fold, d=d,
                                local_optima=int((~better_nb).sum()),
                                strict_local_optima=int(worse_all.sum()),
                                min_error_subsets=int((err == err.min()).sum()),
                                within_one_error=float((err <= err.min() + 1).mean())))
        order = np.argsort(body, kind="stable")
        for q in QS:
            top = order[: max(1, int(round(q * n_sub)))]
            rows_try.append(dict(dataset=ds, fold=fold, top_fraction=q,
                                 test_acc=float(test_acc_all[1:][top].mean()),
                                 val_acc=float(1.0 - pn.fit_errors[1:][top].mean() / pn.n_val)))
opt_df = pd.DataFrame(rows_opt)
struct = pd.DataFrame(rows_struct)
tryhard = pd.DataFrame(rows_try)
save(opt_df, "e_exhaustive_optimum")
save(struct, "e5_structure")
save(tryhard.groupby(["dataset", "top_fraction"])[["val_acc", "test_acc"]].mean(), "e6_harder_you_try")
ex = opt_df.groupby("dataset")[["exhaustive_L_reported", "exhaustive_N_test", "exhaustive_N_val"]].mean()
ex_infl = (ex.exhaustive_L_reported - ex.exhaustive_N_test) * 100
summary["P2"]["exhaustive_optimum"] = dict(inflation_pp=float(ex_infl.mean()), ci=boot_ci(ex_infl))
summary["E6_optimism_of_exhaustive_N_optimum_pp"] = float(((ex.exhaustive_N_val - ex.exhaustive_N_test) * 100).mean())
ceiling = opt_df.groupby("dataset").leaky_ceiling.agg(["min", "mean", "max"])
save(ceiling, "e7_ceiling_leaky_oracle")
summary["E7_datasets_with_100pct_attainable_under_L_every_fold"] = int((ceiling["min"] == 1.0).sum())

# ---------------- P3 and H4: ranking agreement ----------------
best_t = best_of_table(N)
mean_t = dataset_table(N, "test_acc", "mean", SEARCH)
med_t = dataset_table(N, "test_acc", "median", SEARCH)
fit_t = dataset_table(N, "fitness", "median", SEARCH)
r_best, r_mean, r_med = mean_ranks(best_t), mean_ranks(mean_t), mean_ranks(med_t)
r_fit = mean_ranks(fit_t, higher_is_better=False)
ranks = pd.DataFrame(dict(best_of_30=r_best, mean=r_mean, median=r_med, validation_fitness=r_fit))
save(ranks, "p3_mean_ranks_N")
summary["P3"] = dict(tau_best_vs_mean=tau(r_best, r_mean), tau_best_vs_median=tau(r_best, r_med),
                     tau_mean_vs_median=tau(r_mean, r_med), tau_fitness_vs_test=tau(r_fit, r_med))

# ---------------- omnibus and H2 (metaheuristics vs EA and SFS, Holm) ----------------
fr = stats.friedmanchisquare(*[med_t[m].values for m in SEARCH])
summary["friedman_test_acc_N"] = dict(statistic=float(fr.statistic), p=float(fr.pvalue))
rows = []
for c in ("EA", "SFS"):
    for m in METAHEURISTICS:
        a12s = [a12(N[(N.dataset == ds) & (N.method == m)].test_acc.values,
                    N[(N.dataset == ds) & (N.method == c)].test_acc.values) for ds in med_t.index]
        rows.append(dict(method=m, control=c, median_diff_pp=float((med_t[m] - med_t[c]).median() * 100),
                         p=wilcoxon_p(med_t[m], med_t[c]), median_A12=float(np.median(a12s))))
h2 = pd.DataFrame(rows)
h2["p_holm"] = holm(h2.p)
save(h2, "h2_wilcoxon_holm")
h2_beats = h2[(h2.p_holm < 0.05) & (h2.median_diff_pp > 0)]
summary["H2"] = dict(supported=bool(len(h2_beats) == 0), significant_wins=h2_beats.method.tolist(),
                     smallest_p_holm=float(h2.p_holm.min()))

# ---------------- H1, H3, H4 ----------------
small = [ds for ds in hit.index if int(struct[struct.dataset == ds].d.iloc[0]) <= 13]
gap = hit.loc[small, METAHEURISTICS].max(axis=1) - hit.loc[small, "RS"]
summary["H1"] = dict(datasets=small, mean_gap_points=float(gap.mean() * 100),
                     per_dataset_gap_points={k: float(v * 100) for k, v in gap.items()},
                     supported=bool(gap.mean() <= 0.10))
top2 = med_t.mean(axis=0).sort_values(ascending=False)
top_gap = float((top2.iloc[0] - top2.iloc[1]) * 100)
mean_infl = summary["P2"]["all_search_methods"]["inflation_pp"]
summary["H3"] = dict(mean_inflation_pp=mean_infl, top_two=top2.index[:2].tolist(),
                     top_two_gap_pp=top_gap, supported=bool(mean_infl > top_gap))
summary["H4"] = dict(tau=summary["P3"]["tau_fitness_vs_test"],
                     supported=bool(summary["P3"]["tau_fitness_vs_test"] < 0.5))

# ---------------- E1: how many datasets does a ranking need ----------------
full_r = mean_ranks(med_t)
winner = full_r.idxmin()
rows = []
for k in range(3, len(med_t)):
    match, taus, sig = [], [], []
    for _ in range(1000):
        sub = med_t.iloc[RNG.choice(len(med_t), k, replace=False)]
        r = mean_ranks(sub)
        match.append(r.idxmin() == winner)
        taus.append(tau(r, full_r))
        sig.append(stats.friedmanchisquare(*[sub[m].values for m in SEARCH]).pvalue < 0.05)
    rows.append(dict(k=k, p_same_winner=np.mean(match), mean_tau=np.mean(taus), p_friedman_sig=np.mean(sig)))
e1 = pd.DataFrame(rows)
save(e1, "e1_datasets_needed")
summary["E1"] = dict(full_winner=winner, rows=e1.round(3).to_dict("records"))

# ---------------- E2: budget ----------------
bud = pd.concat([runs[runs.experiment == "budget"], N])
bud = bud[bud.method.isin(SEARCH)]
rows = []
for b, g in bud.groupby("budget"):
    h = dataset_table(g, "hit", "mean", SEARCH).mean()
    t = dataset_table(g, "test_acc", "median", SEARCH).mean()
    cov = (g.unique / (2.0 ** g.d - 1)).groupby(g.method).mean()
    for m in SEARCH:
        rows.append(dict(budget=b, method=m, hit_rate=h[m], test_acc=t[m], coverage=cov[m]))
save(pd.DataFrame(rows), "e2_budget_curve")

# ---------------- E3: fitness weight; E4: transfer function ----------------
alp = runs[(runs.experiment == "alpha") & runs.method.isin(SEARCH)]
med_a = dataset_table(alp, "test_acc", "median", SEARCH)
r_a = mean_ranks(med_a)
summary["E3"] = dict(tau_alpha_0p9_vs_0p99=tau(r_a, r_med), winner_0p99=r_med.idxmin(), winner_0p9=r_a.idxmin(),
                     mean_size_0p99=float(N[N.method.isin(SEARCH)]["size"].mean()),
                     mean_size_0p9=float(alp["size"].mean()))
rows = []
for alg in ("PSO", "GWO", "WOA", "SCA"):
    s_, v_ = f"{alg}-S", f"{alg}-V"
    rows.append(dict(algorithm=alg, hit_S_minus_V=float((hit[s_] - hit[v_]).mean()),
                     p_hit=wilcoxon_p(hit[s_], hit[v_]),
                     test_acc_S_minus_V_pp=float((med_t[s_] - med_t[v_]).mean() * 100),
                     p_test=wilcoxon_p(med_t[s_], med_t[v_])))
save(pd.DataFrame(rows), "e4_transfer_function")

# ---------------- E8: tie-rule agreement with scikit-learn ----------------
ver = pd.read_csv(OUT.parent / "reports" / "verification.csv")
save(ver.groupby("dataset").sklearn_error_agreement.median().reindex(EXACT), "e8_sklearn_agreement")

# ---------------- Table 1 ----------------
man = pd.read_csv(OUT.parent / "data" / "manifest.csv").set_index("dataset").loc[EXACT]
t1 = man[["d", "n", "classes", "smallest_class", "duplicate_rows", "conflicting_duplicates"]].copy()
t1["subsets"] = 2 ** t1.d - 1
t1["B0_coverage_pct"] = np.minimum(100.0, 100.0 * 1000 / t1.subsets)
t1["mean_optimal_subsets"] = opt_df.groupby("dataset").n_optimal.mean()
t1["all_features_test_acc"] = test_N["ALL"]
t1["exhaustive_N_test_acc"] = ex.exhaustive_N_test
save(t1, "table1_datasets")

(OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
print(json.dumps({k: summary[k] for k in ("H1", "H2", "H3", "H4")}, indent=2, default=float))
print("Friedman:", summary["friedman_test_acc_N"])
print("Mean inflation (pp):", round(mean_infl, 2), " exhaustive optimum:",
      round(summary["P2"]["exhaustive_optimum"]["inflation_pp"], 2))
print("Wrote results/tables/*.csv and results/summary.json")
