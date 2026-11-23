"""Revision analysis X8 (REVISION_PLAN.md, tag revision-plan-v1): the tie rules of scikit-learn. Exploratory.

(a) For the subsets compared in verify_tables.py (same generator, same draws), the absolute accuracy difference
    between the table and KNeighborsClassifier(n_neighbors=5, algorithm="brute").
(b) For the 7 datasets with d <= 13, complete validation and test tables built with scikit-learn's classifier on
    the same splits and scaling; exact-optimum quantities and protocol N and L runs on them.

  python scripts/revision_x8.py compare --workers 8
  python scripts/revision_x8.py build --workers 8
  python scripts/revision_x8.py run --workers 8
  python scripts/revision_x8.py analyze
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse
import json
import platform
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier

from fsb.algorithms import SEARCH
from fsb.data import EXACT, load
from fsb.problem import Problem
from fsb.tables import EMPTY, K, minmax
from revision_common import boot_ci, dataset_table, run_methods

OUT = Path("results/revision")
TAG = f"{platform.system().lower()}-{platform.machine().lower()}"   # darwin-arm64 or linux-x86_64
TAB = OUT / f"x8_tables_{TAG}"
RAW = OUT / f"x8_raw_{TAG}"
SMALL = [ds for ds in EXACT if load(ds)[0].shape[1] <= 13]


def sk_errors(trs, ytr, evs, yev, mask, d):
    cols = [j for j in range(d) if (mask >> j) & 1]
    knn = KNeighborsClassifier(n_neighbors=K, algorithm="brute").fit(trs[:, cols], ytr)
    return int((knn.predict(evs[:, cols]) != yev).sum())


# --------------------------------------------------- (a) the verification subsets, in accuracy points
def compare_one(args):
    ds, f, kind = args
    X, y = load(ds)
    s = np.load(f"tables/{ds}/fold{f}_splits.npz")
    tr, ev = ("inner_train", "inner_val") if kind == "val" else ("outer_train", "outer_test")
    table = np.load(f"tables/{ds}/fold{f}_{kind}.npy")
    Xtr, ytr, Xev, yev = X[s[tr]], y[s[tr]], X[s[ev]], y[s[ev]]
    trs, evs = minmax(Xtr, Xev)
    d = Xtr.shape[1]
    # replay the generator of fsb/verify.py check_table (seed = fold) so the same subsets are drawn
    rng = np.random.default_rng(f)
    n_masks = (1 << d) - 1
    body = table[1:].astype(np.int64)
    opt_masks = np.flatnonzero(body == body.min()) + 1
    rng.choice(opt_masks, size=min(200, len(opt_masks)), replace=False)
    rng.integers(1, n_masks + 1, size=min(2000, n_masks))
    masks = rng.integers(1, n_masks + 1, size=300).tolist() + [n_masks]
    rows = []
    for m in masks:
        e_sk = sk_errors(trs, ytr, evs, yev, m, d)
        rows.append(dict(dataset=ds, fold=f, table=kind, mask=m, errors_table=int(table[m]), errors_sklearn=e_sk,
                         abs_diff_pp=100 * abs(e_sk - int(table[m])) / len(yev)))
    return rows


def compare(workers):
    jobs = [(ds, f, k) for ds in EXACT for f in range(5) for k in ("val", "test")]
    with Pool(workers) as pool:
        rows = sum(pool.map(compare_one, jobs), [])
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"x8_sklearn_subsets_{TAG}.csv.gz", index=False)
    ver = pd.read_csv("reports/verification.csv")
    agree = df.assign(same=df.errors_table == df.errors_sklearn).groupby(["dataset", "fold", "table"]).same.mean()
    check = ver.set_index(["dataset", "fold", "table"]).sklearn_error_agreement.reindex(agree.index)
    replay_ok = bool(np.allclose(agree.values, check.values))
    g = df.groupby("dataset").abs_diff_pp.agg(["mean", "max"]).reindex(EXACT)
    g["agreement_pct_median"] = 100 * agree.groupby("dataset").median().reindex(EXACT)
    g.to_csv(OUT / f"x8a_sklearn_accuracy_difference_{TAG}.csv", float_format="%.4g")
    print("replay reproduces the recorded agreement shares:", replay_ok)
    print(g.round(3).to_string())
    sp = OUT / "summary_revision.json"
    s = json.loads(sp.read_text()) if sp.exists() else {}
    s.setdefault("X8", {})[f"a_{TAG}"] = dict(replay_matches_verification_csv=replay_ok,
                                       mean_abs_diff_pp=float(df.abs_diff_pp.mean()),
                                       per_dataset_mean_pp={k: float(v) for k, v in g["mean"].items()},
                                       per_dataset_max_pp={k: float(v) for k, v in g["max"].items()})
    sp.write_text(json.dumps(s, indent=2, default=float))


# --------------------------------------------------- (b) complete scikit-learn tables, d <= 13
def build_one(args):
    ds, f, kind = args
    X, y = load(ds)
    s = np.load(f"tables/{ds}/fold{f}_splits.npz")
    tr, ev = ("inner_train", "inner_val") if kind == "val" else ("outer_train", "outer_test")
    trs, evs = minmax(X[s[tr]], X[s[ev]])
    d = X.shape[1]
    out = np.empty(1 << d, dtype=np.uint16)
    out[0] = EMPTY
    for m in range(1, 1 << d):
        out[m] = sk_errors(trs, y[s[tr]], evs, y[s[ev]], m, d)
    (TAB / ds).mkdir(parents=True, exist_ok=True)
    np.save(TAB / ds / f"fold{f}_{kind}.npy", out)
    return args


def build(workers):
    jobs = [(ds, f, k) for ds in SMALL for f in range(5) for k in ("val", "test")
            if not (TAB / ds / f"fold{f}_{k}.npy").exists()]
    t0 = time.time()
    with Pool(workers) as pool:
        for i, j in enumerate(pool.imap_unordered(build_one, jobs), 1):
            print(f"[{i:3d}/{len(jobs)}] {j}  {time.time() - t0:6.0f} s", flush=True)


def run_group(g):
    ds, f, protocol = g
    s = np.load(f"tables/{ds}/fold{f}_splits.npz")
    df = run_methods(np.load(TAB / ds / f"fold{f}_val.npy"), np.load(TAB / ds / f"fold{f}_test.npy"),
                     len(s["inner_val"]), len(s["outer_test"]), ds, f, protocol)
    RAW.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW / f"{ds}_f{f}_{protocol}.csv", index=False)
    return g


def run(workers):
    groups = [(ds, f, p) for ds in SMALL for f in range(5) for p in ("N", "L")]
    todo = [g for g in groups if not (RAW / f"{g[0]}_f{g[1]}_{g[2]}.csv").exists()]
    with Pool(workers) as pool:
        for i, g in enumerate(pool.imap_unordered(run_group, todo), 1):
            print(f"[{i:3d}/{len(todo)}] {g}", flush=True)
    pd.concat([pd.read_csv(p) for p in sorted(RAW.glob("*.csv"))], ignore_index=True) \
        .to_csv(OUT / f"x8_runs_{TAG}.csv.gz", index=False)


def exact_quantities(load_tables):
    rows = []
    for ds in SMALL:
        for f in range(5):
            e_val, e_test, n_val, n_test = load_tables(ds, f)
            pn = Problem.from_arrays(e_val, e_test, n_val, n_test, "N")
            pl = Problem.from_arrays(e_val, e_test, n_val, n_test, "L")
            acc_test = 1.0 - pn.test_errors.astype(float) / n_test
            o_n = np.flatnonzero(pn.fit == pn.f_star)
            o_l = np.flatnonzero(pl.fit == pl.f_star)
            rows.append(dict(dataset=ds, fold=f, opt_N_test=acc_test[o_n].mean(),
                             opt_N_val=1.0 - e_val[o_n[0]] / n_val, opt_L_reported=acc_test[o_l].mean()))
    t = pd.DataFrame(rows).groupby("dataset").mean().drop(columns="fold").reindex(SMALL)
    return pd.DataFrame(dict(leak_free_opt_pct=100 * t.opt_N_test, optimism_pp=100 * (t.opt_N_val - t.opt_N_test),
                             inflation_opt_pp=100 * (t.opt_L_reported - t.opt_N_test)))


def analyze():
    rng = np.random.default_rng(20260919)

    def orig(ds, f):
        s = np.load(f"tables/{ds}/fold{f}_splits.npz")
        return (np.load(f"tables/{ds}/fold{f}_val.npy"), np.load(f"tables/{ds}/fold{f}_test.npy"),
                len(s["inner_val"]), len(s["outer_test"]))

    def skl(ds, f):
        s = np.load(f"tables/{ds}/fold{f}_splits.npz")
        return (np.load(TAB / ds / f"fold{f}_val.npy"), np.load(TAB / ds / f"fold{f}_test.npy"),
                len(s["inner_val"]), len(s["outer_test"]))

    ex = pd.concat({"original": exact_quantities(orig), "sklearn": exact_quantities(skl)}, axis=1)
    ex.to_csv(OUT / f"x8b_exact_optimum_{TAG}.csv", float_format="%.6g")
    res = {}
    main_runs = pd.read_csv("results/runs.csv.gz")
    main_runs = main_runs[(main_runs.experiment == "main") & main_runs.dataset.isin(SMALL)]
    sk = pd.read_csv(OUT / f"x8_runs_{TAG}.csv.gz")
    for name, R in (("original", main_runs), ("sklearn", sk)):
        N, L = R[R.protocol == "N"], R[R.protocol == "L"]
        accN = dataset_table(N, "test_acc", order=SMALL)
        infl = (dataset_table(L, "fit_acc", order=SMALL) - accN) * 100
        fr = stats.friedmanchisquare(*[accN[m].values for m in SEARCH])
        res[name] = dict(n_datasets=len(SMALL), inflation_pp=float(infl.values.mean()),
                         inflation_ci=boot_ci(infl.mean(axis=1), rng), friedman_p=float(fr.pvalue),
                         leak_free_spread_metaheuristics_pp=float(
                             (accN.iloc[:, :9].mean().max() - accN.iloc[:, :9].mean().min()) * 100),
                         exact_optimum_inflation_pp=float(ex[(name, "inflation_opt_pp")].mean()),
                         exact_optimum_optimism_pp=float(ex[(name, "optimism_pp")].mean()))
    sp = OUT / "summary_revision.json"
    s = json.loads(sp.read_text()) if sp.exists() else {}
    s.setdefault("X8", {})[f"b_{TAG}"] = res
    sp.write_text(json.dumps(s, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))
    print(ex.round(2).to_string())


def compare_platforms():
    """Amendment 1, X8c: scikit-learn error counts of the same subsets on the two machines."""
    a = pd.read_csv(OUT / "x8_sklearn_subsets_darwin-arm64.csv.gz")
    b = pd.read_csv(OUT / "x8_sklearn_subsets_linux-x86_64.csv.gz")
    key = ["dataset", "fold", "table", "mask"]
    a = a.reset_index().rename(columns={"index": "draw"})
    b = b.reset_index().rename(columns={"index": "draw"})
    m = a.merge(b, on=["draw"] + key, suffixes=("_mac", "_linux"), validate="one_to_one")
    assert len(m) == len(a) == len(b)
    assert (m.errors_table_mac == m.errors_table_linux).all(), "table values differ between machines"
    m["same"] = m.errors_sklearn_mac == m.errors_sklearn_linux
    g = m.groupby("dataset").agg(share_identical_pct=("same", lambda v: 100 * v.mean()),
                                 tables_with_any_difference=("same", lambda v: 0)).reindex(EXACT)
    g["tables_with_any_difference"] = (~m.groupby(["dataset", "fold", "table"]).same.all()).groupby("dataset").sum()
    g.to_csv(OUT / "x8c_sklearn_across_machines.csv", float_format="%.4g")
    sp = OUT / "summary_revision.json"
    s = json.loads(sp.read_text()) if sp.exists() else {}
    s.setdefault("X8", {})["c"] = dict(table_values_identical=True, subsets_compared=int(len(m)),
                                       share_identical_pct_overall=float(100 * m.same.mean()),
                                       per_dataset_share_identical_pct={k: float(v) for k, v in g.share_identical_pct.items()})
    sp.write_text(json.dumps(s, indent=2, default=float))
    print(g.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["compare", "build", "run", "analyze", "compare_platforms"])
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    {"compare": lambda: compare(a.workers), "build": lambda: build(a.workers), "run": lambda: run(a.workers),
     "analyze": analyze, "compare_platforms": compare_platforms}[a.step]()
