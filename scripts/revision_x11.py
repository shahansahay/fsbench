"""Revision analysis X11 (REVISION_PLAN.md, Amendment 2): size-matched control for inflation. Exploratory.

Each outer test fold T is dealt into five stratified parts P1..P5. Exact tables e_Pk (outer-train -> Pk) are built
with the unchanged rules of fsb/tables.py; scaling is fitted on outer-train as for e_test, so the five part tables
sum to e_test for every subset, which is checked first. A search that leaves out Pk takes its fitness from
e_test - e_Pk, scored on n_test - |Pk| rows, about the size of the inner validation set, with the classifier of
protocol L (trained on all outer-training rows). The subset it returns is scored on Pk, which took no part in it.

  python scripts/revision_x11.py build      # part tables, identity check (resumable)
  python scripts/revision_x11.py run --workers 8
  python scripts/revision_x11.py analyze
Writes results/revision/x11_tables/, x11_runs.csv.gz, x11_*.csv and updates summary_revision.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse
import hashlib
import json
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats

from fsb.algorithms import SEARCH
from fsb.data import EXACT, load
from fsb.problem import fitness_array
from fsb.tables import BASE_SEED, error_table, minmax, seed_for, sqdiff, stratified_folds
from revision_common import run_methods

OUT = Path("results/revision")
TAB = OUT / "x11_tables"
RAW = OUT / "x11_raw"
TABLES = Path("tables")
MAIN_RUNS = Path("results/runs.csv.gz")
DATASETS = EXACT
PARTS, ALPHA, BUDGET = 5, 0.99, 1000
NBOOT = 10000


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def part_ids(y_test, ds, f):
    return stratified_folds(y_test, PARTS, seed_for(ds, "fifth", f, BASE_SEED))


# ------------------------------------------------------------------ build
def build():
    TAB.mkdir(parents=True, exist_ok=True)
    index_path = TAB / "index.csv"
    rows = pd.read_csv(index_path).to_dict("records") if index_path.exists() else []
    done = {(r["dataset"], r["fold"]) for r in rows}
    for ds in DATASETS:
        X, y = load(ds)
        C = int(y.max()) + 1
        for f in range(5):
            if (ds, f) in done:
                continue
            t0 = time.time()
            s = np.load(TABLES / ds / f"fold{f}_splits.npz")
            otr, ote = s["outer_train"], s["outer_test"]
            part = part_ids(y[ote], ds, f)
            tr, ev = minmax(X[otr], X[ote])
            D = sqdiff(ev, tr)
            e_test = np.load(TABLES / ds / f"fold{f}_test.npy")
            folder = TAB / ds
            folder.mkdir(parents=True, exist_ok=True)
            total = np.zeros(len(e_test) - 1, dtype=np.int64)
            row = dict(dataset=ds, fold=f, d=X.shape[1], n_test=len(ote), n_inner_val=len(s["inner_val"]))
            for k in range(PARTS):
                sel = part == k
                e_k = error_table(np.ascontiguousarray(D[sel]), y[otr], y[ote][sel], C)
                total += e_k[1:].astype(np.int64)
                pk = folder / f"fold{f}_P{k + 1}.npy"
                np.save(pk, e_k)
                row[f"n_P{k + 1}"] = int(sel.sum())
                row[f"sha256_P{k + 1}"] = sha(pk)
            np.save(folder / f"fold{f}_parts.npy", part)
            row["identity_sum_of_parts_equals_etest"] = bool(np.array_equal(total, e_test[1:].astype(np.int64)))
            row["seconds"] = round(time.time() - t0, 2)
            rows.append(row)
            pd.DataFrame(rows).to_csv(index_path, index=False)
            sizes = [row[f"n_P{k + 1}"] for k in range(PARTS)]
            print(f"{ds:13s} fold {f}  parts={sizes}  inner_val={row['n_inner_val']:3d}  "
                  f"identity={row['identity_sum_of_parts_equals_etest']}  {row['seconds']:7.1f} s", flush=True)
    idx = pd.read_csv(index_path)
    bad = idx[~idx.identity_sum_of_parts_equals_etest]
    print(f"{len(idx)} folds built; identity sum of parts = e_test fails on {len(bad)}")
    if len(bad):
        sys.exit(1)


# ------------------------------------------------------------------ runs
def groups():
    return [(ds, f, k) for ds in DATASETS for f in range(5) for k in range(1, PARTS + 1)]


def load_fold(ds, f):
    e_test = np.load(TABLES / ds / f"fold{f}_test.npy").astype(np.int64)
    parts = {k: np.load(TAB / ds / f"fold{f}_P{k}.npy").astype(np.int64) for k in range(1, PARTS + 1)}
    part = np.load(TAB / ds / f"fold{f}_parts.npy")
    sizes = {k: int((part == k - 1).sum()) for k in range(1, PARTS + 1)}
    return e_test, parts, sizes


def run_group(g):
    ds, f, k = g
    e_test, parts, sizes = load_fold(ds, f)
    e_fit = e_test - parts[k]                      # rows of the test fold outside part k
    n_fit = int(sum(sizes.values()) - sizes[k])
    # fitness from the other four parts; "test" errors are those on the held-out part k
    df = run_methods(e_fit, parts[k], n_fit, sizes[k], ds, f, protocol="N", seed_protocol=f"X11P{k}")
    df["part"] = k
    RAW.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW / f"{ds}_f{f}_P{k}.csv", index=False)
    return g


def run(workers):
    todo = [g for g in groups() if not (RAW / f"{g[0]}_f{g[1]}_P{g[2]}.csv").exists()]
    print(f"{len(groups())} groups, {len(todo)} to run on {workers} workers", flush=True)
    t0 = time.time()
    with Pool(workers) as pool:
        for i, g in enumerate(pool.imap_unordered(run_group, todo), 1):
            print(f"[{i:3d}/{len(todo)}] {g}  {time.time() - t0:6.0f} s", flush=True)
    runs = pd.concat([pd.read_csv(p) for p in sorted(RAW.glob("*.csv"))], ignore_index=True)
    runs.to_csv(OUT / "x11_runs.csv.gz", index=False)
    print(f"Wrote {OUT / 'x11_runs.csv.gz'} ({len(runs)} runs)")


# ------------------------------------------------------------------ analysis
def boot_ci(v, rng):
    v = np.asarray(v, dtype=float)
    means = v[rng.integers(0, len(v), (NBOOT, len(v)))].mean(axis=1)
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def wilcoxon(v):
    v = np.asarray(v, dtype=float)
    return 1.0 if np.allclose(v, 0) else float(stats.wilcoxon(v).pvalue)


def analyze():
    rng = np.random.default_rng(20260919)
    x = pd.read_csv(OUT / "x11_runs.csv.gz")
    main = pd.read_csv(MAIN_RUNS)
    nmain = main[(main.experiment == "main") & (main.protocol == "N")][
        ["dataset", "fold", "method", "rep", "test_acc"]].rename(columns={"test_acc": "acc_N"})
    # per run (method, fold, seed index): pool the five parts
    x["part_correct"] = x.n_test - x.test_errors          # held-out part rows classified correctly
    x["scoring_acc"] = x.fit_acc                           # accuracy on the rows that scored the search
    g = x.groupby(["dataset", "fold", "method", "rep"])
    per = pd.DataFrame(dict(fresh_acc=g.part_correct.sum() / g.n_test.sum(), scoring_acc=g.scoring_acc.mean(),
                            n_parts=g.part.nunique())).reset_index()
    assert (per.n_parts == PARTS).all()
    per = per.merge(nmain, on=["dataset", "fold", "method", "rep"], how="left", validate="one_to_one")
    assert per.acc_N.notna().all()
    per["leak"] = (per.scoring_acc - per.fresh_acc) * 100
    per["config"] = (per.fresh_acc - per.acc_N) * 100
    per["total"] = (per.scoring_acc - per.acc_N) * 100
    fold = per.groupby(["dataset", "fold", "method"])[["leak", "config", "total"]].median()
    dset = fold.groupby(["dataset", "method"]).mean()
    summary = {}
    for comp in ("leak", "config", "total"):
        t = dset[comp].unstack("method").reindex(index=DATASETS, columns=SEARCH + ["ALL"])
        t.to_csv(OUT / f"x11_{comp}_pp.csv", float_format="%.6g")
        m12 = t[SEARCH].mean(axis=1)
        summary[comp] = dict(mean_pp=float(m12.mean()), ci=boot_ci(m12, rng), wilcoxon_p=wilcoxon(m12),
                             per_method_mean_pp={m: float(t[m].mean()) for m in SEARCH},
                             all_features_mean_pp=float(t["ALL"].mean()),
                             per_dataset_pp={k: float(v) for k, v in m12.items()})
    summary["total_minus_sum_of_components_pp"] = summary["total"]["mean_pp"] - (
        summary["leak"]["mean_pp"] + summary["config"]["mean_pp"])
    # exact optima of the leave-one-part-out tables (ties averaged), against the exact validation optimum
    rows = []
    for ds in DATASETS:
        for f in range(5):
            e_test, parts, sizes = load_fold(ds, f)
            n_test = sum(sizes.values())
            d = int(len(e_test)).bit_length() - 1
            e_val = np.load(TABLES / ds / f"fold{f}_val.npy").astype(np.int64)
            n_val = len(np.load(TABLES / ds / f"fold{f}_splits.npz")["inner_val"])
            fv = fitness_array(e_val, n_val, d, ALPHA)
            opt_val = np.flatnonzero(fv == fv[1:].min())
            acc_N = float((1.0 - e_test[opt_val] / n_test).mean())
            correct, scoring = 0.0, []
            for k in range(1, PARTS + 1):
                e_fit = e_test - parts[k]
                n_fit = n_test - sizes[k]
                ff = fitness_array(e_fit, n_fit, d, ALPHA)
                opt = np.flatnonzero(ff == ff[1:].min())
                correct += float((sizes[k] - parts[k][opt]).mean())
                scoring.append(float((1.0 - e_fit[opt] / n_fit).mean()))
            fresh = correct / n_test
            rows.append(dict(dataset=ds, fold=f, leak=100 * (np.mean(scoring) - fresh), config=100 * (fresh - acc_N),
                             total=100 * (np.mean(scoring) - acc_N)))
    ex = pd.DataFrame(rows)
    ex.to_csv(OUT / "x11_exact_optimum_folds.csv", index=False, float_format="%.6g")
    exd = ex.groupby("dataset")[["leak", "config", "total"]].mean().reindex(DATASETS)
    exd.to_csv(OUT / "x11_exact_optimum_pp.csv", float_format="%.6g")
    summary["exact_optimum"] = {c: dict(mean_pp=float(exd[c].mean()), ci=boot_ci(exd[c], rng),
                                        wilcoxon_p=wilcoxon(exd[c])) for c in ("leak", "config", "total")}
    idx = pd.read_csv(TAB / "index.csv")
    idx["n_scoring_mean"] = idx.n_test * (PARTS - 1) / PARTS
    sizes = idx.groupby("dataset")[["n_scoring_mean", "n_inner_val"]].mean().reindex(DATASETS)
    sizes.to_csv(OUT / "x11_scoring_sizes.csv", float_format="%.6g")
    summary["scoring_rows_mean"] = float(idx.n_scoring_mean.mean())
    summary["inner_val_rows_mean"] = float(idx.n_inner_val.mean())
    summary["identity_folds"] = int(len(idx))
    summary["identity_holds"] = int(idx.identity_sum_of_parts_equals_etest.sum())
    cfg_ci = summary["config"]["ci"]
    summary["decision_rule_config_within_1pp"] = bool(-1.0 <= cfg_ci[0] and cfg_ci[1] <= 1.0)
    sp = OUT / "summary_revision.json"
    s = json.loads(sp.read_text()) if sp.exists() else {}
    s["X11"] = summary
    sp.write_text(json.dumps(s, indent=2, default=float))
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items()
                                                              if kk not in ("per_dataset_pp", "per_method_mean_pp")})
                      for k, v in summary.items()}, indent=2, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "run", "analyze"])
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    {"build": build, "run": lambda: run(a.workers), "analyze": analyze}[a.step]()
