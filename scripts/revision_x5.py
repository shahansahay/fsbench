"""Revision analysis X5 (REVISION_PLAN.md, tag revision-plan-v1): matched control for inflation. Exploratory.

Each outer test fold is split into stratified halves A and B. Exact tables e_A (outer-train -> A) and e_B
(outer-train -> B) are built with the unchanged rules of fsb/tables.py; scaling is fitted on outer-train as for
e_test, so e_A + e_B = e_test for every subset, which is checked first. A search selecting on one half uses the
same classifier (trained on all outer-training rows) as protocol L; scoring its subset on the selection half
reproduces the leak, scoring it on the other half does not.

  python scripts/revision_x5.py build      # half tables, identity check (resumable)
  python scripts/revision_x5.py run --workers 8
  python scripts/revision_x5.py analyze
Writes results/revision/x5_tables/, x5_runs.csv.gz, x5_*.csv and updates summary_revision.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats

from fsb.algorithms import METHODS, METAHEURISTICS, SEARCH
from fsb.data import EXACT, load
from fsb.problem import Oracle, Problem
from fsb.tables import BASE_SEED, error_table, minmax, seed_for, sqdiff, stratified_holdout

OUT = Path("results/revision")
TAB = OUT / "x5_tables"
RAW = OUT / "x5_raw"
REPS, ALPHA, BUDGET = 30, 0.99, 1000
NBOOT = 10000


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ build
def build():
    TAB.mkdir(parents=True, exist_ok=True)
    index_path = TAB / "index.csv"
    rows = pd.read_csv(index_path).to_dict("records") if index_path.exists() else []
    done = {(r["dataset"], r["fold"]) for r in rows}
    for ds in EXACT:
        X, y = load(ds)
        C = int(y.max()) + 1
        for f in range(5):
            if (ds, f) in done:
                continue
            t0 = time.time()
            s = np.load(f"tables/{ds}/fold{f}_splits.npz")
            otr, ote = s["outer_train"], s["outer_test"]
            in_a = stratified_holdout(y[ote], 0.5, seed_for(ds, "half", f, BASE_SEED))
            tr, ev = minmax(X[otr], X[ote])
            D = sqdiff(ev, tr)
            e_a = error_table(np.ascontiguousarray(D[in_a]), y[otr], y[ote][in_a], C)
            e_b = error_table(np.ascontiguousarray(D[~in_a]), y[otr], y[ote][~in_a], C)
            e_test = np.load(f"tables/{ds}/fold{f}_test.npy")
            ident = bool(np.array_equal(e_a[1:].astype(np.int64) + e_b[1:], e_test[1:].astype(np.int64)))
            folder = TAB / ds
            folder.mkdir(parents=True, exist_ok=True)
            pa, pb = folder / f"fold{f}_A.npy", folder / f"fold{f}_B.npy"
            np.save(pa, e_a)
            np.save(pb, e_b)
            np.savez(folder / f"fold{f}_halves.npz", A=ote[in_a], B=ote[~in_a])
            rows.append(dict(dataset=ds, fold=f, d=X.shape[1], n_A=int(in_a.sum()), n_B=int((~in_a).sum()),
                             identity_eA_plus_eB_equals_etest=ident, seconds=round(time.time() - t0, 2),
                             sha256_A=sha(pa), sha256_B=sha(pb)))
            pd.DataFrame(rows).to_csv(index_path, index=False)
            print(f"{ds:13s} fold {f}  n_A={in_a.sum():3d} n_B={(~in_a).sum():3d}  identity={ident}  "
                  f"{time.time() - t0:7.1f} s", flush=True)
    idx = pd.read_csv(index_path)
    bad = idx[~idx.identity_eA_plus_eB_equals_etest]
    print(f"{len(idx)} folds built; identity e_A + e_B = e_test fails on {len(bad)}")
    if len(bad):
        sys.exit(1)


# ------------------------------------------------------------------ runs
def groups():
    return [(ds, f, h) for ds in EXACT for f in range(5) for h in ("A", "B")]


def run_group(g):
    ds, f, h = g
    o_h = "B" if h == "A" else "A"
    e_sel = np.load(TAB / ds / f"fold{f}_{h}.npy")
    e_oth = np.load(TAB / ds / f"fold{f}_{o_h}.npy")
    halves = np.load(TAB / ds / f"fold{f}_halves.npz")
    n_sel, n_oth = len(halves[h]), len(halves[o_h])
    # fitness from the selection half; "test" errors are those on the other half
    pr = Problem.from_arrays(e_sel, e_oth, n_sel, n_oth, protocol="N", alpha=ALPHA)
    protocol = "L" + h
    rows = []
    for method, fn in METHODS.items():
        for rep in range(1 if method == "ALL" else REPS):
            seed = seed_for(ds, f, protocol, method, ALPHA, BUDGET, rep, BASE_SEED)
            o = Oracle(pr, BUDGET)
            fn(o, pr.d, np.random.default_rng(seed))
            m = o.best_mask
            rows.append(dict(dataset=ds, fold=f, protocol=protocol, method=method, rep=rep, seed=f"{seed:016x}",
                             mask=m, fitness=o.best_f, f_star=pr.f_star, hit=int(o.best_f == pr.f_star),
                             sel_errors=int(e_sel[m]), n_sel=n_sel, oth_errors=int(e_oth[m]), n_oth=n_oth))
    RAW.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RAW / f"{ds}_f{f}_{protocol}.csv", index=False)
    return g


def run(workers):
    todo = [g for g in groups() if not (RAW / f"{g[0]}_f{g[1]}_L{g[2]}.csv").exists()]
    print(f"{len(groups())} groups, {len(todo)} to run on {workers} workers", flush=True)
    t0 = time.time()
    with Pool(workers) as pool:
        for i, g in enumerate(pool.imap_unordered(run_group, todo), 1):
            print(f"[{i:3d}/{len(todo)}] {g}  {time.time() - t0:6.0f} s", flush=True)
    runs = pd.concat([pd.read_csv(p) for p in sorted(RAW.glob("*.csv"))], ignore_index=True)
    runs.to_csv(OUT / "x5_runs.csv.gz", index=False)
    print(f"Wrote {OUT / 'x5_runs.csv.gz'} ({len(runs)} runs)")


# ------------------------------------------------------------------ analysis
def boot_ci(v, rng):
    v = np.asarray(v, dtype=float)
    means = v[rng.integers(0, len(v), (NBOOT, len(v)))].mean(axis=1)
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def analyze():
    rng = np.random.default_rng(20260919)
    x5 = pd.read_csv(OUT / "x5_runs.csv.gz")
    main_runs = pd.read_csv("results/runs.csv.gz")
    nmain = main_runs[(main_runs.experiment == "main") & (main_runs.protocol == "N")][
        ["dataset", "fold", "method", "rep", "mask"]].rename(columns={"mask": "mask_N"})
    x5["half"] = x5.protocol.str[1]
    x5 = x5.merge(nmain, on=["dataset", "fold", "method", "rep"], how="left", validate="many_to_one")
    assert x5.mask_N.notna().all()
    # accuracy of the protocol-N subset on the other half
    acc_oth_N = np.empty(len(x5))
    for (ds, f, h), idx in x5.groupby(["dataset", "fold", "half"]).groups.items():
        o_h = "B" if h == "A" else "A"
        e_oth = np.load(TAB / ds / f"fold{f}_{o_h}.npy").astype(np.int64)
        sub = x5.loc[idx]
        acc_oth_N[x5.index.get_indexer(idx)] = 1.0 - e_oth[sub.mask_N.astype(np.int64).values] / sub.n_oth.values
    x5["acc_sel"] = 1.0 - x5.sel_errors / x5.n_sel
    x5["acc_oth"] = 1.0 - x5.oth_errors / x5.n_oth
    x5["acc_oth_N"] = acc_oth_N
    x5["leak"] = (x5.acc_sel - x5.acc_oth) * 100
    x5["config"] = (x5.acc_oth - x5.acc_oth_N) * 100
    x5["total"] = (x5.acc_sel - x5.acc_oth_N) * 100
    # average the two directions per run, then median over seeds within a fold, then mean over folds
    per_run = x5.groupby(["dataset", "fold", "method", "rep"])[["leak", "config", "total"]].mean()
    fold = per_run.groupby(["dataset", "fold", "method"]).median()
    dset = fold.groupby(["dataset", "method"]).mean()
    out, summary = {}, {}
    for comp in ("leak", "config", "total"):
        t = dset[comp].unstack("method").reindex(index=EXACT, columns=SEARCH + ["ALL"])
        t.to_csv(OUT / f"x5_{comp}_pp.csv", float_format="%.6g")
        out[comp] = t
        m12 = t[SEARCH].mean(axis=1)
        summary[comp] = dict(mean_pp=float(m12.mean()), ci=boot_ci(m12, rng),
                             wilcoxon_p=float(stats.wilcoxon(m12).pvalue) if not np.allclose(m12, 0) else 1.0,
                             per_method_mean_pp={m: float(t[m].mean()) for m in SEARCH},
                             all_features_mean_pp=float(t["ALL"].mean()),
                             per_dataset_pp={k: float(v) for k, v in m12.items()})
    # exact optima of the half tables (ties averaged), against the exact validation optimum of protocol N
    rows = []
    for ds in EXACT:
        for f in range(5):
            pn = Problem(ds, f, "N")
            opt_val = np.flatnonzero(pn.fit == pn.f_star)
            for h in ("A", "B"):
                o_h = "B" if h == "A" else "A"
                e_sel = np.load(TAB / ds / f"fold{f}_{h}.npy").astype(np.int64)
                e_oth = np.load(TAB / ds / f"fold{f}_{o_h}.npy").astype(np.int64)
                hv = np.load(TAB / ds / f"fold{f}_halves.npz")
                n_sel, n_oth = len(hv[h]), len(hv[o_h])
                pr = Problem.from_arrays(e_sel, e_oth, n_sel, n_oth, "N", ALPHA)
                opt = np.flatnonzero(pr.fit == pr.f_star)
                a_sel = 1.0 - e_sel[opt] / n_sel
                a_oth = 1.0 - e_oth[opt] / n_oth
                a_oth_N = 1.0 - e_oth[opt_val] / n_oth
                rows.append(dict(dataset=ds, fold=f, half=h, n_optimal=len(opt),
                                 leak=100 * float((a_sel - a_oth).mean()),
                                 config=100 * float(a_oth.mean() - a_oth_N.mean()),
                                 total=100 * float(a_sel.mean() - a_oth_N.mean())))
    ex = pd.DataFrame(rows)
    ex.to_csv(OUT / "x5_exact_optimum_folds.csv", index=False, float_format="%.6g")
    exd = ex.groupby(["dataset", "fold"])[["leak", "config", "total"]].mean().groupby("dataset").mean().reindex(EXACT)
    exd.to_csv(OUT / "x5_exact_optimum_pp.csv", float_format="%.6g")
    summary["exact_optimum"] = {c: dict(mean_pp=float(exd[c].mean()), ci=boot_ci(exd[c], rng),
                                        wilcoxon_p=float(stats.wilcoxon(exd[c]).pvalue)) for c in ("leak", "config", "total")}
    summary["leak_as_multiple_of_0p69pp_spread"] = summary["leak"]["mean_pp"] / 0.69
    cfg_ci = summary["config"]["ci"]
    summary["decision_rule_config_within_1pp"] = bool(-1.0 <= cfg_ci[0] and cfg_ci[1] <= 1.0)
    hit_half = x5[x5.method.isin(SEARCH)].groupby(["dataset", "fold", "method", "half"]).hit.mean()
    summary["mean_hit_rate_on_half_tables"] = float(hit_half.mean())
    sp = OUT / "summary_revision.json"
    s = json.loads(sp.read_text()) if sp.exists() else {}
    s["X5"] = summary
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
