"""Revision analysis X6 (REVISION_PLAN.md, tag revision-plan-v1): duplicate and conflicting rows. Exploratory.

(a) Per dataset and fold, the share of evaluation rows with an identical feature vector among their training rows.
(b) Deduplicated sensitivity: on the 8 datasets with duplicates keep the first copy of every identical
    (features, label) row and drop every row whose feature vector occurs with more than one label; rebuild
    splits and tables with the unchanged code and seed scheme, rerun protocols N and L, and recompute the frozen
    decisions on the suite in which those 8 datasets are replaced.

  python scripts/revision_x6.py describe
  python scripts/revision_x6.py build
  python scripts/revision_x6.py run --workers 8
  python scripts/revision_x6.py analyze
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

from fsb.data import EXACT, load
from fsb.tables import build_pair
from revision_common import endpoints, run_methods

OUT = Path("results/revision")
TAB = OUT / "x6_tables"
RAW = OUT / "x6_raw"
DUP = ["Breastcancer", "Exactly", "Exactly2", "M-of-n", "CongressEW", "Vote", "Zoo", "SpectEW"]


def dedup(X, y):
    """Keep the first copy of identical (features, label) rows; drop all rows of conflicting feature vectors."""
    keys = [x.tobytes() for x in X]
    labels = {}
    for k, c in zip(keys, y):
        labels.setdefault(k, set()).add(int(c))
    keep, seen = [], set()
    for i, (k, c) in enumerate(zip(keys, y)):
        if len(labels[k]) > 1 or (k, int(c)) in seen:
            continue
        seen.add((k, int(c)))
        keep.append(i)
    keep = np.array(keep)
    # relabel to consecutive integers in case a class disappears
    _, y2 = np.unique(y[keep], return_inverse=True)
    return np.ascontiguousarray(X[keep]), y2.astype(np.int64)


def describe():
    rows = []
    for ds in EXACT:
        X, y = load(ds)
        for f in range(5):
            s = np.load(f"tables/{ds}/fold{f}_splits.npz")
            for part, tr, ev in (("inner_validation", "inner_train", "inner_val"), ("outer_test", "outer_train", "outer_test")):
                trX = {}
                for i in s[tr]:
                    trX.setdefault(X[i].tobytes(), set()).add(int(y[i]))
                same = diff = 0
                for i in s[ev]:
                    lab = trX.get(X[i].tobytes())
                    if lab is None:
                        continue
                    if int(y[i]) in lab:
                        same += 1
                    if lab - {int(y[i])}:
                        diff += 1
                rows.append(dict(dataset=ds, fold=f, part=part, n_eval=len(s[ev]),
                                 with_duplicate_same_label=same, with_duplicate_other_label=diff))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "x6_cross_split_duplicates_folds.csv", index=False)
    g = df.groupby(["dataset", "part"])[["n_eval", "with_duplicate_same_label", "with_duplicate_other_label"]].sum()
    g["share_same_pct"] = 100 * g.with_duplicate_same_label / g.n_eval
    g["share_other_pct"] = 100 * g.with_duplicate_other_label / g.n_eval
    g = g.reset_index().pivot(index="dataset", columns="part", values=["share_same_pct", "share_other_pct"]).reindex(EXACT)
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g.to_csv(OUT / "x6_cross_split_duplicates.csv", float_format="%.4g")
    print(g.round(2).to_string())
    info = []
    for ds in DUP:
        X, y = load(ds)
        X2, y2 = dedup(X, y)
        info.append(dict(dataset=ds, n=len(y), n_dedup=len(y2), classes=int(y.max()) + 1, classes_dedup=int(y2.max()) + 1,
                         smallest_class_dedup=int(np.bincount(y2).min())))
    pd.DataFrame(info).to_csv(OUT / "x6_dedup_sizes.csv", index=False)
    print(pd.DataFrame(info).to_string(index=False))


def build():
    index_path = TAB / "index.csv"
    TAB.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(index_path).to_dict("records") if index_path.exists() else []
    done = {(r["dataset"], r["fold"]) for r in rows}
    for ds in DUP:
        X, y = dedup(*load(ds))
        for f in range(5):
            if (ds, f) in done:
                continue
            t0 = time.time()
            e_val, e_test, idx = build_pair(X, y, ds, f)
            folder = TAB / ds
            folder.mkdir(parents=True, exist_ok=True)
            pv, pt = folder / f"fold{f}_val.npy", folder / f"fold{f}_test.npy"
            np.save(pv, e_val)
            np.save(pt, e_test)
            np.savez(folder / f"fold{f}_splits.npz", **idx)
            rows.append(dict(dataset=ds, fold=f, d=X.shape[1], n=len(y), n_inner_val=len(idx["inner_val"]),
                             n_outer_test=len(idx["outer_test"]), seconds=round(time.time() - t0, 2),
                             sha256_val=hashlib.sha256(pv.read_bytes()).hexdigest(),
                             sha256_test=hashlib.sha256(pt.read_bytes()).hexdigest()))
            pd.DataFrame(rows).to_csv(index_path, index=False)
            print(f"{ds:13s} fold {f}  n={len(y)}  {time.time() - t0:7.1f} s", flush=True)


def run_group(g):
    ds, f, protocol = g
    s = np.load(TAB / ds / f"fold{f}_splits.npz")
    df = run_methods(np.load(TAB / ds / f"fold{f}_val.npy"), np.load(TAB / ds / f"fold{f}_test.npy"),
                     len(s["inner_val"]), len(s["outer_test"]), ds, f, protocol)
    RAW.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW / f"{ds}_f{f}_{protocol}.csv", index=False)
    return g


def run(workers):
    groups = [(ds, f, p) for ds in DUP for f in range(5) for p in ("N", "L")]
    todo = [g for g in groups if not (RAW / f"{g[0]}_f{g[1]}_{g[2]}.csv").exists()]
    print(f"{len(groups)} groups, {len(todo)} to run on {workers} workers", flush=True)
    t0 = time.time()
    with Pool(workers) as pool:
        for i, g in enumerate(pool.imap_unordered(run_group, todo), 1):
            print(f"[{i:3d}/{len(todo)}] {g}  {time.time() - t0:6.0f} s", flush=True)
    runs = pd.concat([pd.read_csv(p) for p in sorted(RAW.glob("*.csv"))], ignore_index=True)
    runs.to_csv(OUT / "x6_runs.csv.gz", index=False)
    print(f"Wrote {OUT / 'x6_runs.csv.gz'} ({len(runs)} runs)")


def analyze():
    rng = np.random.default_rng(20260919)
    d_of = {ds: load(ds)[0].shape[1] for ds in EXACT}
    main_runs = pd.read_csv("results/runs.csv.gz")
    main_runs = main_runs[main_runs.experiment == "main"]
    dd = pd.read_csv(OUT / "x6_runs.csv.gz")
    keep = main_runs[~main_runs.dataset.isin(DUP)]
    cols = [c for c in dd.columns if c in keep.columns]
    suite = pd.concat([keep[cols], dd[cols]], ignore_index=True)
    res_orig, infl_o = endpoints(main_runs[main_runs.protocol == "N"], main_runs[main_runs.protocol == "L"], d_of, rng)
    res_dd, infl_d = endpoints(suite[suite.protocol == "N"], suite[suite.protocol == "L"], d_of, rng)
    per_ds = pd.DataFrame(dict(inflation_original_pp=infl_o.drop(columns="ALL").mean(axis=1),
                               inflation_dedup_pp=infl_d.drop(columns="ALL").mean(axis=1))).reindex(EXACT)
    per_ds.to_csv(OUT / "x6_inflation_per_dataset.csv", float_format="%.6g")
    sp = OUT / "summary_revision.json"
    s = json.loads(sp.read_text()) if sp.exists() else {}
    s["X6"] = dict(original_suite=res_orig, deduplicated_suite=res_dd)
    sp.write_text(json.dumps(s, indent=2, default=float))
    print(json.dumps(s["X6"], indent=2, default=float))
    print(per_ds.round(2).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["describe", "build", "run", "analyze"])
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    {"describe": describe, "build": build, "run": lambda: run(a.workers), "analyze": analyze}[a.step]()
