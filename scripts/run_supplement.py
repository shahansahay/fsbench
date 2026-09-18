"""Exploratory replication on the three larger canonical datasets with live KNN evaluation
(BreastEW 30, IonosphereEW 34 and SonarEW 60 features): protocols N and L, budget 1,000,
30 seeds, same methods, splits and tie rules. No exact optimum exists at this size, so only
leakage, the benefit of selection and rankings are measured."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from fsb.algorithms import METHODS
from fsb.live import LiveOracle, LiveProblem
from fsb.tables import BASE_SEED, seed_for

DATASETS = ["BreastEW", "IonosphereEW", "SonarEW"]
OUTDIR = Path("results/supplement")
RAW = OUTDIR / "raw"
REPS = 30


def all_groups():
    return [(ds, fold, protocol) for ds in DATASETS for fold in range(5) for protocol in ("N", "L")]


def key(g):
    return f"supp_{g[0]}_f{g[1]}_{g[2]}"


def run_group(g, reps=REPS, save=True):
    ds, fold, protocol = g
    t0 = time.time()
    pr = LiveProblem(ds, fold, protocol)
    rows = []
    for method, fn in METHODS.items():
        for rep in range(1 if method == "ALL" else reps):
            seed = seed_for(ds, fold, protocol, method, 0.99, 1000, rep, BASE_SEED)
            o = LiveOracle(pr, 1000)
            fn(o, pr.d, np.random.default_rng(seed))
            m = o.best_mask
            rows.append(dict(dataset=ds, fold=fold, protocol=protocol, method=method, rep=rep,
                             seed=f"{seed:016x}", d=pr.d, calls=o.calls, calls_to_best=o.calls_to_best,
                             unique=len(o.seen), mask=str(m), size=bin(m).count("1"),
                             fitness=o.best_f, fit_errors=pr.fit_split.errors(m), n_fit=pr.n_fit,
                             test_errors=pr.test.errors(m), n_test=pr.n_test))
    df = pd.DataFrame(rows)
    df["fit_acc"] = 1.0 - df.fit_errors / df.n_fit
    df["test_acc"] = 1.0 - df.test_errors / df.n_test
    if save:
        df.to_csv(RAW / f"{key(g)}.csv", index=False)
    return key(g), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timing-test", action="store_true",
                    help="run BreastEW fold 0 (both protocols) with 3 seeds and estimate the full run")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    if args.timing_test:
        secs = sum(run_group(("BreastEW", 0, p), reps=3, save=False)[1] for p in ("N", "L"))
        est = secs / 2 * 10 * len(all_groups()) / args.workers / 60
        print(f"Timing test done. Upper estimate for the full run: about {est:.0f} min on "
              f"{args.workers} workers (BreastEW is the slowest dataset; caching makes it faster)")
        return
    todo = [g for g in all_groups() if (RAW / f"{key(g)}.csv").exists() is False]
    print(f"{len(all_groups())} groups, {len(todo)} to run on {args.workers} workers")
    t0 = time.time()
    with Pool(args.workers) as pool:
        for i, (k, secs) in enumerate(pool.imap_unordered(run_group, todo), 1):
            print(f"[{i:2d}/{len(todo)}] {k:28s} {secs:7.1f} s   elapsed {(time.time() - t0) / 60:5.1f} min")
    runs = pd.concat([pd.read_csv(f) for f in sorted(RAW.glob("*.csv"))], ignore_index=True)
    runs.to_csv(OUTDIR / "runs.csv.gz", index=False)
    print(f"Wrote {OUTDIR}/runs.csv.gz ({len(runs)} runs)")


if __name__ == "__main__":
    main()
