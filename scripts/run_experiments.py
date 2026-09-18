"""Run every method on every exact dataset, fold and protocol (parallel, resumable).

Experiments: main (protocols N and L, alpha 0.99, budget 1,000);
budget (protocol N, budgets 100, 300, 3,000); alpha (protocol N, alpha 0.9, budget 1,000).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from multiprocessing import Pool

import numpy as np
import pandas as pd

from fsb.algorithms import METHODS
from fsb.data import EXACT
from fsb.problem import Oracle, Problem
from fsb.tables import BASE_SEED, seed_for

RAW = Path("results/raw")
REPS = 30


def all_groups():
    out = []
    for ds in EXACT:
        for fold in range(5):
            out += [("main", ds, fold, "N", 0.99, 1000), ("main", ds, fold, "L", 0.99, 1000)]
            out += [("budget", ds, fold, "N", 0.99, b) for b in (100, 300, 3000)]
            out.append(("alpha", ds, fold, "N", 0.9, 1000))
    return out


def key(g):
    exp, ds, fold, protocol, alpha, budget = g
    return f"{exp}_{ds}_f{fold}_{protocol}_a{alpha}_b{budget}"


def run_group(g):
    exp, ds, fold, protocol, alpha, budget = g
    t0 = time.time()
    pr = Problem(ds, fold, protocol, alpha)
    rows = []
    for method, fn in METHODS.items():
        for rep in range(1 if method == "ALL" else REPS):
            seed = seed_for(ds, fold, protocol, method, alpha, budget, rep, BASE_SEED)
            o = Oracle(pr, budget)
            fn(o, pr.d, np.random.default_rng(seed))
            m = o.best_mask
            rows.append(dict(
                experiment=exp, dataset=ds, fold=fold, protocol=protocol, alpha=alpha,
                budget=budget, method=method, rep=rep, seed=f"{seed:016x}", d=pr.d,
                calls=o.calls, calls_to_best=o.calls_to_best, unique=len(o.seen),
                mask=m, size=bin(m).count("1"), fitness=o.best_f, f_star=pr.f_star,
                hit=int(o.best_f == pr.f_star), percentile=pr.percentile(m),
                fit_errors=int(pr.fit_errors[m]), n_fit=pr.n_fit,
                test_errors=int(pr.test_errors[m]), n_test=pr.n_test))
    df = pd.DataFrame(rows)
    df["fit_acc"] = 1.0 - df.fit_errors / df.n_fit
    df["test_acc"] = 1.0 - df.test_errors / df.n_test
    df.to_csv(RAW / f"{key(g)}.csv", index=False)
    return key(g), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--timing-test", action="store_true",
                    help="run 8 representative groups and estimate the full run time")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    groups = all_groups()
    if args.timing_test:
        groups = [g for g in groups if g[1] in ("HeartEW", "SpectEW") and g[2] == 0
                  and g[3] == "N" and g[0] in ("main", "budget")]
        for g in groups:
            (RAW / f"{key(g)}.csv").unlink(missing_ok=True)
    todo = [g for g in groups if (RAW / f"{key(g)}.csv").exists() is False]
    print(f"{len(groups)} groups, {len(groups) - len(todo)} already done, {len(todo)} to run "
          f"on {args.workers} workers")
    t0 = time.time()
    done_secs = {}
    with Pool(args.workers) as pool:
        for i, (k, secs) in enumerate(pool.imap_unordered(run_group, todo), 1):
            done_secs[k] = secs
            elapsed = time.time() - t0
            print(f"[{i:3d}/{len(todo)}] {k:40s} {secs:6.1f} s   elapsed {elapsed / 60:5.1f} min")
    if args.timing_test:
        def per_fold(ds):
            t = {int(k.split("_b")[-1]): v for k, v in done_secs.items() if f"_{ds}_" in k}
            return 3 * t[1000] + t[100] + t[300] + t[3000]
        est = (55 * per_fold("HeartEW") + 5 * per_fold("SpectEW")) / args.workers / 60
        print(f"Timing test done. Estimate for the full run: about {est:.0f} min on {args.workers} workers")
        return
    files = sorted(RAW.glob("*.csv"))
    runs = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    runs.to_csv("results/runs.csv.gz", index=False)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    import numba, scipy, sklearn
    prov = dict(created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                git_commit=commit, python=platform.python_version(), platform=platform.platform(),
                numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
                sklearn=sklearn.__version__, numba=numba.__version__,
                runs=len(runs), groups=len(files), base_seed=BASE_SEED, reps=REPS)
    Path("results/provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"Wrote results/runs.csv.gz ({len(runs)} runs) and results/provenance.json")


if __name__ == "__main__":
    main()
