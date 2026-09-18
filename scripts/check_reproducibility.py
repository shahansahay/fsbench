"""Re-run a random sample of stored experiment groups from scratch and compare the new result
files with the stored ones byte for byte (one group per dataset, plus one supplement group).
Writes reports/reproducibility.csv."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse
import filecmp
import tempfile

import numpy as np
import pandas as pd

import run_experiments as rx
import run_supplement as rsu

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=2026)
args = ap.parse_args()
rng = np.random.default_rng(args.seed)
stored_main, stored_supp = Path("results/raw"), Path("results/supplement/raw")

by_ds = {}
for g in rx.all_groups():
    if (stored_main / f"{rx.key(g)}.csv").exists():
        by_ds.setdefault(g[1], []).append(g)
picks = [("main", by_ds[ds][int(rng.integers(len(by_ds[ds])))]) for ds in sorted(by_ds)]
supp = [g for g in rsu.all_groups() if g[0] == "SonarEW" and (stored_supp / f"{rsu.key(g)}.csv").exists()]
if supp:
    picks.append(("supplement", supp[int(rng.integers(len(supp)))]))

tmp = Path(tempfile.mkdtemp())
rx.RAW, rsu.RAW = tmp, tmp
rows = []
for kind, g in picks:
    mod, stored = (rx, stored_main) if kind == "main" else (rsu, stored_supp)
    k, secs = mod.run_group(g)
    same = filecmp.cmp(tmp / f"{k}.csv", stored / f"{k}.csv", shallow=False)
    runs = len(pd.read_csv(tmp / f"{k}.csv"))
    rows.append(dict(kind=kind, group=k, runs=runs, identical=same, seconds=round(secs, 1)))
    print(f"{k:42s} {runs:4d} runs  {'identical' if same else 'DIFFERENT'}  ({secs:.1f} s)")
out = pd.DataFrame(rows)
Path("reports").mkdir(exist_ok=True)
out.to_csv("reports/reproducibility.csv", index=False)
print(f"\n{int(out.identical.sum())} of {len(out)} groups ({int(out.runs.sum())} runs) reproduced byte for byte")
print("Wrote reports/reproducibility.csv")
