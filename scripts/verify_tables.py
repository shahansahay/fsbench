"""Verify every built table against the reference implementation; report sklearn agreement."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse

import numpy as np
import pandas as pd

from fsb.data import load
from fsb.verify import check_table

ap = argparse.ArgumentParser()
ap.add_argument("--random", type=int, default=2000)
ap.add_argument("--optimal", type=int, default=200)
ap.add_argument("--sklearn", type=int, default=300)
args = ap.parse_args()

index = pd.read_csv("tables/index.csv")
rows = []
for r in index.itertuples():
    X, y = load(r.dataset)
    s = np.load(f"tables/{r.dataset}/fold{r.fold}_splits.npz")
    for kind, tr, ev in (("val", "inner_train", "inner_val"), ("test", "outer_train", "outer_test")):
        table = np.load(f"tables/{r.dataset}/fold{r.fold}_{kind}.npy")
        res = check_table(table, X[s[tr]], y[s[tr]], X[s[ev]], y[s[ev]],
                          n_random=args.random, n_opt=args.optimal, n_sklearn=args.sklearn,
                          seed=r.fold)
        rows.append(dict(dataset=r.dataset, fold=r.fold, table=kind, **res))
        print(f"{r.dataset:13s} fold {r.fold} {kind:4s} checked={res['checked']:5d} "
              f"mismatches={res['mismatches']} sklearn_agreement={res['sklearn_error_agreement']:.3f}")
Path("reports").mkdir(exist_ok=True)
out = pd.DataFrame(rows)
out.to_csv("reports/verification.csv", index=False)
total = int(out.mismatches.sum())
print(f"\nTotal subsets checked: {int(out.checked.sum())}, mismatches: {total}")
print("Wrote reports/verification.csv")
sys.exit(1 if total else 0)
