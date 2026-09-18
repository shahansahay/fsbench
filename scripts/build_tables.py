"""Build E_val and E_test for every exact dataset and outer fold (resumable)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import time

import numpy as np
import pandas as pd

from fsb.data import EXACT, load
from fsb.tables import build_pair

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="*", default=EXACT)
ap.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
args = ap.parse_args()

root = Path("tables")
index_path = root / "index.csv"
rows = pd.read_csv(index_path).to_dict("records") if index_path.exists() else []
done = {(r["dataset"], r["fold"]) for r in rows}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


for name in args.datasets:
    X, y = load(name)
    for f in args.folds:
        if (name, f) in done:
            print(f"skip {name} fold {f} (already built)")
            continue
        t0 = time.time()
        e_val, e_test, idx = build_pair(X, y, name, f)
        secs = time.time() - t0
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        pv, pt = folder / f"fold{f}_val.npy", folder / f"fold{f}_test.npy"
        np.save(pv, e_val)
        np.save(pt, e_test)
        np.savez(folder / f"fold{f}_splits.npz", **idx)
        rows.append(dict(dataset=name, fold=f, d=X.shape[1], subsets=2 ** X.shape[1] - 1,
                         n_inner_train=len(idx["inner_train"]), n_inner_val=len(idx["inner_val"]),
                         n_outer_train=len(idx["outer_train"]), n_outer_test=len(idx["outer_test"]),
                         opt_val_errors=int(e_val[1:].min()), opt_test_errors=int(e_test[1:].min()),
                         seconds=round(secs, 2), sha256_val=sha(pv), sha256_test=sha(pt)))
        pd.DataFrame(rows).to_csv(index_path, index=False)
        print(f"{name:13s} fold {f}  d={X.shape[1]:2d}  {secs:8.1f} s  "
              f"optimum val={e_val[1:].min()}/{len(idx['inner_val'])}  "
              f"test={e_test[1:].min()}/{len(idx['outer_test'])}")
print("Index:", index_path)
