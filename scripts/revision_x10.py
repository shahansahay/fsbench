"""Revision analysis X10 (REVISION_PLAN.md, tag revision-plan-v1): rebuild on a second platform. Exploratory.

Rebuilds E_val and E_test of every dataset with d <= 16 (100 of the 120 tables) with the unchanged code into a
scratch folder and compares their SHA-256 digests with tables/index.csv (built on the Apple M2 machine). The X5
identity e_A + e_B = e_test, checked in revision_x5.py, extends the comparison to the remaining datasets.
Writes results/revision/x10_rebuild.csv and updates summary_revision.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hashlib
import io
import json
import platform
import time

import numpy as np
import pandas as pd

from fsb.data import EXACT, load
from fsb.tables import build_pair

OUT = Path("results/revision")


def digest(arr):
    buf = io.BytesIO()
    np.save(buf, arr)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def main():
    index = pd.read_csv("tables/index.csv")
    rows = []
    for ds in EXACT:
        X, y = load(ds)
        if X.shape[1] > 16:
            continue
        for f in range(5):
            t0 = time.time()
            e_val, e_test, _ = build_pair(X, y, ds, f)
            ref = index[(index.dataset == ds) & (index.fold == f)].iloc[0]
            rows.append(dict(dataset=ds, fold=f, d=X.shape[1], val_identical=digest(e_val) == ref.sha256_val,
                             test_identical=digest(e_test) == ref.sha256_test, seconds=round(time.time() - t0, 2)))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "x10_rebuild.csv", index=False)
    x5 = pd.read_csv(OUT / "x5_tables" / "index.csv")
    sp = OUT / "summary_revision.json"
    s = json.loads(sp.read_text()) if sp.exists() else {}
    s["X10"] = dict(platform=platform.platform(), tables_rebuilt=int(2 * len(df)),
                    tables_identical=int(df.val_identical.sum() + df.test_identical.sum()),
                    x5_identity_folds=int(len(x5)), x5_identity_holds=int(x5.identity_eA_plus_eB_equals_etest.sum()))
    sp.write_text(json.dumps(s, indent=2, default=float))
    print(json.dumps(s["X10"], indent=2))


if __name__ == "__main__":
    main()
