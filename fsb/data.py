"""Dataset registry and loader for the canonical 18-dataset suite."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

# name -> (file, published_d, published_n, drop_first_column)
SUITE = {
    "Breastcancer": ("BreastCancer.csv", 9, 699, True),   # col 0 is the sample ID
    "BreastEW":     ("BreastEW.csv", 30, 569, False),
    "CongressEW":   ("CongressEW.csv", 16, 435, False),
    "Exactly":      ("Exactly.csv", 13, 1000, False),
    "Exactly2":     ("Exactly2.csv", 13, 1000, False),
    "HeartEW":      ("HeartEW.csv", 13, 270, False),
    "IonosphereEW": ("Ionosphere.csv", 34, 351, False),
    "KrvskpEW":     ("KrVsKpEW.csv", 36, 3196, False),
    "Lymphography": ("Lymphography.csv", 18, 148, False),
    "M-of-n":       ("M-of-n.csv", 13, 1000, False),
    "PenglungEW":   ("PenglungEW.csv", 325, 73, False),
    "SonarEW":      ("Sonar.csv", 60, 208, False),
    "SpectEW":      ("SpectEW.csv", 22, 267, False),
    "Tic-tac-toe":  ("Tic-tac-toe.csv", 9, 958, False),
    "Vote":         ("Vote.csv", 16, 300, False),
    "WaveformEW":   ("WaveformEW.csv", 40, 5000, False),
    "WineEW":       ("Wine.csv", 13, 178, False),
    "Zoo":          ("Zoo.csv", 16, 101, False),
}

EXACT = [k for k, v in SUITE.items() if v[1] <= 22]          # 12 datasets
SUPPLEMENT = [k for k in SUITE if k not in EXACT]              # 6 datasets


def load(name):
    """Return X (float64, n x d), y (int64, 0..C-1)."""
    fname, _, _, drop_first = SUITE[name]
    df = pd.read_csv(RAW / fname, header=None)
    X = df.iloc[:, :-1].to_numpy(dtype=np.float64)
    if drop_first:
        X = X[:, 1:]
    _, y = np.unique(df.iloc[:, -1].to_numpy(), return_inverse=True)
    return np.ascontiguousarray(X), y.astype(np.int64)


def manifest():
    rows = []
    for name, (fname, pub_d, pub_n, _) in SUITE.items():
        raw = (RAW / fname).read_bytes()
        X, y = load(name)
        counts = np.bincount(y)
        dup_xy = int(pd.DataFrame(np.column_stack([X, y])).duplicated().sum())
        dup_x = int(pd.DataFrame(X).duplicated().sum())
        rows.append({
            "dataset": name, "file": fname,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "n": X.shape[0], "published_n": pub_n,
            "d": X.shape[1], "published_d": pub_d,
            "classes": len(counts), "smallest_class": int(counts.min()),
            "class_counts": "/".join(map(str, counts)),
            "duplicate_rows": dup_xy,
            "conflicting_duplicates": dup_x - dup_xy,
            "exact_reference": name in EXACT,
        })
    return pd.DataFrame(rows)
