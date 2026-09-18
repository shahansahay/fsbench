"""Verification of exhaustive tables against the reference implementation and scikit-learn."""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from .tables import K, EMPTY, minmax, sqdiff, reference_errors


def check_table(table, Xtr, ytr, Xev, yev, n_random=2000, n_opt=200, n_sklearn=300, seed=0):
    d = Xtr.shape[1]
    C = int(max(ytr.max(), yev.max())) + 1
    trs, evs = minmax(Xtr, Xev)
    D = sqdiff(evs, trs)
    rng = np.random.default_rng(seed)
    n_masks = (1 << d) - 1
    assert table.shape == (1 << d,) and table[0] == EMPTY
    assert table[1:].max() <= len(yev)
    body = table[1:].astype(np.int64)
    opt_masks = np.flatnonzero(body == body.min()) + 1
    picks = set([n_masks])  # full feature set
    picks.update(rng.choice(opt_masks, size=min(n_opt, len(opt_masks)), replace=False).tolist())
    picks.update(rng.integers(1, n_masks + 1, size=min(n_random, n_masks)).tolist())
    mismatches = [m for m in sorted(picks) if reference_errors(D, ytr, yev, C, m) - int(table[m])]
    agree_err = []
    for m in rng.integers(1, n_masks + 1, size=n_sklearn).tolist() + [n_masks]:
        cols = [j for j in range(d) if (m >> j) & 1]
        knn = KNeighborsClassifier(n_neighbors=K, algorithm="brute").fit(trs[:, cols], ytr)
        pred = knn.predict(evs[:, cols])
        agree_err.append(int((pred - yev).astype(bool).sum()) == int(table[m]))
    return dict(checked=len(picks), mismatches=len(mismatches), optimum=int(body.min()),
                n_optimal_subsets=int(len(opt_masks)),
                sklearn_error_agreement=float(np.mean(agree_err)))
