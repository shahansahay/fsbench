"""Fitness over the exact tables, with budget accounting and the exact reference.

f(S) = alpha * errors(S) / n_eval + (1 - alpha) * |S| / d ; the empty subset scores 2.0.
Protocol N: fitness from E_val (inner validation). Protocol L: fitness from E_test.
Every returned subset is scored on E_test (outer test fold) for its test accuracy.
"""
from pathlib import Path

import numpy as np

TABLES = Path(__file__).resolve().parents[1] / "tables"
EMPTY_FITNESS = 2.0
_POPCOUNT = {}


def popcounts(d):
    if d not in _POPCOUNT:
        m = np.arange(1 << d, dtype=np.int64)
        c = np.zeros(1 << d, dtype=np.int64)
        for j in range(d):
            c += (m >> j) & 1
        _POPCOUNT[d] = c
    return _POPCOUNT[d]


def fitness_array(errors, n_eval, d, alpha):
    f = alpha * errors.astype(np.float64) / n_eval + (1.0 - alpha) * popcounts(d) / d
    f[0] = EMPTY_FITNESS
    return f


class Problem:
    """Everything a run needs for one dataset, fold, protocol and alpha."""

    def __init__(self, dataset, fold, protocol, alpha=0.99):
        folder = TABLES / dataset
        splits = np.load(folder / f"fold{fold}_splits.npz")
        self._setup(np.load(folder / f"fold{fold}_val.npy"), np.load(folder / f"fold{fold}_test.npy"),
                    len(splits["inner_val"]), len(splits["outer_test"]), protocol, alpha)
        self.dataset, self.fold = dataset, fold

    @classmethod
    def from_arrays(cls, e_val, e_test, n_val, n_test, protocol="N", alpha=0.99):
        obj = cls.__new__(cls)
        obj._setup(np.asarray(e_val), np.asarray(e_test), n_val, n_test, protocol, alpha)
        obj.dataset, obj.fold = "synthetic", 0
        return obj

    def _setup(self, e_val, e_test, n_val, n_test, protocol, alpha):
        self.protocol, self.alpha = protocol, alpha
        self.d = int(len(e_val)).bit_length() - 1
        self.n_val, self.n_test = n_val, n_test
        if protocol == "N":
            self.fit_errors, self.n_fit = e_val, self.n_val
        elif protocol == "L":
            self.fit_errors, self.n_fit = e_test, self.n_test
        else:
            raise ValueError(protocol)
        self.test_errors = e_test
        self.fit = fitness_array(self.fit_errors, self.n_fit, self.d, alpha)
        self.f_star = float(self.fit[1:].min())
        self._sorted = np.sort(self.fit[1:])

    def percentile(self, mask):
        """Fraction of non-empty subsets with strictly better fitness (0 = exact optimum)."""
        return float(np.searchsorted(self._sorted, self.fit[mask], side="left") / len(self._sorted))


class Oracle:
    """Counts every fitness call against the budget and keeps the best subset seen."""

    def __init__(self, problem, budget):
        self.fit = problem.fit
        self.budget = int(budget)
        self.calls = 0
        self.best_f = np.inf
        self.best_mask = 0
        self.calls_to_best = 0
        self.seen = set()

    @property
    def exhausted(self):
        return self.calls >= self.budget

    def __call__(self, masks):
        masks = np.atleast_1d(np.asarray(masks, dtype=np.int64))
        out = np.full(len(masks), np.inf)
        n = min(len(masks), self.budget - self.calls)
        if n > 0:
            mm = masks[:n]
            fv = self.fit[mm]
            out[:n] = fv
            i = int(np.argmin(fv))
            if fv[i] < self.best_f:
                self.best_f = float(fv[i])
                self.best_mask = int(mm[i])
                self.calls_to_best = self.calls + i + 1
            self.seen.update(mm.tolist())
            self.calls += n
        return out
