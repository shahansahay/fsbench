"""Live KNN evaluation for the larger canonical datasets (exploratory supplement).

Same splits, scaling, distance and tie rules as fsb.tables; errors are computed on demand
for every subset a method visits and cached per split, so revisits cost nothing.
"""
import numpy as np
from numba import njit

from .data import load
from .problem import EMPTY_FITNESS, Oracle
from .tables import K, minmax, split_parts, sqdiff


@njit(cache=True)
def errors_for(D, ytr, yev, n_classes, feats):
    ne, nt, _ = D.shape
    nf = len(feats)
    bd = np.empty(K, dtype=np.int64)
    bi = np.empty(K, dtype=np.int64)
    votes = np.zeros(n_classes, dtype=np.int64)
    errors = 0
    for i in range(ne):
        cnt = 0
        for t in range(nt):
            s = 0.0
            for f in range(nf):
                s += D[i, t, feats[f]]
            q = np.int64(np.floor(s * 1e9 + 0.5))
            if cnt < K:
                p = cnt
                cnt += 1
            elif q < bd[K - 1]:
                p = K - 1
            else:
                continue
            while p > 0 and bd[p - 1] > q:
                bd[p] = bd[p - 1]
                bi[p] = bi[p - 1]
                p -= 1
            bd[p] = q
            bi[p] = t
        for c in range(n_classes):
            votes[c] = 0
        top = 0
        for r in range(cnt):
            c = ytr[bi[r]]
            votes[c] += 1
            if votes[c] > top:
                top = votes[c]
        pred = -1
        for r in range(cnt):
            c = ytr[bi[r]]
            if votes[c] == top:
                pred = c
                break
        if pred == yev[i]:
            continue
        errors += 1
    return errors


class Split:
    def __init__(self, Xtr, ytr, Xev, yev, n_classes):
        tr, ev = minmax(Xtr, Xev)
        self.D = sqdiff(ev, tr)
        self.ytr, self.yev, self.C, self.n = ytr, yev, n_classes, len(yev)
        self.cache = {}

    def errors(self, mask):
        e = self.cache.get(mask)
        if e is None:
            d = self.D.shape[2]
            feats = np.array([j for j in range(d) if (mask >> j) & 1], dtype=np.int64)
            e = int(errors_for(self.D, self.ytr, self.yev, self.C, feats))
            self.cache[mask] = e
        return e


class LiveProblem:
    """Same interface as fsb.problem.Problem for what the runner needs, without a full table."""

    def __init__(self, dataset, fold, protocol, alpha=0.99):
        X, y = load(dataset)
        itr, iva, otr, ote = split_parts(X, y, dataset, fold)
        C = int(y.max()) + 1
        self.dataset, self.fold, self.protocol, self.alpha = dataset, fold, protocol, alpha
        self.d = X.shape[1]
        self.val = Split(X[itr], y[itr], X[iva], y[iva], C)
        self.test = Split(X[otr], y[otr], X[ote], y[ote], C)
        self.fit_split = self.val if protocol == "N" else self.test
        self.n_fit, self.n_test = self.fit_split.n, self.test.n

    def fitness(self, mask):
        if mask == 0:
            return EMPTY_FITNESS
        e = self.fit_split.errors(mask)
        return self.alpha * e / self.n_fit + (1.0 - self.alpha) * bin(mask).count("1") / self.d


class LiveOracle(Oracle):
    def __init__(self, problem, budget):
        self.problem = problem
        self.budget = int(budget)
        self.calls = 0
        self.best_f = np.inf
        self.best_mask = 0
        self.calls_to_best = 0
        self.seen = set()

    def __call__(self, masks):
        masks = np.atleast_1d(np.asarray(masks, dtype=np.int64))
        out = np.full(len(masks), np.inf)
        n = min(len(masks), self.budget - self.calls)
        if n > 0:
            mm = [int(m) for m in masks[:n]]
            fv = np.array([self.problem.fitness(m) for m in mm])
            out[:n] = fv
            i = int(np.argmin(fv))
            if fv[i] < self.best_f:
                self.best_f = float(fv[i])
                self.best_mask = mm[i]
                self.calls_to_best = self.calls + i + 1
            self.seen.update(mm)
            self.calls += n
        return out
