"""Exhaustive KNN (k=5) error tables over every non-empty feature subset.

Rules fixed in advance (state them in Methods):
- min-max scaling fitted on the training part only (constant feature: range 1)
- subset distance = squared differences summed in increasing feature order
- distances quantised to 1e-9 before ranking
- distance ties -> lower training index; vote ties -> class of the nearest tied neighbour
- table entry = number of misclassified evaluation rows (uint16); entry 0 (empty set) = 65535
"""
import hashlib

import numpy as np
from numba import njit, prange

K = 5
BASE_SEED = 20260919
EMPTY = 65535


def seed_for(*parts):
    digest = hashlib.md5("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def stratified_folds(y, k, seed):
    """Fold id per row; each class dealt round-robin after a seeded shuffle."""
    rng = np.random.default_rng(seed)
    fold = np.empty(len(y), dtype=np.int64)
    offset = 0
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        fold[idx] = (np.arange(len(idx)) + offset) % k
        offset += len(idx)
    return fold


def stratified_holdout(y, frac, seed):
    """Boolean mask of validation rows: floor(frac * n_c + 0.5) rows per class."""
    rng = np.random.default_rng(seed)
    val = np.zeros(len(y), dtype=bool)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        val[idx[: int(np.floor(frac * len(idx) + 0.5))]] = True
    return val


def minmax(train, other):
    lo = train.min(axis=0)
    rng = train.max(axis=0) - lo
    rng[rng == 0] = 1.0
    return (train - lo) / rng, (other - lo) / rng


def sqdiff(ev, tr):
    """D[i, t, j] = (ev[i, j] - tr[t, j])**2, shape (n_eval, n_train, d)."""
    return np.ascontiguousarray((ev[:, None, :] - tr[None, :, :]) ** 2)


@njit(parallel=True, cache=True)
def error_table(D, ytr, yev, n_classes):
    ne, nt, d = D.shape
    out = np.empty(1 << d, dtype=np.uint16)
    out[0] = EMPTY
    for m in prange(1, 1 << d):
        feats = np.empty(d, dtype=np.int64)
        nf = 0
        for j in range(d):
            if (m >> j) & 1:
                feats[nf] = j
                nf += 1
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
        out[m] = errors
    return out


def reference_errors(D, ytr, yev, n_classes, mask):
    """Independent plain-numpy implementation of the same rules (for verification)."""
    feats = [j for j in range(D.shape[2]) if (mask >> j) & 1]
    s = np.zeros(D.shape[:2])
    for j in feats:
        s = s + D[:, :, j]
    q = np.floor(s * 1e9 + 0.5).astype(np.int64)
    order_idx = np.arange(D.shape[1])
    errors = 0
    for i in range(D.shape[0]):
        nearest = np.lexsort((order_idx, q[i]))[:K]
        labels = ytr[nearest]
        votes = np.bincount(labels, minlength=n_classes)
        top = votes.max()
        pred = next(c for c in labels if votes[c] == top)
        errors += int(pred == yev[i]) ^ 1
    return errors


def split_parts(X, y, name, fold_id, n_folds=5, inner_frac=0.2):
    """Index arrays for one outer fold: inner-train, inner-val, outer-train, outer-test."""
    folds = stratified_folds(y, n_folds, seed_for(name, "outer", BASE_SEED))
    otr = np.flatnonzero(folds - fold_id)
    ote = np.flatnonzero(folds == fold_id)
    val = stratified_holdout(y[otr], inner_frac, seed_for(name, "inner", fold_id, BASE_SEED))
    return otr[~val], otr[val], otr, ote


def build_pair(X, y, name, fold_id):
    """Return E_val, E_test tables and the index sets used."""
    itr, iva, otr, ote = split_parts(X, y, name, fold_id)
    C = int(y.max()) + 1
    tr, ev = minmax(X[itr], X[iva])
    e_val = error_table(sqdiff(ev, tr), y[itr], y[iva], C)
    tr, ev = minmax(X[otr], X[ote])
    e_test = error_table(sqdiff(ev, tr), y[otr], y[ote], C)
    return e_val, e_test, dict(inner_train=itr, inner_val=iva, outer_train=otr, outer_test=ote)
