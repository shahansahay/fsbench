import numpy as np

from fsb.data import load
from fsb.tables import (EMPTY, build_pair, error_table, minmax, reference_errors,
                        sqdiff, stratified_folds)


def test_folds_cover_every_row_once():
    _, y = load("Zoo")
    folds = stratified_folds(y, 5, 1)
    assert sorted(set(folds.tolist())) == [0, 1, 2, 3, 4]
    assert len(folds) == len(y)


def test_distance_tie_goes_to_lower_index():
    tr = np.array([[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]])
    ytr = np.array([0, 0, 0, 1, 1, 1])
    ev = np.array([[0.0]])
    D = sqdiff(ev, tr)
    table = error_table(D, ytr, np.array([0]), 2)
    assert table[1] == 0  # neighbours are rows 0-4: votes 3 vs 2 for class 0


def test_vote_tie_goes_to_nearest_tied_class():
    tr = np.array([[0.1], [0.2], [0.3], [0.4], [0.9]])
    ytr = np.array([1, 0, 0, 1, 2])
    ev = np.array([[0.0]])
    table = error_table(sqdiff(ev, tr), ytr, np.array([1]), 3)
    assert table[1] == 0  # classes 0 and 1 tie 2-2; nearest tied neighbour is class 1


def test_table_matches_reference_heart():
    X, y = load("HeartEW")
    e_val, e_test, idx = build_pair(X, y, "HeartEW", 0)
    assert e_val[0] == EMPTY and e_val.shape == (2 ** 13,)
    tr, ev = minmax(X[idx["outer_train"]], X[idx["outer_test"]])
    D = sqdiff(ev, tr)
    rng = np.random.default_rng(0)
    for m in rng.integers(1, 2 ** 13, size=300):
        assert reference_errors(D, y[idx["outer_train"]], y[idx["outer_test"]], 2, int(m)) == e_test[m]
