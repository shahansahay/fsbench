import numpy as np

from fsb.algorithms import METHODS
from fsb.problem import Oracle, Problem, popcounts

D = 12
TARGET = 0b101101011010


def planted():
    """Errors = Hamming distance to TARGET, so TARGET is the unique optimum."""
    err = popcounts(D)[np.arange(1 << D) ^ TARGET].astype(np.uint16)
    return Problem.from_arrays(err, err, D, D)


def run(pr, method, seed, budget):
    o = Oracle(pr, budget)
    METHODS[method](o, pr.d, np.random.default_rng(seed))
    return o


def test_budget_is_exact():
    pr = planted()
    for method in METHODS:
        for budget in (100, 1000):
            o = run(pr, method, 1, budget)
            if method == "ALL":
                assert o.calls == 1
            elif method == "SFS":
                assert o.calls == min(D * (D + 1) // 2, budget)
            else:
                assert o.calls == budget


def test_same_seed_same_result():
    pr = planted()
    for method in METHODS:
        a, b = run(pr, method, 7, 500), run(pr, method, 7, 500)
        assert (a.best_mask, a.calls_to_best) == (b.best_mask, b.calls_to_best)


def test_best_is_minimum_of_everything_evaluated():
    pr = planted()
    for method in METHODS:
        o = run(pr, method, 3, 300)
        seen = np.fromiter(o.seen, dtype=np.int64)
        assert o.best_f == pr.fit[seen].min() == pr.fit[o.best_mask]


def test_simple_baselines_find_planted_optimum():
    pr = planted()
    assert pr.percentile(TARGET) == 0.0
    for method in ("EA", "SFS"):
        hits = sum(run(pr, method, s, 3000).best_mask == TARGET for s in range(10))
        assert hits == 10
