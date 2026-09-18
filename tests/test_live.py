import numpy as np

from fsb.live import LiveOracle, LiveProblem
from fsb.problem import Oracle, Problem


def test_live_errors_match_exact_tables():
    live = LiveProblem("HeartEW", 0, "N")
    e_val = np.load("tables/HeartEW/fold0_val.npy")
    e_test = np.load("tables/HeartEW/fold0_test.npy")
    rng = np.random.default_rng(0)
    for m in rng.integers(1, 2 ** 13, size=300).tolist():
        assert live.val.errors(m) == e_val[m]
        assert live.test.errors(m) == e_test[m]


def test_live_fitness_matches_exact_fitness():
    for protocol in ("N", "L"):
        live, exact = LiveProblem("HeartEW", 1, protocol), Problem("HeartEW", 1, protocol)
        for m in range(1, 2 ** 13, 97):
            assert live.fitness(m) == exact.fit[m]


def test_live_oracle_matches_exact_oracle():
    from fsb.algorithms import METHODS
    live, exact = LiveProblem("HeartEW", 2, "N"), Problem("HeartEW", 2, "N")
    for method in ("GA", "PSO-S", "EA", "SFS"):
        a, b = LiveOracle(live, 300), Oracle(exact, 300)
        METHODS[method](a, 13, np.random.default_rng(5))
        METHODS[method](b, 13, np.random.default_rng(5))
        assert (a.best_mask, a.calls, a.best_f) == (b.best_mask, b.calls, b.best_f)
