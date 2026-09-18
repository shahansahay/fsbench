"""Binary search methods. Every fitness call goes through the Oracle (budget counted there).

Continuous-space methods (PSO, GWO, WOA, SCA) keep real-valued vectors and map them to bits
with a transfer function (Mirjalili and Lewis 2013):
  S2(x) = 1 / (1 + exp(-x))  : bit = 1 with probability S2(x)
  V2(x) = |tanh(x)|          : bit flips with probability V2(x)
PSO feeds its velocity to the transfer function; GWO, WOA and SCA feed their positions.
Real-valued vectors are clipped to [-6, 6]. Population size 10 for population methods.
"""
import numpy as np

P = 10
CLIP = 6.0


def _w(d):
    return np.int64(1) << np.arange(d, dtype=np.int64)


def _masks(B, w):
    return B.astype(np.int64) @ w


def _s2(x):
    return 1.0 / (1.0 + np.exp(-x))


def _binarize(X, B, rng, tf):
    if tf == "S":
        return rng.random(X.shape) < _s2(X)
    return B ^ (rng.random(X.shape) < np.abs(np.tanh(X)))


def _iters(oracle):
    return max(1, oracle.budget // P - 1)


def _init(d, rng, tf):
    X = rng.uniform(-1.0, 1.0, (P, d))
    B = rng.random((P, d)) < (0.5 if tf == "V" else _s2(X))
    return X, B


def pso(oracle, d, rng, tf):
    """Binary PSO (Kennedy and Eberhart 1997); w 0.9 to 0.4, c1 = c2 = 2, |v| <= 6."""
    w, T = _w(d), _iters(oracle)
    B = rng.random((P, d)) < 0.5
    V = np.zeros((P, d))
    f = oracle(_masks(B, w))
    pb, pf = B.copy(), f.copy()
    t = 0
    while not oracle.exhausted:
        t += 1
        g = pb[np.argmin(pf)].astype(float)
        inertia = 0.9 - 0.5 * min(t, T) / T
        Bf = B.astype(float)
        V = (inertia * V + 2.0 * rng.random((P, d)) * (pb.astype(float) - Bf)
             + 2.0 * rng.random((P, d)) * (g - Bf))
        V = np.clip(V, -CLIP, CLIP)
        B = _binarize(V, B, rng, tf)
        f = oracle(_masks(B, w))
        better = f < pf
        pb[better], pf[better] = B[better], f[better]


def gwo(oracle, d, rng, tf):
    """Grey wolf optimizer (Mirjalili et al. 2014); a from 2 to 0; leaders as in the reference code."""
    w, T = _w(d), _iters(oracle)
    X, B = _init(d, rng, tf)
    lead = np.zeros((3, d))
    lead_f = np.full(3, np.inf)

    def update_leaders(Xc, fc):
        for i in range(P):
            if fc[i] < lead_f[0]:
                lead_f[0], lead[0] = fc[i], Xc[i]
            if fc[i] > lead_f[0] and fc[i] < lead_f[1]:
                lead_f[1], lead[1] = fc[i], Xc[i]
            if fc[i] > lead_f[0] and fc[i] > lead_f[1] and fc[i] < lead_f[2]:
                lead_f[2], lead[2] = fc[i], Xc[i]

    update_leaders(X, oracle(_masks(B, w)))
    t = 0
    while not oracle.exhausted:
        t += 1
        a = 2.0 - 2.0 * min(t, T) / T
        Xn = np.zeros((P, d))
        for k in range(3):
            A = 2.0 * a * rng.random((P, d)) - a
            C = 2.0 * rng.random((P, d))
            Xn += lead[k] - A * np.abs(C * lead[k] - X)
        X = np.clip(Xn / 3.0, -CLIP, CLIP)
        B = _binarize(X, B, rng, tf)
        update_leaders(X, oracle(_masks(B, w)))


def woa(oracle, d, rng, tf):
    """Whale optimization algorithm (Mirjalili and Lewis 2016), following the reference code."""
    w, T = _w(d), _iters(oracle)
    X, B = _init(d, rng, tf)
    f = oracle(_masks(B, w))
    i = int(np.argmin(f))
    lead, lead_f = X[i].copy(), f[i]
    t = 0
    cols = np.arange(d)
    while not oracle.exhausted:
        t += 1
        a = 2.0 - 2.0 * min(t, T) / T
        a2 = -1.0 - min(t, T) / T
        Xn = np.empty((P, d))
        for i in range(P):
            A = 2.0 * a * rng.random() - a
            C = 2.0 * rng.random()
            l = (a2 - 1.0) * rng.random() + 1.0
            if rng.random() < 0.5:
                if abs(A) >= 1.0:
                    Xr = X[rng.integers(P, size=d), cols]
                    Xn[i] = Xr - A * np.abs(C * Xr - X[i])
                else:
                    Xn[i] = lead - A * np.abs(C * lead - X[i])
            else:
                Xn[i] = np.abs(lead - X[i]) * np.exp(l) * np.cos(2.0 * np.pi * l) + lead
        X = np.clip(Xn, -CLIP, CLIP)
        B = _binarize(X, B, rng, tf)
        f = oracle(_masks(B, w))
        i = int(np.argmin(f))
        if f[i] < lead_f:
            lead, lead_f = X[i].copy(), f[i]


def sca(oracle, d, rng, tf):
    """Sine cosine algorithm (Mirjalili 2016); r1 from 2 to 0."""
    w, T = _w(d), _iters(oracle)
    X, B = _init(d, rng, tf)
    f = oracle(_masks(B, w))
    i = int(np.argmin(f))
    dest, dest_f = X[i].copy(), f[i]
    t = 0
    while not oracle.exhausted:
        t += 1
        r1 = 2.0 - 2.0 * min(t, T) / T
        r2 = 2.0 * np.pi * rng.random((P, d))
        r3 = 2.0 * rng.random((P, d))
        r4 = rng.random((P, d))
        step = r1 * np.where(r4 < 0.5, np.sin(r2), np.cos(r2)) * np.abs(r3 * dest - X)
        X = np.clip(X + step, -CLIP, CLIP)
        B = _binarize(X, B, rng, tf)
        f = oracle(_masks(B, w))
        i = int(np.argmin(f))
        if f[i] < dest_f:
            dest, dest_f = X[i].copy(), f[i]


def ga(oracle, d, rng):
    """Generational GA: binary tournament, uniform crossover (0.9), bit-flip 1/d, one elite."""
    w = _w(d)
    B = rng.random((P, d)) < 0.5
    f = oracle(_masks(B, w))
    while not oracle.exhausted:
        t1 = rng.integers(P, size=(P, 2))
        t2 = rng.integers(P, size=(P, 2))
        p1 = np.where(f[t1[:, 0]] <= f[t1[:, 1]], t1[:, 0], t1[:, 1])
        p2 = np.where(f[t2[:, 0]] <= f[t2[:, 1]], t2[:, 0], t2[:, 1])
        mix = (rng.random(P) < 0.9)[:, None] & (rng.random((P, d)) < 0.5)
        child = np.where(mix, B[p2], B[p1]) ^ (rng.random((P, d)) < 1.0 / d)
        fc = oracle(_masks(child, w))
        e, worst = int(np.argmin(f)), int(np.argmax(fc))
        if f[e] < fc[worst]:
            child[worst], fc[worst] = B[e], f[e]
        B, f = child, fc


def random_search(oracle, d, rng):
    """Uniform random subsets (each bit 1 with probability 0.5)."""
    w = _w(d)
    while not oracle.exhausted:
        n = min(1000, oracle.budget - oracle.calls)
        oracle(_masks(rng.random((n, d)) < 0.5, w))


def one_plus_one_ea(oracle, d, rng):
    """(1+1) EA: bit-flip rate 1/d (at least one bit flipped), accepts equal fitness."""
    w = _w(d)
    x = rng.random(d) < 0.5
    fx = oracle(_masks(x, w))[0]
    while not oracle.exhausted:
        flip = rng.random(d) < 1.0 / d
        if int(flip.sum()) == 0:
            flip[rng.integers(d)] = True
        y = x ^ flip
        fy = oracle(_masks(y, w))[0]
        if fy <= fx:
            x, fx = y, fy


def sfs(oracle, d, rng):
    """Sequential forward selection to the full set; ties broken at random; best subset on the path."""
    w = _w(d)
    selected = np.zeros(d, dtype=bool)
    while int(selected.sum()) < d and not oracle.exhausted:
        cand = np.flatnonzero(~selected)
        f = oracle(_masks(selected, w) + (np.int64(1) << cand))
        ok = np.isfinite(f)
        if int(ok.sum()) == 0:
            break
        best = f[ok].min()
        selected[rng.choice(cand[ok & (f == best)])] = True


def all_features(oracle, d, rng):
    """No search: the full feature set."""
    oracle(np.array([(1 << d) - 1], dtype=np.int64))


METHODS = {
    "GA": lambda o, d, r: ga(o, d, r),
    "PSO-S": lambda o, d, r: pso(o, d, r, "S"),
    "PSO-V": lambda o, d, r: pso(o, d, r, "V"),
    "GWO-S": lambda o, d, r: gwo(o, d, r, "S"),
    "GWO-V": lambda o, d, r: gwo(o, d, r, "V"),
    "WOA-S": lambda o, d, r: woa(o, d, r, "S"),
    "WOA-V": lambda o, d, r: woa(o, d, r, "V"),
    "SCA-S": lambda o, d, r: sca(o, d, r, "S"),
    "SCA-V": lambda o, d, r: sca(o, d, r, "V"),
    "RS": lambda o, d, r: random_search(o, d, r),
    "EA": lambda o, d, r: one_plus_one_ea(o, d, r),
    "SFS": lambda o, d, r: sfs(o, d, r),
    "ALL": lambda o, d, r: all_features(o, d, r),
}
METAHEURISTICS = ["GA", "PSO-S", "PSO-V", "GWO-S", "GWO-V", "WOA-S", "WOA-V", "SCA-S", "SCA-V"]
CONTROLS = ["RS", "EA", "SFS"]
SEARCH = METAHEURISTICS + CONTROLS
