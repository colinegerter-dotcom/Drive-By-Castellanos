"""
Distribution calibration (design 8) and gate G2 (design 9.3).

A model hands back, per game, a grid of probabilities for every (home, away)
score. Calibration corrects the SHAPE of two distributions read off that grid:

  total    home + away
  margin   home - away

with one gentle monotone correction each, then puts the grid back together.

The correction, per distribution: each game's cumulative probabilities F(k)
go through a Beta cumulative curve g, so the corrected chance of k is
g(F(k)) - g(F(k-1)). With both Beta parameters at 1, g is the identity (no
change); below 1 the tails get fatter (the model was too confident), above 1
thinner, and unequal parameters lean one tail. Two numbers per distribution,
fitted by maximum likelihood with a ridge pulling them toward 1 ("pulled
toward no change", design 8). Scores the grid gives zero chance (a tied
final) keep zero chance.

The average is not calibration's job (design 8: the model sets the level
through league_env), so after the shape change each game's distribution is
tilted back to its original mean (multiply P(k) by e^(t k) and renormalize,
t solved per game). The fit is done on the tilted result, so the correction
learns shape only.

Putting the grid back together: iterative proportional fitting. The grid is
rescaled along its diagonals until its total and margin distributions match
the corrected ones, so every market comes from one consistent grid. On a
real grid the total and margin always have the same odd/even split (h + a
and h - a are both odd or both even); two separate corrections can disagree
slightly on it, so both targets first move to the average split. A last
tilt of the grid restores each team's expected runs exactly.

Cross-fitting (the gates, design 8): 2023 is scored with a correction fitted
on 2024's predictions and 2024 with one fitted on 2023's. A correction is
kept only if it improves the joint log score in both seasons.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betainc
from scipy.stats import norm

RIDGE = 10.0          # pull toward "no change", on (log a, log b); fixed in advance


# ---------------------------------------------------------------- grid layout
class Layout:
    """Index helpers for a K x K score grid (home runs down, away across)."""

    def __init__(self, K: int):
        h, a = np.meshgrid(np.arange(K), np.arange(K), indexing="ij")
        self.K = K
        self.h, self.a = h.ravel(), a.ravel()
        self.L = 2 * K - 1
        self.s = self.h + self.a                 # total, index = total
        self.d = self.h - self.a + K - 1         # margin, index = margin + K - 1
        self.total_k = np.arange(self.L).astype(float)
        self.margin_k = np.arange(self.L) - (K - 1.0)

    def marginals(self, X):
        """X: (n, K*K). Returns total and margin distributions, each (n, L)."""
        return _group_sum(X, self.s, self.L), _group_sum(X, self.d, self.L)

    def cell(self, h, a):
        h = np.minimum(np.asarray(h, int), self.K - 1)
        a = np.minimum(np.asarray(a, int), self.K - 1)
        return h * self.K + a


_ONEHOT: dict = {}


def _group_sum(X, idx, L):
    """Sum columns of X by group index idx (length X.shape[1])."""
    key = (X.shape[1], L, idx.tobytes())
    if key not in _ONEHOT:
        onehot = np.zeros((X.shape[1], L))
        onehot[np.arange(X.shape[1]), idx] = 1.0
        _ONEHOT[key] = onehot
    return X @ _ONEHOT[key]


# ------------------------------------------------------------ the correction
def _beta_map(P, a, b):
    """Corrected distribution g(F(k)) - g(F(k-1)), g = Beta(a, b) CDF.
    Uses the upper-tail form where F is above one half, for precision."""
    P = np.clip(P, 0.0, None)
    P = P / P.sum(axis=1, keepdims=True)
    F = np.clip(np.cumsum(P, axis=1), 0.0, 1.0)
    Fprev = np.clip(F - P, 0.0, 1.0)
    S = np.clip(np.cumsum(P[:, ::-1], axis=1)[:, ::-1] - P, 0.0, 1.0)   # P(> k)
    Sprev = np.clip(S + P, 0.0, 1.0)                                     # P(>= k)
    lower = betainc(a, b, F) - betainc(a, b, Fprev)
    upper = betainc(b, a, Sprev) - betainc(b, a, S)
    Q = np.where(Fprev < 0.5, lower, upper)
    Q = np.where(P > 0, np.clip(Q, 0.0, None), 0.0)
    return Q / Q.sum(axis=1, keepdims=True)


def _tilt_to_mean(Q, k, m0, iters=12):
    """Exponential tilt of each row of Q so its mean equals m0 (Newton)."""
    kc = k[None, :] - m0[:, None]
    t = np.zeros(len(Q))
    logQ = np.log(np.where(Q > 0, Q, 1.0))
    for _ in range(iters):
        z = np.where(Q > 0, logQ + t[:, None] * kc, -np.inf)
        z -= z.max(axis=1, keepdims=True)
        w = np.exp(z)
        w /= w.sum(axis=1, keepdims=True)
        m = (w * kc).sum(axis=1)
        v = (w * kc * kc).sum(axis=1) - m * m
        t -= m / np.maximum(v, 1e-9)
    z = np.where(Q > 0, logQ + t[:, None] * kc, -np.inf)
    z -= z.max(axis=1, keepdims=True)
    w = np.exp(z)
    return w / w.sum(axis=1, keepdims=True)


@dataclass
class ShapeCorrection:
    kind: str                 # "total" or "margin"
    log_a: float = 0.0
    log_b: float = 0.0
    ridge: float = RIDGE
    n_fit: int = 0

    def apply(self, P, k):
        if self.log_a == 0.0 and self.log_b == 0.0:
            return P / P.sum(axis=1, keepdims=True)
        m0 = (P * k[None, :]).sum(axis=1) / P.sum(axis=1)
        Q = _beta_map(P, np.exp(self.log_a), np.exp(self.log_b))
        return _tilt_to_mean(Q, k, m0)

    def fit(self, P, y_idx, k):
        rows = np.arange(len(P))
        m0 = (P * k[None, :]).sum(axis=1) / P.sum(axis=1)

        def obj(th):
            Q = _tilt_to_mean(_beta_map(P, np.exp(th[0]), np.exp(th[1])), k, m0)
            ll = np.log(np.maximum(Q[rows, y_idx], 1e-300)).sum()
            return -ll + self.ridge * (th[0] ** 2 + th[1] ** 2)

        res = minimize(obj, np.zeros(2), method="Nelder-Mead",
                       options={"xatol": 1e-9, "fatol": 1e-9, "maxiter": 4000})   # tight (design E11)
        self.log_a, self.log_b = float(res.x[0]), float(res.x[1])
        self.n_fit = int(len(P))
        return self

    def to_json(self):
        return {"kind": self.kind, "a": float(np.exp(self.log_a)), "b": float(np.exp(self.log_b)),
                "ridge": self.ridge, "n_fit": self.n_fit}


# ------------------------------------------------------ putting it together
def _parity_align(T, M, lay):
    """Move the total and margin targets to their average odd/even split."""
    ev_t = (np.arange(lay.L) % 2 == 0)
    ev_m = ((np.arange(lay.L) - (lay.K - 1)) % 2 == 0)
    pt, pm = T[:, ev_t].sum(1), M[:, ev_m].sum(1)
    pe = (pt + pm) / 2
    T = T.copy()
    M = M.copy()
    T[:, ev_t] *= (pe / np.maximum(pt, 1e-300))[:, None]
    T[:, ~ev_t] *= ((1 - pe) / np.maximum(1 - pt, 1e-300))[:, None]
    M[:, ev_m] *= (pe / np.maximum(pm, 1e-300))[:, None]
    M[:, ~ev_m] *= ((1 - pe) / np.maximum(1 - pm, 1e-300))[:, None]
    return T, M


def _tilt_grid_means(X, lay, eh, ea, iters=8):
    """Tilt each grid by e^(t1 h + t2 a) so E[home] = eh and E[away] = ea."""
    hc = lay.h[None, :] - eh[:, None]
    ac = lay.a[None, :] - ea[:, None]
    t1 = np.zeros(len(X))
    t2 = np.zeros(len(X))
    logX = np.log(np.where(X > 0, X, 1.0))
    for _ in range(iters):
        z = np.where(X > 0, logX + t1[:, None] * hc + t2[:, None] * ac, -np.inf)
        z -= z.max(axis=1, keepdims=True)
        w = np.exp(z)
        w /= w.sum(axis=1, keepdims=True)
        mh, ma = (w * hc).sum(1), (w * ac).sum(1)
        vhh = (w * hc * hc).sum(1) - mh * mh
        vaa = (w * ac * ac).sum(1) - ma * ma
        vha = (w * hc * ac).sum(1) - mh * ma
        det = np.maximum(vhh * vaa - vha * vha, 1e-12)
        t1 -= (vaa * mh - vha * ma) / det
        t2 -= (-vha * mh + vhh * ma) / det
    z = np.where(X > 0, logX + t1[:, None] * hc + t2[:, None] * ac, -np.inf)
    z -= z.max(axis=1, keepdims=True)
    w = np.exp(z)
    return w / w.sum(axis=1, keepdims=True)


def calibrate(X, lay, total: ShapeCorrection | None, margin: ShapeCorrection | None,
              iters=200, tol=1e-9):
    """X: (n, K*K) grids. Returns calibrated grids, same shape."""
    if total is None and margin is None:
        return X
    X = np.clip(X, 0.0, None)
    X = X / X.sum(axis=1, keepdims=True)
    eh, ea = X @ lay.h.astype(float), X @ lay.a.astype(float)
    T, M = lay.marginals(X)
    if total is not None:
        T = total.apply(T, lay.total_k)
    if margin is not None:
        M = margin.apply(M, lay.margin_k)
    T, M = _parity_align(T, M, lay)
    Y = X.copy()
    for _ in range(iters):
        cur_t = _group_sum(Y, lay.s, lay.L)
        f = np.divide(T, cur_t, out=np.zeros_like(T), where=cur_t > 0)
        Y *= f[:, lay.s]
        cur_m = _group_sum(Y, lay.d, lay.L)
        f = np.divide(M, cur_m, out=np.zeros_like(M), where=cur_m > 0)
        Y *= f[:, lay.d]
        if np.abs(_group_sum(Y, lay.s, lay.L) - T).max() < tol:
            break
    Y /= Y.sum(axis=1, keepdims=True)
    return _tilt_grid_means(Y, lay, eh, ea)


def log_score(X, lay, h, a):
    return np.log(np.maximum(X[np.arange(len(X)), lay.cell(h, a)], 1e-300))


def fit_corrections(X, lay, h, a):
    """Fit total and margin corrections on one season's grids and outcomes."""
    T, M = lay.marginals(X)
    s = np.minimum(np.asarray(h) + np.asarray(a), lay.L - 1)
    d = np.clip(np.asarray(h) - np.asarray(a), -(lay.K - 1), lay.K - 1) + lay.K - 1
    ct = ShapeCorrection("total").fit(T, s.astype(int), lay.total_k)
    cm = ShapeCorrection("margin").fit(M, d.astype(int), lay.margin_k)
    return ct, cm


def cross_fit(seasons: dict, lay):
    """seasons: {label: (X, h, a)} for exactly two seasons. Each season is
    scored with corrections fitted on the other. Returns the decision and the
    calibrated grids under the kept corrections."""
    labels = list(seasons)
    assert len(labels) == 2
    fits = {}
    for i, lab in enumerate(labels):
        other = labels[1 - i]
        Xo, ho, ao = seasons[other]
        fits[lab] = fit_corrections(Xo, lay, ho, ao)   # used to score `lab`
    variants = {"none": (False, False), "total": (True, False), "margin": (False, True), "both": (True, True)}
    scores = {v: {} for v in variants}
    for lab in labels:
        X, h, a = seasons[lab]
        ct, cm = fits[lab]
        for v, (ut, um) in variants.items():
            Y = calibrate(X, lay, ct if ut else None, cm if um else None)
            scores[v][lab] = float(log_score(Y, lay, h, a).mean())
    keep_t = all(scores["total"][l] > scores["none"][l] for l in labels)
    keep_m = all(scores["margin"][l] > scores["none"][l] for l in labels)
    if keep_t and keep_m and not all(scores["both"][l] > scores["none"][l] for l in labels):
        # each helps alone but not together: keep the better single one
        keep_t = sum(scores["total"].values()) >= sum(scores["margin"].values())
        keep_m = not keep_t
    kept = {(False, False): "none", (True, False): "total", (False, True): "margin", (True, True): "both"}[(keep_t, keep_m)]
    out = {}
    for lab in labels:
        X, h, a = seasons[lab]
        ct, cm = fits[lab]
        out[lab] = calibrate(X, lay, ct if keep_t else None, cm if keep_m else None)
    report = {"kept": kept, "log_score": scores,
              "fitted_on_other_season": {lab: {"total": fits[lab][0].to_json(), "margin": fits[lab][1].to_json()}
                                         for lab in labels}}
    return out, report


# ------------------------------------------------------------------ gate G2
EVENTS = [  # (segment, name, probability from grid, outcome from score)
    ("full", "home_win", lambda lay: lay.h > lay.a),
    ("f5", "home_win", lambda lay: lay.h > lay.a),
    ("full", "over_7.5", lambda lay: lay.s >= 8),
    ("full", "over_8.5", lambda lay: lay.s >= 9),
    ("full", "over_9.5", lambda lay: lay.s >= 10),
    ("f5", "over_3.5", lambda lay: lay.s >= 4),
    ("f5", "over_4.5", lambda lay: lay.s >= 5),
    ("f5", "over_5.5", lambda lay: lay.s >= 6),
]
SLOPE_RANGE = (0.8, 1.2)
FAMILY_ALPHA = 0.20       # a perfectly calibrated model passes the whole gate >= 80%
N_RANDOM_CHECKS = 3 * len(EVENTS)   # slope interval, intercept interval, ECE per event
CHECK_ALPHA = FAMILY_ALPHA / N_RANDOM_CHECKS
Z_WIDE = float(norm.ppf(1 - CHECK_ALPHA / 2))
ECE_BINS = 10


def _logistic_batch(x, Y, iters=25):
    """Logistic fit of each column of Y on [1, x]. Returns intercept, slope and
    their standard errors, each of length Y.shape[1]."""
    Y = Y.astype(float)
    S = Y.shape[1]
    c = np.zeros(S)
    s = np.ones(S)
    for _ in range(iters):
        eta = c[None, :] + s[None, :] * x[:, None]
        p = 1 / (1 + np.exp(-eta))
        w = p * (1 - p)
        r = Y - p
        g0, g1 = r.sum(0), (r * x[:, None]).sum(0)
        h00, h01, h11 = w.sum(0), (w * x[:, None]).sum(0), (w * (x * x)[:, None]).sum(0)
        det = h00 * h11 - h01 * h01
        dc = (h11 * g0 - h01 * g1) / det
        ds = (-h01 * g0 + h00 * g1) / det
        c += dc
        s += ds
        if max(np.abs(dc).max(), np.abs(ds).max()) < 1e-10:
            break
    eta = c[None, :] + s[None, :] * x[:, None]
    p = 1 / (1 + np.exp(-eta))
    w = p * (1 - p)
    h00, h01, h11 = w.sum(0), (w * x[:, None]).sum(0), (w * (x * x)[:, None]).sum(0)
    det = h00 * h11 - h01 * h01
    return c, s, np.sqrt(h11 / det), np.sqrt(h00 / det)


def _ece(p, Y):
    """Expected calibration error, 10 equal-count bins of p; Y (n, S)."""
    order = np.argsort(p, kind="stable")
    b = np.empty(len(p), int)
    b[order] = np.arange(len(p)) * ECE_BINS // len(p)
    B = np.zeros((len(p), ECE_BINS))
    B[np.arange(len(p)), b] = 1.0
    cnt = B.sum(0)
    pbar = (B.T @ p) / cnt
    ybar = (B.T @ Y.astype(float)) / cnt[:, None]
    return ((cnt / len(p))[:, None] * np.abs(pbar[:, None] - ybar)).sum(0)


def reliability(p, y, bins=ECE_BINS):
    order = np.argsort(p, kind="stable")
    out = []
    for chunk in np.array_split(order, bins):
        out.append({"n": int(len(chunk)), "pred": float(p[chunk].mean()), "actual": float(y[chunk].mean())})
    return out


def _sample_scores(X, lay, S, rng):
    """S draws of (home, away) per game from its grid."""
    cdf = np.cumsum(X, axis=1)
    cdf /= cdf[:, -1:]
    u = rng.random((len(X), S))
    idx = np.empty((len(X), S), dtype=np.int32)
    for i in range(len(X)):
        idx[i] = np.searchsorted(cdf[i], u[i], side="right")
    idx = np.minimum(idx, X.shape[1] - 1)
    return lay.h[idx], lay.a[idx]


def gate_g2(grids: dict, lays: dict, obs: dict, n_sim=2000, seed=20260926):
    """grids: {"full": X, "f5": X} pooled 2023+2024 (calibrated), lays: their
    layouts, obs: {"full": (h, a), "f5": (h, a)}. Checks each event on the
    real outcomes, and on n_sim simulated seasons in which the model is
    perfectly calibrated by construction (scores drawn from its own grids).
    The simulated seasons give the error thresholds and how often a perfect
    model passes each event and the whole gate at this sample size."""
    rng = np.random.default_rng(seed)
    sims = {seg: _sample_scores(grids[seg], lays[seg], n_sim, rng) for seg in grids}
    res = {}
    sim_pass = []
    for seg, name, cells in EVENTS:
        lay, X = lays[seg], grids[seg]
        m = cells(lay)
        p = np.clip(X[:, m].sum(1), 1e-6, 1 - 1e-6)
        x = np.log(p / (1 - p))
        h, a = obs[seg]
        y = cells_outcome(name, np.asarray(h), np.asarray(a))
        hs, as_ = sims[seg]
        ys = cells_outcome(name, hs, as_)
        c, s, se_c, se_s = _logistic_batch(x, y[:, None])
        e = _ece(p, y[:, None])[0]
        cs, ss, se_cs, se_ss = _logistic_batch(x, ys)
        es = _ece(p, ys)
        e_thr = float(np.quantile(es, 1 - CHECK_ALPHA))

        def checks(c, s, se_c, se_s, e):
            in_range = (s >= SLOPE_RANGE[0]) & (s <= SLOPE_RANGE[1])
            slope_ok = np.abs(s - 1) <= Z_WIDE * se_s
            icpt_ok = np.abs(c) <= Z_WIDE * se_c
            ece_ok = e <= e_thr
            return in_range, slope_ok, icpt_ok, ece_ok

        ir, so, io, eo = checks(c, s, se_c, se_s, np.array([e]))
        irs, sos, ios, eos = checks(cs, ss, se_cs, se_ss, es)
        passes_sim = irs & sos & ios & eos
        sim_pass.append(passes_sim)
        rate = float(passes_sim.mean())
        passed = bool(ir[0] and so[0] and io[0] and eo[0])
        res[f"{seg}/{name}"] = {
            "n": int(len(p)), "mean_pred": float(p.mean()), "actual_rate": float(y.mean()),
            "slope": float(s[0]), "slope_lo": float(s[0] - Z_WIDE * se_s[0]), "slope_hi": float(s[0] + Z_WIDE * se_s[0]),
            "intercept": float(c[0]), "intercept_lo": float(c[0] - Z_WIDE * se_c[0]), "intercept_hi": float(c[0] + Z_WIDE * se_c[0]),
            "ece": float(e), "ece_threshold": e_thr,
            "checks": {"slope_in_0.8_1.2": bool(ir[0]), "slope_interval_covers_1": bool(so[0]),
                       "intercept_interval_covers_0": bool(io[0]), "ece_within_chance": bool(eo[0])},
            "perfect_model_pass_rate": rate,
            "verdict": "pass" if passed else ("inconclusive" if rate < 0.8 else "fail"),
            "reliability": reliability(p, y),
        }
    whole = np.all(np.vstack(sim_pass), axis=0)
    blocking = [k for k, v in res.items() if v["verdict"] == "fail"]
    return {"events": res, "z_widened": Z_WIDE, "check_alpha": CHECK_ALPHA,
            "perfect_model_whole_gate_pass_rate": float(whole.mean()),
            "n_sim": n_sim, "passes": not blocking, "failing_events": blocking}


def cells_outcome(name, h, a):
    if name == "home_win":
        return h > a
    line = float(name.split("_")[1])
    return (h + a) > line


# ------------------------------------------------------------ gate G2, v2
# Repaired 26 Sep 2026 (Colin approved), after the first read showed the v1
# gate would fail a perfectly calibrated model 25-33% of the time (target:
# at most 20%). Written down and checked on simulated seasons before being
# applied to any real outcome. What changed and why:
#   1. Totals: the three lines per segment are one event (stacked, three
#      rows per game), not three nearly identical events
#   2. Full-game home -1.5 (run line) added as an event: run line is one of
#      the markets the model is meant for and v1 never tested it
#   3. Standard errors are clustered by 7-day block (the same blocks as the
#      bootstrap), so shared weeks of weather or scoring, and the three rows
#      per game, aren't counted as independent evidence
#   4. The 0.8-1.2 slope window is judged together with the interval: the
#      slope check fails only when the interval excludes 1 AND the estimate
#      is outside 0.8-1.2 (a miss that is both real and material). v1 failed
#      on the estimate alone, which a perfect model does often when the
#      estimate is noisy
#   5. Precision: an event whose 95% slope interval is wider than +/-0.2
#      can't tell a slope of 0.8 from 1, so if it doesn't fail it reads
#      "inconclusive" (reported, neither passes nor blocks)
#   6. Widening: Bonferroni over every check of every event, family error
#      20%, then verified on simulated seasons in which the model is right
#      by construction; if a perfect model passes the whole gate less than
#      80% of the time, the widening grows until it does
# Unchanged: intercept interval must cover 0; calibration error within what
# a perfect model shows by chance; pooled 2023+2024; F5 tie is a reliability
# curve only.
EVENTS_V2 = [  # (segment, name, cells of the grid counted as "yes", lines stacked)
    ("full", "home_win", [lambda lay: lay.h > lay.a]),
    ("f5", "home_win", [lambda lay: lay.h > lay.a]),
    ("full", "home_m1.5", [lambda lay: (lay.h - lay.a) >= 2]),
    ("full", "totals", [lambda lay, t=t: lay.s > t for t in (7.5, 8.5, 9.5)]),
    ("f5", "totals", [lambda lay, t=t: lay.s > t for t in (3.5, 4.5, 5.5)]),
]
V2_FAMILY_ALPHA = 0.20
V2_CHECKS = 3 * len(EVENTS_V2)
V2_PRECISION = 0.2


def _outcome_v2(name, h, a, k):
    if name == "home_win":
        return h > a
    if name == "home_m1.5":
        return (h - a) >= 2
    return (h + a) > k


def _logistic_cluster(x, Y, cl, n_cl, offset=None, iters=25):
    """Logistic fit of each column of Y on [1, x] (plus a fixed offset);
    cluster-robust standard errors (clusters cl in 0..n_cl-1, small-sample
    factor C/(C-1))."""
    Y = Y.astype(float)
    S = Y.shape[1]
    c = np.zeros(S)
    s = np.ones(S)
    x2 = x[:, None]
    o2 = (np.zeros(len(x)) if offset is None else offset)[:, None]
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(o2 + c[None, :] + s[None, :] * x2)))
        w = p * (1 - p)
        r = Y - p
        g0, g1 = r.sum(0), (r * x2).sum(0)
        h00, h01, h11 = w.sum(0), (w * x2).sum(0), (w * x2 * x2).sum(0)
        det = h00 * h11 - h01 * h01
        dc, ds = (h11 * g0 - h01 * g1) / det, (-h01 * g0 + h00 * g1) / det
        c += dc
        s += ds
        if max(np.abs(dc).max(), np.abs(ds).max()) < 1e-10:
            break
    p = 1 / (1 + np.exp(-(o2 + c[None, :] + s[None, :] * x2)))
    w = p * (1 - p)
    r = Y - p
    h00, h01, h11 = w.sum(0), (w * x2).sum(0), (w * x2 * x2).sum(0)
    det = h00 * h11 - h01 * h01
    i00, i01, i11 = h11 / det, -h01 / det, h00 / det            # inverse Hessian
    B = np.zeros((len(x), n_cl))
    B[np.arange(len(x)), cl] = 1.0
    u0, u1 = B.T @ r, B.T @ (r * x2)                             # cluster scores (C, S)
    f = n_cl / (n_cl - 1)
    m00, m01, m11 = f * (u0 * u0).sum(0), f * (u0 * u1).sum(0), f * (u1 * u1).sum(0)
    # V = Hinv M Hinv
    v00 = i00 * (i00 * m00 + i01 * m01) + i01 * (i00 * m01 + i01 * m11)
    v11 = i01 * (i01 * m00 + i11 * m01) + i11 * (i01 * m01 + i11 * m11)
    return c, s, np.sqrt(v00), np.sqrt(v11)


def _block_ids(dates):
    """7-day blocks counted from each season's first date."""
    d = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    yr = d.dt.year
    start = d.groupby(yr).transform("min")
    key = yr.astype(str) + "-" + ((d - start).dt.days // 7).astype(str)
    codes, uniq = pd.factorize(key)
    return codes.astype(int), len(uniq)


def _event_rows(seg, name, cells, lay, X, h, a, cl):
    """Stacked rows for one event: model probability, outcome, cluster."""
    ps, ys, cs, gs = [], [], [], []
    lines = {"full": (7.5, 8.5, 9.5), "f5": (3.5, 4.5, 5.5)}[seg] if name == "totals" else [None]
    for g, (fn, k) in enumerate(zip(cells, lines)):
        m = fn(lay)
        ps.append(np.clip(X[:, m].sum(1), 1e-6, 1 - 1e-6))
        ys.append(_outcome_v2(name, h, a, k))
        cs.append(cl)
        gs.append(np.full(len(X), g))
    return np.concatenate(ps), np.concatenate(ys, axis=0), np.concatenate(cs), np.concatenate(gs)


def _within_line(x, g):
    """Stacked total lines: the slope is measured across games WITHIN each
    line (log-odds minus that line's average), with each line's average
    log-odds as a fixed offset. A perfect model still has slope 1 and
    intercept 0, but the large, easy differences between lines (over 7.5 vs
    over 9.5 in the same game) no longer drive the slope."""
    xbar = np.array([x[g == k].mean() for k in range(g.max() + 1)])[g]
    return x - xbar, xbar


def gate_g2_v2(grids, lays, obs, dates, n_sim=2000, seed=20260926, within_line=True):
    """grids, obs (home, away) and dates per segment ("full", "f5"), pooled
    2023+2024. Returns per-event results, the widening used and how often a
    perfectly calibrated model (scores drawn from its own grids) passes."""
    rng = np.random.default_rng(seed)
    sims = {seg: _sample_scores(grids[seg], lays[seg], n_sim, rng) for seg in grids}
    clus = {seg: _block_ids(dates[seg]) for seg in grids}
    per = []
    for seg, name, cells in EVENTS_V2:
        lay, X = lays[seg], grids[seg]
        cl, ncl = clus[seg]
        h, a = (np.asarray(v) for v in obs[seg])
        p, y, c, g = _event_rows(seg, name, cells, lay, X, h, a, cl)
        hs, as_ = sims[seg]
        _, ys, _, _ = _event_rows(seg, name, cells, lay, X, hs, as_, cl)
        x = np.log(p / (1 - p))
        off = None
        if within_line and name == "totals":
            x, off = _within_line(x, g)
        real = _logistic_cluster(x, y[:, None], c, ncl, off)
        sim = _logistic_cluster(x, ys, c, ncl, off)
        per.append({"seg": seg, "name": name, "p": p, "y": y, "real": real, "sim": sim,
                    "ece": _ece(p, y[:, None])[0], "ece_sim": _ece(p, ys)})

    def evaluate(z):
        alpha = 2 * (1 - norm.cdf(z))
        fails_sim = []
        for e in per:
            thr = float(np.quantile(e["ece_sim"], 1 - alpha))
            c, s, sc, ss = e["sim"]
            f = (((np.abs(s - 1) > z * ss) & ((s < SLOPE_RANGE[0]) | (s > SLOPE_RANGE[1])))
                 | (np.abs(c) > z * sc) | (e["ece_sim"] > thr))
            fails_sim.append(f)
            e["thr"] = thr
        fails_sim = np.vstack(fails_sim)
        return float((~fails_sim.any(0)).mean()), fails_sim

    z = float(norm.ppf(1 - (V2_FAMILY_ALPHA / V2_CHECKS) / 2))
    rate, fails_sim = evaluate(z)
    while rate < 1 - V2_FAMILY_ALPHA and z < 4.0:
        z = round(z + 0.05, 2)
        rate, fails_sim = evaluate(z)
    res = {}
    for e, fs in zip(per, fails_sim):
        c, s, sc, ss = (float(v[0]) for v in e["real"])
        slope_fail = abs(s - 1) > z * ss and not (SLOPE_RANGE[0] <= s <= SLOPE_RANGE[1])
        icpt_fail = abs(c) > z * sc
        ece_fail = e["ece"] > e["thr"]
        precise = 1.96 * ss <= V2_PRECISION
        verdict = "fail" if (slope_fail or icpt_fail or ece_fail) else ("pass" if precise else "inconclusive")
        res[f"{e['seg']}/{e['name']}"] = {
            "rows": int(len(e["p"])), "mean_pred": float(e["p"].mean()), "actual_rate": float(e["y"].mean()),
            "slope": s, "slope_lo": s - z * ss, "slope_hi": s + z * ss, "slope_se_clustered": ss,
            "intercept": c, "intercept_lo": c - z * sc, "intercept_hi": c + z * sc,
            "ece": float(e["ece"]), "ece_threshold": e["thr"],
            "checks": {"slope_real_and_material_miss": bool(slope_fail), "intercept_excludes_0": bool(icpt_fail),
                       "ece_above_chance": bool(ece_fail), "precise_enough": bool(precise)},
            "perfect_model_no_fail_rate": float(1 - fs.mean()),
            "verdict": verdict, "reliability": reliability(e["p"], e["y"].astype(float))}
    return {"version": 2, "events": res, "z_widened": z, "family_alpha": V2_FAMILY_ALPHA,
            "perfect_model_whole_gate_pass_rate": rate, "n_sim": n_sim,
            "passes": not any(v["verdict"] == "fail" for v in res.values()),
            "failing_events": [k for k, v in res.items() if v["verdict"] == "fail"],
            "inconclusive_events": [k for k, v in res.items() if v["verdict"] == "inconclusive"]}
