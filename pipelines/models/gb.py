"""
M2: gradient-boosted trees for a team's expected runs (design E22).

M2 = M1 + trees (revised after the independent pre-build review, Colin's
call 4 Oct 2026). The mean has the same two parts as M1's linear predictor:

    log(mu) = base + team

  base   offset (log league level) + constant + the SHARED features (park,
         weather, home, umpire...), from the linear negative binomial model
         M1 fitted on the same rows. Unchanged, so the spread fix (phase_c,
         design E4) shrinks the same pieces as for M1.
  team   M1's linear team part PLUS shallow LightGBM trees on the non-SHARED
         features, started from M1's full prediction (init_score). With zero
         trees M2 is exactly M1, so any difference is a curve or an
         interaction M1 misses.

The trees use M1's own likelihood as their objective: the negative binomial
(zero-adjusted for F5) with M1's dispersion and zero adjustment held fixed.
Number of trees: chosen by time-ordered validation inside the training rows
(fit on rows before 1 July of the last training season, choose on the rest,
same likelihood), then refit on all training rows with that count.
Dispersion (and the F5 zero adjustment) is then refitted by maximum
likelihood with the mean held fixed.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import digamma, gammaln

from .nb import MAX_RUNS, NBModel, nb_logpmf

PARAMS = {
    "learning_rate": 0.03, "max_depth": 2, "num_leaves": 4,
    "min_data_in_leaf": 200, "lambda_l2": 10.0, "metric": "None", "feature_fraction": 0.8, "bagging_fraction": 1.0,
    "verbosity": -1, "seed": 20261004, "deterministic": True, "force_row_wise": True, "num_threads": 1,
}
MAX_ROUNDS = 2000
EARLY_STOP = 100
# more runs for the batting team when these are higher (+1) or lower (-1)
MONOTONE = {
    "lineup_woba_f5": 1, "lineup_woba_8": 1, "lineup_platoon": 1,
    "lineup_k_bb_f5": -1, "lineup_k_bb_8": -1,
    "sp_skill": 1, "pen_skill_f5": 1, "pen_skill_8": 1,
    "pitching_composite_f5": 1, "pitching_composite_8": 1,
}


def _nb_terms(eta, y, a0, a1, d, za):
    """Log-likelihood per row, its gradient in eta (negative, for LightGBM)
    and a positive Fisher-type weight, with a0, a1 (and d) held fixed. The
    dispersion depends on the mean, alpha = exp(a0 + a1 * eta), so the
    gradient includes that path, as in nb.py (fixed 4 Oct 2026 after the
    E22 result review found it missing)."""
    eta = np.clip(eta, -5, 5)
    mu = np.exp(eta)
    la_raw = a0 + a1 * eta
    la = np.clip(la_raw, -8, 4)
    m_la = (la_raw > -8) & (la_raw < 4)
    r = np.exp(-la)
    rm = r + mu
    log_r_rm = np.log(r / rm)
    lp = gammaln(y + r) - gammaln(r) - gammaln(y + 1) + r * log_r_rm + y * np.log(mu / rm)
    d_mu = y / mu - (r + y) / rm
    d_r = digamma(y + r) - digamma(r) + log_r_rm + (mu - y) / rm
    if za:
        zero = y == 0
        lp0 = r * log_r_rm
        p0 = np.exp(lp0)
        l1m = np.log(-np.expm1(lp0))
        lo = lp0 - l1m + d
        p0n = 1.0 / (1.0 + np.exp(-lo))
        lp = np.where(zero, -np.logaddexp(0.0, -lo), lp - np.logaddexp(0.0, lo) - l1m)
        c = np.where(zero, (1 - p0n) / (1 - p0), (p0 - p0n) / (1 - p0))
        d_mu = np.where(zero, 0.0, d_mu) + c * (-r / rm)
        d_r = np.where(zero, 0.0, d_r) + c * (log_r_rm + mu / rm)
    g_eta = d_mu * mu + d_r * (-r) * m_la * a1
    return lp, -g_eta, mu * r / rm


class GBModel:
    def __init__(self, features, offset, shared, zero_adj=False):
        self.features = list(features)
        self.offset = offset
        self.shared = set(shared)
        self.zero_adj = zero_adj
        self.team_feats = [f for f in self.features if f not in self.shared]
        self.lin: NBModel | None = None
        self.booster = None
        self.n_rounds = 0
        self.a0 = self.a1 = self.zdelta = 0.0
        self.notes: dict = {}

    def _Xt(self, df):
        return df[self.team_feats].astype(float).to_numpy()

    def _base(self, df):
        return self.lin.eta_parts(df, self.shared)[0]

    def fit(self, df, y):
        y = np.asarray(y, dtype=float)
        self.lin = NBModel(self.features, offset=self.offset, zero_adj=self.zero_adj).fit(df, y)
        L = self.lin
        base, lteam = L.eta_parts(df, self.shared)
        init = base + lteam                                   # M1's full linear predictor
        X = self._Xt(df)
        za = self.zero_adj
        fobj = lambda preds, ds: _nb_terms(preds, ds.get_label(), L.a0, L.a1, L.zdelta, za)[1:]
        feval = lambda preds, ds: ("nb_nll", float(-_nb_terms(preds, ds.get_label(), L.a0, L.a1, L.zdelta, za)[0].mean()), False)
        params = dict(PARAMS, objective=fobj, monotone_constraints=[MONOTONE.get(f, 0) for f in self.team_feats])
        d = pd.to_datetime(df["date"]).to_numpy()
        last = pd.to_datetime(df["date"]).dt.year.max()
        cut = np.datetime64(f"{last}-07-01")
        tr, va = d < cut, d >= cut
        dtr = lgb.Dataset(X[tr], y[tr], init_score=init[tr], feature_name=self.team_feats, free_raw_data=False)
        dva = lgb.Dataset(X[va], y[va], init_score=init[va], reference=dtr, free_raw_data=False)
        b = lgb.train(params, dtr, num_boost_round=MAX_ROUNDS, valid_sets=[dva], feval=feval,
                      callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        start_nll = float(-_nb_terms(init[va], y[va], L.a0, L.a1, L.zdelta, za)[0].mean())
        best_nll = float(b.best_score["valid_0"]["nb_nll"])
        # zero trees allowed: if no number of trees beats M1 on the inner
        # validation rows, M2 is exactly M1
        self.n_rounds = int(b.best_iteration or 0) if best_nll < start_nll else 0
        if self.n_rounds:
            dall = lgb.Dataset(X, y, init_score=init, feature_name=self.team_feats, free_raw_data=False)
            self.booster = lgb.train(params, dall, num_boost_round=self.n_rounds)
        if self.n_rounds:
            self._fit_dispersion(init + self._trees(df), y)
        else:                       # exactly M1
            self.a0, self.a1, self.zdelta = L.a0, L.a1, L.zdelta
        self.notes = {"n_rounds": self.n_rounds, "inner_cut": str(cut), "n_inner_train": int(tr.sum()),
                      "n_inner_valid": int(va.sum()), "inner_valid_nll_m1": start_nll,
                      "inner_valid_nll_best": best_nll}
        return self

    def _fit_dispersion(self, eta, y):
        eta = np.clip(eta, -5, 5)
        mu = np.exp(eta)
        zero = y == 0

        def nll(th):
            a0, a1 = th[0], th[1]
            alpha = np.exp(np.clip(a0 + a1 * eta, -8, 4))
            lp = nb_logpmf(y, mu, alpha)
            if self.zero_adj:
                d = th[2]
                lp0 = nb_logpmf(np.zeros_like(y), mu, alpha)
                l1m = np.log(-np.expm1(lp0))
                lo = lp0 - l1m + d
                lp = np.where(zero, -np.logaddexp(0.0, -lo), lp - np.logaddexp(0.0, lo) - l1m)
            return -lp.sum()

        th0 = np.array([np.log(0.3), 0.0] + ([0.0] if self.zero_adj else []))
        res = minimize(nll, th0, method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 20000})
        res = minimize(nll, res.x, method="BFGS", options={"gtol": 1e-6})
        self.a0, self.a1 = float(res.x[0]), float(res.x[1])
        self.zdelta = float(res.x[2]) if self.zero_adj else 0.0

    def _trees(self, df):
        return self.booster.predict(self._Xt(df), raw_score=True) if self.booster is not None else np.zeros(len(df))

    def _team(self, df):
        return self.lin.eta_parts(df, self.shared)[1] + self._trees(df)

    def _eta(self, df):
        return self._base(df) + self._team(df)

    def predict(self, df):
        eta = np.clip(self._eta(df), -5, 5)
        return np.exp(eta), np.exp(np.clip(self.a0 + self.a1 * eta, -8, 4))

    def eta_parts(self, df, shared):
        assert set(shared) == self.shared, "M2 was fitted with a different SHARED set"
        return self._base(df), self._team(df)

    def pmf_eta(self, eta):
        eta = np.clip(eta, -5, 5)
        mu = np.exp(eta)
        alpha = np.exp(np.clip(self.a0 + self.a1 * eta, -8, 4))
        ks = np.arange(MAX_RUNS + 1)
        p = np.exp(nb_logpmf(ks[None, :], mu[:, None], alpha[:, None]))
        if self.zero_adj:
            p0 = p[:, 0].copy()
            p0n = 1 / (1 + np.exp(-(np.log(p0) - np.log1p(-p0) + self.zdelta)))
            p[:, 0] = p0n
            p[:, 1:] *= ((1 - p0n) / (1 - p0))[:, None]
        return p / p.sum(axis=1, keepdims=True)

    def pmf_grid(self, df):
        return self.pmf_eta(self._eta(df))

    def to_json(self):
        imp = (dict(zip(self.team_feats, map(float, self.booster.feature_importance("gain"))))
               if self.booster is not None else {})
        return {"type": "M2 gradient boosting", "features": self.features, "team_features": self.team_feats,
                "offset": self.offset, "params": PARAMS, "monotone": MONOTONE, "n_rounds": self.n_rounds,
                "a0": self.a0, "a1": self.a1, "zero_adj": self.zero_adj, "zdelta": self.zdelta,
                "gain_importance": imp, "notes": self.notes, "linear_m1": self.lin.to_json(),
                "trees": self.booster.model_to_string() if self.booster is not None else ""}
