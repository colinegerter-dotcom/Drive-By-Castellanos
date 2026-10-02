"""
Negative binomial regression with a log link and mean-dependent dispersion.

Why negative binomial: runs are counts, and they spread out more than a
Poisson allows (variance about 2.2x the mean, design 2.1). The negative
binomial has one extra "dispersion" parameter for that extra spread.

Why mean-dependent dispersion: the data shows the dispersion grows with the
expected runs (design 6.2), so dispersion alpha = exp(a0 + a1 * log(mu)).

    y ~ NegBin(mean mu, variance mu + alpha * mu^2)
    log(mu)   = offset + b0 + b . x

The offset is the log of the league run environment for the segment, with
its coefficient fixed at 1. The model then explains a team-game's runs
RELATIVE to the league level at the time, and the level itself tracks the
league as it moves (design 8: "the average level is the model's job, through
league_env"). Learning that coefficient instead fails: within one or two
training seasons the league level barely varies, so the fit can't learn it,
and the 2023 rule changes then leave every model predicting 2022 scoring.
    log(alpha) = a0 + a1 * log(mu)

Zero adjustment (model M1z, design 6.1): F5 has about 6.5% more scoreless
outcomes than a negative binomial expects (design 2.1). With zero_adj on,
one more parameter d moves the chance of zero on the log-odds scale,
logit P'(0) = logit P(0) + d, and every other count is rescaled by the same
factor so the probabilities still add to 1. d is fitted jointly with the rest.

Fitted by maximum likelihood (scipy L-BFGS with the exact gradient, tight
tolerances and a convergence check, design E11) with a small ridge penalty on the
feature coefficients (features are standardized first, so one penalty fits
all). Coefficients are plain numbers, saved as JSON: no pickles (design 10).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import digamma, gammaln

MAX_RUNS = 25  # score grids run 0..25 (design 4)


def nb_logpmf(y, mu, alpha):
    """log P(Y = y) for NB with mean mu and dispersion alpha (vectorized)."""
    r = 1.0 / alpha
    return (gammaln(y + r) - gammaln(r) - gammaln(y + 1)
            + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu)))


def _zero_adjust_logpmf(lp, zero, lp0, d):
    """log P' for the zero-adjusted distribution (see module notes)."""
    l1m = np.log(-np.expm1(lp0))                   # log(1 - P(0))
    lo = lp0 - l1m + d                             # logit P'(0)
    lp0n = -np.logaddexp(0.0, -lo)                 # log P'(0)
    l1mn = -np.logaddexp(0.0, lo)                  # log(1 - P'(0))
    return np.where(zero, lp0n, lp + l1mn - l1m)


@dataclass
class NBModel:
    features: list[str]
    offset: str | None = None
    ridge: float = 1.0
    mean_scaled_dispersion: bool = True
    zero_adj: bool = False
    # fitted
    center: dict = field(default_factory=dict)
    scale: dict = field(default_factory=dict)
    fill: dict = field(default_factory=dict)
    coef: dict = field(default_factory=dict)
    a0: float = 0.0
    a1: float = 0.0
    zdelta: float = 0.0
    n_train: int = 0
    loglik: float = 0.0
    fit_max_grad: float = 0.0

    def _X(self, df):
        cols = []
        for f in self.features:
            v = df[f].astype(float).to_numpy()
            v = np.where(np.isnan(v), self.fill[f], v)
            cols.append((v - self.center[f]) / self.scale[f])
        return np.column_stack([np.ones(len(df))] + cols)

    def _off(self, df):
        return df[self.offset].astype(float).to_numpy() if self.offset else np.zeros(len(df))

    def fit(self, df, y):
        for f in self.features:
            v = df[f].astype(float)
            m = float(v.mean())
            self.fill[f] = m if np.isfinite(m) else 0.0   # a feature missing on every training row (2021 velocity change) carries no information
            vv = v.fillna(self.fill[f])
            self.center[f] = float(vv.mean())
            self.scale[f] = float(vv.std()) or 1.0
        X = self._X(df)
        off = self._off(df)
        y = np.asarray(y, dtype=float)
        k = X.shape[1]
        pen = np.r_[0.0, np.full(k - 1, self.ridge)]

        zero = y == 0
        msd = self.mean_scaled_dispersion
        za = self.zero_adj

        def nll_grad(theta):
            """Penalized negative log-likelihood and its exact gradient
            (design E11: the earlier finite-difference gradient let the fit
            stop at slightly different points for inputs that differ only by
            rounding)."""
            b, a0 = theta[:k], theta[k]
            a1 = theta[k + 1] if msd else 0.0
            eta_raw = off + X @ b
            eta = np.clip(eta_raw, -5, 5)
            m_eta = (eta_raw > -5) & (eta_raw < 5)
            mu = np.exp(eta)
            la_raw = a0 + a1 * eta
            la = np.clip(la_raw, -8, 4)
            m_la = (la_raw > -8) & (la_raw < 4)
            r = np.exp(-la)
            rm = r + mu
            log_r_rm = np.log(r / rm)
            lp = gammaln(y + r) - gammaln(r) - gammaln(y + 1) + r * log_r_rm + y * np.log(mu / rm)
            d_mu = y / mu - (r + y) / rm                                  # d lp / d mu
            d_r = digamma(y + r) - digamma(r) + log_r_rm + (mu - y) / rm  # d lp / d r
            if za:
                d = theta[k + 2]
                lp0 = r * log_r_rm
                p0 = np.exp(lp0)
                l1m = np.log(-np.expm1(lp0))
                lo = lp0 - l1m + d
                p0n = 1.0 / (1.0 + np.exp(-lo))
                lp_out = np.where(zero, -np.logaddexp(0.0, -lo), lp - np.logaddexp(0.0, lo) - l1m)
                c = np.where(zero, (1 - p0n) / (1 - p0), (p0 - p0n) / (1 - p0))   # d lp' / d lp0
                d0_mu = -r / rm
                d0_r = log_r_rm + mu / rm
                g_mu = np.where(zero, 0.0, d_mu) + c * d0_mu
                g_r = np.where(zero, 0.0, d_r) + c * d0_r
                g_d = np.where(zero, 1 - p0n, -p0n).sum()
            else:
                lp_out, g_mu, g_r = lp, d_mu, d_r
            g_la = g_r * (-r) * m_la                     # r = exp(-la)
            g_eta = (g_mu * mu + g_la * a1) * m_eta      # through mu and through la = a0 + a1 * eta
            grad = np.empty_like(theta)
            grad[:k] = -(X.T @ g_eta) + pen * b
            grad[k] = -g_la.sum()
            if msd:
                grad[k + 1] = -(g_la * eta).sum()
            else:
                grad[k + 1] = 0.0
            if za:
                grad[k + 2] = -g_d
            return -lp_out.sum() + 0.5 * np.sum(pen * b * b), grad

        b0 = np.r_[np.log(y.mean()) - off.mean(), np.zeros(k - 1)]
        theta0 = np.r_[b0, np.log(0.3), 0.0] if not za else np.r_[b0, np.log(0.3), 0.0, 0.0]
        res = minimize(nll_grad, theta0, jac=True, method="L-BFGS-B",
                       options={"maxiter": 20000, "maxcor": 30, "ftol": 1e-15, "gtol": 1e-9})
        gmax = float(np.max(np.abs(nll_grad(res.x)[1])))
        # Converged = gradient at the floating-point floor for a sum over n rows.
        # Measured on 2021-2023 training fits (E11): stops by relative change
        # leave max |gradient| 3e-8 to 2.4e-4 (n 4,600-9,700), and a further
        # BFGS polish moves no parameter by more than 1.5e-8, so these are at
        # the optimum. Allowed: 1e-7 per training row
        if gmax > 1e-7 * max(1.0, len(y)):
            raise RuntimeError(f"NB fit did not converge: {res.message}, max gradient {gmax:.2e}")
        self.fit_max_grad = gmax
        b = res.x[:k]
        self.coef = {"const": float(b[0]), **{f: float(c) for f, c in zip(self.features, b[1:])}}
        self.a0, self.a1 = float(res.x[k]), float(res.x[k + 1]) if self.mean_scaled_dispersion else 0.0
        self.zdelta = float(res.x[k + 2]) if self.zero_adj else 0.0
        self.n_train = len(y)
        self.loglik = float(-res.fun)
        return self

    def predict(self, df):
        """Returns (mu, alpha) arrays."""
        X = self._X(df)
        b = np.array([self.coef["const"]] + [self.coef[f] for f in self.features])
        eta = np.clip(self._off(df) + X @ b, -5, 5)
        mu = np.exp(eta)
        alpha = np.exp(np.clip(self.a0 + self.a1 * eta, -8, 4))
        return mu, alpha

    def pmf_grid(self, df):
        """P(runs = 0..MAX_RUNS) per row, renormalized to sum to 1."""
        X = self._X(df)
        b = np.array([self.coef["const"]] + [self.coef[f] for f in self.features])
        return self.pmf_eta(self._off(df) + X @ b)

    def eta_parts(self, df, shared):
        """Split the linear predictor into a base part (offset, constant and
        the features in `shared`) and a team part (every other feature)."""
        X = self._X(df)
        b = np.array([self.coef[f] for f in self.features])
        C = X[:, 1:] * b
        sh = np.array([f in shared for f in self.features], dtype=bool)
        return self._off(df) + self.coef["const"] + C[:, sh].sum(axis=1), C[:, ~sh].sum(axis=1)

    def pmf_eta(self, eta):
        """P(runs = 0..MAX_RUNS) for given linear predictors."""
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

    def to_json(self):
        return {"features": self.features, "offset": self.offset, "ridge": self.ridge,
                "mean_scaled_dispersion": self.mean_scaled_dispersion,
                "center": self.center, "scale": self.scale, "fill": self.fill,
                "coef": self.coef, "a0": self.a0, "a1": self.a1,
                "zero_adj": self.zero_adj, "zdelta": self.zdelta,
                "n_train": self.n_train, "loglik": self.loglik, "fit_max_grad": self.fit_max_grad}
