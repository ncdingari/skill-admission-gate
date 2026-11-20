"""Outcome model for paired skill exams.

Variant i has latent difficulty theta_i ~ N(0, sd_theta^2) on the logit scale. The comparator
arm passes with p0_i = sigmoid(b0 + theta_i); the skill arm with p1_i = sigmoid(b0 + g_i + theta_i),
g_i ~ N(g, sd_lift^2). With probability rho_crn the two arms share one sampling uniform (common
random numbers); otherwise they draw independently. r attempts per arm; D_i is the mean paired
difference (Eq. 1 of the paper). The true lift is Delta = E[p1 - p0], computed by Gauss-Hermite
quadrature, and `solve_g` finds the g that gives a target Delta exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.optimize import brentq
from scipy.special import expit

_GH_X, _GH_W = hermegauss(80)
_GH_W = _GH_W / _GH_W.sum()          # weights for E[f(Z)], Z ~ N(0, 1)


def mean_sigmoid(mu: float, sd: float) -> float:
    """E[sigmoid(mu + sd Z)]."""
    return float(np.dot(_GH_W, expit(mu + sd * _GH_X)))


def base_logit(rate: float, sd_theta: float) -> float:
    """b0 such that the comparator pass rate E[sigmoid(b0 + theta)] equals `rate`."""
    return float(brentq(lambda b: mean_sigmoid(b, sd_theta) - rate, -30, 30, xtol=1e-14))


def true_lift(g: float, b0: float, sd_theta: float, sd_lift: float) -> float:
    return mean_sigmoid(b0 + g, np.hypot(sd_theta, sd_lift)) - mean_sigmoid(b0, sd_theta)


def solve_g(target: float, b0: float, sd_theta: float, sd_lift: float) -> float:
    p0 = mean_sigmoid(b0, sd_theta)
    if not (-p0 < target < 1 - p0):
        raise ValueError(f"lift {target} infeasible for comparator rate {p0:.3f}")
    return float(brentq(lambda g: true_lift(g, b0, sd_theta, sd_lift) - target, -40, 40, xtol=1e-14))


@dataclass(frozen=True)
class Model:
    b0: float
    g: float
    sd_theta: float = 1.5
    sd_lift: float = 0.5
    rho_crn: float = 0.5
    r: int = 1
    drift: float = 0.0       # logit shift applied to the skill arm only (stale-baseline scenario)

    @classmethod
    def for_lift(cls, lift: float, base_rate: float, sd_theta=1.5, sd_lift=0.5, rho_crn=0.5, r=1,
                 drift=0.0) -> "Model":
        b0 = base_logit(base_rate, sd_theta)
        g = solve_g(lift, b0, sd_theta, sd_lift)
        return cls(b0, g, sd_theta, sd_lift, rho_crn, r, drift)

    @property
    def lift(self) -> float:
        """True lift of the skill (the drift term is not part of the skill's effect)."""
        return true_lift(self.g, self.b0, self.sd_theta, self.sd_lift)

    def with_(self, **kw) -> "Model":
        return replace(self, **kw)


def simulate(rng: np.random.Generator, m: Model, reps: int, n: int) -> np.ndarray:
    """Paired differences D, shape (reps, n), values in [-1, 1] (multiples of 1/r)."""
    theta = rng.standard_normal((reps, n)) * m.sd_theta
    gi = m.g + rng.standard_normal((reps, n)) * m.sd_lift
    p0 = expit(m.b0 + theta)
    p1 = expit(m.b0 + gi + theta + m.drift)
    D = np.zeros((reps, n))
    for _ in range(m.r):
        u0 = rng.random((reps, n))
        u1 = np.where(rng.random((reps, n)) < m.rho_crn, u0, rng.random((reps, n)))
        D += (u1 < p1).astype(float) - (u0 < p0).astype(float)
    return D / m.r


def simulate_two_point(rng: np.random.Generator, lift: float, reps: int, n: int) -> np.ndarray:
    """Worst case for the betting test: every pair discordant, D in {-1, +1}, E[D] = lift."""
    return np.where(rng.random((reps, n)) < (1 + lift) / 2, 1.0, -1.0)


def discordant_share(m: Model) -> float:
    """P(D=+1 | D != 0) for r = 1, by Monte Carlo-free quadrature over theta and g_i (rho_crn aware)."""
    z = _GH_X
    w = _GH_W
    # 2-D quadrature over (theta, g_i)
    th = m.sd_theta * z[:, None]
    gi = m.g + m.sd_lift * z[None, :]
    W = w[:, None] * w[None, :]
    p0 = expit(m.b0 + th)
    p1 = expit(m.b0 + gi + th + m.drift)
    # coupled (shared uniform): P(Y1=1,Y0=0) = max(p1-p0,0); independent: p1(1-p0)
    c = m.rho_crn
    p10 = c * np.maximum(p1 - p0, 0) + (1 - c) * p1 * (1 - p0)
    p01 = c * np.maximum(p0 - p1, 0) + (1 - c) * p0 * (1 - p1)
    a, b = float((W * p10).sum()), float((W * p01).sum())
    return a / (a + b)


def sd_D(m: Model, n_mc: int = 400_000, seed: int = 0) -> float:
    """SD of the paired difference D (Monte Carlo), for the fixed-n sample-size formula."""
    rng = np.random.default_rng(seed)
    return float(simulate(rng, m, 1, n_mc).std())
