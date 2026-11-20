"""Multiple-testing rules for a skill library.

bh        Benjamini-Hochberg on p-values.
ebh       e-BH (Wang and Ramdas): FDR <= alpha for arbitrarily dependent e-values.
elond     online e-LOND (Xu and Ramdas): test t runs at level alpha_t = alpha * gamma_t * (R_{t-1} + 1),
          i.e. rejects when e_t >= 1 / alpha_t. FDR <= alpha over the whole stream, any dependence.
gamma_seq normalized gamma_t proportional to t^(-s), sum over t >= 1 equal to 1.
lineage_levels  the alpha-spending rule for resubmissions: the k-th exam of one skill lineage runs at
          alpha * gamma_k, so the chance a null lineage is ever admitted is at most alpha.
"""
from __future__ import annotations

import numpy as np
from scipy.special import zeta


def bh(p: np.ndarray, alpha: float) -> np.ndarray:
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    ok = p[order] <= alpha * np.arange(1, m + 1) / m
    k = (np.nonzero(ok)[0].max() + 1) if ok.any() else 0
    out = np.zeros(m, bool)
    out[order[:k]] = True
    return out


def ebh(e: np.ndarray, alpha: float) -> np.ndarray:
    e = np.asarray(e, float)
    m = len(e)
    order = np.argsort(-e)
    ok = e[order] >= m / (alpha * np.arange(1, m + 1))
    k = (np.nonzero(ok)[0].max() + 1) if ok.any() else 0
    out = np.zeros(m, bool)
    out[order[:k]] = True
    return out


def gamma_seq(n: int, s: float = 1.6) -> np.ndarray:
    t = np.arange(1, n + 1, dtype=float)
    return t ** (-s) / zeta(s)


def elond_levels(rejections_before: int, t: int, alpha: float, s: float = 1.6) -> float:
    """Level for the t-th test (1-based) given the number of earlier rejections."""
    return alpha * (t ** (-s) / zeta(s)) * (rejections_before + 1)


def elond(e: np.ndarray, alpha: float, s: float = 1.6) -> np.ndarray:
    """Offline replay of e-LOND on a stream of e-values (in arrival order)."""
    out = np.zeros(len(e), bool)
    R = 0
    for t, et in enumerate(e, start=1):
        if et >= 1 / elond_levels(R, t, alpha, s):
            out[t - 1] = True
            R += 1
    return out


def lineage_levels(k: int, alpha: float, s: float = 1.6) -> np.ndarray:
    return alpha * gamma_seq(k, s)


# ------------------------------------------------------------------ E20: e-BH over e-processes
def ebh_path(E: np.ndarray, alpha: float) -> np.ndarray:
    """e-BH rejection set at every time: E is (m, n) e-process values; returns bool (m, n)."""
    E = np.asarray(E, float)
    m = E.shape[0]
    srt = -np.sort(-E, axis=0)
    ok = srt >= m / (alpha * np.arange(1, m + 1))[:, None]
    k = np.where(ok.any(axis=0), m - np.argmax(ok[::-1], axis=0), 0)
    thr = np.where(k > 0, m / (alpha * np.maximum(k, 1)), np.inf)
    return E >= thr[None, :]


def _monitor(E: np.ndarray, alpha: float, done: np.ndarray):
    """Continuous monitoring: at each time t, e-BH on the current values; an exam stops (its value is frozen)
    the first time it is in the e-BH set, or when `done` (its own futility stop) turns True. Final rejections
    are the e-BH set on the frozen values at the end. Returns (rejected, stop index per exam)."""
    E = np.asarray(E, float)
    m, n = E.shape
    cur = E[:, 0].copy()
    stop = np.full(m, n - 1)
    open_ = np.ones(m, bool)
    for t in range(n):
        cur[open_] = E[open_, t]
        r = ebh(cur, alpha)
        hit = open_ & (r | done[:, t])
        stop[hit] = t
        open_ &= ~hit
        if not open_.any():
            break
    return ebh(cur, alpha), stop


def stopped_ebh(E: np.ndarray, alpha: float, done: np.ndarray):
    """Stopped e-BH (se-BH; Wang, Dandapanthula and Ramdas 2025, arXiv 2502.08539): e-BH applied to the
    e-processes at a data-dependent stopping time. FDR <= alpha when the streams are independent (their
    causal condition). Streams finished by futility are frozen (constant after `done` turns True)."""
    return _monitor(E, alpha, done)


def adjust_sqrt(M: np.ndarray) -> np.ndarray:
    """Adjuster A2(M) = sqrt(M) - 1 (Tavyrikov, Goeman and de Heide 2025, arXiv 2501.19360, Eq. 5)."""
    return np.sqrt(np.maximum(M, 1.0)) - 1.0


def ebh_suprema(E: np.ndarray, alpha: float, done: np.ndarray):
    """e-BH on adjusted running suprema A2(max_{s<=t} E_s) (Tavyrikov et al. 2025, Theorem 1):
    FDR-sup <= alpha under arbitrary dependence."""
    return _monitor(adjust_sqrt(np.maximum.accumulate(np.asarray(E, float), axis=1)), alpha, done)
