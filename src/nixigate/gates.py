"""Admission gates, vectorized over Monte Carlo replicates.

Every gate takes D, an array (reps, n_max) of paired differences in [-1, 1], tests
H0: Delta <= delta against H1: Delta > delta, and returns a GateResult with, per replicate,
`admit` (bool), `n` (variants used: the stopping time) and `dbar` (mean of D over the first n).

G1 fixed_t      one-sided paired t-test at a fixed n.
G2 peek_t       the same t-test re-run after every variant from n_min (invalid: optional stopping).
G3 peek_ci      admit when the lower end of a two-sided normal CI exceeds delta, checked every k variants.
G4 sprt         Wald SPRT on discordant pairs (r = 1, delta = 0 only).
G5 msprt        normal-mixture SPRT (Johari et al.), plug-in variance; asymptotically valid.
G6 betting      Waudby-Smith and Ramdas betting test with predictable bets; futility by a
                downward bet at alpha_f. Valid at every stopping time (Ville's inequality).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class GateResult:
    admit: np.ndarray
    n: np.ndarray
    dbar: np.ndarray
    log_wealth: np.ndarray | None = None   # betting only: log K at the stopping time
    futility: np.ndarray | None = None


def _first(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index of the first True in each row, and whether any True exists."""
    has = mask.any(axis=1)
    idx = np.where(has, mask.argmax(axis=1), mask.shape[1] - 1)
    return idx, has


def _dbar_at(D: np.ndarray, n: np.ndarray) -> np.ndarray:
    cs = np.cumsum(D, axis=1)
    return cs[np.arange(D.shape[0]), n - 1] / n


def _running_mean_sd(D: np.ndarray):
    n = np.arange(1, D.shape[1] + 1)
    s1 = np.cumsum(D, axis=1)
    s2 = np.cumsum(D * D, axis=1)
    mean = s1 / n
    var = np.where(n > 1, (s2 - n * mean**2) / np.maximum(n - 1, 1), np.nan)
    return n, mean, np.sqrt(np.maximum(var, 0))


# ------------------------------------------------------------------ G1
def fixed_t(D: np.ndarray, n: int, delta: float, alpha: float) -> GateResult:
    x = D[:, :n]
    mean = x.mean(axis=1)
    sd = x.std(axis=1, ddof=1)
    crit = stats.t.ppf(1 - alpha, n - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (mean - delta) / (sd / np.sqrt(n))
    admit = np.where(sd > 0, t > crit, False)
    return GateResult(admit, np.full(len(x), n), mean)


def sample_size(sd: float, lift1: float, delta: float, alpha: float, power: float) -> int:
    """Normal-approximation paired sample size: n = ((z_{1-a} + z_{1-b}) sd / (lift1 - delta))^2, rounded up."""
    z = stats.norm.ppf(1 - alpha) + stats.norm.ppf(power)
    return int(np.ceil((z * sd / (lift1 - delta)) ** 2))


# ------------------------------------------------------------------ G2
def peek_t(D: np.ndarray, delta: float, alpha: float, n_min: int) -> GateResult:
    n, mean, sd = _running_mean_sd(D)
    crit = stats.t.ppf(1 - alpha, np.maximum(n - 1, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (mean - delta) / (sd / np.sqrt(n))
    hit = (n >= n_min) & (sd > 0) & (t > crit)
    idx, has = _first(hit)
    N = idx + 1
    return GateResult(has, N, _dbar_at(D, N))


# ------------------------------------------------------------------ G3
def peek_ci(D: np.ndarray, delta: float, z: float, n_min: int, every: int) -> GateResult:
    n, mean, sd = _running_mean_sd(D)
    look = (n >= n_min) & ((n - n_min) % every == 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lower = mean - z * sd / np.sqrt(n)
    hit = look & (sd > 0) & (lower > delta)
    idx, has = _first(hit)
    N = idx + 1
    return GateResult(has, N, _dbar_at(D, N))


# ------------------------------------------------------------------ G4
def sprt(D: np.ndarray, p1: float, alpha: float, beta: float) -> GateResult:
    """Wald SPRT of p = P(D=+1 | D != 0) <= 1/2 against p = p1 (delta = 0, r = 1)."""
    b = np.cumsum(D > 0.5, axis=1)
    bp = np.cumsum(D < -0.5, axis=1)
    llr = b * np.log(p1 / 0.5) + bp * np.log((1 - p1) / 0.5)
    up, lo = np.log((1 - beta) / alpha), np.log(beta / (1 - alpha))
    i_up, h_up = _first(llr >= up)
    i_lo, h_lo = _first(llr <= lo)
    big = D.shape[1] + 10
    t_up = np.where(h_up, i_up, big)
    t_lo = np.where(h_lo, i_lo, big)
    admit = h_up & (t_up <= t_lo)
    stop = np.minimum(np.minimum(t_up, t_lo), D.shape[1] - 1)
    N = stop + 1
    # the likelihood ratio is a test supermartingale under p <= 1/2, so its stopped value is an e-value
    return GateResult(admit, N, _dbar_at(D, N), llr[np.arange(len(D)), stop])


# ------------------------------------------------------------------ G5
def msprt(D: np.ndarray, delta: float, alpha: float, tau: float, n_min: int) -> GateResult:
    n, mean, sd = _running_mean_sd(D)
    s2 = np.maximum(sd**2, 1e-6)
    with np.errstate(over="ignore", invalid="ignore"):
        log_lam = 0.5 * np.log(s2 / (s2 + n * tau**2)) + (n**2 * tau**2 * (mean - delta) ** 2) / (
            2 * s2 * (s2 + n * tau**2))
    hit = (n >= n_min) & (mean > delta) & (log_lam >= np.log(1 / alpha))
    idx, has = _first(hit)
    N = idx + 1
    return GateResult(has, N, _dbar_at(D, N))


# ------------------------------------------------------------------ G6
def _plugin_var(X: np.ndarray, prior_mean=0.5, prior_var=0.25):
    """WSR predictable plug-in mean and variance: values *before* observing X_t (index t-1)."""
    t = np.arange(1, X.shape[1] + 1)
    mu = (prior_mean + np.cumsum(X, axis=1)) / (t + 1)
    var = (prior_var + np.cumsum((X - mu) ** 2, axis=1)) / (t + 1)
    mu_prev = np.concatenate([np.full((X.shape[0], 1), prior_mean), mu[:, :-1]], axis=1)
    var_prev = np.concatenate([np.full((X.shape[0], 1), prior_var), var[:, :-1]], axis=1)
    return t, mu_prev, var_prev


def _log_inv(alpha) -> np.ndarray | float:
    """log(1/alpha); an array of per-row levels becomes a column for broadcasting."""
    a = np.asarray(alpha, float)
    return np.log(1 / a)[:, None] if a.ndim == 1 else float(np.log(1 / a))


def prpl_bets(X: np.ndarray, alpha, cap: float) -> np.ndarray:
    """Predictable plug-in bets (WSR Eq. 26 form, one-sided): min(sqrt(2 log(1/a)/(v t log(1+t))), cap).
    `alpha` may be a scalar or one level per row."""
    t, _, var_prev = _plugin_var(X)
    lam = np.sqrt(2 * _log_inv(alpha) / (var_prev * t * np.log1p(t)))
    return np.minimum(lam, cap)


def agrapa_bets(X: np.ndarray, m0: float, c: float) -> np.ndarray:
    _, mu_prev, var_prev = _plugin_var(X)
    lam = (mu_prev - m0) / (var_prev + (mu_prev - m0) ** 2)
    return np.clip(lam, 0, c / m0)



# ------------------------------------------------ level-tuned grid mixture (Theorem thm:adaptive)
_K = np.arange(1, 10**6 + 1, dtype=float)
C_U = 1.0 / (np.sum(1.0 / ((_K + 1) * np.log(_K + 1) ** 2)) + 1.0 / np.log(10**6 + 1.5))
del _K


def grid_logK(X: np.ndarray, m0: float, alpha: float, jmax_floor: float = 1e-4, chunk: int = 500) -> np.ndarray:
    """log of M_n = sum_j w_j prod_{i<=n} (1 + lam_j (X_i - m0)), lam_j = rho^(j+1)/m0, rho = L/(L+1),
    L = log(1/alpha), w_j = c_u / {(j+2) log^2(j+2)} (sum_j w_j <= 1, so M is an e-process).
    Same grid and weights as experiments/e13 (Theorem thm:adaptive). alpha must be a scalar."""
    L = float(np.log(1 / alpha))
    rho = L / (L + 1)
    J = int(np.ceil(np.log(1 / jmax_floor) / np.log(1 / rho)))
    j = np.arange(J + 1)
    lam = rho ** (j + 1) / m0
    logw = np.log(C_U) - np.log(j + 2) - 2 * np.log(np.log(j + 2))
    out = np.empty(X.shape)
    for s in range(0, X.shape[0], chunk):
        S = np.cumsum(np.log1p(lam[None, None, :] * (X[s:s + chunk, :, None] - m0)), axis=1)
        z = S + logw[None, None, :]
        mx = z.max(axis=2, keepdims=True)
        out[s:s + chunk] = (mx + np.log(np.exp(z - mx).sum(axis=2, keepdims=True)))[..., 0]
    return out


def betting(D: np.ndarray, delta: float, alpha, alpha_f: float | None = 0.10, cap: float = 0.5,
            rule: str = "prpl", admit_stop: bool = True, delta_f: float | None = None) -> GateResult:
    """Anytime-valid betting gate. rule: 'prpl' (cap = c), 'cap_m0' (cap = m0), 'agrapa' (cap = c / m0), 'grid' (level-tuned grid mixture).
    `alpha` may be one level per row (e-LOND). With admit_stop=False the exam does not stop at the
    admission boundary; it runs to n_max (or futility) and `log_wealth` is the e-value at that time.
    delta_f (>= delta): futility margin. The futility bet tests Delta >= delta_f, so the exam stops early
    when the lift is confidently below a useful size. Futility can only remove admissions, so the
    type-I guarantee for H0: Delta <= delta is unchanged."""
    X = (1 + D) / 2
    m0 = (1 + delta) / 2
    if rule == "grid":
        if np.ndim(alpha) == 0:
            logK = grid_logK(X, m0, float(alpha))
        else:  # one level per row (e-LOND): the grid is tuned to each level, so group rows by level
            al = np.asarray(alpha, float)
            logK = np.empty(X.shape)
            for u in np.unique(al):
                rows = al == u
                logK[rows] = grid_logK(X[rows], m0, float(u))
    else:
        if rule == "prpl":
            lam = prpl_bets(X, alpha, cap)
        elif rule == "cap_m0":
            lam = prpl_bets(X, alpha, m0)
        elif rule == "agrapa":
            lam = agrapa_bets(X, m0, cap)
        else:
            raise ValueError(rule)
        logK = np.cumsum(np.log1p(lam * (X - m0)), axis=1)
    big = D.shape[1] + 10
    if admit_stop:
        i_up, h_up = _first(logK >= _log_inv(alpha))
    else:
        i_up, h_up = np.full(len(D), big), np.zeros(len(D), bool)
    t_up = np.where(h_up, i_up, big)
    if alpha_f is not None:
        lam_f = prpl_bets(X, alpha_f, cap)
        mf = (1 + (delta if delta_f is None else delta_f)) / 2
        logKf = np.cumsum(np.log1p(-lam_f * (X - mf)), axis=1)
        i_f, h_f = _first(logKf >= np.log(1 / alpha_f))
        t_f = np.where(h_f, i_f, big)
    else:
        t_f = np.full(len(D), big)
    admit = h_up & (t_up <= t_f)   # admit checked first at a tie
    stop = np.minimum(np.minimum(t_up, t_f), D.shape[1] - 1)
    N = stop + 1
    fut = (~admit) & (t_f < big)
    return GateResult(admit, N, _dbar_at(D, N), logK[np.arange(len(D)), stop], fut)


# ------------------------------------------------------------------ confidence sequence
def lcb_path(D_row: np.ndarray, alpha: float, cap: float = 0.5, grid: int = 2001) -> np.ndarray:
    """Anytime-valid lower confidence bound for Delta at every t (running intersection).

    Uses the same predictable bets as `betting(rule='prpl')`, so the gate admits at time t
    exactly when the bound at t exceeds delta (up to grid resolution 2/(grid-1)).
    """
    X = (1 + D_row[None, :]) / 2
    lam = prpl_bets(X, alpha, cap)[0]
    m = np.linspace(0, 1, grid)
    logK = np.cumsum(np.log1p(lam[:, None] * (X[0][:, None] - m[None, :])), axis=0)   # (t, grid)
    rejected = np.maximum.accumulate(logK >= np.log(1 / alpha), axis=0)
    # rejected set is [0, LCB): the first non-rejected grid point is the bound
    first_ok = np.argmin(rejected, axis=1)
    all_rej = rejected.all(axis=1)
    lcb_m = np.where(all_rej, 1.0, m[first_ok])
    return 2 * lcb_m - 1
