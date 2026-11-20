"""E13: numerical checks for the level-tuned grid mixture (Theorem thm:adaptive, Section 5).

The exam at level a mixes fixed bets lambda_j = rho^(j+1)/m0, j >= 0, with rho = L/(L+1), L = log(1/a), and
weights w_j = c_u/((j+2) log^2(j+2)) (the universal weights of Corollary cor:universal). It needs no knowledge
of P. Theorem: K_inf E[tau] <= (1 + 1/L){L + log(1/m0) + ell(j*)}, ell(j) = log(1/w_j),
j* = floor(log(1/(m0 lambda*)) / log(1/rho)).

1. Every SkillsBench skill with positive released lift, at the first-exam level alpha q_1 and at alpha:
   E[tau] for the Kelly bet (needs P), the tuned grid, a grid with fixed rho = 1/2, and the paper's PrPl bet
   (cap 1/2) and aGRAPA; the theorem's bound for the tuned grid.
2. The median skill (as in E11/E12) over levels L = log(1/alpha) + 0..16: K_inf E[tau] - L for the tuned
   grid, the fixed grid and the Kelly bet, against the bound.
3. Validity: rejection rate of the tuned-grid exam under the least-favourable null Q* of each skill."""
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import logsumexp, zeta

from _common import P, QUICK, Run, rng_for, universal_cu, v
from _realdata import skillsbench
from nixigate import gates

run = Run("E13")
alpha = v(P, "alpha")
m0 = 0.5
rng = rng_for("E13")
q1 = 1 / zeta(1.6)
cu = universal_cu()

w_ = skillsbench()
tasks = sorted(w_.task.unique())
cfgs = sorted(w_.config.unique())
Xall = (1 + np.stack([w_[w_.task == t].set_index("config").loc[cfgs].D.values for t in tasks])) / 2


def g(lam, x):
    return np.mean(np.log1p(lam * (x - m0)))


def kinf(x):
    r = optimize.minimize_scalar(lambda l: -g(l, x), bounds=(0, 1 / m0 - 1e-12), method="bounded",
                                 options={"xatol": 1e-12})
    lam = r.x
    if g(1 / m0, x) >= -r.fun:      # all-in bet is the maximiser (no mass at 0)
        return g(1 / m0, x), 1 / m0
    return -r.fun, lam


def grid(L, rho=None, jmax_floor=1e-4):
    rho = L / (L + 1) if rho is None else rho
    J = int(np.ceil(np.log(1 / jmax_floor) / np.log(1 / rho)))
    j = np.arange(J + 1)
    lam = rho ** (j + 1) / m0
    logw = np.log(cu) - np.log(j + 2) - 2 * np.log(np.log(j + 2))
    return lam, logw, rho


def ell(j):
    return -np.log(cu) + np.log(j + 2) + 2 * np.log(np.log(j + 2))


def bound(L, K, lam_star):
    rho = L / (L + 1)
    jstar = int(np.floor(np.log(1 / (m0 * lam_star)) / np.log(1 / rho) + 1e-12))
    return (1 + 1 / L) * (L + np.log(1 / m0) + ell(jstar)) / K, jstar


def tau_mix(Xs, lam, logw, L, chunk=400):
    """First n with sum_j w_j prod_i (1 + lam_j (X_i - m0)) >= e^L. Returns taus (horizon+1 if never)."""
    R, n = Xs.shape
    out = np.full(R, n + 1)
    for s in range(0, R, chunk):
        X = Xs[s:s + chunk]
        S = np.cumsum(np.log1p(lam[None, None, :] * (X[:, :, None] - m0)), axis=1)   # R x n x J
        logM = logsumexp(S + logw[None, None, :], axis=2)
        hit = logM >= L
        ok = hit.any(1)
        out[s:s + chunk] = np.where(ok, hit.argmax(1) + 1, n + 1)
    return out


def tau_fixed(Xs, lam, L):
    S = np.cumsum(np.log1p(lam * (Xs - m0)), axis=1)
    hit = S >= L
    return np.where(hit.any(1), hit.argmax(1) + 1, Xs.shape[1] + 1)


helpful = [i for i in range(len(tasks)) if Xall[i].mean() > m0]
R1 = 200 if QUICK else 1000
rows = []
for i in helpful:
    x = Xall[i]
    K, ls = kinf(x)
    for lvl_name, a in (("alpha*q1", alpha * q1), ("alpha", alpha)):
        L = np.log(1 / a)
        hz = int(4 * (L + 12) / K) + 100
        Xs = rng.choice(x, size=(R1, hz))
        lam_t, logw_t, _ = grid(L)
        lam_f, logw_f, _ = grid(L, rho=0.5)
        t_grid = tau_mix(Xs, lam_t, logw_t, L)
        t_fix = tau_mix(Xs, lam_f, logw_f, L)
        t_kel = tau_fixed(Xs, ls, L)
        Ds = 2 * Xs - 1
        pr = gates.betting(Ds, 0, a, None, 0.5)
        ag = gates.betting(Ds, 0, a, None, v(P, "bet_cap_c"), rule="agrapa")
        b, jstar = bound(L, K, ls)
        assert (t_grid <= hz).all() and (t_kel <= hz).all() and (t_fix <= hz).all()
        rows.append({"task": tasks[i], "level": lvl_name, "a": a, "L": L, "kinf": K, "lam_star": ls, "jstar": jstar,
                     "n_ideal": L / K, "n_kelly": t_kel.mean(), "n_grid_tuned": t_grid.mean(),
                     "n_grid_fixed_half": t_fix.mean(), "n_prpl": pr.n.mean(), "prpl_censored": 1 - pr.admit.mean(),
                     "n_agrapa": ag.n.mean(), "agrapa_censored": 1 - ag.admit.mean(), "bound_grid_tuned": b,
                     "se_grid": t_grid.std() / np.sqrt(R1)})
per = pd.DataFrame(rows)
assert (per.n_grid_tuned <= per.bound_grid_tuned).all()
run.save(per, "e13_per_skill")

# ---- 2. median skill over levels
e11 = pd.read_csv(run.dir.parent.parent / "results" / "E11" / "e11_lineage_sim.csv")
task = e11.task.iloc[0]
x = Xall[tasks.index(task)]
K, ls = kinf(x)
R2 = 300 if QUICK else 1500
rows = []
for dL in (np.arange(0, 17, 4) if QUICK else np.arange(0, 17, 1)):
    L = np.log(1 / alpha) + dL
    hz = int(4 * (L + 12) / K) + 100
    Xs = rng.choice(x, size=(R2, hz))
    lam_t, logw_t, rho = grid(L)
    lam_f, logw_f, _ = grid(L, rho=0.5)
    tg, tf, tk = tau_mix(Xs, lam_t, logw_t, L), tau_mix(Xs, lam_f, logw_f, L), tau_fixed(Xs, ls, L)
    b, jstar = bound(L, K, ls)
    rows.append({"task": task, "L": L, "rho": rho, "kinf": K, "excess_kelly": K * tk.mean() - L,
                 "excess_grid_tuned": K * tg.mean() - L, "excess_grid_fixed_half": K * tf.mean() - L,
                 "excess_bound": K * b - L, "log_L": np.log(L)})
lev = pd.DataFrame(rows)
assert (lev.excess_grid_tuned <= lev.excess_bound).all()
run.save(lev, "e13_levels")

# ---- 3. validity under Q* (reject within a long horizon)
R3 = 400 if QUICK else 4000
H3 = 2000
rows = []
L = np.log(1 / (alpha * q1))
lam_t, logw_t, _ = grid(L)
for i in helpful:
    x = Xall[i]
    K, ls = kinf(x)
    sup, cnt = np.unique(x, return_counts=True)
    p = cnt / cnt.sum()
    q = p / (1 + ls * (sup - m0))
    supp, qq = np.append(sup, 0.0), np.append(q, max(0.0, 1 - q.sum()))
    Xq = rng.choice(supp, p=qq / qq.sum(), size=(R3, H3))
    t = tau_mix(Xq, lam_t, logw_t, L, chunk=200)
    rows.append({"task": tasks[i], "a": alpha * q1, "reject_rate_Qstar": float((t <= H3).mean()), "mean_Qstar": float(qq @ supp / qq.sum())})
val = pd.DataFrame(rows)
run.save(val, "e13_validity")
run.done(task=task, n_skills=len(helpful), max_reject=float(val.reject_rate_Qstar.max()))
