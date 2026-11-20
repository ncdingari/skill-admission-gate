"""E14: referee checks M5 to M7 (free simulation, no agent runs).

1. M5, matched power. At the design lift and the curated lift, the SPRT with type-II error beta in
   {0.20, 0.10, 0.05, 0.01}, the betting gate with futility margin truncated at N_max in {150, 200, 300, 600},
   and the fixed-n t-test planned for power {0.80, 0.90, 0.95, 0.99}: power and mean variants.
2. M6, moment-matched model. Fit (s_theta, s_g, rho) at r = 3 to three within-skill moments of the released
   SkillsBench differences (tie share, SD, share with |D| >= 2/3), with the released lift and comparator rate.
   Then add a correlation kappa between a variant's easiness theta and the skill effect g_i, and rerun the
   cost claims of Section 6.3 (fixed n, betting power and variants at the design and curated lifts) and the
   null check (Delta = 0) for each model.
3. M7, library procedure and selection. Resampling cost of certifying SkillsBench skills when each exam's
   level comes from e-LOND (the library procedure the gate uses) over random arrival orders, against the
   per-exam and Bonferroni levels; and a split-half check of the selection on observed lift > 0: skills are
   selected on nine configurations and their cost is resampled from the other nine.
"""
import os

import numpy as np
import pandas as pd
from scipy.special import expit, zeta

from _common import P, QUICK, Run, rng_for, v
from _common import RULE
from _realdata import skillsbench
from nixigate import dgp, gates, suite

_bet = lambda *a_, **k_: gates.betting(*a_, rule=RULE, **k_)  # NIXI_RULE=grid reruns with the grid bet
WIDE = os.environ.get("NIXI_E23WIDE") == "1"  # E23: wider moment-matching grid, results under E23
run = Run("E23" if WIDE else "E14")
a, af, cap = v(P, "alpha"), v(P, "alpha_futility"), v(P, "bet_cap_c")
base, L1, LC, dfut = v(P, "base_pass_rate"), v(P, "target_lift"), v(P, "curated_lift"), v(P, "futility_margin")
sdt, sdl, rho = v(P, "sd_theta"), v(P, "sd_lift"), v(P, "rho_crn")
NMAX = int(v(P, "n_max"))
R = 400 if QUICK else 4000

# ---------------------------------------------------------------- 1. matched power (M5)
rows = []
for lift in (L1, LC):
    m = dgp.Model.for_lift(lift, base, sdt, sdl, rho, 1)
    m1 = dgp.Model.for_lift(L1, base, sdt, sdl, rho, 1)
    p1 = dgp.discordant_share(m1)
    sd1 = dgp.sd_D(m1)
    D = dgp.simulate(rng_for(f"E14mp{lift:.4f}"), m, R, 2000)
    for beta in (0.20, 0.10, 0.05, 0.01):
        g = gates.sprt(D[:, :NMAX], p1, a, beta)
        rows.append({"true_lift": lift, "gate": "sprt", "setting": beta, "power": g.admit.mean(), "mean_n": g.n.mean()})
    for nm in (150, 200, 300, 600):
        g = _bet(D[:, :nm], 0.0, a, af, cap=cap, delta_f=dfut)
        rows.append({"true_lift": lift, "gate": "bet_futm", "setting": nm, "power": g.admit.mean(), "mean_n": g.n.mean()})
    for pw in (0.80, 0.90, 0.95, 0.99):
        n = gates.sample_size(sd1, L1, 0.0, a, pw)
        g = gates.fixed_t(D, n, 0.0, a)
        rows.append({"true_lift": lift, "gate": "fixed_t", "setting": pw, "power": g.admit.mean(), "mean_n": float(n)})
run.save(pd.DataFrame(rows), "e14_matched_power")

# ---------------------------------------------------------------- 2. moment-matched model (M6)
w = skillsbench()
tasks = sorted(w.task.unique())
cfgs = sorted(w.config.unique())
X = np.stack([w[w.task == t].set_index("config").loc[cfgs].D.values for t in tasks])   # (87, 18)
Xc = X - X.mean(1, keepdims=True)
target = {"tie": float((X == 0).mean()), "sd_within": float(np.sqrt((Xc ** 2).sum() / (X.size - len(X)))),
          "big": float((np.abs(X) >= 2 / 3 - 1e-9).mean())}
lift_obs, p0_obs = float(X.mean()), float(w.p0.mean())


def b0_for(rate, s):
    return dgp.base_logit(rate, s)


def lift_k(g, b0, st, sg, k):
    return dgp.mean_sigmoid(b0 + g, np.sqrt(st ** 2 + sg ** 2 + 2 * k * st * sg)) - dgp.mean_sigmoid(b0, st)


def g_for(lift, b0, st, sg, k):
    from scipy.optimize import brentq
    return float(brentq(lambda g: lift_k(g, b0, st, sg, k) - lift, -40, 40, xtol=1e-12))


def sim(rng, lift, rate, st, sg, rh, k, r, reps, n):
    """dgp.simulate with corr(theta_i, g_i) = k."""
    b0 = b0_for(rate, st)
    g = g_for(lift, b0, st, sg, k)
    z1 = rng.standard_normal((reps, n))
    z2 = rng.standard_normal((reps, n))
    theta = st * z1
    gi = g + sg * (k * z1 + np.sqrt(1 - k ** 2) * z2)
    p0, p1 = expit(b0 + theta), expit(b0 + gi + theta)
    D = np.zeros((reps, n))
    for _ in range(r):
        u0 = rng.random((reps, n))
        u1 = np.where(rng.random((reps, n)) < rh, u0, rng.random((reps, n)))
        D += (u1 < p1).astype(float) - (u0 < p0).astype(float)
    return D / r


def moments(D):
    return {"tie": float((D == 0).mean()), "sd_within": float(D.std()), "big": float((np.abs(D) >= 2 / 3 - 1e-9).mean())}


def loss(mo):
    return sum(((mo[k] - target[k]) / target[k]) ** 2 for k in target)


fit_rows = []
grid_st = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
grid_sg = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0) + ((4.0, 5.0) if WIDE else ())
grid_rho = (0.0, 0.25, 0.5, 0.75, 0.9) + ((0.95, 0.98, 0.995, 0.999) if WIDE else ())
nfit = 20000 if QUICK else 100000
for st in grid_st:
    for sg in grid_sg:
        for rh in grid_rho:
            try:
                mo = moments(sim(rng_for("E14fit"), lift_obs, p0_obs, st, sg, rh, 0.0, 3, 1, nfit)[0])
            except ValueError:
                continue
            fit_rows.append({"sd_theta": st, "sd_lift": sg, "rho": rh, **mo, "loss": loss(mo)})
fit = pd.DataFrame(fit_rows).sort_values("loss")
best = fit.iloc[0]
# local refinement around the best grid point
ref = []
for st in np.linspace(max(0.25, best.sd_theta - 0.5), best.sd_theta + 0.5, 5):
    for sg in np.linspace(max(0.0, best.sd_lift - 0.5), best.sd_lift + 0.5, 5):
        for rh in (np.clip(np.linspace(best.rho - 0.02, best.rho + 0.02, 5), 0, 0.999) if WIDE else np.clip(np.linspace(best.rho - 0.2, best.rho + 0.2, 5), 0, 0.95)):
            try:
                mo = moments(sim(rng_for("E14fit"), lift_obs, p0_obs, st, sg, rh, 0.0, 3, 1, nfit)[0])
            except ValueError:
                continue
            ref.append({"sd_theta": st, "sd_lift": sg, "rho": rh, **mo, "loss": loss(mo)})
fit = pd.concat([fit, pd.DataFrame(ref)]).drop_duplicates(["sd_theta", "sd_lift", "rho"]).sort_values("loss").reset_index(drop=True)
best = fit.iloc[0]
fit.insert(0, "target", str(target))
run.save(fit.head(25), "e14_moment_fit")

# cost claims under each model; kappa = corr(easiness, skill effect)
models = [("default", sdt, sdl, rho, 0.0)] + [(f"fitted_kappa_{k:+.1f}", float(best.sd_theta), float(best.sd_lift),
                                               float(best.rho), k) for k in (-0.5, 0.0, 0.5)]
models += [("default_kappa_-0.5", sdt, sdl, rho, -0.5), ("default_kappa_+0.5", sdt, sdl, rho, 0.5)]
RS = 400 if QUICK else 3000
srows = []
for name, st, sg, rh, k in models:
    for r in (1, 3):
        sd1 = float(sim(rng_for(f"E14sd{name}{r}"), L1, base, st, sg, rh, k, r, 1, 400000)[0].std())
        nfx = gates.sample_size(sd1, L1, 0.0, a, 0.80)
        for lift in (0.0, L1, LC):
            D = sim(rng_for(f"E14c{name}{r}{lift:.3f}"), lift, base, st, sg, rh, k, r, RS, max(NMAX, nfx))
            fx = gates.fixed_t(D, nfx, 0.0, a)
            bt = _bet(D[:, :NMAX], 0.0, a, af, cap=cap, delta_f=dfut)
            srows.append({"model": name, "sd_theta": st, "sd_lift": sg, "rho": rh, "kappa": k, "r": r, "true_lift": lift,
                          "sd_D_design": sd1, "n_fixed": nfx, "power_fixed": fx.admit.mean(),
                          "power_bet": bt.admit.mean(), "mean_n_bet": bt.n.mean(), "reps": RS})
sens = pd.DataFrame(srows)
run.save(sens, "e14_cost_sensitivity")

# ---------------------------------------------------------------- 3. library procedure and selection (M7)
M = len(tasks)
NM = 2000
levels = a * np.logspace(0, -6, 25)            # from alpha down to alpha * 1e-6
Rb = 100 if QUICK else 400
rng = rng_for("E14lib")
ADM = np.zeros((M, len(levels), Rb), bool)
NN = np.zeros((M, len(levels), Rb))
for i in range(M):
    Db = rng.choice(X[i], size=(Rb, NM))
    for j, lv in enumerate(levels):
        g = _bet(Db, 0, lv, None, cap)
        ADM[i, j], NN[i, j] = g.admit, np.where(g.admit, g.n, np.inf)


def draw(i, lv, rr):
    """Outcome of one resampled exam of skill i at level lv: the largest grid level <= lv (conservative)."""
    j = int(np.searchsorted(-levels, -lv, side="left"))
    j = min(j, len(levels) - 1)
    b = rr.integers(Rb)
    return ADM[i, j, b], NN[i, j, b]


ORD = 100 if QUICK else 1000
rr = rng_for("E14ord")
gam_inf = lambda t: t ** -1.6 / zeta(1.6)  # noqa: E731
res = {k: [[] for _ in range(M)] for k in ("elond_zeta", "elond_horizon", "per_exam", "bonferroni")}
for o in range(ORD):
    order = rr.permutation(M)
    for proc, gam in (("elond_zeta", gam_inf), ("elond_horizon", lambda t: 1 / M)):
        R_ = 0
        for t, i in enumerate(order, start=1):
            lv = a * gam(t) * (R_ + 1)
            ad, n = draw(i, min(lv, a), rr)
            R_ += int(ad and n <= 600)
            res[proc][i].append(n if ad else np.inf)
    for i in range(M):
        res["per_exam"][i].append(draw(i, a, rr)[1])
        res["bonferroni"][i].append(draw(i, a / M, rr)[1])
pos = X.mean(1) > 0
lrows = []
for proc, lst in res.items():
    med = np.array([np.median(np.array(x)) for x in lst])
    c600 = np.array([np.mean(np.array(x) <= 600) for x in lst])
    lrows.append({"procedure": proc, "n_pos": int(pos.sum()), "median_of_skill_medians": float(np.median(med[pos])),
                  "share_cert_600_pos": float(c600[pos].mean()), "admitted_600_all_mean": float(c600.sum())})
run.save(pd.DataFrame(lrows), "e14_library_procedure")

# selection on observed lift > 0: select on half A, resample from half A or from the held-out half B
S = 20 if QUICK else 100
Rs = 100 if QUICK else 200
rsel = rng_for("E14sel")
srow = []
for s in range(S):
    perm = rsel.permutation(len(cfgs))
    A_, B_ = perm[:9], perm[9:]
    selA = X[:, A_].mean(1) > 0
    for src, cols in (("same_half", A_), ("held_out_half", B_)):
        meds = []
        for i in np.where(selA)[0]:
            Db = rsel.choice(X[i, cols], size=(Rs, NM))
            g = _bet(Db, 0, a / M, None, cap)
            meds.append(float(np.median(np.where(g.admit, g.n, np.inf))))
        meds = np.array(meds)
        srow.append({"split": s, "resample_from": src, "selected": int(selA.sum()),
                     "median_of_medians": float(np.median(meds)), "share_finite": float(np.isfinite(meds).mean())})
sel = pd.DataFrame(srow)
run.save(sel, "e14_selection_split")
run.done(reps=R, fit_best=dict(best[["sd_theta", "sd_lift", "rho", "loss"]]), orders=ORD, splits=S)
