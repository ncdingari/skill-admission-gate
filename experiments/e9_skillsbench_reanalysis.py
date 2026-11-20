"""E9: SkillsBench reanalysis (C1, C4): per-skill rules, split-half replication over configurations,
calibration of the outcome model, and variants needed to certify each curated skill."""
import numpy as np
import pandas as pd
from scipy import stats

from _common import P, QUICK, Run, rng_for, v
from _realdata import skillsbench
from nixigate import dgp, fdr, gates

run = Run("E9")
a, af, cap, dfut = v(P, "alpha"), v(P, "alpha_futility"), v(P, "bet_cap_c"), v(P, "futility_margin")
w = skillsbench()
tasks = sorted(w.task.unique())
cfgs = sorted(w.config.unique())
M = len(tasks)
X = np.stack([w[w.task == t].set_index("config").loc[cfgs].D.values for t in tasks])   # (87, 18)
rng = rng_for("E9")


def rules_on(Xs):
    out = {}
    mean = Xs.mean(1)
    sd = Xs.std(1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = mean / (sd / np.sqrt(Xs.shape[1]))
    p_t = np.where(sd > 0, stats.t.sf(t, Xs.shape[1] - 1), np.where(mean > 0, 0.0, 1.0))
    out["t_uncorrected"] = p_t < a
    out["t_bh"] = fdr.bh(p_t, a)
    out["betting_per_exam"] = gates.betting(Xs, 0, a, None, cap).admit
    out["betting_c075"] = gates.betting(Xs, 0, a, None, 0.75).admit
    out["betting_grid"] = gates.betting(Xs, 0, a, None, rule="grid").admit
    out["betting_grid_ebh"] = fdr.ebh(np.exp(gates.betting(Xs, 0, a, None, rule="grid", admit_stop=False).log_wealth), a)
    out["betting_ebh"] = fdr.ebh(np.exp(gates.betting(Xs, 0, a, None, cap, admit_stop=False).log_wealth), a)
    return out


full = rules_on(X)
rows = [{"rule": k, "admitted": int(A.sum()), "of": M, "mean_lift_admitted": X[A].mean() if A.any() else np.nan}
        for k, A in full.items()]
run.save(pd.DataFrame(rows), "e9_full_rules")

# Revision B: what a registry would run on these units. e-LOND (gamma_t ~ t^-1.6) over random arrival orders,
# and Bonferroni at alpha/M, each with the grid bet and with PrPl (cap 1/2); 18 units per skill, no resampling.
_g = fdr.gamma_seq(M, 1.6)
_rng = np.random.default_rng(2026)
_lib = []
for _rule in ("grid", "prpl"):
    _bet = (lambda x, l: gates.betting(x, 0, l, None, rule="grid")) if _rule == "grid" else (lambda x, l: gates.betting(x, 0, l, None, cap))
    for _rep in range(100 if QUICK else 1000):
        _o, _R = _rng.permutation(M), 0
        for _t, _i in enumerate(_o):
            _R += int(_bet(X[_i][None], a * _g[_t] * (_R + 1)).admit[0])
        _lib.append({"rule": _rule, "rep": _rep, "elond_admitted": _R, "bonferroni_admitted": int(_bet(X, a / M).admit.sum())})
run.save(pd.DataFrame(_lib), "e9_library_rules")

# split-half replication over configurations
S = 50 if QUICK else 500
rep = []
for b in range(S):
    perm = rng.permutation(len(cfgs))
    A_, B_ = perm[: len(cfgs) // 2], perm[len(cfgs) // 2:]
    dec = rules_on(X[:, A_])
    liftB = X[:, B_].mean(1)
    for k, A in dec.items():
        rep.append({"split": b, "rule": k, "admitted": int(A.sum()),
                    "mean_lift_B_admitted": liftB[A].mean() if A.any() else np.nan,
                    "share_B_positive": (liftB[A] > 0).mean() if A.any() else np.nan,
                    "mean_lift_B_not_admitted": liftB[~A].mean() if (~A).any() else np.nan})
rp = pd.DataFrame(rep).groupby("rule").mean(numeric_only=True).drop(columns="split").reset_index()
run.save(rp, "e9_split_half_replication")

# calibration: released data vs the outcome model at the same lift
lift_obs = float(X.mean())
p0_obs = float(w.p0.mean())
m = dgp.Model.for_lift(lift_obs, p0_obs, v(P, "sd_theta"), v(P, "sd_lift"), v(P, "rho_crn"), r=3)
Dm = dgp.simulate(rng, m, 1, 200000)[0]
cal = pd.DataFrame([{"source": "skillsbench_released", "mean_D": lift_obs, "sd_D": float(X.std()),
                     "tie_share": float((X == 0).mean()), "p0": p0_obs},
                    {"source": "outcome_model_r3", "mean_D": float(Dm.mean()), "sd_D": float(Dm.std()),
                     "tie_share": float((Dm == 0).mean()), "p0": p0_obs}])
run.save(cal, "e9_calibration")

# variants needed to certify each curated skill, resampling its 18 per-configuration differences
NM = 2000
Rb = 100 if QUICK else 1000
need = []
for i, t in enumerate(tasks):
    if X[i].mean() <= 0:
        need.append({"task": t, "lift": X[i].mean(), "p_cert_600_alpha": 0.0, "p_cert_600_bonf": 0.0, "p_cert_600_bonf_grid": 0.0,
                     "median_n_alpha": np.nan, "median_n_bonf": np.nan})
        continue
    Db = rng.choice(X[i], size=(Rb, NM))
    r1 = gates.betting(Db, 0, a, None, cap)
    r2 = gates.betting(Db, 0, a / M, None, cap)
    g1 = gates.betting(Db, 0, a, None, rule="grid")
    g2 = gates.betting(Db, 0, a / M, None, rule="grid")
    need.append({"task": t, "lift": X[i].mean(),
                 "p_cert_600_alpha": float((r1.admit & (r1.n <= 600)).mean()),
                 "p_cert_600_bonf": float((r2.admit & (r2.n <= 600)).mean()),
                 "median_n_alpha": float(np.median(np.where(r1.admit, r1.n, np.inf))),
                 "median_n_bonf": float(np.median(np.where(r2.admit, r2.n, np.inf))),
                 "median_n_alpha_grid": float(np.median(np.where(g1.admit, g1.n, np.inf))),
                 "median_n_bonf_grid": float(np.median(np.where(g2.admit, g2.n, np.inf))),
                 "p_cert_600_bonf_grid": float((g2.admit & (g2.n <= 600)).mean())})
run.save(pd.DataFrame(need), "e9_variants_needed")
run.done(tasks=M, configs=len(cfgs), splits=S, boot=Rb)
