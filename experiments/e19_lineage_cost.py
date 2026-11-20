"""E19: total lineage cost under the deployed rule (Sai's Plan 2, E19). Measures what Theorem thm:price excludes:
useless versions, N_max truncation, futility, and the interaction with e-LOND.

An honest lineage has J - 1 useless versions (Delta_u in {0, -0.05}) and then a helpful version (Delta_h in {0.10,
0.166}); J ~ pi in {geometric with mean 3, Zipf with exponent 2 truncated at 20, uniform on 1..20}. A null lineage
submits J ~ pi useless versions (Delta = 0). Exam k of a lineage runs at the deployed level
min(alpha_t, alpha * q_k), alpha_t = alpha * gamma_t * (R_t + 1) the e-LOND level at registry index t, with
the lineage opening at registry age A in {50, 200, 500, 2000}, one exam per 10 registry indices, and R_t = 0.1 t
earlier admissions (the admission rate of E3b). Schedules q: zeta(1.6) (deployed), universal (Corollary
cor:universal), uniform on 1..20 (minimax over Pi_20), and the oracle q = pi (the limit of the learned schedule).
Grid bet, futility at delta_f = 0.05 (alpha_f = 0.10), N_max = 600. A lineage stops at its first admission; a
helpful exam that misses ends the lineage unadmitted.
Reports per cell: mean variants per lineage (useless, helpful, total), honest and null lineage admission
probabilities, the share of exams where the lineage term sets the level, and the helpful-exam cost against the
prediction (log(1/alpha) + E log(1/q_J)) / g, with g the measured slope of helpful cost in log(1/level).
"""
import numpy as np
import pandas as pd
from scipy.special import zeta

from _common import QUICK, P, Run, rng_for, v
from nixigate import dgp, fdr, gates

run = Run("E19")
a, af, dfut, nmax = v(P, "alpha"), v(P, "alpha_futility"), v(P, "futility_margin"), int(v(P, "n_max"))
base, sdt, sdl, rho = v(P, "base_pass_rate"), v(P, "sd_theta"), v(P, "sd_lift"), v(P, "rho_crn")
L = 300 if QUICK else 2000
KMAX = 20
k = np.arange(1, KMAX + 1)
PI = {"geometric_mean3": (1 / 3) * (2 / 3) ** (k - 1), "zipf2": k ** -2.0, "uniform20": np.ones(KMAX)}
PI = {n: p / p.sum() for n, p in PI.items()}
c_u = 1 / sum(1 / ((j + 1) * np.log(j + 1) ** 2) for j in range(1, 200000))
SCHED = {"zeta1.6": k ** -1.6 / zeta(1.6), "universal": c_u / ((k + 1) * np.log(k + 1) ** 2), "uniform20": np.ones(KMAX) / KMAX}
AGES = (50, 200, 500, 2000)
SPACING, ADM_RATE = 10, 0.1
gamma = fdr.gamma_seq(AGES[-1] + SPACING * KMAX + 1, v(P, "elond_gamma_exp"))
_models = {}


def model(lift):
    if lift not in _models:
        _models[lift] = dgp.Model.for_lift(lift, base, sdt, sdl, rho, 1)
    return _models[lift]


def levels(age, q):
    t = age + SPACING * (k - 1)
    a_t = a * gamma[t - 1] * (np.floor(ADM_RATE * t) + 1)
    lin = a * q
    return np.minimum(a_t, lin), lin < a_t


def exam(rng, lift, level, n):
    D = dgp.simulate(rng, model(lift), n, nmax)
    res = gates.betting(D, 0.0, float(level), af, rule="grid", delta_f=dfut)
    return res.admit, res.n


rows = []
slope_rows = []
for piname, pi in PI.items():
    for sname, q0 in [*SCHED.items(), ("oracle_pi", None)]:
        q = pi if q0 is None else q0
        for age in AGES:
            lv, lin_binds = levels(age, q)
            for du in (0.0, -0.05):
                for dh in (0.10, 0.166):
                    rng = rng_for(f"E19{piname}{sname}{age}{du}{dh}")
                    J = rng.choice(k, size=L, p=pi)
                    alive = np.ones(L, bool)
                    adm_useless = np.zeros(L, bool)
                    adm_help = np.zeros(L, bool)
                    var_u = np.zeros(L)
                    var_h = np.zeros(L)
                    lin_exams = tot_exams = 0
                    for kk in range(1, KMAX + 1):
                        u = alive & (J > kk)
                        h = alive & (J == kk)
                        for mask, lift in ((u, du), (h, dh)):
                            if mask.any():
                                ad, n = exam(rng, lift, lv[kk - 1], int(mask.sum()))
                                idx = np.flatnonzero(mask)
                                if lift == dh:
                                    var_h[idx] += n
                                    adm_help[idx] = ad
                                else:
                                    var_u[idx] += n
                                    adm_useless[idx] = ad
                                tot_exams += mask.sum()
                                lin_exams += mask.sum() * lin_binds[kk - 1]
                        alive &= ~(h | adm_useless)          # lineage ends at its helpful exam or a false admission
                    # null lineage: J useless versions at Delta = 0
                    Jn = rng.choice(k, size=L, p=pi)
                    alive_n = np.ones(L, bool)
                    adm_n = np.zeros(L, bool)
                    for kk in range(1, KMAX + 1):
                        mask = alive_n & (Jn >= kk)
                        if mask.any():
                            ad, _ = exam(rng, 0.0, lv[kk - 1], int(mask.sum()))
                            adm_n[np.flatnonzero(mask)] |= ad
                            alive_n &= ~adm_n
                    Hpi = float(-(pi * np.log(pi)).sum())
                    rows.append({"pi": piname, "schedule": sname, "age": age, "delta_useless": du, "delta_helpful": dh,
                                 "variants_useless": var_u.mean(), "variants_helpful": var_h.mean(),
                                 "variants_total": (var_u + var_h).mean(),
                                 "honest_admitted": adm_help.mean(), "honest_false_admit": adm_useless.mean(),
                                 "null_lineage_admitted": adm_n.mean(), "share_lineage_binds": lin_exams / tot_exams,
                                 "H_pi": Hpi, "E_log_inv_q": float((pi * np.log(1 / q)).sum()),
                                 "E_log_inv_level_J": float((pi * np.log(1 / lv)).sum())})
            print(piname, sname, age, "done", flush=True)

# slope of helpful-exam cost in log(1/level), per helpful lift (Kelly-like rate g of the grid bet)
for dh in (0.10, 0.166):
    rng = rng_for(f"E19slope{dh}")
    pts = []
    for lvl in np.geomspace(a, a * 1e-4, 9):
        _, n = exam(rng, dh, lvl, 1500 if not QUICK else 200)
        pts.append((np.log(1 / lvl), n.mean()))
    x, y = np.array(pts).T
    b1, b0_ = np.polyfit(x, y, 1)
    slope_rows.append({"delta_helpful": dh, "slope_variants_per_nat": b1, "intercept": b0_})
df = pd.DataFrame(rows)
sl = pd.DataFrame(slope_rows).set_index("delta_helpful")
df["predicted_helpful"] = df.apply(lambda r: sl.intercept[r.delta_helpful] + sl.slope_variants_per_nat[r.delta_helpful] * r.E_log_inv_level_J, axis=1)
run.save(df, "e19_lineage_cost")
run.save(sl.reset_index(), "e19_slope")
print(df[(df.age == 500) & (df.delta_useless == 0)][["pi", "schedule", "delta_helpful", "variants_useless", "variants_helpful",
                                                      "predicted_helpful", "honest_admitted", "null_lineage_admitted",
                                                      "share_lineage_binds"]].round(3).to_string())
run.done(lineages=L)
