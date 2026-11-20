"""E3: library-level false admissions over windows, and the resubmission attack (RQ3)."""
import numpy as np
import pandas as pd

from _common import P, QUICK, Run, rng_for, v
from _common import RULE
from nixigate import dgp, fdr, gates

_bet = lambda *a_, **k_: gates.betting(*a_, rule=RULE, **k_)  # NIXI_RULE=grid reruns with the grid bet
run = Run("E3")
base, a, af = v(P, "base_pass_rate"), v(P, "alpha"), v(P, "alpha_futility")
nmax, cap, dfut = int(v(P, "n_max")), v(P, "bet_cap_c"), v(P, "futility_margin")
sdt, sdl, rho = v(P, "sd_theta"), v(P, "sd_lift"), v(P, "rho_crn")
m, W, pi0, s = int(v(P, "window_size")), int(v(P, "n_windows")), v(P, "null_share"), v(P, "elond_gamma_exp")
H = 40 if QUICK else 400
T = m * W
b0 = dgp.base_logit(base, sdt)
grid = np.linspace(-0.05, 0.40, 181)
g_of = np.array([dgp.solve_g(x, b0, sdt, sdl) for x in grid])


def sim_lifts(rng, lifts):
    """Paired differences for a vector of true lifts (one row each)."""
    g = np.interp(lifts, grid, g_of)[:, None]
    mdl = dgp.Model(b0, g, sdt, sdl, rho, 1)
    return dgp.simulate(rng, mdl, len(lifts), nmax)


rng = rng_for("E3A")
recs = []
for hh in range(H):
    null = rng.random(T) < pi0
    lift = np.where(null, 0.0, rng.uniform(0.05, 0.25, T))
    D = sim_lifts(rng, lift)
    out = {}
    per = _bet(D, 0, a, af, cap, delta_f=dfut)
    out["M1_per_exam"] = per.admit
    adm = np.zeros(T, bool)
    for w in range(W):
        sl = slice(w * m, (w + 1) * m)
        adm[sl] = per.admit[sl] & fdr.ebh(np.exp(per.log_wealth[sl]), a)
    out["M2_ebh_stop_1_over_alpha"] = adm.copy()
    st = _bet(D, 0, a / m, af, cap, delta_f=dfut)
    adm[:] = False
    for w in range(W):
        sl = slice(w * m, (w + 1) * m)
        adm[sl] = fdr.ebh(np.exp(st.log_wealth[sl]), a) & st.admit[sl]
    out["M3_ebh_stop_m_over_alpha"] = adm.copy()
    full = _bet(D, 0, a, af, cap, admit_stop=False, delta_f=dfut)
    adm[:] = False
    for w in range(W):
        sl = slice(w * m, (w + 1) * m)
        adm[sl] = fdr.ebh(np.exp(full.log_wealth[sl]), a)
    out["M4_ebh_run_to_nmax"] = adm.copy()
    # online e-LOND: levels use rejections from earlier windows only (decisions known at exam start)
    for name, gam in (("M5_elond_infinite", fdr.gamma_seq(T, s)), ("M6_elond_horizon", np.full(T, 1.0 / T))):
        adm[:] = False
        ns = np.zeros(T)
        R = 0
        for w in range(W):
            sl = slice(w * m, (w + 1) * m)
            lv = a * gam[sl] * (R + 1)
            res = _bet(D[sl], 0, lv, af, cap, delta_f=dfut)
            adm[sl] = res.admit
            ns[sl] = res.n
            R += int(res.admit.sum())
        out[name] = adm.copy()
        out[name + "_n"] = ns
    nmap = {"M1_per_exam": per.n, "M2_ebh_stop_1_over_alpha": per.n, "M3_ebh_stop_m_over_alpha": st.n,
            "M4_ebh_run_to_nmax": full.n, "M5_elond_infinite": out["M5_elond_infinite_n"],
            "M6_elond_horizon": out["M6_elond_horizon_n"]}
    for k, nn in nmap.items():
        A = out[k]
        recs.append({"history": hh, "method": k, "admitted": int(A.sum()), "false": int((A & null).sum()),
                     "fdp": (A & null).sum() / max(A.sum(), 1), "power": (A & ~null).sum() / max((~null).sum(), 1),
                     "mean_n": float(np.mean(nn))})
df = pd.DataFrame(recs)
summ = df.groupby("method").agg(fdr=("fdp", "mean"), fdr_se=("fdp", lambda x: x.std() / np.sqrt(len(x))),
                                power=("power", "mean"), admitted=("admitted", "mean"),
                                false_admits=("false", "mean"), mean_n=("mean_n", "mean")).reset_index()
run.save(summ, "e3_window_fdr")

# ------------------------------------------------ resubmission attack: one null lineage, K attempts
K = int(v(P, "resubmissions"))
Ra = 2000 if QUICK else 20000
rng = rng_for("E3B")
Dn = sim_lifts(rng, np.zeros(Ra * K)).reshape(Ra, K, nmax)
rows = []
lin = fdr.lineage_levels(K, a, s)
for name, levels in (("per_exam_alpha", np.full(K, a)), ("lineage_spending", lin)):
    adm = np.zeros((Ra, K), bool)
    for k in range(K):
        adm[:, k] = _bet(Dn[:, k], 0, levels[k], af, cap, delta_f=dfut).admit
    cum = np.cumsum(adm, axis=1) > 0
    for k in range(K):
        rows.append({"rule": name, "attempt": k + 1, "p_admitted_by": cum[:, k].mean(), "level": levels[k]})
# re-checked CI gate (G3) per attempt, for contrast
adm = np.zeros((Ra, K), bool)
for k in range(K):
    adm[:, k] = gates.peek_ci(Dn[:, k], 0, v(P, "ci_level_z"), int(v(P, "peek_n_min")), int(v(P, "peek_every"))).admit
cum = np.cumsum(adm, axis=1) > 0
for k in range(K):
    rows.append({"rule": "peek_ci_per_attempt", "attempt": k + 1, "p_admitted_by": cum[:, k].mean(), "level": np.nan})
run.save(pd.DataFrame(rows), "e3_resubmission")
run.done(histories=H, candidates=T, attack_reps=Ra)
