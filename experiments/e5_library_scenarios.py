"""E5: calibrated library scenarios: admissions, false-admission share and agent runs (RQ5)."""
import numpy as np
import pandas as pd

from _common import P, QUICK, Run, rng_for, v
from nixigate import dgp, fdr, gates, suite

run = Run("E5")
base, a, af = v(P, "base_pass_rate"), v(P, "alpha"), v(P, "alpha_futility")
nmax, cap, dfut = int(v(P, "n_max")), v(P, "bet_cap_c"), v(P, "futility_margin")
sdt, sdl, rho, s = v(P, "sd_theta"), v(P, "sd_lift"), v(P, "rho_crn"), v(P, "elond_gamma_exp")
M = 500 if QUICK else 4000
b0 = dgp.base_logit(base, sdt)
grid = np.linspace(-0.32, 0.60, 369)
g_of = np.array([dgp.solve_g(x, b0, sdt, sdl) for x in grid])
d = suite.Design(alpha=a, lift1=v(P, "target_lift"))
pl = suite.plan(d, dgp.Model.for_lift(0, base, sdt, sdl, rho), base)
sd = v(P, "scenario_sd")


def draw(rng, name):
    if name == "curated_like":
        x = rng.normal(v(P, "curated_lift"), sd, M)
    elif name == "selfgen_like":
        x = rng.normal((v(P, "selfgen_lift_lo") + v(P, "selfgen_lift_hi")) / 2, sd, M)
    elif name == "swe_like":
        x = np.where(rng.random(M) < v(P, "swe_null_share"), 0.0, rng.normal(v(P, "swe_golden_lift"), sd / 2, M))
    elif name == "webdev_like":
        x = rng.normal((v(P, "webdev_lift_lo") + v(P, "webdev_lift_hi")) / 2, sd, M)
    return np.clip(x, grid[0], grid[-1])


rows = []
for name in ("curated_like", "selfgen_like", "swe_like", "webdev_like"):
    rng = rng_for("E5" + name)
    lift = draw(rng, name)
    g = np.interp(lift, grid, g_of)[:, None]
    D = dgp.simulate(rng, dgp.Model(b0, g, sdt, sdl, rho, 1), M, max(nmax, pl["n_fixed"]))
    S = D[:, :nmax]
    res = {
        "fixed_t_uncorrected": gates.fixed_t(D, pl["n_fixed"], 0, a),
        "peek_ci": gates.peek_ci(S, 0, v(P, "ci_level_z"), int(v(P, "peek_n_min")), int(v(P, "peek_every"))),
        "betting_per_exam": gates.betting(S, 0, a, af, cap),
        "betting_futility_margin": gates.betting(S, 0, a, af, cap, delta_f=dfut),
        "grid_futility_margin": gates.betting(S, 0, a, af, rule="grid", delta_f=dfut),
        "sprt_discordant": gates.sprt(S, pl["p1"], a, v(P, "sprt_beta")),
    }
    lv = a * fdr.gamma_seq(M, s)
    # e-LOND in arrival order; candidates of one 20-skill window run concurrently
    adm = np.zeros(M, bool)
    nn = np.zeros(M)
    R = 0
    for w0 in range(0, M, 20):
        sl = slice(w0, w0 + 20)
        r_ = gates.betting(S[sl], 0, lv[sl] * (R + 1), af, cap, delta_f=dfut)
        adm[sl], nn[sl] = r_.admit, r_.n
        R += int(r_.admit.sum())
    res["betting_elond"] = gates.GateResult(adm, nn.astype(int), np.zeros(M))
    adm, nn, R = np.zeros(M, bool), np.zeros(M), 0                  # same, with the grid bet
    for w0 in range(0, M, 20):
        sl = slice(w0, w0 + 20)
        r_ = gates.betting(S[sl], 0, lv[sl] * (R + 1), af, rule="grid", delta_f=dfut)
        adm[sl], nn[sl] = r_.admit, r_.n
        R += int(r_.admit.sum())
    res["grid_elond"] = gates.GateResult(adm, nn.astype(int), np.zeros(M))
    for k, x in res.items():
        A = x.admit
        runs = 2 * x.n
        rows.append({"scenario": name, "gate": k, "candidates": M, "true_helpful_share": (lift > 0).mean(),
                     "admitted_share": A.mean(), "false_admit_share_of_admitted": ((lift <= 0) & A).sum() / max(A.sum(), 1),
                     "helpful_admitted_share": (A & (lift > 0)).sum() / max((lift > 0).sum(), 1),
                     "mean_runs_per_candidate": runs.mean(),
                     "runs_per_true_admission": runs.sum() / max((A & (lift > 0)).sum(), 1),
                     "mean_true_lift_admitted": lift[A].mean() if A.any() else np.nan})
    print(name, flush=True)
run.save(pd.DataFrame(rows), "e5_library_scenarios")
run.done(candidates=M, n_fixed=pl["n_fixed"])
