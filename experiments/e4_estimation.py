"""E4: winner's curse of the reported lift and coverage of the anytime-valid lower bound (RQ4)."""
import numpy as np
import pandas as pd

from _common import P, QUICK, Run, rng_for, v
from nixigate import dgp, gates, suite

run = Run("E4")
base, a, af = v(P, "base_pass_rate"), v(P, "alpha"), v(P, "alpha_futility")
nmax, cap, dfut = int(v(P, "n_max")), v(P, "bet_cap_c"), v(P, "futility_margin")
sdt, sdl, rho = v(P, "sd_theta"), v(P, "sd_lift"), v(P, "rho_crn")
R = 300 if QUICK else 3000
d = suite.Design(alpha=a, lift1=v(P, "target_lift"))
pl = suite.plan(d, dgp.Model.for_lift(0, base, sdt, sdl, rho), base)
rows = []
for lift in [0.025, 0.05, 0.075, 0.10, v(P, "curated_lift"), 0.25]:
    m = dgp.Model.for_lift(lift, base, sdt, sdl, rho)
    D = dgp.simulate(rng_for(f"E4{lift:.4f}"), m, R, max(nmax, pl["n_fixed"]))
    bet = gates.betting(D[:, :nmax], 0, a, af, cap, delta_f=dfut)
    fx = gates.fixed_t(D, pl["n_fixed"], 0, a)
    lcb = np.full(R, np.nan)
    cover_path = np.zeros(R, bool)
    for j in range(R):
        path = gates.lcb_path(D[j, :nmax], a, cap)
        cover_path[j] = (path <= lift + 1e-9).all()
        lcb[j] = path[bet.n[j] - 1]
    A = bet.admit
    rows.append({"true_lift": lift, "reps": R, "power_bet": A.mean(), "power_fixed": fx.admit.mean(),
                 "mean_dhat_admitted_bet": bet.dbar[A].mean(), "bias_bet": bet.dbar[A].mean() - lift,
                 "rel_bias_bet": (bet.dbar[A].mean() - lift) / lift,
                 "mean_dhat_admitted_fixed": fx.dbar[fx.admit].mean(), "bias_fixed": fx.dbar[fx.admit].mean() - lift,
                 "mean_lcb_admitted": lcb[A].mean(), "lcb_cover_at_stop": (lcb[A] <= lift + 1e-9).mean(),
                 "time_uniform_cover": cover_path.mean(), "mean_n_admitted": bet.n[A].mean()})
    print(lift, "done", flush=True)
run.save(pd.DataFrame(rows), "e4_estimation")
run.done(reps=R)
