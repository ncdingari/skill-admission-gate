"""E2: power and expected variants E[N] versus true lift, with ablations (RQ2)."""
import numpy as np
import pandas as pd

from _common import P, Run, reps, rng_for, v
from _common import RULE
from nixigate import dgp, gates, suite

_bet = lambda *a_, **k_: gates.betting(*a_, rule=RULE, **k_)  # NIXI_RULE=grid reruns with the grid bet
run = Run("E2")
base = v(P, "base_pass_rate")
d0 = suite.Design(alpha=v(P, "alpha"), alpha_f=v(P, "alpha_futility"), n_max=int(v(P, "n_max")),
                  lift1=v(P, "target_lift"), power=v(P, "power_target"), n_min=int(v(P, "peek_n_min")),
                  every=int(v(P, "peek_every")), z=v(P, "ci_level_z"), beta=v(P, "sprt_beta"),
                  tau=v(P, "msprt_tau"), cap=v(P, "bet_cap_c"), cap_alt=v(P, "bet_cap_c_alt"),
                  delta_f=v(P, "futility_margin"))
sdt, sdl, rho = v(P, "sd_theta"), v(P, "sd_lift"), v(P, "rho_crn")
lifts = [-0.05, 0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, v(P, "curated_lift"), 0.20, 0.25, 0.30]
R = reps("mc_reps_power")

configs = [("main", 1, 0.0, d0.n_max), ("r_3", int(v(P, "r_sens")), 0.0, d0.n_max),
           ("margin_0.05", 1, v(P, "delta_sens"), d0.n_max), ("nmax_300", 1, 0.0, 300), ("nmax_1200", 1, 0.0, 1200)]
rows = []
for cname, r, delta, nmax in configs:
    d = suite.Design(**{**d0.__dict__, "delta": delta, "n_max": nmax})
    nuis = dgp.Model.for_lift(0.0, base, sdt, sdl, rho, r)
    pl = suite.plan(d, nuis, base)
    for lift in lifts:
        m = dgp.Model.for_lift(lift, base, sdt, sdl, rho, r)
        rng = rng_for(f"E2{cname}{lift:.4f}")
        D = dgp.simulate(rng, m, R, max(nmax, pl["n_fixed"]))
        res = suite.run_all(D, d, pl, r) if cname == "main" else {
            "G1_fixed_t": gates.fixed_t(D, pl["n_fixed"], d.delta, d.alpha),
            "G6_bet_c050": _bet(D[:, :nmax], d.delta, d.alpha, d.alpha_f, cap=d.cap),
            "G7_grid": gates.betting(D[:, :nmax], d.delta, d.alpha, d.alpha_f, rule="grid")}
        if cname == "main":
            res["G6_bet_c050_nofutility"] = _bet(D[:, :nmax], d.delta, d.alpha, None, cap=d.cap)
        for row in suite.summarize(res, r, lift):
            rows.append({"config": cname, "r": r, "delta": delta, "n_max": nmax, "n_fixed": pl["n_fixed"],
                         "sd1": pl["sd1"], **row})
    print(cname, "done", flush=True)

df = pd.DataFrame(rows)
fx = df[df.gate == "G1_fixed_t"].set_index(["config", "true_lift"])["mean_runs"]
df["runs_vs_fixed"] = [r / fx[(c, l)] for c, l, r in zip(df.config, df.true_lift, df.mean_runs)]
run.save(df, "e2_power_cost")
run.done(reps=R, lifts=len(lifts))
