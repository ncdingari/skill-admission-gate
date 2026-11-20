"""E1: false-admission rate of each gate under the null (RQ1)."""
import numpy as np
import pandas as pd

from _common import P, Run, reps, rng_for, v
from nixigate import dgp, suite

run = Run("E1")
base = v(P, "base_pass_rate")
d0 = suite.Design(alpha=v(P, "alpha"), alpha_f=v(P, "alpha_futility"), n_max=int(v(P, "n_max")),
                  lift1=v(P, "target_lift"), power=v(P, "power_target"), n_min=int(v(P, "peek_n_min")),
                  every=int(v(P, "peek_every")), z=v(P, "ci_level_z"), beta=v(P, "sprt_beta"),
                  tau=v(P, "msprt_tau"), cap=v(P, "bet_cap_c"), cap_alt=v(P, "bet_cap_c_alt"),
                  delta_f=v(P, "futility_margin"))
sdt, sdl, rho = v(P, "sd_theta"), v(P, "sd_lift"), v(P, "rho_crn")

# (label, lift, sd_theta, sd_lift, rho, r, delta, drift, kind)
cells = [
    ("boundary", 0.0, sdt, sdl, rho, 1, 0.0, 0.0, "model"),
    ("placebo_webdev", v(P, "webdev_lift_hi"), sdt, sdl, rho, 1, 0.0, 0.0, "model"),
    ("two_point_worst", 0.0, sdt, sdl, rho, 1, 0.0, 0.0, "two_point"),
    ("sd_theta_0.5", 0.0, 0.5, sdl, rho, 1, 0.0, 0.0, "model"),
    ("sd_theta_3.0", 0.0, 3.0, sdl, rho, 1, 0.0, 0.0, "model"),
    ("sd_lift_0", 0.0, sdt, 0.0, rho, 1, 0.0, 0.0, "model"),
    ("sd_lift_1", 0.0, sdt, 1.0, rho, 1, 0.0, 0.0, "model"),
    ("rho_0", 0.0, sdt, sdl, 0.0, 1, 0.0, 0.0, "model"),
    ("rho_0.9", 0.0, sdt, sdl, 0.9, 1, 0.0, 0.0, "model"),
    ("r_3", 0.0, sdt, sdl, rho, int(v(P, "r_sens")), 0.0, 0.0, "model"),
    ("margin_0.05", v(P, "delta_sens"), sdt, sdl, rho, 1, v(P, "delta_sens"), 0.0, "model"),
    ("stale_baseline_drift", 0.0, sdt, sdl, 0.0, 1, 0.0, v(P, "drift_logit"), "model"),
]

rows = []
R = reps("mc_reps")
for (label, lift, st, sl, rh, r, delta, drift, kind) in cells:
    d = suite.Design(**{**d0.__dict__, "delta": delta})
    m = dgp.Model.for_lift(lift, base, st, sl, rh, r, drift=drift)
    nuis = m.with_(drift=0.0)
    pl = suite.plan(d, nuis, base)
    ncol = max(d.n_max, pl["n_fixed"])
    rng = rng_for("E1" + label)
    chunks = []
    for c0 in range(0, R, 5000):
        k = min(5000, R - c0)
        D = dgp.simulate(rng, m, k, ncol) if kind == "model" else dgp.simulate_two_point(rng, lift, k, ncol)
        chunks.append(suite.run_all(D, d, pl, r))
    merged = {g: type(chunks[0][g])(*[np.concatenate([getattr(c[g], f) for c in chunks])
                                      if getattr(chunks[0][g], f) is not None else None
                                      for f in ("admit", "n", "dbar", "log_wealth", "futility")])
              for g in chunks[0]}
    for row in suite.summarize(merged, r, lift):
        rows.append({"cell": label, "delta": delta, "sd_theta": st, "sd_lift": sl, "rho_crn": rh, "r": r,
                     "drift_logit": drift, "n_fixed": pl["n_fixed"], **row})
    print(label, "done", flush=True)

df = pd.DataFrame(rows)
run.save(df, "e1_false_admission")
run.done(cells=len(cells), reps=R)
