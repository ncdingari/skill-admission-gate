"""E2b: the main power-cost comparison (E2) re-run with the level-tuned grid bet recommended in Section 5
(referee item: the evaluated gate used PrPl c = 1/2 while the theory recommends the grid). Same outcome model,
lifts, alpha, futility margin and N_max as E2's main configuration. Single exams at level alpha; for library
procedures see E3. Reports power, E[N] and false-admission rate for PrPl (cap 1/2), aGRAPA and the grid."""
import numpy as np
import pandas as pd

from _common import P, Run, reps, rng_for, v
from nixigate import dgp, gates

run = Run("E2b")
base, a, af = v(P, "base_pass_rate"), v(P, "alpha"), v(P, "alpha_futility")
nmax, cap, dfut = int(v(P, "n_max")), v(P, "bet_cap_c"), v(P, "futility_margin")
sdt, sdl, rho = v(P, "sd_theta"), v(P, "sd_lift"), v(P, "rho_crn")
lifts = [-0.05, 0.0, 0.05, 0.10, v(P, "target_lift"), 0.15, v(P, "curated_lift"), 0.20, 0.25, 0.30]
lifts = sorted(set(round(x, 4) for x in lifts))
R = reps("mc_reps_power")
rows = []
for lift in lifts:
    m = dgp.Model.for_lift(lift, base, sdt, sdl, rho, 1)
    D = dgp.simulate(rng_for(f"E2b{lift:.4f}"), m, R, nmax)
    for name, rule in (("prpl_c050", "prpl"), ("agrapa", "agrapa"), ("grid", "grid")):
        res = gates.betting(D, 0.0, a, af, cap, rule=rule, delta_f=dfut)
        rows.append({"lift": lift, "bet": name, "admit_rate": res.admit.mean(), "mean_N": res.n.mean(),
                     "se_admit": res.admit.std() / np.sqrt(R), "futility_rate": res.futility.mean()})
    print(lift, "done", flush=True)
df = pd.DataFrame(rows)
run.save(df, "e2b_grid_vs_prpl")
null_max = df[df.lift <= 0].admit_rate.max()
assert null_max <= a + 3 * df.se_admit.max(), "false-admission rate above alpha"
run.done(R=R, null_max=float(null_max))
print(df.pivot(index="lift", columns="bet", values="admit_rate").round(3).to_string())
print(df.pivot(index="lift", columns="bet", values="mean_N").round(1).to_string())
