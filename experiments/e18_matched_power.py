"""E18: matched-power comparison (Sai's Plan 2, E18).

Each gate is tuned to the same power (80% or 95%) at the design lift (0.10) and the curated lift (0.166):
  fixed_t   paired t-test, n from the planning formula (gates.sample_size) at the target power
  sprt      discordant-pair SPRT (r = 1, delta = 0 only) with beta = 1 - target
  msprt     mixture SPRT truncated at N_max, the smallest N_max with power >= target
  prpl      betting, PrPl c = 1/2, futility margin, truncated at the smallest N_max with power >= target
  grid      betting, level-tuned grid (deployed), futility margin, same tuning
Truncated gates are tuned on one long run (columns = 3,000): a cap at N admits exactly the exams that admitted by N,
and its cost is E[min(n, N)]. Models: default and moment-matched (E14 fit); r = 1 and r = 3; 4,000 replicates per
cell. Reports the tuned setting, achieved power, expected variants and agent calls (2 arms x r) under the
alternative, and expected variants under nulls at Delta = 0 and Delta = -0.05 with the same setting.
"""
import numpy as np
import pandas as pd

from _common import QUICK, P, Run, rng_for, v
from nixigate import dgp, gates

run = Run("E18")
a, af, dfut = v(P, "alpha"), v(P, "alpha_futility"), v(P, "futility_margin")
base, sdt, sdl, rho = v(P, "base_pass_rate"), v(P, "sd_theta"), v(P, "sd_lift"), v(P, "rho_crn")
tau, nmin = v(P, "msprt_tau"), int(v(P, "peek_n_min"))
R = 400 if QUICK else 4000
NC = 1500 if QUICK else 3000
MODELS = {"default": (sdt, sdl, rho), "moment_matched": (2.25, 2.25, 0.95)}   # E14 fit (rho at the grid edge; see E23)


def run_trunc(D, kind):
    if kind == "msprt":
        return gates.msprt(D, 0.0, a, tau, nmin)
    if kind == "prpl":
        return gates.betting(D, 0.0, a, af, cap=0.5, delta_f=dfut)
    return gates.betting(D, 0.0, a, af, rule="grid", delta_f=dfut)


def tune(res, target):
    """Smallest cap N with P(admit by N) >= target; None if not reached within NC."""
    adm_n = np.where(res.admit, res.n, np.inf)
    for N in range(10, NC + 1, 5):
        if (adm_n <= N).mean() >= target:
            return N
    return None


rows = []
for mname, (st, sg, rh) in MODELS.items():
    for r in (1, 3):
        for lift in (0.10, 0.166):
            m1 = dgp.Model.for_lift(lift, base, st, sg, rh, r)
            sd1 = dgp.sd_D(m1)
            rng = rng_for(f"E18{mname}{r}{lift}")
            D1 = dgp.simulate(rng, m1, R, NC)
            nulls = {d0: dgp.simulate(rng, dgp.Model.for_lift(d0, base, st, sg, rh, r), R, NC) for d0 in (0.0, -0.05)}
            full = {k: run_trunc(D1, k) for k in ("msprt", "prpl", "grid")}
            fulln = {(k, d0): run_trunc(Dn, k) for k in ("msprt", "prpl", "grid") for d0, Dn in nulls.items()}
            for target in (0.80, 0.95):
                # fixed-n
                n = gates.sample_size(sd1, lift, 0.0, a, target)
                if n <= NC:
                    res = gates.fixed_t(D1, n, 0.0, a)
                    row = {"gate": "fixed_t", "setting": f"n={n}", "power": res.admit.mean(), "variants_alt": float(n)}
                    for d0, Dn in nulls.items():
                        row[f"variants_null_{d0}"] = float(n)
                        row[f"admit_null_{d0}"] = gates.fixed_t(Dn, n, 0.0, a).admit.mean()
                    rows.append({"model": mname, "r": r, "lift": lift, "target": target, **row})
                # SPRT
                if r == 1:
                    p1 = dgp.discordant_share(m1)
                    res = gates.sprt(D1, p1, a, 1 - target)
                    row = {"gate": "sprt", "setting": f"beta={1 - target:.2f}", "power": res.admit.mean(),
                           "variants_alt": res.n.mean()}
                    for d0, Dn in nulls.items():
                        rn = gates.sprt(Dn, p1, a, 1 - target)
                        row[f"variants_null_{d0}"] = rn.n.mean()
                        row[f"admit_null_{d0}"] = rn.admit.mean()
                    rows.append({"model": mname, "r": r, "lift": lift, "target": target, **row})
                # truncated sequential gates
                for k, res in full.items():
                    N = tune(res, target)
                    if N is None:
                        rows.append({"model": mname, "r": r, "lift": lift, "target": target, "gate": k,
                                     "setting": f"N_max>{NC}", "power": np.nan})
                        continue
                    adm_n = np.where(res.admit, res.n, np.inf)
                    row = {"gate": k, "setting": f"N_max={N}", "power": (adm_n <= N).mean(),
                           "variants_alt": np.minimum(res.n, N).mean()}
                    for d0 in nulls:
                        rn = fulln[(k, d0)]
                        row[f"variants_null_{d0}"] = np.minimum(rn.n, N).mean()
                        row[f"admit_null_{d0}"] = (np.where(rn.admit, rn.n, np.inf) <= N).mean()
                    rows.append({"model": mname, "r": r, "lift": lift, "target": target, **row})
            print(mname, r, lift, "done", flush=True)

df = pd.DataFrame(rows)
df["calls_alt"] = df.variants_alt * 2 * df.r
run.save(df, "e18_matched_power")
print(df[["model", "r", "lift", "target", "gate", "setting", "power", "variants_alt", "variants_null_0.0", "admit_null_0.0"]].round(3).to_string())
run.done(reps=R, cols=NC)
