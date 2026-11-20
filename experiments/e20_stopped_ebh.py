"""E20 (numbered E19 before 2026-11-20): library control by e-LOND (deployed) against stopped e-BH and e-BH on adjusted suprema.

Same library histories as E3 (seed tag E3A, windows of m candidates, 80% nulls, lifts U(0.05, 0.25)).
Each candidate's e-process is the betting wealth at level alpha (rule from --rule, default grid), with the
futility bet frozen in. Library-level procedures:
  elond         deployed: e-LOND levels, each exam stops at 1/alpha_t (as E3 M5)
  ebh_nmax      every exam runs to N_max (or futility), then e-BH (as E3 M4)
  stopped_ebh   se-BH (Wang et al. 2025): e-BH monitored continuously; an exam stops when it enters the e-BH set or at futility
  ebh_suprema   e-BH on A2(running max) (Tavyrikov et al. 2025), same monitoring
Cost = variants spent per candidate (exams stop at futility, at their own admission, or at the window stop).
  python experiments/e20_stopped_ebh.py [--rule grid|prpl]
"""
import argparse

import numpy as np
import pandas as pd

from _common import P, QUICK, Run, rng_for, v
from nixigate import dgp, fdr, gates

ap = argparse.ArgumentParser()
ap.add_argument("--rule", choices=["grid", "prpl"], default="grid")
RULE = ap.parse_args().rule

run = Run("E20")
base, a, af = v(P, "base_pass_rate"), v(P, "alpha"), v(P, "alpha_futility")
nmax, cap, dfut = int(v(P, "n_max")), v(P, "bet_cap_c"), v(P, "futility_margin")
sdt, sdl, rho = v(P, "sd_theta"), v(P, "sd_lift"), v(P, "rho_crn")
m, W, pi0, s = int(v(P, "window_size")), int(v(P, "n_windows")), v(P, "null_share"), v(P, "elond_gamma_exp")
H = 20 if QUICK else 200
T = m * W
b0 = dgp.base_logit(base, sdt)
grid = np.linspace(-0.05, 0.40, 181)
g_of = np.array([dgp.solve_g(x, b0, sdt, sdl) for x in grid])


def sim_lifts(rng, lifts):
    g = np.interp(lifts, grid, g_of)[:, None]
    return dgp.simulate(rng, dgp.Model(b0, g, sdt, sdl, rho, 1), len(lifts), nmax)


def paths(D, level):
    """e-process paths at `level`, frozen after a futility stop; and the futility mask."""
    X = (1 + D) / 2
    if RULE == "grid":
        logK = gates.grid_logK(X, 0.5, float(level))
    else:
        logK = np.cumsum(np.log1p(gates.prpl_bets(X, level, cap) * (X - 0.5)), axis=1)
    res = gates.betting(D, 0, level, af, cap, rule=RULE, admit_stop=False, delta_f=dfut)
    t = np.arange(D.shape[1])[None, :]
    stopf = np.where(res.futility, res.n - 1, D.shape[1])[:, None]
    logK = np.where(t > stopf, np.take_along_axis(logK, np.minimum(stopf, D.shape[1] - 1), 1), logK)
    return np.exp(logK), t >= stopf, res


rng = rng_for("E3A")
recs = []
for hh in range(H):
    null = rng.random(T) < pi0
    D = sim_lifts(rng, np.where(null, 0.0, rng.uniform(0.05, 0.25, T)))
    out, ncost = {k: np.zeros(T, bool) for k in ("elond", "ebh_nmax", "stopped_ebh", "ebh_suprema")}, {}
    for k in out:
        ncost[k] = np.zeros(T)
    R = 0
    gam = fdr.gamma_seq(T, s)
    for w in range(W):
        sl = slice(w * m, (w + 1) * m)
        for j in range(sl.start, sl.stop):
            lv = a * gam[j] * (R + 1)
            r = gates.betting(D[j:j + 1], 0, lv if RULE == "grid" else np.array([lv]), af, cap, rule=RULE,
                              delta_f=dfut)
            out["elond"][j], ncost["elond"][j] = r.admit[0], r.n[0]
        R += int(out["elond"][sl].sum())
        E, done, res = paths(D[sl], a)
        out["ebh_nmax"][sl] = fdr.ebh(E[:, -1], a)
        ncost["ebh_nmax"][sl] = res.n
        for name, fn in (("stopped_ebh", fdr.stopped_ebh), ("ebh_suprema", fdr.ebh_suprema)):
            rej, st = fn(E, a, done)
            out[name][sl] = rej
            ncost[name][sl] = st + 1
    for k, A in out.items():
        recs.append({"history": hh, "method": k, "fdp": (A & null).sum() / max(A.sum(), 1),
                     "power": (A & ~null).sum() / max((~null).sum(), 1), "mean_n": float(ncost[k].mean())})
df = pd.DataFrame(recs)
summ = df.groupby("method").agg(fdr=("fdp", "mean"), fdr_se=("fdp", lambda x: x.std() / np.sqrt(len(x))),
                                power=("power", "mean"), mean_n=("mean_n", "mean")).reset_index()
summ.insert(0, "rule", RULE)
run.save(summ, f"e20_library_{RULE}")
print(summ.to_string(index=False))
run.done(histories=H, candidates=T, rule=RULE)
