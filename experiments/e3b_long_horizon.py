"""E3b: e-LOND over a long library history (referee item: E3 has only 10 windows x 20 = 200 exams).

Same outcome model, bets, futility and null share as E3. A history has W_long windows of m candidates
(T = m * W_long exams). Levels use rejections from earlier windows only. Methods:
  elond_infinite   gamma_t proportional to t^-1.6, normalised over the infinite sequence (deployed rule)
  elond_horizon    gamma_t = 1/T (horizon known in advance)
  per_exam_alpha   every exam at level alpha (no library control; reference)
Reports overall FDR and power, power by block of the history (to show decay under t^-1.6), and the
deployed level min(alpha_t, alpha * gamma_1^lin) for a first attempt, to show which term binds."""
import os

import numpy as np
import pandas as pd

from _common import P, QUICK, Run, rng_for, v
from _common import RULE
from nixigate import dgp, fdr, gates

_bet = lambda *a_, **k_: gates.betting(*a_, rule=RULE, **k_)  # NIXI_RULE=grid reruns with the grid bet
_H_ENV = os.environ.get("NIXI_E3B_H")
run = Run("E3b" if _H_ENV in (None, "60") else f"E3b_h{_H_ENV}")  # E24 reruns do not overwrite E3b
base, a, af = v(P, "base_pass_rate"), v(P, "alpha"), v(P, "alpha_futility")
nmax, cap, dfut = int(v(P, "n_max")), v(P, "bet_cap_c"), v(P, "futility_margin")
sdt, sdl, rho = v(P, "sd_theta"), v(P, "sd_lift"), v(P, "rho_crn")
m, pi0, s = int(v(P, "window_size")), v(P, "null_share"), v(P, "elond_gamma_exp")
W = 20 if QUICK else 100
H = 10 if QUICK else int(os.environ.get("NIXI_E3B_H", "60"))  # E24: NIXI_E3B_H=600 for tighter FDR bounds
T = m * W
B = 10                                   # blocks for power-by-position
b0 = dgp.base_logit(base, sdt)
grid = np.linspace(-0.05, 0.40, 181)
g_of = np.array([dgp.solve_g(x, b0, sdt, sdl) for x in grid])


def sim_lifts(rng, lifts):
    g = np.interp(lifts, grid, g_of)[:, None]
    return dgp.simulate(rng, dgp.Model(b0, g, sdt, sdl, rho, 1), len(lifts), nmax)


rng = rng_for("E3b")
recs, blocks, levels = [], [], []
gam_inf = fdr.gamma_seq(T, s)
lin1 = fdr.lineage_levels(1, a, s)[0]
for hh in range(H):
    null = rng.random(T) < pi0
    lift = np.where(null, 0.0, rng.uniform(0.05, 0.25, T))
    D = sim_lifts(rng, lift)
    out = {"per_exam_alpha": _bet(D, 0, a, af, cap, delta_f=dfut).admit}
    for name, gam in (("elond_infinite", gam_inf), ("elond_horizon", np.full(T, 1.0 / T))):
        adm = np.zeros(T, bool)
        R = 0
        for w in range(W):
            sl = slice(w * m, (w + 1) * m)
            lv = a * gam[sl] * (R + 1)
            if name == "elond_infinite" and hh == 0:
                levels += [{"t": int(t), "elond_level": float(x), "lineage_first": lin1} for t, x in zip(range(w * m + 1, (w + 1) * m + 1), lv)]
            adm[sl] = _bet(D[sl], 0, lv, af, cap, delta_f=dfut).admit
            R += int(adm[sl].sum())
        out[name] = adm
    for k, A in out.items():
        recs.append({"history": hh, "method": k, "fdp": (A & null).sum() / max(A.sum(), 1),
                     "power": (A & ~null).sum() / max((~null).sum(), 1), "admitted": int(A.sum())})
        for b in range(B):
            sl = slice(b * T // B, (b + 1) * T // B)
            nn = ~null[sl]
            blocks.append({"history": hh, "method": k, "block": b + 1, "first_exam": b * T // B + 1,
                           "power": (A[sl] & nn).sum() / max(nn.sum(), 1)})
df = pd.DataFrame(recs)
summ = df.groupby("method").agg(fdr=("fdp", "mean"), fdr_se=("fdp", lambda x: x.std() / np.sqrt(len(x))),
                                power=("power", "mean"), admitted=("admitted", "mean")).reset_index()
bl = pd.DataFrame(blocks).groupby(["method", "block", "first_exam"]).power.mean().reset_index()
lvl = pd.DataFrame(levels)
lvl["binding"] = np.where(lvl.elond_level <= a * lvl.lineage_first, "elond", "lineage")
run.save(summ, "e3b_summary")
run.save(bl, "e3b_power_by_block")
run.save(lvl, "e3b_levels_history0")
run.done(T=T, histories=H, share_elond_binding=float((lvl.binding == "elond").mean()))
print(summ.to_string(index=False))
print(bl.pivot(index="block", columns="method", values="power").round(3).to_string())
