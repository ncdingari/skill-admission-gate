"""E8: SWE-Skills-Bench reanalysis (C1, C2, C3): per-skill decisions, multiplicity, replication across
configurations, configuration shopping, and gate false-admission on an empirical null."""
import numpy as np
import pandas as pd
from scipy import stats

from _common import P, QUICK, Run, rng_for, v
from _realdata import signflip_p, swe
from nixigate import fdr, gates

run = Run("E8")
a, af, cap, dfut, nmax = v(P, "alpha"), v(P, "alpha_futility"), v(P, "bet_cap_c"), v(P, "futility_margin"), int(v(P, "n_max"))
d = swe()
rng = rng_for("E8")

# ---------------------------------------------- per (config, skill) cell decisions, 10 batches each
cells = []
for (cfg, sk), g in d.groupby(["config", "skill"]):
    x = g.sort_values("requirement_batch").D.values
    p1 = signflip_p(x)
    p2 = min(1.0, 2 * min(p1, signflip_p(-x)))
    cells.append({"config": cfg, "skill": sk, "n": len(x), "mean": x.mean(), "p_one": p1, "p_two": p2,
                  "bet_c050": bool(gates.betting(x[None], 0, a, None, 0.5).admit[0]),
                  "bet_c075": bool(gates.betting(x[None], 0, a, None, 0.75).admit[0]),
                  "bet_grid": bool(gates.betting(x[None], 0, a, None, rule="grid").admit[0]),
                  "logK_final_grid": float(gates.betting(x[None], 0, a, None, rule="grid", admit_stop=False).log_wealth[0]),
                  "logK_final": float(gates.betting(x[None], 0, a, None, 0.75, admit_stop=False).log_wealth[0]),
                  "max_logK_possible": float(10 * np.log1p(0.75 * 0.5))})
c = pd.DataFrame(cells)
c["golden_source_rule"] = (c.p_two < a) & (c["mean"] > 0)
c["harmful_source_rule"] = (c.p_two < a) & (c["mean"] < 0)
c["one_sided"] = c.p_one < a
c["bh"] = fdr.bh(c.p_one.values, a)
c["ebh"] = fdr.ebh(np.exp(c.logK_final.values), a)
pool = d.groupby(["skill", "config"]).D.mean().unstack()
c["lift_other_configs"] = [pool.loc[s].drop(cf).mean() for s, cf in zip(c.skill, c.config)]
run.save(c, "e8_cells")

rules = ["golden_source_rule", "one_sided", "bh", "bet_c050", "bet_c075", "bet_grid", "ebh"]
summ = []
for rname in rules:
    A = c[rname].values
    summ.append({"rule": rname, "cells": len(c), "admitted_cells": int(A.sum()), "skills": c.skill[A].nunique(),
                 "mean_lift_admitted": c["mean"][A].mean() if A.any() else np.nan,
                 "mean_lift_other_configs": c.lift_other_configs[A].mean() if A.any() else np.nan,
                 "share_other_configs_positive": (c.lift_other_configs[A] > 0).mean() if A.any() else np.nan})
summ.append({"rule": "all_cells", "cells": len(c), "admitted_cells": len(c), "skills": c.skill.nunique(),
             "mean_lift_admitted": c["mean"].mean(), "mean_lift_other_configs": c.lift_other_configs.mean(),
             "share_other_configs_positive": (c.lift_other_configs > 0).mean()})
summ.append({"rule": "harmful_source_rule", "cells": len(c), "admitted_cells": int(c.harmful_source_rule.sum()),
             "skills": c.skill[c.harmful_source_rule].nunique()})

# pooled skill-level tests over 40 units (4 configs x 10 batches)
sk_rows = []
for sk, g in d.groupby("skill"):
    x = g.sort_values(["config", "requirement_batch"]).D.values
    sk_rows.append({"skill": sk, "n": len(x), "mean": x.mean(), "p_one": signflip_p(x, rng, 20000),
                    "logK_final": float(gates.betting(x[None], 0, a, None, 0.5, admit_stop=False).log_wealth[0]),
                    "bet": bool(gates.betting(x[None], 0, a, None, 0.5).admit[0]),
                    "bet_grid": bool(gates.betting(x[None], 0, a, None, rule="grid").admit[0])})
s = pd.DataFrame(sk_rows)
s["bh"] = fdr.bh(s.p_one.values, a)
s["ebh"] = fdr.ebh(np.exp(s.logK_final.values), a)
run.save(s, "e8_skills_pooled")
for rname, A in (("pooled_one_sided", s.p_one < a), ("pooled_bh", s.bh), ("pooled_betting", s.bet), ("pooled_grid", s.bet_grid), ("pooled_ebh", s.ebh)):
    summ.append({"rule": rname, "cells": len(s), "admitted_cells": int(A.sum()), "skills": int(A.sum()),
                 "mean_lift_admitted": s["mean"][A].mean() if A.any() else np.nan})
run.save(pd.DataFrame(summ), "e8_summary")

# ---------------------------------------------- configuration shopping on a sign-flip null
B = 200 if QUICK else 2000
skills = sorted(d.skill.unique())
X = np.stack([np.stack([d[(d.skill == sk) & (d.config == cf)].sort_values("requirement_batch").D.values
                        for cf in sorted(d.config.unique())]) for sk in skills])       # (98, 4, 10)
absX = np.abs(X)
n = X.shape[2]
signs_all = 1 - 2 * ((np.arange(2**n)[:, None] >> np.arange(n)) & 1)
shop = []
for b in range(B):
    Y = absX * rng.choice([-1.0, 1.0], size=absX.shape)
    # exact one-sided sign-flip p per cell, vectorized over the 2^10 sign patterns
    means = np.einsum("kn,scn->sck", signs_all, absX) / n
    p = ((means >= Y.mean(2)[..., None] - 1e-12).mean(2))
    per_cfg = p < a
    shop.append({"rep": b, "per_config_rate": per_cfg.mean(), "any_of_4_rate": per_cfg.any(1).mean(),
                 "first_config_only_rate": per_cfg[:, 0].mean(),
                 "bonferroni_any_rate": (p < a / 4).any(1).mean()})
sh = pd.DataFrame(shop).mean().to_frame().T.drop(columns="rep")
run.save(sh, "e8_config_shopping_null")

# ---------------------------------------------- gates on an empirical null with the released tie structure
R = 2000 if QUICK else 20000
vals = d.D.values
for name, draw in (("empirical_symmetrized", lambda k: rng.choice(np.abs(vals), size=(k, nmax)) *
                    rng.choice([-1.0, 1.0], size=(k, nmax))),):
    Dn = draw(R)
    from nixigate import suite
    des = suite.Design(alpha=a, lift1=v(P, "target_lift"), delta_f=dfut)
    sd_emp = float(np.sqrt((vals ** 2).mean()))
    nf = gates.sample_size(sd_emp, des.lift1, 0, a, des.power)
    res = suite.run_all(Dn, des, {"n_fixed": min(nf, nmax), "sd1": sd_emp, "p1": np.nan}, 1)
    rows = suite.summarize(res, 1, 0.0)
    for r_ in rows:
        r_.update(null=name, tie_share=float((vals == 0).mean()), sd_D=sd_emp, n_fixed=nf)
    run.save(pd.DataFrame(rows), "e8_empirical_null_gates")
run.done(cells=len(c), shop_reps=B, null_reps=R)
