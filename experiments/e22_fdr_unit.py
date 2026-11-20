"""E22: the unit of FDR and lineage error (Sai's Plan 2, E22).

Registries of lineages with several versions, under the convention of Section 4 ("The unit of error"): an exam
tests one version against its comparator; a lineage is admitted at most once per comparator, and later versions are
replacements tested against the admitted version; every exam uses an e-LOND index and a lineage attempt index.
Deployed level min(alpha_t, alpha * gamma^lin_k); e-LOND counts admissions known when a window starts.

Lineage types (lifts are relative to the version's comparator):
  null    1 + Geom versions (mean 3, at most 8), every one useless (Delta = 0)
  honest  J - 1 useless versions (J ~ Geom, mean 2), then a helpful version (Delta ~ U(0.05, 0.25)), then 0-2
          replacement edits against the admitted version: each useless (Delta = 0) or a further improvement
          (Delta ~ U(0.05, 0.15)) with probability 1/2. Edits arrive only after the lineage's helpful version is decided.
Each window of m = 20 exams takes the next pending version of up to m lineages (at most one per lineage per window),
oldest lineage first, opening new lineages (null with probability 0.6) to fill the window. 400 histories of 10
windows. Grid bet, alpha = 0.05, futility at delta_f = 0.05, N_max 600.

Reports, per history and averaged: exam-level FDR (false admissions / admissions; what e-LOND controls), the share
of null lineages with any admission (lineage error, R3), the share of honest lineages with a false admission among
their useless versions (mixed-lineage error), and the worked two-version example from history 0.
"""
import json

import numpy as np
import pandas as pd

from _common import QUICK, Run, P, rng_for, v
from nixigate import dgp, fdr, gates

run = Run("E22")
a, af = v(P, "alpha"), v(P, "alpha_futility")
nmax, dfut = int(v(P, "n_max")), v(P, "futility_margin")
base, sdt, sdl, rho = v(P, "base_pass_rate"), v(P, "sd_theta"), v(P, "sd_lift"), v(P, "rho_crn")
S = v(P, "elond_gamma_exp")
m, W = 20, 10
H = 40 if QUICK else 400
b0 = dgp.base_logit(base, sdt)
grid = np.linspace(-0.05, 0.40, 181)
g_of = np.array([dgp.solve_g(x, b0, sdt, sdl) for x in grid])
gam_lin = fdr.gamma_seq(64, S)


def sim(rng, lifts):
    g = np.interp(lifts, grid, g_of)[:, None]
    return dgp.simulate(rng, dgp.Model(b0, g, sdt, sdl, rho, 1), len(lifts), nmax)


def new_lineage(rng, lid):
    if rng.random() < 0.6:
        k = min(1 + rng.geometric(1 / 3), 8)
        return {"id": lid, "type": "null", "pre": [0.0] * k, "edits": [], "admitted": False, "k": 0, "log": []}
    J = rng.geometric(1 / 2)
    edits = [0.0 if rng.random() < 0.5 else float(rng.uniform(0.05, 0.15)) for _ in range(rng.integers(0, 3))]
    return {"id": lid, "type": "honest", "pre": [0.0] * (J - 1) + [float(rng.uniform(0.05, 0.25))],
            "edits": edits, "admitted": False, "k": 0, "log": []}


def next_lift(L):
    """Lift of the lineage's next version, or None if it has no more versions."""
    if not L["admitted"]:
        n_pre_done = sum(1 for e in L["log"] if e["phase"] == "pre")
        if n_pre_done < len(L["pre"]):
            return "pre", L["pre"][n_pre_done]
        return None
    n_ed = sum(1 for e in L["log"] if e["phase"] == "edit")
    return ("edit", L["edits"][n_ed]) if n_ed < len(L["edits"]) else None


rng = rng_for("E22")
recs, example = [], None
for hh in range(H):
    lineages, active, R, t = [], [], 0, 0
    for w in range(W):
        batch = []
        for L in list(active):
            if len(batch) == m:
                break
            nx = next_lift(L)
            if nx is None:
                active.remove(L)
                continue
            batch.append((L, *nx))
        while len(batch) < m:
            L = new_lineage(rng, len(lineages))
            lineages.append(L)
            active.append(L)
            batch.append((L, *next_lift(L)))
        levels, lifts = [], []
        for L, phase, lift in batch:
            t += 1
            L["k"] += 1
            levels.append(min(fdr.elond_levels(R, t, a, S), a * gam_lin[L["k"] - 1]))
            lifts.append(lift)
        D = sim(rng, np.array(lifts))
        res = gates.betting(D, 0.0, np.array(levels), af, rule="grid", delta_f=dfut)
        for (L, phase, lift), adm, lv, n in zip(batch, res.admit, levels, res.n):
            L["log"].append({"phase": phase, "lift": lift, "null": lift <= 0.0, "admit": bool(adm), "level": lv,
                             "attempt": L["k"], "n": int(n)})
            if adm and not L["admitted"]:
                L["admitted"] = True
        R += int(res.admit.sum())                     # admissions known at the next window's start
    ex = [e for L in lineages for e in L["log"]]
    adm = sum(e["admit"] for e in ex)
    false_adm = sum(e["admit"] and e["null"] for e in ex)
    nulls = [L for L in lineages if L["type"] == "null" and L["log"]]
    hon = [L for L in lineages if L["type"] == "honest" and L["log"]]
    recs.append({"history": hh, "exams": len(ex), "admissions": adm, "false_admissions": false_adm,
                 "fdp_exam": false_adm / max(adm, 1),
                 "null_lineages": len(nulls), "null_lineage_admitted": float(np.mean([any(e["admit"] for e in L["log"]) for L in nulls])) if nulls else np.nan,
                 "honest_lineages": len(hon),
                 "honest_false_admit": float(np.mean([any(e["admit"] and e["null"] for e in L["log"]) for L in hon])) if hon else np.nan,
                 "honest_helpful_admitted": float(np.mean([any(e["admit"] and not e["null"] for e in L["log"]) for L in hon])) if hon else np.nan,
                 "lineages_two_admissions": sum(sum(e["admit"] for e in L["log"]) >= 2 for L in lineages)})
    if example is None:
        for L in lineages:
            if L["type"] == "honest" and L["admitted"] and len(L["log"]) >= 2 and any(e["phase"] == "edit" for e in L["log"]):
                example = {"history": hh, "lineage": L["id"], "versions": L["log"]}
                break

df = pd.DataFrame(recs)
summ = pd.DataFrame([{"histories": H, "fdr_exam": df.fdp_exam.mean(), "fdr_exam_se": df.fdp_exam.std() / np.sqrt(H),
                      "null_lineage_error": df.null_lineage_admitted.mean(),
                      "null_lineage_error_se": df.null_lineage_admitted.std() / np.sqrt(H),
                      "honest_lineage_false_admit": df.honest_false_admit.mean(),
                      "honest_helpful_admitted": df.honest_helpful_admitted.mean(),
                      "mean_exams": df.exams.mean(), "mean_admissions": df.admissions.mean(),
                      "share_histories_with_two_admission_lineage": (df.lineages_two_admissions > 0).mean()}])
run.save(df, "e22_histories")
run.save(summ, "e22_summary")
(run.dir / "e22_example.json").write_text(json.dumps(example, indent=1))
print(summ.T.to_string())
print(json.dumps(example, indent=1))
run.done(histories=H)
