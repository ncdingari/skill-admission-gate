"""Pre-registered power plan for live study 5 (E25), from the paired Haiku calibration (written before commit).

Exams: one lineage per skill (attempt 1), in the order of SHA-256(skill name), fixed before calibration.
For each skill, the paired outcome law (P(D=+1), P(D=-1)) is estimated from its 20 public calibration pairs;
the conservative law lowers the lift by one standard error and uses at least the placebo's two-way discordance.
N_max is the smallest of 600, 900, 1200 at which every skill with calibration lift >= 0.05 has power >= 0.80
at its own conservative law and e-LOND level (no earlier admissions assumed); 600 if no skill qualifies, 1200
if none of the three reaches 0.80. Writes pilot2/power_plan_study5.json and refuses to overwrite after commit.
  python pilot2/study5_power.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))
sys.path.insert(0, str(ROOT))
from nixigate import fdr, gates  # noqa: E402
import family_dm as family  # noqa: E402

ALPHA, ALPHA_F, DELTA_F, CAP, S = 0.05, 0.10, 0.05, 0.5, 1.6
OUT = ROOT / "power_plan_study5.json"
LEDGER = ROOT.parent / "pilot" / "ledger.jsonl"
REPS = 2000
ORDER = sorted(family.SKILL_NAMES, key=lambda n: hashlib.sha256(n.encode()).hexdigest())


def level(t, k, R=0):
    g = fdr.gamma_seq(max(t, k), S)
    return float(min(ALPHA * g[t - 1] * (R + 1), ALPHA * g[k - 1]))


def laws():
    recs = [json.loads(l) for l in (ROOT / "transcripts5" / "calibration.jsonl").read_text().splitlines()]
    out = {}
    for name in family.SKILL_NAMES:
        D = [int(r["with_skill"]["pass"]) - int(r["no_skill"]["pass"]) for r in recs
             if r["skill"] == name and r["with_skill"]["ok"] and r["no_skill"]["ok"]]
        n = len(D)
        pp, pm = sum(d == 1 for d in D) / n, sum(d == -1 for d in D) / n
        se = float(np.std(D, ddof=1) / np.sqrt(n)) if n > 1 else 0.5
        out[name] = {"pairs": n, "p_plus": pp, "p_minus": pm, "lift": pp - pm, "se": se}
    P = out["placebo"]
    noise = (P["p_plus"] + P["p_minus"]) / 2
    for v in out.values():
        v["p_minus_conservative"] = max(v["p_minus"], noise)
        v["p_plus_conservative"] = max(v["p_plus"] - v["se"], 0.0) + v["p_minus_conservative"]
    return out, noise


def power(pp, pm, a, nmax, rng):
    D = rng.choice([1.0, 0.0, -1.0], p=[pp, 1 - pp - pm, pm], size=(REPS, nmax))
    res = gates.betting(D, 0.0, a, ALPHA_F, CAP, rule="grid", delta_f=DELTA_F)
    return float(res.admit.mean()), float(np.mean(res.n))


def main():
    if any(json.loads(l)["kind"] in ("study5_commit", "study5n_commit") for l in LEDGER.read_text().splitlines()):
        sys.exit("study 5 already committed; the power plan is frozen")
    L, noise = laws()
    rng = np.random.default_rng(20261120)
    exams = [["L_" + n, 1, n] for n in ORDER]
    targets = [n for n in ORDER if L[n]["lift"] >= 0.05]
    curve, chosen = [], None
    for nmax in (600, 900, 1200):
        row = {"n_max": nmax, "skills": {}}
        for t, n in enumerate(ORDER, 1):
            a = level(t, 1)
            pw_c, mean_c = power(L[n]["p_plus_conservative"], L[n]["p_minus_conservative"], a, nmax, rng)
            pw, mean_n = power(L[n]["p_plus"], L[n]["p_minus"], a, nmax, rng)
            row["skills"][n] = {"t": t, "level": a, "power_cal": pw, "power_conservative": pw_c,
                                "mean_pairs_cal": mean_n, "mean_pairs_conservative": mean_c}
        curve.append(row)
        if chosen is None and targets and all(row["skills"][n]["power_conservative"] >= 0.80 for n in targets):
            chosen = nmax
    n_max = chosen if chosen else (1200 if targets else 600)
    plan = {"calibration": L, "placebo_noise_per_side": noise, "order": ORDER, "targets": targets,
            "rule": "smallest N_max in 600, 900, 1200 with conservative power >= 0.80 for every skill with lift >= 0.05; "
                    "600 if none qualifies, 1200 if none reaches 0.80", "reps": REPS, "curve": curve,
            "exams": exams, "n_max": n_max, "chosen": chosen is not None}
    OUT.write_text(json.dumps(plan, indent=1))
    print(json.dumps({k: plan[k] for k in ("order", "targets", "exams", "n_max", "chosen")}, indent=1))
    for n in ORDER:
        print(n, {k: round(v, 3) for k, v in L[n].items()})
    for r in curve:
        print(r["n_max"], {n: (round(v["power_cal"], 2), round(v["power_conservative"], 2), round(v["mean_pairs_cal"])) for n, v in r["skills"].items()})


if __name__ == "__main__":
    main()
