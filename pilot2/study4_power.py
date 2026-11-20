"""Pre-registered power plan for live study 4, from the Haiku calibration (written before commit).

Reads the study4_calibration ledger entry and pilot2/transcripts4/calibration.jsonl, estimates the paired
outcome law of the real skill (P(D=+1), P(D=-1)), and simulates the grid-bet exam at that law and at a
conservative law whose lift is one standard error lower and which adds the two-way discordance seen on the
placebo arm (amendment v2, 18 Nov 2026, before commit: v1 ignored discordant D=-1 pairs, which the placebo
calibration shows occur; v1 output kept as power_plan_study4_v1_no_noise.json). Candidate designs, in
order of preference (comparability with studies 2 and 3 first):
  1. placebo first, real skill second (t=2, k=2), N_max in 600, 900, 1200
  2. real skill first (t=1, k=1), N_max in 600, 900, 1200
The first design with power >= 0.80 at the conservative law is chosen. Writes pilot2/power_plan_study4.json
(exams, n_max, power curve, projected pairs) and refuses to overwrite it once study 4 is committed.
  python pilot2/study4_power.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))
from nixigate import fdr, gates  # noqa: E402

ALPHA, ALPHA_F, DELTA_F, CAP, S = 0.05, 0.10, 0.05, 0.5, 1.6
OUT = ROOT / "power_plan_study4.json"
LEDGER = ROOT.parent / "pilot" / "ledger.jsonl"
REPS = 2000


def level(t, k, R=0):
    g = fdr.gamma_seq(max(t, k), S)
    return float(min(ALPHA * g[t - 1] * (R + 1), ALPHA * g[k - 1]))


def cal_law():
    recs = [json.loads(l) for l in (ROOT / "transcripts4" / "calibration.jsonl").read_text().splitlines()]
    D = [int(r["with_skill"]["pass"]) - int(r["no_skill"]["pass"]) for r in recs
         if r["skill"] == "real" and r["with_skill"]["ok"] and r["no_skill"]["ok"]]
    n = len(D)
    pp, pm = sum(d == 1 for d in D) / n, sum(d == -1 for d in D) / n
    lift = pp - pm
    se = float(np.std(D, ddof=1) / np.sqrt(n)) if n > 1 else 0.5
    P = [int(r["with_skill"]["pass"]) - int(r["no_skill"]["pass"]) for r in recs
         if r["skill"] == "placebo" and r["with_skill"]["ok"] and r["no_skill"]["ok"]]
    noise = (sum(d != 0 for d in P) / len(P)) / 2          # per-side discordance rate under no effect
    return n, pp, pm, lift, se, noise


def power(pp, pm, a, nmax, rng):
    D = rng.choice([1.0, 0.0, -1.0], p=[pp, 1 - pp - pm, pm], size=(REPS, nmax))
    res = gates.betting(D, 0.0, a, ALPHA_F, CAP, rule="grid", delta_f=DELTA_F)
    return float(res.admit.mean()), float(np.mean(res.n)), float(np.median(res.n))


def main():
    if any(json.loads(l)["kind"] in ("study4_commit", "study4n_commit") for l in LEDGER.read_text().splitlines()):
        sys.exit("study 4 already committed; the power plan is frozen")
    n, pp, pm, lift, se, noise = cal_law()
    pm_c = max(pm, noise)                                # conservative: at least the placebo's two-way noise
    pp_c = max(pp - se, 0.0) + pm_c                      # ... and a lift one SE lower
    rng = np.random.default_rng(20261118)
    designs = [([["S", 1, "placebo"], ["S", 2, "real"]], 2, 2), ([["S", 1, "real"]], 1, 1)]
    curve, chosen = [], None
    for exams, t, k in designs:
        a = level(t, k)
        for nmax in (600, 900, 1200, 1800):
            pw_c, mean_c, med_c = power(pp_c, pm_c, a, nmax, rng)
            pw, mean_n, med_n = power(pp, pm, a, nmax, rng)
            row = {"exams": exams, "real_t": t, "level": a, "n_max": nmax, "power_cal": pw, "power_conservative": pw_c,
                   "mean_pairs_cal": mean_n, "mean_pairs_conservative": mean_c}
            curve.append(row)
            if chosen is None and pw_c >= 0.80:
                chosen = row
    plan = {"calibration": {"pairs": n, "p_plus": pp, "p_minus": pm, "lift": lift, "se": se, "p_plus_conservative": pp_c,
                            "p_minus_conservative": pm_c, "placebo_noise_per_side": noise, "amendment": "v2"},
            "rule": "first design with power >= 0.80 at the conservative law; order placebo-first, then real-first; "
                    "N_max 600, 900, 1200, 1800", "reps": REPS, "curve": curve,
            "chosen": chosen is not None}
    if chosen:
        plan.update({"exams": chosen["exams"], "n_max": chosen["n_max"]})
    else:  # no design reaches 0.80: run the most powerful one and report whatever happens (pre-registered)
        best = max(curve, key=lambda r: r["power_conservative"])
        plan.update({"exams": best["exams"], "n_max": best["n_max"], "note": "no design reached 0.80; most powerful used"})
    OUT.write_text(json.dumps(plan, indent=1))
    print(json.dumps({k: plan[k] for k in ("calibration", "exams", "n_max", "chosen")}, indent=1))
    for r in curve:
        print(r["real_t"], r["n_max"], f"level {r['level']:.4f} power {r['power_cal']:.2f} / cons {r['power_conservative']:.2f}"
              f" mean pairs {r['mean_pairs_cal']:.0f} / {r['mean_pairs_conservative']:.0f}")


if __name__ == "__main__":
    main()
