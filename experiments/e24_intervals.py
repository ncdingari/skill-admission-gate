"""E24: uncertainty on headline numbers (Sai's Plan 2, E24).

1. Bootstrap over SkillsBench skills (10,000 resamples) for:
   - the median ratio of PrPl (c = 1/2) variants to the information lower bound (E11, 2.78x);
   - the median variants needed per helpful skill at alpha and at Bonferroni alpha/87, grid bet (E9).
2. Monte Carlo 95% intervals for the false-admission rates of E1 (re-checked t and interval rules, betting gates)
   and the FDR values of E3 and E3b. Wald intervals from the stored standard errors; for an estimate of exactly
   zero, the exact one-sided 95% upper bound 1 - 0.05^(1/n) over n independent histories or exams.
Writes results/E24/e24_intervals.csv.
"""
import numpy as np
import pandas as pd

from _common import RES, Run

run = Run("E24")
rng = np.random.default_rng(20261120)
B = 10_000
rows = []


def boot_median(x, name, what):
    x = np.asarray(x, float)
    meds = np.median(rng.choice(x, size=(B, x.size), replace=True), axis=1)
    rows.append({"quantity": name, "what": what, "estimate": float(np.median(x)), "lo": float(np.quantile(meds, 0.025)),
                 "hi": float(np.quantile(meds, 0.975)), "method": f"bootstrap over {x.size} skills, {B} resamples"})


e11 = pd.read_csv(RES / "E11" / "e11_kinf_and_cost.csv")
boot_median(e11.n_prpl / e11.n_ideal, "NThPrplRatio", "median PrPl variants / information lower bound")
boot_median(e11.n_agrapa / e11.n_ideal, "NThAgrapaRatio", "median aGRAPA variants / information lower bound")
vn = pd.read_csv(RES / "E9" / "e9_variants_needed.csv")
pos = vn[vn.lift > 1e-9]
for c, n in (("median_n_alpha_grid", "NSbMedNAlpha"), ("median_n_bonf_grid", "NSbMedNBonf"),
             ("median_n_alpha", "NSbMedNAlphaPrpl"), ("median_n_bonf", "NSbMedNBonfPrpl")):
    boot_median(pos[c].replace(np.inf, 1e9), n, f"median variants per helpful skill ({c})")


def wald(name, what, est, se, n):
    if est == 0:
        rows.append({"quantity": name, "what": what, "estimate": 0.0, "lo": 0.0, "hi": 1 - 0.05 ** (1 / n),
                     "method": f"exact one-sided 95% upper bound, n = {n}"})
    else:
        rows.append({"quantity": name, "what": what, "estimate": est, "lo": max(est - 1.96 * se, 0.0),
                     "hi": est + 1.96 * se, "method": f"Wald, Monte Carlo SE, n = {n}"})


e1 = pd.read_csv(RES / "E1" / "e1_false_admission.csv")
for g in ("G2_peek_t", "G3_peek_ci", "G7_grid", "G7_grid_futm", "G6_bet_c050"):
    x = e1[e1.gate == g]
    for _, r in x.iterrows():
        wald(f"E1:{r.cell}:{g}", "false-admission rate", float(r.admit_rate), float(r.admit_se), int(r.reps))
for exp, f, n in (("E3", "e3_window_fdr_grid", 400), ("E3b", "e3b_summary_grid", 60)):
    x = pd.read_csv(RES / exp / f"{f}.csv")
    for _, r in x.iterrows():
        wald(f"{exp}:{r.method}", "FDR", float(r.fdr), float(r.fdr_se), n)
h600 = RES / "E3b_h600" / "e3b_summary_grid.csv"
if h600.exists():
    for _, r in pd.read_csv(h600).iterrows():
        wald(f"E3b_h600:{r.method}", "FDR", float(r.fdr), float(r.fdr_se), 600)

df = pd.DataFrame(rows)
run.save(df, "e24_intervals")
print(df[~df.quantity.str.startswith("E1:")].to_string())
run.done(resamples=B)
