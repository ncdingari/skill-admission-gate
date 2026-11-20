"""E10: valid fixed-sample library procedures on the released SkillsBench data (referee item M1).

One-off analyses that are valid at a fixed n: BH, BY and Bonferroni on paired t-test p-values and on
exact sign-flip p-values. Reported next to the anytime-valid rules of E9, on the full data and on
split halves of the configurations (decide on half A, check lift on half B)."""
import numpy as np
import pandas as pd
from scipy import stats

from _common import P, QUICK, Run, rng_for, v
from _realdata import signflip_p, skillsbench
from nixigate import fdr

run = Run("E10")
a = v(P, "alpha")
w = skillsbench()
tasks = sorted(w.task.unique())
cfgs = sorted(w.config.unique())
M = len(tasks)
X = np.stack([w[w.task == t].set_index("config").loc[cfgs].D.values for t in tasks])
rng = rng_for("E10")


def by(p, alpha):
    m = len(p)
    return fdr.bh(p, alpha / np.sum(1.0 / np.arange(1, m + 1)))


def p_t(Xs):
    mean, sd = Xs.mean(1), Xs.std(1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = mean / (sd / np.sqrt(Xs.shape[1]))
    return np.where(sd > 0, stats.t.sf(t, Xs.shape[1] - 1), np.where(mean > 0, 0.0, 1.0))


def p_sf(Xs):
    return np.array([signflip_p(x) for x in Xs])


def rules(Xs):
    out = {}
    for name, p in (("t", p_t(Xs)), ("signflip", p_sf(Xs))):
        out[f"{name}_bh"] = fdr.bh(p, a)
        out[f"{name}_by"] = by(p, a)
        out[f"{name}_bonferroni"] = p < a / M
    return out


full = rules(X)
run.save(pd.DataFrame([{"rule": k, "admitted": int(A.sum()), "of": M,
                        "mean_lift_admitted": X[A].mean() if A.any() else np.nan}
                       for k, A in full.items()]), "e10_full_rules")

S = 20 if QUICK else 200
rep = []
for b in range(S):
    perm = rng.permutation(len(cfgs))
    A_, B_ = perm[: len(cfgs) // 2], perm[len(cfgs) // 2:]
    liftB = X[:, B_].mean(1)
    for k, A in rules(X[:, A_]).items():
        rep.append({"split": b, "rule": k, "admitted": int(A.sum()),
                    "mean_lift_B_admitted": liftB[A].mean() if A.any() else np.nan,
                    "share_B_positive": (liftB[A] > 0).mean() if A.any() else np.nan})
run.save(pd.DataFrame(rep).groupby("rule").mean(numeric_only=True).drop(columns="split").reset_index(),
         "e10_split_half_replication")
run.done(tasks=M, configs=len(cfgs), splits=S)
