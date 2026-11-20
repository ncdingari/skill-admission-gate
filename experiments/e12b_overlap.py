"""E12b: learned schedule with overlapping lineages (Theorem thm:learned (b)-(c)).

Lineage i opens at time i and closes at time i + J_i (one attempt per time unit), so with N = 20 every lineage
closes before D = N further lineages open. Two counting rules for the schedule of lineage i:
  naive        count every lineage that has closed by time i (selected on duration, hence on J)
  fixed-delay  count lineages 1..i-1-D (Theorem thm:learned (c)), which are closed and fixed by index alone.
Laws: geometric mean 3 and zipf (short lineages dominate), uniform and increasing (long lineages common).
Report E KL(pi || q_hat^(i)) and the estimated first-attempt mass against pi_1, by lineage index, and the
cumulative excess against the bound of part (c)."""
import numpy as np
import pandas as pd

from _common import QUICK, Run, rng_for

run = Run("E12b")
rng = rng_for("E12b")
N, eps = 20, 0.05
D = N
k = np.arange(1, N + 1)
laws = {"geometric mean 3": (1 / 3) * (2 / 3) ** (k - 1), "zipf s=2": k ** -2.0, "uniform": np.ones(N), "increasing": k.astype(float)}
n = 400
R = 200 if QUICK else 2000
checkpoints = [25, 50, 100, 200, 400]


def qhat(c):
    m = c.sum()
    return (1 - eps) * (c + 1) / (m + N) + eps / N


rows, cum_rows = [], []
for name, pi in laws.items():
    pi = pi / pi.sum()
    kl_naive = np.zeros((R, n))
    kl_delay = np.zeros((R, n))
    q1_naive = np.zeros((R, n))
    q1_delay = np.zeros((R, n))
    for r in range(R):
        J = rng.choice(k, size=n, p=pi)
        close = np.arange(1, n + 1) + J          # lineage i (1-based) closes at i + J_i
        for i in range(1, n + 1):
            closed = np.flatnonzero(close[: i - 1] <= i)          # lineages 1..i-1 closed by time i
            cn = np.bincount(J[closed] - 1, minlength=N)
            cd = np.bincount(J[: max(0, i - 1 - D)] - 1, minlength=N)
            qn, qd = qhat(cn), qhat(cd)
            kl_naive[r, i - 1] = np.sum(pi * np.log(pi / qn))
            kl_delay[r, i - 1] = np.sum(pi * np.log(pi / qd))
            q1_naive[r, i - 1], q1_delay[r, i - 1] = qn[0], qd[0]
    for i in checkpoints:
        rows.append({"pi": name, "lineage": i, "pi_1": pi[0],
                     "KL_naive": kl_naive[:, i - 1].mean(), "KL_fixed_delay": kl_delay[:, i - 1].mean(),
                     "q1_naive": q1_naive[:, i - 1].mean(), "q1_fixed_delay": q1_delay[:, i - 1].mean(),
                     "se_KL_naive": kl_naive[:, i - 1].std() / np.sqrt(R)})
    ns = np.array(checkpoints)
    bound = ns * np.log(1 / (1 - eps)) + (D + 1) * np.log(N) + (N - 1) * np.log(ns)
    for j, i in enumerate(ns):
        cum_rows.append({"pi": name, "n": int(i), "cum_KL_fixed_delay": kl_delay[:, :i].sum(1).mean(),
                         "cum_KL_naive": kl_naive[:, :i].sum(1).mean(), "bound_c": bound[j]})
per = pd.DataFrame(rows)
cum = pd.DataFrame(cum_rows)
assert (cum.cum_KL_fixed_delay <= cum.bound_c).all()
run.save(per, "e12b_overlap")
run.save(cum, "e12b_cumulative")
g = per[(per.pi == "geometric mean 3") & (per.lineage == 400)].iloc[0]
run.done(KL_naive_400=g.KL_naive, KL_delay_400=g.KL_fixed_delay, q1_naive=g.q1_naive, q1_delay=g.q1_fixed_delay, pi1=g.pi_1)
print(per.to_string(index=False))
print(cum.to_string(index=False))
