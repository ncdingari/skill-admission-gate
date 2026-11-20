"""E11: numerical checks of Theorems 2 and 3 (planning/Theory_Resubmission_v1.md) on SkillsBench data.

1. K_inf for each helpful SkillsBench skill by the dual (max over bets), and the least-favourable null law Q*
   built from the dual optimum: Q* must be a null law (mean <= m0, leftover mass >= 0) with KL(P, Q*) = K_inf.
2. Expected variants to admit at level a: lower bound, oracle bet, PrPl cap 1/2, aGRAPA, against
   log(1/a)/K_inf (Theorem 2).
3. Lineage price: extra nats log(1/q_k) for three spending schedules, simulated extra variants
   E[tau_k] - E[tau_1] for the oracle bet, and the expected price H(pi) + KL(pi||q) for attempt-count
   laws pi (Theorem 3)."""
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import zeta

from _common import P, QUICK, Run, rng_for, universal_cu, v
from _realdata import skillsbench
from nixigate import gates

run = Run("E11")
a = v(P, "alpha")
m0 = 0.5                                  # delta = 0
w = skillsbench()
tasks = sorted(w.task.unique())
cfgs = sorted(w.config.unique())
Xall = (1 + np.stack([w[w.task == t].set_index("config").loc[cfgs].D.values for t in tasks])) / 2
rng = rng_for("E11")


def g(lam, x):
    return np.mean(np.log1p(lam * (x - m0)))


def kinf_dual(x):
    r = optimize.minimize_scalar(lambda l: -g(l, x), bounds=(0, 1 / m0 - 1e-9), method="bounded",
                                 options={"xatol": 1e-10})
    return -r.fun, r.x


def least_favourable(x, lam):
    """Q* from the dual optimum (proof of Theorem 2c): Q*(x) = P(x)/(1 + lam (x - m0)) on supp P, leftover
    mass at 0. Returns (KL(P, Q*), E_Q* X, total mass of the P-part). Feasible and KL = g(lam*) proves duality."""
    sup, cnt = np.unique(x, return_counts=True)
    p = cnt / cnt.sum()
    q = p / (1 + lam * (sup - m0))
    left = 1 - q.sum()
    return float(np.sum(p * np.log(p / q))), float(np.dot(q, sup)), float(q.sum()), float(left)


def oracle_tau(Xs, lam, lvl):
    logK = np.cumsum(np.log1p(lam * (Xs - m0)), axis=1)
    hit = logK >= np.log(1 / lvl)
    ok = hit.any(1)
    return np.where(ok, hit.argmax(1) + 1, Xs.shape[1]), ok


# ---- 1 and 2: per-skill K_inf and expected variants at level a
helpful = [i for i in range(len(tasks)) if Xall[i].mean() > m0]
NM = 1500 if QUICK else 6000
R = 100 if QUICK else 400
rows = []
for i in helpful:
    x = Xall[i]
    K, lam_star = kinf_dual(x)
    Kq, meanQ, massP, left = least_favourable(x, lam_star)
    nm = int(min(60000, max(NM, 10 * np.log(1 / a) / K)))
    Xs = rng.choice(x, size=(R, nm))
    t_or, ok_or = oracle_tau(Xs, lam_star, a)
    Ds = 2 * Xs - 1
    pr = gates.betting(Ds, 0, a, None, 0.5)
    ag = gates.betting(Ds, 0, a, None, v(P, "bet_cap_c"), rule="agrapa")
    ag9 = gates.betting(Ds, 0, a, None, 0.9, rule="agrapa")
    rows.append({"task": tasks[i], "lift": 2 * x.mean() - 1, "kinf_dual": K, "kl_P_Qstar": Kq, "mean_Qstar": meanQ,
                 "leftover_mass_at_0": left, "lam_star": lam_star, "horizon": nm,
                 "cap_half_efficiency": g(min(lam_star, 0.5), x) / K,
                 "n_lower": max(0.0, (np.log(1 / a) - np.log(2)) / K), "n_ideal": np.log(1 / a) / K,
                 "n_oracle": t_or.mean(), "oracle_censored": 1 - ok_or.mean(),
                 "n_prpl": pr.n.mean(), "prpl_censored": 1 - pr.admit.mean(),
                 "n_agrapa": ag.n.mean(), "agrapa_censored": 1 - ag.admit.mean(),
                 "n_agrapa_c09": ag9.n.mean(), "agrapa_c09_censored": 1 - ag9.admit.mean()})
per = pd.DataFrame(rows)
run.save(per, "e11_kinf_and_cost")

# ---- 3: lineage price
Kmax = 20
k = np.arange(1, Kmax + 1, dtype=float)
cu = universal_cu()
sched = {"zeta_1.6 (current)": k ** -1.6 / zeta(1.6),
         "universal 1/((k+1)log^2(k+1))": cu / ((k + 1) * np.log(k + 1) ** 2),
         "geometric (1/2)^k": 0.5 ** k}
nats = pd.DataFrame({"k": k.astype(int), **{f"nats_{s}": np.log(1 / q) for s, q in sched.items()}})
run.save(nats, "e11_schedule_nats")

# simulated extra variants for the median-K_inf skill, oracle bet, zeta schedule (the deployed one)
im = per.kinf_dual.sub(per.kinf_dual.median()).abs().idxmin()
x = Xall[helpful[im]]
K, lam_star = kinf_dual(x)
gl = g(lam_star, x)
R2 = 200 if QUICK else 2000
sim = []
for name, q in sched.items():
    Xs = rng.choice(x, size=(R2, 3000 if QUICK else 8000))
    base = None
    for j in (1, 2, 5, 10, 20):
        t, ok = oracle_tau(Xs, lam_star, a * q[j - 1])
        m = t.mean()
        base = m if j == 1 else base
        sim.append({"task": tasks[helpful[im]], "kinf": K, "schedule": name, "k": j, "level": a * q[j - 1], "mean_tau": m, "censored": 1 - ok.mean(),
                    "extra_vs_k1_sim": m - base, "extra_vs_k1_theory": np.log(q[0] / q[j - 1]) / gl})
run.save(pd.DataFrame(sim), "e11_lineage_sim")

# expected price for attempt-count laws pi (truncated at 200 attempts, renormalized)
J = np.arange(1, 201, dtype=float)
laws = {"one_attempt": np.r_[1.0, np.zeros(199)],
        "geometric mean 1.5": (1 / 1.5) * (1 - 1 / 1.5) ** (J - 1),
        "geometric mean 3": (1 / 3) * (2 / 3) ** (J - 1),
        "zipf s=2": J ** -2.0}
qfull = {"zeta_1.6 (current)": J ** -1.6 / zeta(1.6), "universal 1/((k+1)log^2(k+1))": cu / ((J + 1) * np.log(J + 1) ** 2),
         "geometric (1/2)^k": 0.5 ** J}
price = []
for ln, pi in laws.items():
    pi = pi / pi.sum()
    nz = pi > 0
    H = -np.sum(pi[nz] * np.log(pi[nz]))
    for sn, q in qfull.items():
        cross = np.sum(pi[nz] * np.log(1 / q[nz]))
        price.append({"pi": ln, "schedule": sn, "H_pi_nats": H, "KL_pi_q_nats": cross - H, "price_nats": cross,
                      "price_variants_median_skill": cross / K, "floor_variants_median_skill": H / K})
run.save(pd.DataFrame(price), "e11_expected_price")
run.done(helpful=len(helpful), median_task=tasks[helpful[im]], median_kinf=K, NM=NM, R=R)
