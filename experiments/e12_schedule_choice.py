"""E12: numerical checks for choosing the spending schedule (Section 5, minimax and learned schedules).

1. Minimax over Pi_N (all attempt-count laws on 1..N): R = log N with the uniform schedule; the worst case of
   the deployed zeta(1.6) schedule and the universal schedule on the same class.
2. Non-increasing laws on 1..N: a certified bracket for R by Blahut-Arimoto on the extreme points U_m
   (lower: mutual information of the computed prior; upper: worst KL to the computed output law), against
   the explicit bound log H_N of the schedule q_k = 1/(k H_N).
3. Learned schedule: add-one estimate from m completed lineages, mixed with weight eps on the uniform law.
   Monte Carlo E KL(pi || q_hat) against the bound log(1/(1-eps)) + log(1 + (N - 1 + r)/(m + 1)), with r
   poisoned counts placed where pi is smallest.
4. Variants: for the E11 median skill, the oracle bet's expected variants as a function of the level, and
   the expected cost of a lineage under the learned schedule against KL(pi || q_hat)/g_P(lambda*)."""
import numpy as np
import pandas as pd
from scipy.special import zeta

from _common import P, QUICK, Run, rng_for, universal_cu, v
from _realdata import skillsbench

run = Run("E12")
a = v(P, "alpha")
m0 = 0.5
rng = rng_for("E12")


def kl(p, q):
    nz = p > 0
    return float(np.sum(p[nz] * np.log(p[nz] / q[nz])))


# ---- 1. all laws on 1..N
cu = universal_cu()
rows = []
for N in (5, 10, 20, 50, 100, 1000):
    k = np.arange(1, N + 1, dtype=float)
    zq = k ** -1.6 / zeta(1.6)
    uq = cu / ((k + 1) * np.log(k + 1) ** 2)
    rows.append({"N": N, "minimax_logN": np.log(N), "worst_zeta16": np.max(np.log(1 / zq)),
                 "worst_universal": np.max(np.log(1 / uq))})
run.save(pd.DataFrame(rows), "e12_minimax_all")


# ---- 2. non-increasing laws on 1..N
def blahut_arimoto(W, iters=4000, tol=1e-11):
    """W[m, k]: channel rows (here U_m). Returns prior w, output q, lower bound I(w), upper max_m KL(W_m||q)."""
    M = W.shape[0]
    w = np.full(M, 1 / M)
    logW = np.where(W > 0, np.log(np.where(W > 0, W, 1)), 0.0)
    for _ in range(iters):
        q = w @ W
        d = np.sum(W * (logW - np.log(q)[None, :]), axis=1)        # KL(W_m || q)
        lo, hi = float(w @ d), float(d.max())
        if hi - lo < tol:
            break
        w = w * np.exp(d)
        w /= w.sum()
    q = w @ W
    d = np.sum(W * (logW - np.log(q)[None, :]), axis=1)
    return w, q, float(w @ d), float(d.max())


rows = []
for N in ((10, 100, 300) if QUICK else (10, 30, 100, 300, 1000, 3000)):
    m = np.arange(1, N + 1)
    W = (np.arange(1, N + 1)[None, :] <= m[:, None]) / m[:, None]   # U_m = uniform on 1..m
    w, q, lo, hi = blahut_arimoto(W, iters=1500 if N > 1000 else 4000)
    k = np.arange(1, N + 1, dtype=float)
    HN = np.sum(1 / k)
    harm = 1 / (k * HN)
    zq = k ** -1.6 / zeta(1.6)
    worst = lambda qq: float(np.max(np.sum(W * (np.log(np.where(W > 0, W, 1)) - np.log(qq)[None, :]) * (W > 0), axis=1)))
    rows.append({"N": N, "R_lower_BA": lo, "R_upper_BA": hi, "logHN": np.log(HN), "worst_harmonic": worst(harm),
                 "worst_zeta16": worst(zq), "logN": np.log(N), "loglogN": np.log(np.log(N))})
run.save(pd.DataFrame(rows), "e12_minimax_monotone")

# ---- 3. learned schedule, KL level
N, eps = 20, 0.05
J = np.arange(1, N + 1, dtype=float)
laws = {"geometric mean 3": (1 / 3) * (2 / 3) ** (J - 1), "zipf s=2": J ** -2.0, "uniform": np.ones(N)}
laws = {k_: p / p.sum() for k_, p in laws.items()}
uni = np.full(N, 1 / N)
R3 = 400 if QUICK else 4000
rows = []
for ln, pi in laws.items():
    worst_k = int(np.argmin(pi))
    for r in (0, 10, 50):
        for m in (0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000):
            c = rng.multinomial(m, pi, size=R3).astype(float) if m else np.zeros((R3, N))
            c[:, worst_k] += r
            lap = (c + 1) / (m + r + N)
            qh = (1 - eps) * lap + eps * uni
            klv = np.sum(pi * np.log(pi / qh), axis=1)
            rows.append({"pi": ln, "r_poisoned": r, "m_completed": m, "E_KL_sim": klv.mean(), "se": klv.std() / np.sqrt(R3),
                         "bound": np.log(1 / (1 - eps)) + np.log(1 + (N - 1 + r) / (m + 1)),
                         "floor_bound": kl(pi, uni) + np.log(1 / eps), "KL_pi_zeta16": kl(pi, J ** -1.6 / zeta(1.6)),
                         "KL_pi_uniform": kl(pi, uni)})
run.save(pd.DataFrame(rows), "e12_learned_kl")

# ---- 4. variants: oracle bet cost as a function of level, median E11 skill
e11 = pd.read_csv(run.dir.parent / "E11" / "e11_lineage_sim.csv")
task = e11.task.iloc[0]
w_ = skillsbench()
cfgs = sorted(w_.config.unique())
x = (1 + w_[w_.task == task].set_index("config").loc[cfgs].D.values) / 2
from scipy import optimize  # noqa: E402
gfun = lambda l: np.mean(np.log1p(l * (x - m0)))
lam = optimize.minimize_scalar(lambda l: -gfun(l), bounds=(0, 1 / m0 - 1e-9), method="bounded", options={"xatol": 1e-10}).x
gl = gfun(lam)
Lgrid = np.log(1 / a) + np.arange(0, 16.01, 1.0)
R4 = 300 if QUICK else 2000
horizon = int(3 * Lgrid.max() / gl) + 200
Xs = rng.choice(x, size=(R4, horizon))
logK = np.cumsum(np.log1p(lam * (Xs - m0)), axis=1)
Etau = []
for L in Lgrid:
    hit = logK >= L
    assert hit.any(1).all()
    Etau.append((hit.argmax(1) + 1).mean())
Etau = np.array(Etau)
slope, icpt = np.polyfit(Lgrid, Etau, 1)
cost = lambda L: np.interp(L, Lgrid, Etau)
rows = [{"task": task, "g_lambda_star": gl, "slope_fit": slope, "one_over_g": 1 / gl, "L": L_, "E_tau": e_} for L_, e_ in zip(Lgrid, Etau)]
run.save(pd.DataFrame(rows), "e12_cost_vs_level")

pi = laws["geometric mean 3"]
oracle_cost = float(np.sum(pi * cost(np.log(1 / (a * pi)))))
rows = []
for m in (0, 5, 20, 100, 500):
    c = rng.multinomial(m, pi, size=400).astype(float) if m else np.zeros((400, N))
    qh = (1 - eps) * (c + 1) / (m + N) + eps * uni
    Lk = np.log(1 / (a * qh))
    assert Lk.max() <= Lgrid.max()
    extra = (np.sum(pi * cost(Lk), axis=1) - oracle_cost)
    klv = np.sum(pi * np.log(pi / qh), axis=1)
    rows.append({"task": task, "pi": "geometric mean 3", "m_completed": m, "extra_variants_sim": extra.mean(),
                 "KL_over_g": klv.mean() / gl})
zq = J ** -1.6 / zeta(1.6)
rows.append({"task": task, "pi": "geometric mean 3", "m_completed": -1, "extra_variants_sim": float(np.sum(pi * cost(np.log(1 / (a * zq / zq.sum()))))) - oracle_cost,
             "KL_over_g": kl(pi, zq / zq.sum()) / gl})
run.save(pd.DataFrame(rows), "e12_learned_variants")
run.done(task=task, g=gl, slope=slope)
