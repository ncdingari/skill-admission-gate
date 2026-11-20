"""E16: robustness of Theorem thm:price to how the author chooses the attempt index J.

The theorem assumed J independent of the exams. The proof only needs {J = k} to be decided before exam k's
hidden variants are revealed (a stopping time of the lineage history): exam k then runs on fresh i.i.d. variants,
so E[tau_J | J = k] = E_P[tau_k] and averaging over the marginal law pi of J goes through unchanged.
This experiment checks that claim, and that the condition cannot be dropped, on helpful SkillsBench skills:

 A  independent J (pi fixed in advance), the theorem's original setting;
 B  adaptive J: after each useless version's exam the author sees whether its wealth ever passed 2 and, if so,
    submits the helpful version next with higher probability (J depends on past exams only);
 C  "clairvoyant" J: the author sees the hidden variants of exams 1-3 in advance and submits the helpful version
    at whichever of them would stop soonest (forbidden by commit-reveal);
 D  as B, with exams capped at N_max = 600 so that power is below one (bound with beta and h(beta)).

Bounds use the empirical marginal pi of J. Lower: [(1-b){log(1/alpha) + sum_k pi_k log(1/q_k)} - h(b)] / K_inf,
using Q*(A_k) <= alpha q_k (exam k is valid at alpha q_k). Upper: {log(1/alpha) + sum pi log(1/q) + log(1/m0)} / g_P(lam).
"""
import numpy as np
import pandas as pd
from scipy import optimize

from _common import P, QUICK, Run, rng_for, v
from _realdata import skillsbench

run = Run("E16")
alpha, m0 = v(P, "alpha"), 0.5
KMAX, NMAX = 12, 600
q = (6 / np.pi ** 2) / np.arange(1, KMAX + 1) ** 2            # schedule q_k, sum <= 1
L_k = np.log(1 / (alpha * q))
R = 1500 if QUICK else 6000
HZ = 20000                                                    # horizon for untruncated exams (checked below)

w_ = skillsbench()
tasks = sorted(w_.task.unique())
cfgs = sorted(w_.config.unique())
Xall = (1 + np.stack([w_[w_.task == t].set_index("config").loc[cfgs].D.values for t in tasks])) / 2


def g(lam, x):
    return np.mean(np.log1p(lam * (x - m0)))


def kinf(x):
    r = optimize.minimize_scalar(lambda l: -g(l, x), bounds=(0, 1 / m0 - 1e-12), method="bounded", options={"xatol": 1e-12})
    return max(-r.fun, g(1 / m0 - 1e-12, x)), r.x


def qstar(x, lam):
    vals, cnt = np.unique(x, return_counts=True)
    p = cnt / cnt.sum()
    w = p / (1 + lam * (vals - m0))
    rest = 1 - w.sum()
    if rest > 1e-12:
        vals, w = np.append(vals, 0.0), np.append(w, rest)
    return vals, w / w.sum()


def taus(rng, vals, probs, lam, L, n, cap):
    X = rng.choice(vals, p=probs, size=(n, cap))
    S = np.cumsum(np.log1p(lam * (X - m0)), axis=1)
    hit = S >= L
    return np.where(hit.any(1), hit.argmax(1) + 1, cap + 1), hit.any(1)


def h(b):
    return 0.0 if b <= 0 or b >= 1 else -b * np.log(b) - (1 - b) * np.log(1 - b)


helpful = [i for i in range(len(tasks)) if Xall[i].mean() > m0 + 1e-9]
ks = sorted(helpful, key=lambda i: kinf(Xall[i])[0])
pick = [ks[len(ks) // 4], ks[len(ks) // 2], ks[3 * len(ks) // 4]]   # lower quartile, median, upper quartile K_inf
rows = []
for i in pick:
    x = Xall[i]
    K, lam_star = kinf(x)
    lam = min(lam_star, 0.95 / m0)                            # keep wealth positive when lam* = 1/m0
    gl = g(lam, x)
    vals, probs = np.unique(x, return_counts=True)
    probs = probs / probs.sum()
    qv, qp = qstar(x, lam_star)
    rng = rng_for(f"E16{tasks[i]}")
    # exam stopping times of the helpful version at each attempt k (fresh variants per exam)
    T = np.stack([taus(rng, vals, probs, lam, L_k[k], R, HZ)[0] for k in range(KMAX)], 1)          # R x K
    assert (T <= HZ).all(), "untruncated exam hit the horizon"
    Tc = np.minimum(T, NMAX + 1)
    adm_c = T <= NMAX
    # useless versions: their exams' wealth paths on Q* data; "promising" if wealth ever passed 2 within 200 variants
    U = rng.choice(qv, p=qp, size=(R, KMAX, 200))
    prom = (np.cumsum(np.log1p(lam * (U - m0)), axis=2) >= np.log(2)).any(2)
    pi0 = q / q.sum()
    JA = rng.choice(np.arange(KMAX), p=pi0, size=R)
    JB = np.full(R, KMAX - 1)
    u = rng.random((R, KMAX))
    for r in range(R):
        for k in range(KMAX - 1):
            p_next = 0.5 if (k > 0 and prom[r, k - 1]) else 0.15
            if u[r, k] < p_next:
                JB[r] = k
                break
    JC = T[:, :3].argmin(1)                                   # sees exams 1-3's variants, submits at the fastest
    for name, J, TT, capped in (("A_independent", JA, T, False), ("B_adaptive", JB, T, False),
                                ("C_clairvoyant", JC, T, False), ("D_adaptive_capped", JB, Tc, True)):
        pi = np.bincount(J, minlength=KMAX) / R
        tauJ = TT[np.arange(R), J]
        C = np.minimum(tauJ, NMAX).mean() if capped else tauJ.mean()
        se = (np.minimum(tauJ, NMAX) if capped else tauJ).std() / np.sqrt(R)
        b = 1 - adm_c[np.arange(R), J].mean() if capped else 0.0
        ent = float(np.sum(pi * np.log(1 / q)))
        lower = ((1 - b) * (np.log(1 / alpha) + ent) - h(b)) / K
        upper = (np.log(1 / alpha) + ent + np.log(1 / m0)) / gl
        rows.append({"task": tasks[i], "kinf": K, "lam": lam, "case": name, "C": C, "se": se, "beta": b,
                     "lower": lower, "upper": upper, "H_pi": float(-np.sum(pi[pi > 0] * np.log(pi[pi > 0]))),
                     "mean_J": float(J.mean() + 1), "lower_holds": C + 3 * se >= lower,
                     "upper_holds": capped or C - 3 * se <= upper})
df = pd.DataFrame(rows)
run.save(df, "e16_adaptive_attempts")
ok = df[df.case != "C_clairvoyant"]
assert ok.lower_holds.all() and ok.upper_holds.all(), "a bound failed for an admissible author"
run.done(R=R, n_skills=len(pick), clairvoyant_lower_violated=int((~df[df.case == "C_clairvoyant"].lower_holds).sum()))
pd.set_option("display.width", 200)
print(df[["task", "case", "C", "se", "lower", "upper", "beta", "mean_J", "lower_holds", "upper_holds"]].round(2).to_string())
