"""E21: stress tests of the conditional null (Sai's Plan 2, E21).

Every cell has marginal lift E[D_i] = 0 (delta = 0), but departs from the conditional null
E[X_i | F_{i-1}] <= m0 in a way a real deployment could. Reports the false-admission rate of the deployed
grid gate, PrPl c = 1/2, mSPRT and fixed-n t against the 5% target.

  control            the design of Section 3 (i.i.d. variants, concurrent arms)
  coin_hard          one fair coin per exam sets every D_i to +1 or -1 (the referee's example)
  coin_shift_s       one fair coin per exam sets the exam's lift to +s or -s (s = 0.05, 0.10)
  snapshot_both      provider change at a uniform variant index shifts both arms' logits by +-1
  snapshot_skill     the change shifts only the skill arm's logit by +-0.5 (random sign per exam)
  block_b_s          blocks of b variants share a random lift effect N(0, s^2) on the logit (shared cache/batch)
  repeat_f           a share f of variants re-use an earlier variant's outcome (repeated seeds)
  exclude_q          useless skill; variants the skill arm lost (D = -1) are excluded as "harness failures"
                     with probability q (arm- and outcome-dependent exclusion; q = 0.01, 0.05, 0.20)
  python experiments/e21_null_stress.py           (NIXI_QUICK=1 for a smoke run)
"""
import numpy as np
import pandas as pd
from scipy.special import expit

from _common import P, Run, reps, rng_for, v
from nixigate import dgp, suite

run = Run("E21")
base = v(P, "base_pass_rate")
d = suite.Design(alpha=v(P, "alpha"), alpha_f=v(P, "alpha_futility"), n_max=int(v(P, "n_max")),
                 lift1=v(P, "target_lift"), power=v(P, "power_target"), n_min=int(v(P, "peek_n_min")),
                 every=int(v(P, "peek_every")), z=v(P, "ci_level_z"), beta=v(P, "sprt_beta"),
                 tau=v(P, "msprt_tau"), cap=v(P, "bet_cap_c"), cap_alt=v(P, "bet_cap_c_alt"),
                 delta_f=v(P, "futility_margin"))
sdt, sdl, rho = v(P, "sd_theta"), v(P, "sd_lift"), v(P, "rho_crn")
m_null = dgp.Model.for_lift(0.0, base, sdt, sdl, rho, 1)
pl = suite.plan(d, m_null, base)
NCOL = max(d.n_max, pl["n_fixed"])
KEEP = {"G7_grid_futm": "grid", "G6_bet_futm": "prpl_c050", "G5_msprt": "msprt", "G1_fixed_t": "fixed_t",
        "G4_sprt": "sprt"}


def sim(rng, k, n, g_shift=None, sd_lift=sdl, drift0=None, drift1=None):
    """Paired differences with per-element extra logit terms (arrays broadcastable to (k, n))."""
    theta = rng.standard_normal((k, n)) * sdt
    gi = m_null.g + rng.standard_normal((k, n)) * sd_lift
    if g_shift is not None:
        gi = gi + g_shift
    s0 = 0.0 if drift0 is None else drift0
    s1 = 0.0 if drift1 is None else drift1
    p0 = expit(m_null.b0 + theta + s0)
    p1 = expit(m_null.b0 + gi + theta + s1)
    u0 = rng.random((k, n))
    u1 = np.where(rng.random((k, n)) < rho, u0, rng.random((k, n)))
    return (u1 < p1).astype(float) - (u0 < p0).astype(float)


def coin_shift(rng, k, n, s):
    gp = dgp.solve_g(s, m_null.b0, sdt, sdl) - m_null.g
    gm = dgp.solve_g(-s, m_null.b0, sdt, sdl) - m_null.g
    sign = rng.random((k, 1)) < 0.5
    return sim(rng, k, n, g_shift=np.where(sign, gp, gm))


def snapshot(rng, k, n, both):
    cp = rng.integers(0, d.n_max, size=(k, 1))
    after = np.arange(n)[None, :] >= cp
    sgn = np.where(rng.random((k, 1)) < 0.5, 1.0, -1.0)
    if both:
        return sim(rng, k, n, drift0=after * sgn * 1.0, drift1=after * sgn * 1.0)
    return sim(rng, k, n, drift1=after * sgn * 0.5)


def block(rng, k, n, b, s):
    # shared block effect on the skill arm; base g re-solved so the marginal lift stays 0
    g0 = dgp.solve_g(0.0, m_null.b0, sdt, float(np.hypot(sdl, s)))
    eff = np.repeat(rng.standard_normal((k, -(-n // b))) * s, b, axis=1)[:, :n]
    return sim(rng, k, n, g_shift=(g0 - m_null.g) + eff)


def repeat(rng, k, n, f):
    D = sim(rng, k, n)
    idx = np.tile(np.arange(n), (k, 1))
    rep = rng.random((k, n)) < f
    rep[:, 0] = False
    src = np.floor(rng.random((k, n)) * np.arange(n)[None, :]).astype(int)
    idx = np.where(rep, src, idx)
    # resolve chains so a copy points at an original outcome
    for _ in range(8):
        idx = np.take_along_axis(idx, idx, axis=1)
    return np.take_along_axis(D, idx, axis=1)


def exclude(rng, k, n, q):
    # useless skill (pre-exclusion lift 0); variants where the skill arm lost (D = -1) are excluded as
    # "harness failures" with probability q, so the surviving variants show a spurious lift
    long = int(n / (1 - q) * 1.3) + 20
    D = sim(rng, k, long)
    drop = (D == -1) & (rng.random(D.shape) < q)
    out = np.zeros((k, n))
    for j in range(k):
        row = D[j][~drop[j]]
        out[j] = row[:n] if row.size >= n else np.pad(row, (0, n - row.size))
    return out


cells = [("control", None, lambda r, k, n: sim(r, k, n)),
         ("coin_hard", None, lambda r, k, n: np.repeat(np.where(r.random((k, 1)) < 0.5, 1.0, -1.0), n, axis=1)),
         ("coin_shift_0.05", 0.05, lambda r, k, n: coin_shift(r, k, n, 0.05)),
         ("coin_shift_0.10", 0.10, lambda r, k, n: coin_shift(r, k, n, 0.10)),
         ("snapshot_both", None, lambda r, k, n: snapshot(r, k, n, True)),
         ("snapshot_skill", None, lambda r, k, n: snapshot(r, k, n, False)),
         ("block_5_0.5", 0.5, lambda r, k, n: block(r, k, n, 5, 0.5)),
         ("block_10_0.5", 0.5, lambda r, k, n: block(r, k, n, 10, 0.5)),
         ("block_10_1.0", 1.0, lambda r, k, n: block(r, k, n, 10, 1.0)),
         ("repeat_0.01", 0.01, lambda r, k, n: repeat(r, k, n, 0.01)),
         ("repeat_0.05", 0.05, lambda r, k, n: repeat(r, k, n, 0.05)),
         ("repeat_0.20", 0.20, lambda r, k, n: repeat(r, k, n, 0.20)),
         ("exclude_0.01", 0.01, lambda r, k, n: exclude(r, k, n, 0.01)),
         ("exclude_0.05", 0.05, lambda r, k, n: exclude(r, k, n, 0.05)),
         ("exclude_0.20", 0.20, lambda r, k, n: exclude(r, k, n, 0.20))]

R = reps("mc_reps")
rows = []
for label, sev, gen in cells:
    rng = rng_for("E21" + label)
    adm = {g: [] for g in KEEP}
    nn = {g: [] for g in KEEP}
    dmean = []
    for c0 in range(0, R, 5000):
        k = min(5000, R - c0)
        D = gen(rng, k, NCOL)
        dmean.append(D[:, : d.n_max].mean())
        res = suite.run_all(D, d, pl, 1)
        for g in KEEP:
            if g in res:
                adm[g].append(res[g].admit)
                nn[g].append(res[g].n)
    for g, name in KEEP.items():
        if not adm[g]:
            continue
        a = np.concatenate(adm[g])
        rows.append({"cell": label, "severity": sev, "gate": name, "reps": a.size,
                     "false_admit": a.mean(), "se": a.std(ddof=1) / np.sqrt(a.size),
                     "mean_n": np.concatenate(nn[g]).mean(), "marginal_mean_D": float(np.mean(dmean))})
    print(label, {KEEP[g]: round(float(np.concatenate(adm[g]).mean()), 4) for g in KEEP if adm[g]}, flush=True)

df = pd.DataFrame(rows)
run.save(df, "e21_null_stress")
run.done(cells=len(cells), reps=R)
