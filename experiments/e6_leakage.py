"""E6: recall and false-positive rate of leakage checks on planted lookup-table leaks (RQ6)."""
import numpy as np
import pandas as pd

from _common import P, QUICK, Run, rng_for, v
from nixigate import leakage as L

run = Run("E6")
S = int(v(P, "gen_space_small"))
lam, cmax, ns = v(P, "leak_lambda_max"), v(P, "leak_cover_max"), int(v(P, "leak_mc_sample"))
n_sk = 20 if QUICK else 100
rng = rng_for("E6")
dets = ("literal", "normalized", "decoded")
rows = []
for f in (0.02, 0.05, 0.25, 1.0):
    for kind in ("plain", "spaced", "split", "base64", "hex", "computed"):
        hits = {dd: [] for dd in dets}
        for _ in range(n_sk):
            keys = rng.choice(S, size=max(1, int(round(f * S))), replace=False)
            sk = L.leaky_skill(rng, keys, kind)
            sample = rng.integers(0, S, ns)
            for dd in dets:
                hits[dd].append(L.coverage(sk, sample, dd, lam) > cmax)
        for dd in dets:
            rows.append({"table_share": f, "kind": kind, "detector": dd, "recall": float(np.mean(hits[dd])),
                         "skills": n_sk, "exam_pass_gain_bound": f})
fp = {dd: [] for dd in dets}
covs = {dd: [] for dd in dets}
for _ in range(n_sk * 5):
    sk = L.clean_skill(rng)
    sample = rng.integers(0, S, ns)
    for dd in dets:
        c = L.coverage(sk, sample, dd, lam)
        covs[dd].append(c)
        fp[dd].append(c > cmax)
for dd in dets:
    rows.append({"table_share": 0.0, "kind": "clean", "detector": dd, "recall": float(np.mean(fp[dd])),
                 "skills": n_sk * 5, "exam_pass_gain_bound": 0.0, "mean_cover_clean": float(np.mean(covs[dd]))})
run.save(pd.DataFrame(rows), "e6_leakage")
run.done(skills=n_sk, space=S)
