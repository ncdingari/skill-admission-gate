"""Shared experiment plumbing: params, seeds, quick mode, CSV output and a JSONL run log."""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nixigate import __version__  # noqa: E402
from nixigate.params import load_params, v  # noqa: E402

P = load_params()
QUICK = os.environ.get("NIXI_QUICK") == "1"
RULE = os.environ.get("NIXI_RULE", "prpl")  # bet for gates.betting: "prpl" (published runs) or "grid"


def out_name(name: str) -> str:
    """Result file name; grid-bet reruns get a _grid suffix so the published PrPl files are untouched."""
    return name if RULE == "prpl" else f"{name}_{RULE}"
RES = ROOT / ("results_quick" if QUICK else "results")


def reps(name: str = "mc_reps") -> int:
    return int(v(P, "mc_reps_quick")) if QUICK else int(v(P, name))


def rng_for(tag: str) -> np.random.Generator:
    """Independent stream per experiment cell, derived from the master seed and a tag."""
    key = [int(v(P, "seed"))] + [ord(c) for c in tag]
    return np.random.default_rng(np.random.SeedSequence(key))


class Run:
    def __init__(self, exp: str):
        self.exp = exp
        self.t0 = time.time()
        self.dir = RES / exp
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, df: pd.DataFrame, name: str) -> Path:
        p = self.dir / f"{out_name(name)}.csv"
        df.to_csv(p, index=False, float_format="%.6g")
        return p

    def done(self, **extra):
        rec = {"exp": self.exp, "quick": QUICK, "wall_s": round(time.time() - self.t0, 2),
               "seed": int(v(P, "seed")), "nixigate": __version__, "python": platform.python_version(),
               "numpy": np.__version__, "pandas": pd.__version__,
               "time": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "rule": RULE, **extra}
        with open(RES / "runlog.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec))


def universal_cu(N=10**6):
    """Normaliser of q_k = c_u/((k+1) log^2(k+1)), k >= 1: c_u = 1/sum_k 1/((k+1) log^2(k+1)).
    The series converges like 1/log N, so the partial sum is completed with the Euler-Maclaurin tail
    1/log(N + 3/2) (error below 1e-12 at N = 1e6; the sum is 2.1097428012...)."""
    import numpy as _np
    k = _np.arange(1, N + 1, dtype=float)
    return 1 / (_np.sum(1 / ((k + 1) * _np.log(k + 1) ** 2)) + 1 / _np.log(N + 1.5))
