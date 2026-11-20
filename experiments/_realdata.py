"""Loaders for the released benchmark data (data/external, see PROVENANCE.md)."""
import json

import numpy as np
import pandas as pd

from _common import ROOT

EXT = ROOT / "data" / "external"


def swe() -> pd.DataFrame:
    """SWE-Skills-Bench: one row per (config, skill, batch); D = (with - without) pass rate in [-1, 1]."""
    d = pd.read_csv(EXT / "swe_skills_bench_ALL_combined_cells.csv")
    d["D"] = d.delta_pp / 100.0
    return d.rename(columns={"harness_model": "config"})


def skillsbench() -> pd.DataFrame:
    """SkillsBench: one row per (task, config); D = (passes with - passes without) / 3 in [-1, 1]."""
    d = pd.DataFrame(json.loads((EXT / "skillsbench_results_registry.json").read_text()))
    w = d.pivot_table(index=["task", "model"], columns="condition", values="passCount").dropna().reset_index()
    w["D"] = (w["With Skills"] - w["No Skills"]) / 3.0
    w["p0"] = w["No Skills"] / 3.0
    return w.rename(columns={"model": "config"})


def signflip_p(x: np.ndarray, rng: np.random.Generator | None = None, mc: int = 0) -> float:
    """One-sided paired sign-flip p-value for mean(x) > 0 (exact if mc = 0)."""
    x = np.asarray(x, float)
    obs = x.mean()
    if mc == 0:
        n = len(x)
        signs = 1 - 2 * ((np.arange(2**n)[:, None] >> np.arange(n)) & 1)
        return float(((signs * np.abs(x)).mean(1) >= obs - 1e-12).mean())
    signs = rng.choice([-1.0, 1.0], size=(mc, len(x)))
    return float((1 + ((signs * np.abs(x)).mean(1) >= obs - 1e-12).sum()) / (mc + 1))
