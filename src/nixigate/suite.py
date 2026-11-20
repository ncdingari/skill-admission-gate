"""Run every gate on one array of paired differences, with design constants planned from a model."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import dgp, gates


@dataclass(frozen=True)
class Design:
    alpha: float = 0.05
    alpha_f: float = 0.10
    delta: float = 0.0
    n_max: int = 600
    lift1: float = 0.10
    power: float = 0.80
    n_min: int = 10
    every: int = 10
    z: float = 1.959964
    beta: float = 0.20
    tau: float = 0.10
    cap: float = 0.5
    cap_alt: float = 0.75
    delta_f: float = 0.05


def plan(design: Design, nuisance: dgp.Model, base_rate: float) -> dict:
    """Design constants a registry would fix in advance: fixed n (Eq. 7) and the SPRT alternative."""
    m1 = dgp.Model.for_lift(design.lift1, base_rate, nuisance.sd_theta, nuisance.sd_lift, nuisance.rho_crn,
                            nuisance.r)
    sd1 = dgp.sd_D(m1)
    n_fixed = gates.sample_size(sd1, design.lift1, design.delta, design.alpha, design.power)
    p1 = dgp.discordant_share(m1) if (nuisance.r == 1 and design.delta == 0) else np.nan
    return {"n_fixed": n_fixed, "sd1": sd1, "p1": p1}


GATES = ["G1_fixed_t", "G2_peek_t", "G3_peek_ci", "G4_sprt", "G5_msprt",
         "G6_bet_capm0", "G6_bet_c050", "G6_bet_c075", "G6_bet_agrapa", "G6_bet_futm", "G7_grid", "G7_grid_futm"]


def run_all(D: np.ndarray, d: Design, pl: dict, r: int = 1) -> dict[str, gates.GateResult]:
    S = D[:, : d.n_max]
    out = {
        "G1_fixed_t": gates.fixed_t(D, pl["n_fixed"], d.delta, d.alpha),
        "G2_peek_t": gates.peek_t(S, d.delta, d.alpha, d.n_min),
        "G3_peek_ci": gates.peek_ci(S, d.delta, d.z, d.n_min, d.every),
        "G5_msprt": gates.msprt(S, d.delta, d.alpha, d.tau, d.n_min),
        "G6_bet_capm0": gates.betting(S, d.delta, d.alpha, d.alpha_f, rule="cap_m0"),
        "G6_bet_c050": gates.betting(S, d.delta, d.alpha, d.alpha_f, cap=d.cap),
        "G6_bet_c075": gates.betting(S, d.delta, d.alpha, d.alpha_f, cap=d.cap_alt),
        "G6_bet_agrapa": gates.betting(S, d.delta, d.alpha, d.alpha_f, cap=d.cap, rule="agrapa"),
        "G6_bet_futm": gates.betting(S, d.delta, d.alpha, d.alpha_f, cap=d.cap,
                                     delta_f=max(d.delta_f, d.delta)),
        # level-tuned grid mixture bet (Theorem thm:adaptive): the deployed bet from revision B on
        "G7_grid": gates.betting(S, d.delta, d.alpha, d.alpha_f, rule="grid"),
        "G7_grid_futm": gates.betting(S, d.delta, d.alpha, d.alpha_f, rule="grid", delta_f=max(d.delta_f, d.delta)),
    }
    if not np.isnan(pl["p1"]):
        out["G4_sprt"] = gates.sprt(S, pl["p1"], d.alpha, d.beta)
    return out


def summarize(res: dict[str, gates.GateResult], r: int, true_lift: float) -> list[dict]:
    rows = []
    for g, x in res.items():
        R = len(x.admit)
        rows.append({"gate": g, "reps": R, "admit_rate": x.admit.mean(),
                     "admit_se": np.sqrt(x.admit.mean() * (1 - x.admit.mean()) / R),
                     "mean_n": x.n.mean(), "median_n": float(np.median(x.n)),
                     "mean_runs": 2 * r * x.n.mean(), "true_lift": true_lift,
                     "bias_admitted": float((x.dbar[x.admit] - true_lift).mean()) if x.admit.any() else np.nan})
    return rows
