"""Live-study transcript counts for Table 3: passes at the stopping variant tau, and a post hoc re-decision
with another bet. Uses the same filter as pilot2/run2.py D_of (pairs where both arms returned ok)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nixigate import gates  # noqa: E402

ALPHA_F, DELTA_F, CAP, NMAX = 0.10, 0.05, 0.5, 600  # as committed in run2.py / run3.py


def _ok(path):
    recs = [json.loads(l) for l in Path(path).read_text().splitlines()]
    return [r for r in recs if r["with_skill"]["ok"] and r["no_skill"]["ok"]]


def counts(path, n_tau: int) -> dict:
    """Passes with/without among the first n_tau ok pairs (at tau) and among all ok pairs run."""
    ok = _ok(path)
    w = [int(r["with_skill"]["pass"]) for r in ok]
    wo = [int(r["no_skill"]["pass"]) for r in ok]
    return {"with_tau": sum(w[:n_tau]), "without_tau": sum(wo[:n_tau]), "with_all": sum(w),
            "without_all": sum(wo), "n_ok": len(ok)}


def decide(path, level: float, rule: str = "grid") -> tuple[str, int]:
    """Re-decide the exam from its transcript with `rule` at the committed level (post hoc analysis)."""
    ok = _ok(path)
    D = np.array([[int(r["with_skill"]["pass"]) - int(r["no_skill"]["pass"]) for r in ok]], float)
    res = gates.betting(D, 0.0, level, ALPHA_F, CAP, rule=rule, delta_f=DELTA_F)
    n = int(res.n[0])
    if res.admit[0]:
        return "admit", n
    if res.futility[0]:
        return "futility", n
    return ("nmax", NMAX) if D.shape[1] >= NMAX else ("open", n)
