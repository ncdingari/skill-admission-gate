"""Table 3 at the stopping variant tau: counts use the same filtered (both-arms-ok) sequence as decide()."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import live_counts as lc  # noqa: E402

LED = [json.loads(l) for l in (ROOT / "pilot" / "ledger.jsonl").read_text().splitlines()]


def _cred(kind):
    return [e for e in LED if e["kind"] == kind][-1]["payload"]["body"]


def test_study2_helpful_tau_counts_add_up():
    e = {x["skill"]: x for x in _cred("study2_credential")["exams"]}["helpful"]
    c = lc.counts(ROOT / "pilot2" / "transcripts" / "exam4.jsonl", e["decision_pair"])
    assert (c["with_all"], c["without_all"]) == (e["pass_with"], e["pass_without"])
    assert c["with_tau"] <= c["with_all"] and c["without_tau"] <= c["without_all"]
    assert c["n_ok"] >= e["decision_pair"]


def test_grid_recompute_matches_prpl_path_for_prpl():
    e = {x["skill"]: x for x in _cred("study2_credential")["exams"]}["helpful"]
    dec, n = lc.decide(ROOT / "pilot2" / "transcripts" / "exam4.jsonl", e["level"], rule="prpl")
    assert (dec, n) == (e["decision"], e["decision_pair"])
