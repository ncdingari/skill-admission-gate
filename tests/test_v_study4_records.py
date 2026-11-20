"""Study 4 records (final review gaps): real calibration seed form never equals an exam seed; at-tau counts
from tools/live_counts match the signed credentials' decision pairs on the filtered (both-arms-ok) sequence."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "pilot2"))
import live_counts as L  # noqa: E402
import family_s4, family_s4_apa, family_s4_commit  # noqa: E402,F401

ARMS = {"study4": "transcripts4", "study4n": "transcripts_n4", "study4b": "transcripts4b", "study4bn": "transcripts_n4b",
        "study4c": "transcripts4c", "study4cn": "transcripts_n4c"}


def test_calibration_seeds_disjoint_from_exam_seeds():
    cal = {hashlib.sha256(f"calibration-{m.FAMILY}-{i}".encode()).hexdigest()
           for m in (family_s4, family_s4_apa, family_s4_commit) for i in range(20)}
    for d in ARMS.values():
        logged_cal = {json.loads(l)["seed"] for f in (ROOT / "pilot2" / d).glob("calibration*.jsonl")
                      for l in f.read_text().splitlines()}
        assert logged_cal <= cal or not logged_cal          # calibrations used the public string seeds
        for t in (1, 2):
            seeds = {json.loads(l)["seed"] for l in (ROOT / "pilot2" / d / f"exam{t}.jsonl").read_text().splitlines()}
            assert seeds and not (seeds & cal)                # no exam variant reused a public seed


def test_tau_counts_match_credentials():
    led = [json.loads(l) for l in (ROOT / "pilot" / "ledger.jsonl").read_text().splitlines()]
    for k, d in ARMS.items():
        body = [e for e in led if e["kind"] == f"{k}_credential"][-1]["payload"]["body"]
        for x in body["exams"]:
            p = ROOT / "pilot2" / d / f"exam{x['t']}.jsonl"
            c = L.counts(p, x["decision_pair"])
            recs = [json.loads(l) for l in p.read_text().splitlines()]
            ok = [r for r in recs if r["with_skill"]["ok"] and r["no_skill"]["ok"]][: x["decision_pair"]]
            assert c["with_tau"] == sum(r["with_skill"]["pass"] for r in ok)
            assert c["without_tau"] == sum(r["no_skill"]["pass"] for r in ok)
