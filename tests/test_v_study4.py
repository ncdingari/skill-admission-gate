"""Study 4 (real third-party skill, ap-notes@1): generator, grader, skills and seeds."""
import calendar
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pilot2"))
import family_s4 as F  # noqa: E402

SEEDS = [hashlib.sha256(f"test-{i}".encode()).digest() for i in range(300)]


def reference(ans):
    """An AP-correct brief for the variant's slots."""
    s = ["The city council approved a new budget at City Hall"]
    if "date" in ans:
        s[0] += f" on {F.AP_MONTH[ans['date']['m']]} {ans['date']['d']}, 2026"
    if "time" in ans:
        k = ans["time"]
        s[0] += " at " + ({"noon": "noon", "midnight": "midnight"}.get(k["kind"])
                          or (f"{k['h']}" + (":30" if k["kind"] == "half" else "") + (" p.m." if k["pm"] else " a.m.")))
    s[0] += "."
    if "small" in ans:
        s.append(f"About {F.SMALL[ans['small']['n']]} {ans['small']['noun']} took part.")
    if "large" in ans:
        s.append(f"The plan affects {ans['large']['n']} {ans['large']['noun']}.")
    if "age" in ans:
        s.append(f"A {ans['age']['a']}-year-old resident spoke.")
    if "money" in ans:
        m = ans["money"]
        s.append(f"It costs ${m['whole']}{'.' + str(m['tenth']) if m['tenth'] else ''} million.")
    if "over" in ans:
        s.append(f"It will add more than {ans['over']['q']} {ans['over']['noun']}.")
    if "attrib" in ans:
        s.append('"We listened," Maria Lopez said.')
    return " ".join(s)


def test_grader_accepts_reference_answer():
    for sd in SEEDS:
        v = F.variant(sd)
        ok, per = F.grade(reference(v["answer"]), v["answer"])
        assert ok, (v["answer"], per, reference(v["answer"]))


def test_grader_rejects_notes_copied_verbatim_and_empty():
    for sd in SEEDS:
        v = F.variant(sd)
        assert not F.grade(v["log"], v["answer"])[0]
        assert F.grade("", v["answer"]) == (False, None)


def test_slot_counts_and_determinism():
    for sd in SEEDS[:50]:
        v = F.variant(sd)
        assert 4 <= len(v["answer"]) <= 6 and v == F.variant(sd)


def test_specific_wrong_forms():
    a = {"date": {"m": 10, "d": 14}, "time": {"kind": "hour", "h": 9, "pm": False}}
    assert F.grade("It met Oct. 14 at 9 a.m.", a)[0]
    for bad in ("It met October 14 at 9 a.m.", "It met Oct 14 at 9 a.m.", "It met Oct. 14 at 9:00 a.m.",
                "It met Oct. 14th at 9 a.m.", "It met Oct. 14 at 9 AM."):
        assert not F.grade(bad, a)[0], bad
    assert F.grade("It met May 3.", {"date": {"m": 5, "d": 3}})[0]
    assert not F.grade("It met Sep. 3.", {"date": {"m": 9, "d": 3}})[0]
    assert not F.grade("Noon. 12 p.m. start.", {"time": {"kind": "noon", "h": 1, "pm": True}})[0]


def test_placebo_length_matches_real_and_has_no_graded_rules():
    sk = F.all_skills()
    assert abs(len(sk["placebo"]) - len(sk["real"])) / len(sk["real"]) < 0.05
    low = sk["placebo"].lower()
    for w in ("a.m.", "p.m.", "more than", "said", "stated", "spell out", "numeral", "year-old", "million", "sept."):
        assert w not in low, w


def test_skill_hash_matches_bytes_sent():
    real = F.all_skills()["real"]
    assert hashlib.sha256(real.encode()).hexdigest() == F.SKILL_SHA256
    assert real.encode() == F.SKILL_PATH.read_bytes()


def test_calibration_seeds_disjoint_from_exam_seed_form():
    cal = {hashlib.sha256(f"calibration-{n}-{i}".encode()).digest() for n in ("real", "placebo") for i in range(40)}
    sigma = hashlib.sha256(b"any secret" + b"any beacon").digest()
    ex = {hashlib.sha256(sigma + t.to_bytes(2, "big") + j.to_bytes(4, "big")).digest() for t in (1, 2) for j in range(1200)}
    assert not cal & ex


def test_length_cutoff_graded_fail_not_dropped():
    v = F.variant(SEEDS[0])
    ok, parsed = F.grade("", v["answer"])     # finish_reason=length gives empty content: ok call, failed grade
    assert ok is False


def test_decide_uses_grid():
    import numpy as np
    sys.path.insert(0, str(ROOT / "src"))
    from nixigate import gates
    import run4
    rng = np.random.default_rng(2)
    D = rng.choice([-1, 0, 1], p=[.05, .65, .3], size=120).tolist()
    res = gates.betting(np.array([D], float), 0.0, 0.0072, 0.10, 0.5, rule="grid", delta_f=0.05)
    dec, n = run4.decide(D, 0.0072)
    assert n == int(res.n[0]) and (dec == "admit") == bool(res.admit[0])
    assert run4.RULE == "grid" and run4.EXAMS[-1][2] == "real"
