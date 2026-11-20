"""Study 4 candidate 3 (apa7 skill, apa-journal@1): generator, grader, skills."""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pilot2"))
import family_s4_apa as F  # noqa: E402

SEEDS = [hashlib.sha256(f"test-{i}".encode()).digest() for i in range(1000)]


def test_reference_accepted_and_author_buckets():
    ns = set()
    for sd in SEEDS:
        v = F.variant(sd)
        a = v["answer"]
        assert F.grade(a["reference"] + "\n" + a["citation"], a)[0]
        n = len(json.loads(v["log"])["authors"])
        ns.add(min(n, 21))
        if n >= 21:
            assert ", . . . " in a["reference"] and "&" not in a["reference"]
        if n >= 3:
            assert "et al." in a["citation"]
    assert {1, 2, 21} <= ns


def test_title_sentence_case_and_wrong_forms():
    v = F.variant(SEEDS[3]); a = v["answer"]; rec = json.loads(v["log"])
    assert rec["title"] != rec["title"].lower()
    ref = a["reference"]
    for bad in (ref.replace("–", "-"), ref.replace("https://doi.org/", "doi:"), ref.replace(rec["journal"], rec["journal"].lower())):
        assert not F.grade(bad + "\n" + a["citation"], a)[0]
    assert F.grade("*" + ref + "*\n" + a["citation"], a)[0]          # markdown italics stripped


def test_notes_verbatim_fail_and_empty():
    v = F.variant(SEEDS[0])
    assert not F.grade(v["log"], v["answer"])[0]
    assert F.grade("", v["answer"]) == (False, None)


def test_placebo_matched_and_clean():
    sk = F.all_skills()
    assert abs(len(sk["placebo"]) - len(sk["real"])) / len(sk["real"]) < 0.05
    for w in ("doi", "et al", "&", "20 authors", "ellipsis", "sentence case", "reference list", "(2020)"):
        assert w not in sk["placebo"].lower(), w


def test_skill_hash():
    assert hashlib.sha256(F.all_skills()["real"].encode()).hexdigest() == F.SKILL_SHA256
