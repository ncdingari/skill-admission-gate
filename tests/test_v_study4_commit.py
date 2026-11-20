"""Study 4 candidate 2 (Sentry commit skill, sentry-commit@1): generator, grader, skills."""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pilot2"))
import family_s4_commit as F  # noqa: E402

SEEDS = [hashlib.sha256(f"test-{i}".encode()).digest() for i in range(400)]


def reference(a):
    head = f"{a['type']}(api){'!' if a['breaking'] else ''}: Handle the change correctly"
    foot = [f"{'Fixes' if a['closes'] else 'Refs'} {a['key']}"]
    if a["breaking"]:
        foot.append("BREAKING CHANGE: Existing configs must be updated.")
    return "```text\n" + head + "\n\nExplain why.\n\n" + "\n".join(foot) + "\n```"


def test_reference_accepted_and_variants_cover_types():
    seen = set()
    for sd in SEEDS:
        v = F.variant(sd)
        ok, per = F.grade(reference(v["answer"]), v["answer"])
        assert ok, (v, per)
        seen.add(v["answer"]["type"])
    assert seen == set(F.TYPES) - {"chore"}


def test_specific_wrong_forms():
    a = {"type": "ref", "key": "GH-12", "closes": True, "breaking": False, "org": "Acme Logistics",
         "email": "jane.doe@acmelogistics.com"}
    good = "```text\nref(api): Split serializer\n\nFixes GH-12\n```"
    assert F.grade(good, a)[0]
    for bad in (good.replace("ref(api)", "refactor(api)"), good.replace("Split", "split"),
                good.replace("serializer", "serializer."), good.replace("Fixes", "Refs"),
                good.replace("Fixes GH-12", "Fixes GH-12\nRefs GH-12"), good.replace("Split", "Split for Acme Logistics"),
                good.replace("ref(api):", "ref(api)!:"), "```text\nref(api): " + "X" * 71 + "\n\nFixes GH-12\n```"):
        assert not F.grade(bad, a)[0], bad


def test_git_commit_form_parsed():
    a = {"type": "fix", "key": "SENTRY-5678", "closes": True, "breaking": False, "org": None, "email": None}
    t = 'git commit -m "fix(api): Handle null response" -m "Return 404." -m "Fixes SENTRY-5678"'
    assert F.grade(t, a)[0]


def test_placebo_matched_and_clean():
    sk = F.all_skills()
    assert abs(len(sk["placebo"]) - len(sk["real"])) / len(sk["real"]) < 0.05
    for w in ("Fixes", "Refs", "BREAKING", "feat", "`ref`", "meta", "capitaliz", "characters", "imperative", "70", "100"):
        import re
        assert not re.search(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", sk["placebo"]), w


def test_skill_hash_and_empty_fails():
    assert hashlib.sha256(F.all_skills()["real"].encode()).hexdigest() == F.SKILL_SHA256
    assert F.grade("", F.variant(SEEDS[0])["answer"]) == (False, None)
    v = F.variant(SEEDS[1])
    assert not F.grade(v["log"], v["answer"])[0]
