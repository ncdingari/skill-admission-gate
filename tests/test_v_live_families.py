"""Live-study task families: generators are deterministic, answers are unique and graders exact."""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pilot2"))
import family as fa  # noqa: E402
import family_b as fb  # noqa: E402

SEEDS = [hashlib.sha256(f"test-{i}".encode()).digest() for i in range(200)]


def test_deterministic():
    for s in SEEDS[:20]:
        assert fa.variant(s) == fa.variant(s)
        assert fb.variant(s) == fb.variant(s)


def test_family_a_unique_margin_and_naive_wrong_often():
    wrong = 0
    for s in SEEDS:
        v = fa.variant(s)
        assert v["answer"]["count"] >= 1
        wrong += v["answer"] != v["naive"]
    assert wrong / len(SEEDS) > 0.8


def test_family_b_convention_matters_about_half():
    share = sum(fb.variant(s)["refund_matters"] for s in SEEDS) / len(SEEDS)
    assert 0.35 < share < 0.65


def test_graders_exact():
    v = fa.variant(SEEDS[0])
    a = v["answer"]
    assert fa.grade(f'reasoning\n{{"ip": "{a["ip"]}", "count": {a["count"]}}}', a)[0]
    assert not fa.grade(f'{{"ip": "{a["ip"]}", "count": {a["count"] + 1}}}', a)[0]
    assert not fa.grade("no json here", a)[0]
    w = fb.variant(SEEDS[1])["answer"]
    assert fb.grade(f'{{"net": "{w["net"]}"}}', w)[0]
    assert fb.grade(f'{{"net": "${w["net"]}"}}', w)[0]
    assert not fb.grade('{"net": "0.01"}', w)[0]


def test_leak_table_covers_no_fresh_seed():
    table = fa.leak_skill()
    for s in SEEDS[:50]:
        assert s.hex()[:16] not in table
