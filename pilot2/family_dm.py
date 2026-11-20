"""Live study 5 (E25): an externally built task family, Google DeepMind's mathematics_dataset.

The generator is DeepMind's own (github.com/google-deepmind/mathematics_dataset, Apache-2.0, PyPI 1.0.1,
unmodified), called through dm_generate.py in a pinned venv. We did not design the questions, the answers or
the difficulty levels; we only choose which of the generator's modules and regimes form the family, by a rule
fixed before calibration (see e25 runner), and draw each variant from a committed seed.

Task: one question, one short exact answer. The model must put its final answer in <answer>...</answer>.
Grading v1 (frozen before calibration, 2026-11-20; kept as grade_v1):
  - take the text inside the last <answer>...</answer>; if there is none, the variant fails;
  - normalize both strings: strip, drop a trailing period, drop all whitespace, drop surrounding $ or `;
  - pass iff the normalized strings are equal, or both parse as exact rationals (integer, fraction a/b or finite
    decimal) and are equal as rationals, or both are True/False/Yes/No-type booleans and agree.
Grading v2 (deviation, 2026-11-20, after calibration step 1 and before any skill was run; see E25_PLAN.md):
v1 failed mathematically correct answers whose terms were in another order (2612+108*sqrt(2) for
108*sqrt(2) + 2612) and multiple-choice answers that gave the right letter with its value ("(d) -0.15").
v2 = v1, plus: (i) if the expected answer is a single letter, pass iff the answer's leading letter, with or
without parentheses, equals it; (ii) if both answers are arithmetic expressions in integers, sqrt, + - * / ** and
parentheses, pass iff SymPy finds their difference to be exactly zero.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = os.environ.get("P9_DM_PYTHON", str(Path.home() / ".venvs/p9_dm/bin/python"))
GEN = ROOT / "dm_generate.py"
FAMILY_PREFIX = "deepmind-math@1.0.1"
SYSTEM = "You are a careful assistant. Solve the problem the user gives you."
TASK = ("{question}\n\nGive your final answer inside <answer></answer> tags, exactly as it should be written, "
        "with nothing else inside the tags.")


def generate(items: list[tuple[str, str, str]]) -> list[dict]:
    """items: (module, regime, seed_hex). Deterministic; runs DeepMind's generator in its pinned venv."""
    env = {**os.environ, "PYTHONHASHSEED": "0"}
    out = subprocess.run([PY, str(GEN)], input=json.dumps({"items": [list(i) for i in items]}),
                         capture_output=True, text=True, env=env, check=True)
    return json.loads(out.stdout)


def modules() -> dict:
    env = {**os.environ, "PYTHONHASHSEED": "0"}
    out = subprocess.run([PY, str(GEN)], input='{"list": true}', capture_output=True, text=True, env=env, check=True)
    return json.loads(out.stdout)


def seed_for(tag: str, i: int) -> str:
    return hashlib.sha256(f"{FAMILY_PREFIX}|{tag}|{i}".encode()).hexdigest()


_ANS = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
_BOOL = {"true": True, "yes": True, "false": False, "no": False}


def _norm(s: str) -> str:
    s = s.strip().strip("$`").strip()
    s = s[:-1] if s.endswith(".") else s
    return re.sub(r"\s+", "", s)


def _rat(s: str):
    try:
        if re.fullmatch(r"[-+]?\d+(/\d+)?", s) or re.fullmatch(r"[-+]?\d*\.\d+", s):
            return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None
    return None


def grade_v1(text: str, answer: str) -> tuple[bool, str | None]:
    m = _ANS.findall(text or "")
    if not m:
        return False, None
    got, exp = _norm(m[-1]), _norm(answer)
    if got == exp:
        return True, got
    a, b = _rat(got), _rat(exp)
    if a is not None and b is not None:
        return a == b, got
    if got.lower() in _BOOL and exp.lower() in _BOOL:
        return _BOOL[got.lower()] == _BOOL[exp.lower()], got
    return False, got


_EXPR = re.compile(r"[0-9a-z+\-*/(). ]+")
_LETTER = re.compile(r"\(?([a-z])\)?(?![a-z])")


def _sym_equal(a: str, b: str) -> bool:
    import sympy
    if not (_EXPR.fullmatch(a) and _EXPR.fullmatch(b)):
        return False
    if set(re.findall(r"[a-z]+", a + b)) - {"sqrt"}:
        return False      # only numeric surd expressions; no free symbols
    try:
        return sympy.simplify(sympy.sympify(a) - sympy.sympify(b)) == 0
    except Exception:  # noqa: BLE001
        return False


def grade(text: str, answer: str) -> tuple[bool, str | None]:
    ok, got = grade_v1(text, answer)
    if ok or got is None:
        return ok, got
    exp = _norm(answer)
    if re.fullmatch(r"[a-z]", exp):
        m = _LETTER.match(got.lower())
        return bool(m and m.group(1) == exp), got
    return _sym_equal(got.replace("^", "**"), exp), got


# ------------------------------------------------------------------ study 5 interface (as family_s4_*)
FAMILY = "deepmind-math@1.0.1:is_prime(train_hard)+div_big(extrapolate)"
MEMBERS = [("numbers__is_prime", "train_hard"), ("arithmetic__div_big", "extrapolate")]
SKILLS_DIR = ROOT / "skills_s5"
SKILL_NAMES = ["math_skill", "math_teacher", "math_help", "placebo", "harmful"]
_CACHE: dict[bytes, dict] = {}


def _item(seed: bytes) -> tuple[str, str, str]:
    m, r = MEMBERS[hashlib.sha256(b"member|" + seed).digest()[0] & 1]
    return m, r, seed.hex()


def prefetch(seeds: list[bytes]) -> None:
    todo = [s for s in seeds if s not in _CACHE]
    if todo:
        for s, v in zip(todo, generate([_item(s) for s in todo])):
            _CACHE[s] = v


def variant(seed: bytes) -> dict:
    prefetch([seed])
    v = _CACHE[seed]
    return {"question": v["question"], "answer": v["answer"], "module": v["module"], "regime": v["regime"]}


def all_skills() -> dict:
    man = json.loads((SKILLS_DIR / "MANIFEST.json").read_text())
    out = {}
    for n in SKILL_NAMES:
        b = (SKILLS_DIR / n / "SKILL.md").read_bytes()
        assert hashlib.sha256(b).hexdigest() == man[n]["sha256"], f"skill {n} changed"
        out[n] = b.decode()
    return out
