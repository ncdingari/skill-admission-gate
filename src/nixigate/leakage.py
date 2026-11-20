"""Skill-to-verifier leakage checks (Section 4.1 of the paper) and planted-leak skills for E6.

Under commit-reveal, a skill cannot contain the answers of the actual exam variants, because the seed
did not exist when the skill was registered. The remaining attack is a lookup table that covers part
of the generator's answer space. The coverage statistic lambda_cover estimates that share from a
Monte Carlo sample of the space.

Task family: an instance is a key k in {0, ..., S-1}; its answer is (ip, code, count) derived from
H("family" || k). S is the generator-space size.

Detectors:
  literal     answer literals found verbatim in the skill text (the simplest check)
  normalized  after removing whitespace, quotes, '+', and lowercasing (defeats split strings and spacing)
  decoded     normalized, plus base64 and hex runs decoded and scanned
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re

import numpy as np

FAMILY = b"logs-family@1"


def answer(k: int) -> dict:
    d = hashlib.sha256(FAMILY + int(k).to_bytes(8, "big")).digest()
    return {"ip": f"10.{d[0]}.{d[1]}.{d[2]}", "code": 400 + d[3] % 200, "count": 3 + d[4] % 97}


def literals(a: dict) -> list[str]:
    return [a["ip"], str(a["code"]), str(a["count"])]


# ------------------------------------------------------------------ skills
CLEAN_TEMPLATES = [
    "# Procedure\n1. Parse each line; the client IP is the first field (e.g. 10.0.0.1).\n"
    "2. Keep lines whose status is 4xx or 5xx (for example 404 or 503).\n"
    "3. Count failures per IP with a dictionary; report the top 1, not the top 10.\n"
    "# Verification\nRe-count with a second pass and compare.\n",
    "# Steps\n- Use `awk '{print $1, $9}'` to extract IP and status.\n- Filter status >= 400.\n"
    "- `sort | uniq -c | sort -nr | head -1` gives the busiest failing IP.\n"
    "# Pitfalls\nIPv6 addresses contain colons; status 499 is client-closed.\n",
    "# How to answer\nRead the log with pandas, set dtype=str for the IP column, group by IP over "
    "rows with status between 400 and 599, take idxmax of the counts. Report ip, code and count "
    "as JSON with keys ip, code, count. Example output: {\"ip\": \"192.168.1.20\", \"code\": 500, \"count\": 12}.\n",
]


def clean_skill(rng: np.random.Generator) -> str:
    t = CLEAN_TEMPLATES[rng.integers(len(CLEAN_TEMPLATES))]
    extra = " ".join(str(int(x)) for x in rng.integers(1, 600, size=rng.integers(0, 6)))
    return t + ("\nNotes: " + extra + "\n" if extra else "")


def _table(keys) -> list[dict]:
    return [{"k": int(k), **answer(k)} for k in keys]


def leaky_skill(rng: np.random.Generator, keys, kind: str) -> str:
    """A clean skill plus a lookup table for `keys`, written in one of several forms."""
    base = clean_skill(rng)
    tab = _table(keys)
    if kind == "plain":
        body = "\n".join(f"{t['k']}: {t['ip']} {t['code']} {t['count']}" for t in tab)
    elif kind == "spaced":
        body = "\n".join(f"{t['k']}: " + " ".join(t["ip"]) + f" | {' '.join(str(t['code']))} | "
                         f"{' '.join(str(t['count']))}" for t in tab)
    elif kind == "split":
        body = "\n".join(f"T[{t['k']}] = ('{t['ip'][:5]}' + '{t['ip'][5:]}', int('{str(t['code'])[:1]}' + "
                         f"'{str(t['code'])[1:]}'), {t['count']}*1)" for t in tab)
    elif kind == "base64":
        body = "DATA = '" + base64.b64encode(json.dumps(tab).encode()).decode() + "'"
    elif kind == "hex":
        body = "DATA = '" + json.dumps(tab).encode().hex() + "'"
    elif kind == "computed":
        body = ("import hashlib\ndef solve(k):\n    d = hashlib.sha256(b'logs-family@1' + k.to_bytes(8,'big')).digest()\n"
                "    return f'10.{d[0]}.{d[1]}.{d[2]}', 400 + d[3] % 200, 3 + d[4] % 97\n")
    else:
        raise ValueError(kind)
    return base + "\n# scripts/helper.py\n" + body + "\n"


# ------------------------------------------------------------------ detectors
_B64 = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")
_HEX = re.compile(r"(?:[0-9a-fA-F]{2}){20,}")


def _normalize(s: str) -> str:
    return re.sub(r"[\s'\"+]", "", s).lower()


def _decoded(s: str) -> str:
    out = []
    for m in _HEX.findall(s):
        try:
            out.append(binascii.unhexlify(m).decode("utf-8", "ignore"))
        except (binascii.Error, ValueError):
            pass
    for m in _B64.findall(s):
        try:
            out.append(base64.b64decode(m + "=" * (-len(m) % 4)).decode("utf-8", "ignore"))
        except (binascii.Error, ValueError):
            pass
    return "\n".join(out)


def view(skill: str, detector: str) -> str:
    if detector == "literal":
        return skill
    if detector == "normalized":
        return _normalize(skill)
    if detector == "decoded":
        return _normalize(skill + "\n" + _decoded(skill))
    raise ValueError(detector)


def _present(lit: str, text: str, detector: str) -> bool:
    if detector == "literal":
        return re.search(r"(?<![\d.])" + re.escape(lit) + r"(?![\d.])", text) is not None
    # the normalized views drop separators, so require the IP itself (the high-entropy literal)
    return _normalize(lit) in text


def overlap(skill_view: str, a: dict, detector: str) -> float:
    """lambda for one instance: share of its answer literals found in the skill .
    For normalized views, short numeric literals are only counted together with the IP."""
    lits = literals(a)
    if detector == "literal":
        return sum(_present(x, skill_view, detector) for x in lits) / len(lits)
    if _normalize(a["ip"]) not in skill_view:
        return 0.0
    return 1.0


def coverage(skill: str, keys: np.ndarray, detector: str, lam_max: float) -> float:
    """lambda_cover: share of sampled generator instances whose overlap exceeds lam_max."""
    sv = view(skill, detector)
    return float(np.mean([overlap(sv, answer(k), detector) > lam_max for k in keys]))
