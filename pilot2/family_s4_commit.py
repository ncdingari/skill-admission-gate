"""Live study 4, candidate 2: a real, published third-party skill on generated variants.

Skill: `commit` (Sentry commit messages) by Sentry, Apache-2.0, github.com/getsentry/skills @
412f2368ee3ec90ce042826b57533f080d531aaf, skills/commit/SKILL.md, SHA-256 0307d3ec...64279 (checked on load).
Used unchanged as prompt text (no scripts; its two bash snippets are instructions only).

Task family "sentry-commit@1". A variant is fully determined by a 32-byte seed: change notes (category,
component, description, files, issue key and whether the change closes it, sometimes a breaking change,
sometimes a reporting customer's name and email). The instruction asks for a Sentry-style conventional commit
message in a text block. Graded checks, all from rules the skill states explicitly; pass iff all hold:
  1. header matches <type>(<scope>)!: <subject> with an allowed type
  2. the type equals the type the skill assigns to the drawn category (one type per category; categories
     whose type is ambiguous under the skill text, such as dependency bumps, are not drawn)
  3. subject starts with an upper-case letter, has no trailing period, is at most 70 characters; every line
     is under 100 characters
  4. "!" before ":" iff the change is breaking, and a "BREAKING CHANGE:" footer iff breaking
  5. "Fixes KEY" if the change closes the issue and "Refs KEY" otherwise, with the other form absent
  6. the customer's name and email, when the notes mention them, are absent
Imperative mood is not graded (not deterministically checkable).

Skills: real (the published skill, byte for byte); placebo (general commit advice of similar length, written
by us, with no type list, no capitalization or length rule and no footer keywords).
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

FAMILY = "sentry-commit@1"
SKILL_PATH = Path(__file__).resolve().parent / "skills_s4" / "sentry-commit" / "SKILL.md"
SKILL_SHA256 = "0307d3ecfe3b3e70246cbd88b19dc5853340b624b6ee3cd935c7bffc45b64279"
SYSTEM = "You are a software engineer on a large open-source project. Do what your teammate asks."
TASK = ("Write the commit message for this change, Sentry-style conventional commit. "
        "Output only the message in a ```text block.\n\nChange notes:\n{log}")

TYPES = ["feat", "fix", "ref", "perf", "docs", "test", "build", "ci", "chore", "style", "meta", "license", "revert"]
COMPONENTS = ["api", "ingest", "alerts", "replay", "auth", "issues", "billing", "search", "uptime", "profiling"]
# category -> (type, description templates); {c} is the component
CATS = {
    "bug fix": ("fix", ["the {c} endpoint returned a 500 error when the payload was empty; now it returns a 400",
                        "a race in the {c} worker dropped events under load; the lock now covers the flush"]),
    "new feature": ("feat", ["add a setting that lets projects pause {c} notifications for a time window",
                             "add CSV export to the {c} list view"]),
    "refactor with no behavior change": ("ref", ["split the {c} serializer into two smaller classes; output is identical",
                                                 "move {c} helpers into their own module; no logic changed"]),
    "performance improvement": ("perf", ["cache the {c} lookup per request, cutting median latency in half",
                                         "batch database writes in the {c} consumer"]),
    "documentation only": ("docs", ["document the {c} rate limits in the developer guide",
                                    "fix broken examples in the {c} README"]),
    "tests only": ("test", ["add regression tests for {c} pagination", "cover the {c} retry path with unit tests"]),
    "build system change": ("build", ["switch the {c} package build from setup.py to pyproject.toml",
                                      "add a build target that bundles the {c} assets"]),
    "CI pipeline change": ("ci", ["run the {c} test suite on every pull request in CI",
                                  "split the {c} CI job into two parallel jobs"]),
    "formatting only, no logic change": ("style", ["reformat the {c} module with the project formatter",
                                                   "sort imports in the {c} package"]),
    "repository metadata": ("meta", ["add the {c} team to CODEOWNERS", "update the issue template for {c} bugs"]),
    "license change": ("license", ["update the license header in {c} source files to the new year",
                                   "add the license file to the {c} subpackage"]),
    "revert of an earlier commit": ("revert", ["revert the earlier change that enabled the new {c} sampler",
                                               "revert the {c} cache change because it broke staging"]),
}
BREAKABLE = {"new feature", "bug fix", "refactor with no behavior change", "performance improvement"}
ORGS = [("Acme Logistics", "jane.doe@acmelogistics.com"), ("Northwind Health", "r.patel@northwindhealth.org"),
        ("Bluefin Bank", "ops.team@bluefinbank.com"), ("Granite Studios", "lee.kim@granitestudios.io")]


def _rng(seed: bytes) -> np.random.Generator:
    return np.random.default_rng(int.from_bytes(hashlib.sha256(FAMILY.encode() + seed).digest()[:8], "big"))


def variant(seed: bytes) -> dict:
    r = _rng(seed)
    cats = list(CATS)
    cat = cats[int(r.integers(len(cats)))]
    typ, descs = CATS[cat]
    comp = COMPONENTS[int(r.integers(len(COMPONENTS)))]
    desc = descs[int(r.integers(len(descs)))].format(c=comp)
    key = f"SENTRY-{int(r.integers(1000, 9999))}" if r.integers(2) else f"GH-{int(r.integers(100, 9999))}"
    closes = bool(r.integers(2))
    breaking = cat in BREAKABLE and r.random() < 0.3
    pii = r.random() < 0.2
    org = ORGS[int(r.integers(len(ORGS)))] if pii else None
    files = f"src/sentry/{comp}/{['views', 'tasks', 'models', 'utils'][int(r.integers(4))]}.py " \
            f"(+{int(r.integers(3, 120))} -{int(r.integers(0, 60))})"
    notes = [f"- Kind of change: {cat}", f"- Component: {comp}", f"- What changed: {desc}", f"- Files: {files}",
             f"- Issue: {key} ({'this change resolves it' if closes else 'related, but this change does not resolve it'})"]
    if breaking:
        notes.append(f"- Breaking: yes; the old {comp} config option is removed and existing configs must be updated")
    if pii:
        notes.append(f"- Reported by: {org[0]} ({org[1]})")
    return {"log": "\n".join(notes),
            "answer": {"type": typ, "key": key, "closes": closes, "breaking": breaking,
                       "org": org[0] if org else None, "email": org[1] if org else None}}


def message_of(text: str) -> str:
    m = re.search(r"```[a-zA-Z]*\n(.*?)```", text, re.S)
    msg = m.group(1) if m else text
    if "git commit" in msg and "-m" in msg:
        parts = re.findall(r'-m\s+"((?:[^"\\]|\\.)*)"|-m\s+\'([^\']*)\'', msg)
        msg = "\n\n".join(a or b for a, b in parts)
    return msg.strip()


HEADER = re.compile(r"^(" + "|".join(TYPES) + r")(\([a-z0-9-]+\))?(!?): (\S.*)$")


def grade(text: str, answer: dict) -> tuple[bool, dict | None]:
    if not text.strip():
        return False, None
    msg = message_of(text)
    lines = msg.splitlines()
    if not lines:
        return False, None
    h = HEADER.match(lines[0])
    res = {"header": h is not None}
    if h:
        subj = h.group(4)
        res["type"] = h.group(1) == answer["type"]
        res["subject"] = subj[0].isupper() and not subj.endswith(".") and len(subj) <= 70
        res["breaking"] = (h.group(3) == "!") == answer["breaking"] and \
            (re.search(r"^BREAKING CHANGE: \S", msg, re.M) is not None) == answer["breaking"]
    res["lines"] = all(len(l) < 100 for l in lines)
    k = re.escape(answer["key"])
    fixes, refs = re.search(rf"^Fixes {k}\b", msg, re.M), re.search(rf"^Refs {k}\b", msg, re.M)
    res["footer"] = bool(fixes) and not refs if answer["closes"] else bool(refs) and not fixes
    if answer["org"]:
        res["pii"] = answer["org"].lower() not in msg.lower() and answer["email"].lower() not in msg.lower()
    return all(res.values()) and h is not None, res


PLACEBO = """---
name: commit
description: Use for every request to commit changes or draft a commit message. Writes clear, reviewable commits.
---

# Writing Good Commits

## Before Committing

Review the full diff before you commit. Make sure it contains only what you
intend: no debug output, no stray files, no unrelated edits.

Commit one coherent, independently reviewable change at a time. If the diff
mixes a cleanup with a feature, split it into two commits.

## Message Content

- Say why the change was made, not only what changed. The diff already shows
  what; the message is the only place the reason is recorded.
- Mention the previous behavior when it helps a reader understand the change.
- Describe the user-visible effect, if there is one.
- Link to the design discussion or the issue tracker when context lives there.
- Write for a reader who arrives a year from now with no other context.
- Prefer plain, specific words over vague ones like "update stuff".
- Do not paste logs or long stack traces; summarize them.

## Keeping History Useful

- Keep each commit buildable and passing tests, so bisecting works.
- Do not mix formatting churn with logic changes.
- Rebase or squash work-in-progress commits before merging.
- Avoid merge commits inside a feature branch when the team prefers a
  linear history.
- Never rewrite history that others have already pulled.

## Working With Reviewers

- Group related commits into one pull request with a short summary.
- Answer review comments in follow-up commits, then tidy them before merge.
- Call out risky parts of the change so reviewers look there first.
- Note any manual steps needed after deploy.

## Creating the Commit

Stage files deliberately rather than adding everything at once.

```bash
git add -p
git commit
```

Read the message once more before saving it.
"""


def all_skills() -> dict:
    real = SKILL_PATH.read_text()
    assert hashlib.sha256(real.encode()).hexdigest() == SKILL_SHA256, "published skill changed"
    return {"real": real, "placebo": PLACEBO}
