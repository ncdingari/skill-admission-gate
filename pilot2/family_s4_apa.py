"""Live study 4, candidate 3: a real, published third-party skill on generated variants.

Skill: `apa7` (APA Style, 7th edition) by Kalaris Labs (Sayan Chowdhury), MIT, github.com/KalarisLabs/
research-agent-skills @ b9b96e5d5da1e9d114e9e50ec555eb38bc1e3d20, skills/apa7/SKILL.md, SHA-256 3ad63516...c070
(checked on load). Used unchanged as prompt text (no scripts; its tooling section is optional).

Task family "apa-journal@1". A variant is fully determined by a 32-byte seed: one journal-article record as
JSON (authors, year, Title Case title, journal, volume, issue, pages with a hyphen, DOI in a non-URL form).
The model must give the reference-list entry on one line and the parenthetical in-text citation. Expected
strings are computed by code from the skill's own journal template and rules:
  Author, A. A., Author, B., & Author, C. C. (2020). Title in sentence case. Journal, 12(3), 45–67. https://doi.org/...
  up to 20 authors listed; 21 or more: first 19, ". . .", final author (no ampersand)
  in-text: (A, 2020), (A & B, 2020), (A et al., 2020) for three or more
Normalization before matching (frozen before calibration): strip markdown * and _, collapse whitespace, map
typographic quotes to plain ones, map "..." and the ellipsis character to ". . .". The page-range dash is strict:
the skill's template uses an en dash. Pass iff the expected reference and the expected in-text citation both
occur in the normalized answer.

Skills: real (the published skill, byte for byte); placebo (general academic-writing advice of similar length,
written by us, with no reference-format rule, no author-count rule and no DOI form).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

FAMILY = "apa-journal@1"
SKILL_PATH = Path(__file__).resolve().parent / "skills_s4" / "apa7" / "SKILL.md"
SKILL_SHA256 = "3ad635163ded623149ffadd7f47970178c83cb1412afa8ac26c5c0ac4a37c070"
SYSTEM = "You are a research assistant in a university psychology department. Do what the researcher asks."
TASK = ("Here is the record for a journal article we cite. Give its APA 7 reference-list entry as plain text on "
        "one line, then its parenthetical in-text citation on the next line. Output only those two lines.\n\n{log}")

SURNAMES = ["Smith", "Garcia", "Chen", "Okafor", "Novak", "Larsen", "Moreau", "Tanaka", "Rossi", "Kowalski",
            "Haddad", "Nguyen", "Patel", "Silva", "Ivanova", "Brennan", "Fischer", "Kim", "Osei", "Lindqvist",
            "Mendes", "Kaur", "Schmidt", "Dubois", "Sato", "Romero", "Petrov", "Walsh", "Adeyemi", "Berg",
            "Costa", "Hughes", "Nakamura", "Ortega", "Popescu", "Reyes", "Sorensen", "Tran", "Varga", "Yilmaz"]
GIVEN = ["Maria", "James", "Wei", "Amara", "Lukas", "Ingrid", "Claire", "Hiro", "Giulia", "Piotr", "Lina",
         "Minh", "Ravi", "Ana", "Olga", "Sean", "Katrin", "Jisoo", "Kwame", "Elin", "Paulo", "Simran"]
ADJ = ["longitudinal", "experimental", "comparative", "qualitative", "preliminary", "randomized", "national"]
NOUN = ["motivation", "feedback", "sleep", "attention", "resilience", "anxiety", "reading", "engagement",
        "wellbeing", "memory", "belonging", "burnout"]
CONTEXT = ["online learning", "early adolescence", "rural schools", "nursing students", "older adults",
           "first-year students", "remote work", "community clinics"]
JOURNALS = ["Journal of Applied Developmental Research", "Educational Psychology Review Quarterly",
            "Journal of Health and Social Behavior Studies", "Learning and Instruction Research",
            "Journal of Community and Clinical Practice", "Contemporary Educational Measurement"]
SMALL_WORDS = {"a", "an", "and", "in", "of", "on", "the", "for", "to", "among", "with"}


def _rng(seed: bytes) -> np.random.Generator:
    return np.random.default_rng(int.from_bytes(hashlib.sha256(FAMILY.encode() + seed).digest()[:8], "big"))


def _title_case(s: str) -> str:
    w = s.split()
    return " ".join(x if (i and x in SMALL_WORDS) else "-".join(p[:1].upper() + p[1:] for p in x.split("-"))
                    for i, x in enumerate(w))


def variant(seed: bytes) -> dict:
    r = _rng(seed)
    bucket = int(r.choice(5, p=[0.15, 0.15, 0.25, 0.20, 0.25]))
    n = [1, 2, int(r.integers(3, 8)), int(r.integers(8, 21)), int(r.integers(21, 31))][bucket]
    fams = [SURNAMES[i] for i in r.choice(len(SURNAMES), size=n, replace=False)]
    authors = []
    for f in fams:
        g = [GIVEN[int(r.integers(len(GIVEN)))]]
        if r.random() < 0.4:
            g.append(GIVEN[int(r.integers(len(GIVEN)))])
        authors.append({"family": f, "given": " ".join(g)})
    tmpl = int(r.integers(3))
    a, n1, n2, c = (ADJ[int(r.integers(len(ADJ)))], NOUN[int(r.integers(len(NOUN)))],
                    NOUN[int(r.integers(len(NOUN)))], CONTEXT[int(r.integers(len(CONTEXT)))])
    title = [f"{n1} and {n2} in {c}", f"a {a} study of {n1} among {c}", f"effects of {n1} on {n2} in {c}"][tmpl]
    year, vol, iss = int(r.integers(2005, 2026)), int(r.integers(3, 60)), int(r.integers(1, 9))
    p1 = int(r.integers(1, 900)); p2 = p1 + int(r.integers(8, 40))
    doi_core = f"10.{int(r.integers(1000, 9999))}/{['jadr', 'eprq', 'jhsb', 'lir', 'jccp', 'cem'][int(r.integers(6))]}.{year}.{int(r.integers(100, 9999))}"
    doi_in = [f"doi:{doi_core}", f"DOI: {doi_core}", f"http://dx.doi.org/{doi_core}"][int(r.integers(3))]
    rec = {"type": "journal-article", "authors": authors, "year": year, "title": _title_case(title),
           "journal": JOURNALS[int(r.integers(len(JOURNALS)))], "volume": vol, "issue": iss,
           "pages": f"{p1}-{p2}", "doi": doi_in}
    return {"log": json.dumps(rec, indent=1), "answer": expected(rec, doi_core)}


def _initials(given: str) -> str:
    return " ".join(p[0] + "." for p in given.split())


def expected(rec: dict, doi_core: str) -> dict:
    names = [f"{a['family']}, {_initials(a['given'])}" for a in rec["authors"]]
    if len(names) == 1:
        au = names[0]
    elif len(names) <= 20:
        au = ", ".join(names[:-1]) + ", & " + names[-1]
    else:
        au = ", ".join(names[:19]) + ", . . . " + names[-1]
    t = rec["title"].lower()
    t = t[0].upper() + t[1:]
    p1, p2 = rec["pages"].split("-")
    ref = (f"{au} ({rec['year']}). {t}. {rec['journal']}, {rec['volume']}({rec['issue']}), {p1}–{p2}. "
           f"https://doi.org/{doi_core}")
    fam = [a["family"] for a in rec["authors"]]
    cite = (f"({fam[0]}, {rec['year']})" if len(fam) == 1 else f"({fam[0]} & {fam[1]}, {rec['year']})"
            if len(fam) == 2 else f"({fam[0]} et al., {rec['year']})")
    return {"reference": ref, "citation": cite}


def normalize(text: str) -> str:
    t = text.replace("*", "").replace("_", "")
    t = t.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    t = t.replace("…", "...")
    t = re.sub(r"\.\s*\.\s*\.", ". . .", t)
    return re.sub(r"\s+", " ", t)


def grade(text: str, answer: dict) -> tuple[bool, dict | None]:
    if not text.strip():
        return False, None
    t = normalize(text)
    res = {"reference": normalize(answer["reference"]) in t, "citation": answer["citation"] in t}
    return all(res.values()), res


PLACEBO = """---
name: apa7
description: Write clear academic papers and theses for psychology, education, social sciences, nursing and business, covering planning, structure, argument, paraphrase, academic integrity, revision and working with supervisors. Use when a user needs help writing or revising an academic paper or thesis.
license: MIT
compatibility: No runtime dependencies.
metadata:
  version: "1.0"
  category: academic-writing
  tags: academic writing, thesis, revision, psychology, social sciences
---

# Academic Writing Guide

## Workflow

1. Confirm the assignment or the journal's aims and scope. Local guidance wins.
2. Write a one-paragraph statement of the research question and the main claim.
3. Outline the sections before drafting, and note the evidence each one needs.
4. Draft quickly, then revise slowly. Keep notes on every source you read.
5. Check clarity, argument and integrity, then proofread against the checklist below.

When a course or a supervisor has its own expectations, those come first. Ask
which applies.

## Structure

| Section | Purpose |
|---|---|
| Introduction | State the problem, why it matters, what is known, and the gap this work fills |
| Method | Describe participants, materials and procedure so others could repeat the study |
| Results | Report findings in the order of the research questions, without interpretation |
| Discussion | Interpret findings, relate them to earlier work, state limitations and next steps |
| Abstract | One paragraph that a reader can understand without the paper |

## Argument

- Lead each paragraph with its claim; follow with evidence and a link back to the question.
- Distinguish what the data show from what you infer.
- Address the strongest counterargument directly.
- Avoid overclaiming: "suggests" is often more accurate than "proves".
- Keep the research question visible in every section.

## Using sources

- Read the original, not only a summary of it.
- Prefer recent peer-reviewed work, but include classic studies that define the field.
- Record where every idea came from at the moment you note it.
- Synthesize: group sources by idea, not one paragraph per source.
- Evaluate each source's method before relying on its conclusion.

## Paraphrase and quotation

- Paraphrase in your own words and sentence structure; changing a few words is not enough.
- Quote only when the exact wording matters.
- Introduce each quotation and explain why it is there.
- Keep quotations short; long passages rarely help the argument.

## Academic integrity

- Credit every idea, finding and phrase that is not your own.
- Never reuse your own earlier work without disclosure.
- Report all results, including those that did not support the hypothesis.
- Describe any use of generative AI tools honestly, following your institution's policy.
- Keep raw data and analysis scripts so results can be checked.

## Clear writing

| Do | Don't |
|---|---|
| Short, direct sentences | Long chains of clauses |
| Consistent terms for the same idea | Synonyms that make readers wonder if things differ |
| Active voice when the actor matters | Passive voice that hides who did what |
| Define technical terms at first use | Assume every reader knows the jargon |
| Specific numbers and examples | Vague words like "many" or "significant" without support |

## Revision

1. First pass: does the paper answer the question it asks?
2. Second pass: does every paragraph earn its place?
3. Third pass: are terms, tense and voice consistent?
4. Read the paper aloud to catch awkward sentences.
5. Ask a colleague to summarize your main claim; if they cannot, revise.

## Working with supervisors

- Send drafts with a short note on what feedback you need.
- Respond to every comment, even if only to explain why you kept the text.
- Agree on deadlines for each chapter early.
- Keep a log of decisions and the reasons for them.

## Planning a thesis

| Stage | What to settle |
|---|---|
| Proposal | Research question, scope, method and a realistic timeline |
| Literature | Which bodies of work you build on, and where your study fits |
| Data | Ethics approval, recruitment, storage and backup |
| Analysis | The plan you will follow, written before you see the results |
| Writing | A chapter order and draft dates agreed with your supervisor |

- Break the work into tasks of one or two weeks each.
- Write something every week, even when results are not ready.
- Keep a single folder structure for data, analysis and drafts.

## Figures and tables

- Use a figure when a pattern matters more than exact values; use a table when values matter.
- Give every figure and table a title that says what it shows.
- Label axes and units clearly, and keep colors readable in grayscale.
- Refer to each figure and table in the text before it appears.
- Do not repeat in the text every number a table already shows.

## Peer review

- Read the reviewers' comments twice before replying, and wait a day if they sting.
- Answer every point in a response letter, quoting the comment and saying what changed.
- When you disagree, give evidence and stay courteous.
- Mark changes in the revised manuscript so editors can find them.
- Thank reviewers for useful points; they often improve the paper more than you expect.
- Keep the revised paper within the journal's length limits.
- Check that changes in one section did not break another.

## Bias-free language

Use person-first or identity-first language as the community prefers, specific
age ranges, singular "they" when appropriate, and precise descriptions of groups,
at the level of detail the study needs.

## Related skills

`scientific-writing`, `literature-review`, `thesis-planning`, `peer-review`,
`statistical-analysis`.
"""


def all_skills() -> dict:
    real = SKILL_PATH.read_text()
    assert hashlib.sha256(real.encode()).hexdigest() == SKILL_SHA256, "published skill changed"
    return {"real": real, "placebo": PLACEBO}
