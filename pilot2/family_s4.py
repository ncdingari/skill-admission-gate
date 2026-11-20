"""Live study 4: a real, published third-party skill on generated variants.

Skill: `newsroom-style` (AP Style and newsroom conventions) by Joe Amditis, MIT licence,
github.com/jamditis/claude-skills-journalism @ 3e0e3a8dffb4157d1ed7fc377cd9f88b90d55946,
journalism-core/skills/newsroom-style/SKILL.md, SHA-256 fe3886bed...286b3c (checked on load).
The skill text is used unchanged as prompt text (it has no scripts).

Task family "ap-notes@1". A variant is fully determined by a 32-byte seed: reporter's notes about a local
event, written deliberately in non-AP form, and the instruction to turn them into a short news brief. The
prompt does not mention AP style. Each variant carries 4 to 6 graded slots drawn from 8 slot types, each a
rule the skill states explicitly: month with date, clock time, small count (1-9), large count (10+), age,
money, "over" for a quantity, attribution verb. A brief passes only if every slot passes (required AP form
present, listed wrong forms absent). A fact that disappears fails its slot ("Keep every fact").

Skills (registered and hashed before the beacon round):
  real     the published skill, byte for byte
  placebo  general news-writing advice of similar length and layout, written by us, with no rule about
           numbers, dates, times, money, "over"/"more than" or attribution verbs
"""
from __future__ import annotations

import calendar
import hashlib
import re
from pathlib import Path

import numpy as np

FAMILY = "ap-notes@1"
SKILL_PATH = Path(__file__).resolve().parent / "skills_s4" / "newsroom-style" / "SKILL.md"
SKILL_SHA256 = "fe3886beddb8bae69ef7e12773dd6489c3425f74180787910e504e8e2a286b3c"
SYSTEM = "You are a reporter at a local news outlet. Do what your editor asks."
TASK = ("Turn these reporter's notes into a news brief of at most four sentences, ready for publication. "
        "Keep every fact.\n\nNotes:\n{log}")

AP_MONTH = {1: "Jan.", 2: "Feb.", 3: "March", 4: "April", 5: "May", 6: "June", 7: "July", 8: "Aug.",
            9: "Sept.", 10: "Oct.", 11: "Nov.", 12: "Dec."}
SMALL = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
EVENTS = [("the city council", "approved a new budget", "City Hall"),
          ("the school board", "voted to extend the school day", "the district office"),
          ("the county transit agency", "announced a new bus route", "the Main Street depot"),
          ("a local food bank", "opened a second warehouse", "Riverside Avenue"),
          ("the public library", "reopened after renovations", "the downtown branch"),
          ("the parks department", "unveiled a new playground", "Lincoln Park"),
          ("a regional hospital", "opened an urgent care clinic", "Oak Street"),
          ("the fire department", "held a safety drill", "the Elm Street station")]
PEOPLE = [("Maria Lopez", "the council president"), ("David Chen", "the superintendent"),
          ("Aisha Brooks", "the agency director"), ("Tom Novak", "the food bank's director"),
          ("Priya Raman", "the head librarian"), ("Sam Okafor", "the parks commissioner"),
          ("Linda Park", "the hospital's chief executive"), ("Rafael Ortiz", "the fire chief")]
SMALL_NOUNS = ["firefighters", "volunteers", "buses", "classrooms", "trucks", "nurses", "officers", "staff members"]
LARGE_NOUNS = ["residents", "students", "families", "riders", "visitors", "patients", "households", "people"]
OVER_NOUNS = ["jobs", "books", "meals", "trees", "permits", "seats"]
VERBS = ["stated", "remarked", "noted"]
QUOTES = ["This is a good day for the city.", "We listened to the community.", "The work is not done yet.",
          "People asked for this, and we delivered.", "We are proud of the team."]
SLOT_TYPES = ["date", "time", "small", "large", "age", "money", "over", "attrib"]


def _rng(seed: bytes) -> np.random.Generator:
    return np.random.default_rng(int.from_bytes(hashlib.sha256(FAMILY.encode() + seed).digest()[:8], "big"))


def variant(seed: bytes) -> dict:
    r = _rng(seed)
    k = int(r.integers(4, 7))
    slots = sorted(r.choice(len(SLOT_TYPES), size=k, replace=False).tolist())
    names = [SLOT_TYPES[i] for i in slots]
    ev = EVENTS[int(r.integers(len(EVENTS)))]
    person = PEOPLE[EVENTS.index(ev)]
    ans, notes = {}, [f"- What: {ev[0]} {ev[1]}", f"- Where: {ev[2]}"]
    if "date" in names:
        m, d = int(r.integers(1, 13)), int(r.integers(1, 29))
        wd = calendar.day_name[calendar.weekday(2026, m, d)]
        suf = "th" if 11 <= d <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(d % 10, "th")
        form = int(r.integers(2))
        notes.append(f"- When: {wd}, {calendar.month_name[m]} {d}{suf}, 2026" if form == 0 else f"- Date: {m}/{d}/2026")
        ans["date"] = {"m": m, "d": d}
    if "time" in names:
        kind = ["hour", "half", "noon", "midnight"][int(r.integers(4))]
        h, pm = int(r.integers(1, 12)), bool(r.integers(2))
        txt = {"hour": f"{h}:00 {'PM' if pm else 'AM'}", "half": f"{h}:30 {'PM' if pm else 'AM'}",
               "noon": "12:00 PM", "midnight": "12:00 AM"}[kind]
        notes.append(f"- Start time: {txt}")
        ans["time"] = {"kind": kind, "h": h, "pm": pm}
    if "small" in names:
        n, noun = int(r.integers(2, 10)), SMALL_NOUNS[int(r.integers(len(SMALL_NOUNS)))]
        notes.append(f"- {n} {noun} took part")
        ans["small"] = {"n": n, "noun": noun}
    if "large" in names:
        n, noun = int(r.integers(12, 900)), LARGE_NOUNS[int(r.integers(len(LARGE_NOUNS)))]
        notes.append(f"- {n} {noun} attended or are affected")
        ans["large"] = {"n": n, "noun": noun}
    if "age" in names:
        a = int(r.integers(2, 10)) if r.integers(2) else int(r.integers(10, 95))
        notes.append(f"- A {a} year old {'child' if a < 13 else 'resident'} spoke at the event")
        ans["age"] = {"a": a}
    if "money" in names:
        whole, tenth = int(r.integers(2, 96)), int(r.integers(0, 10)) if r.integers(2) else 0
        notes.append(f"- Cost: ${whole * 1_000_000 + tenth * 100_000:,}")
        ans["money"] = {"whole": whole, "tenth": tenth}
    if "over" in names:
        q, noun = int(r.integers(30, 900)), OVER_NOUNS[int(r.integers(len(OVER_NOUNS)))]
        notes.append(f"- Will add over {q} {noun}")
        ans["over"] = {"q": q, "noun": noun}
    if "attrib" in names:
        verb, quote = VERBS[int(r.integers(len(VERBS)))], QUOTES[int(r.integers(len(QUOTES)))]
        notes.append(f'- {person[0]}, {person[1]}, {verb}: "{quote}"')
        ans["attrib"] = {"verb": verb}
    order = r.permutation(len(notes) - 2) + 2
    notes = notes[:2] + [notes[i] for i in order]
    return {"log": "\n".join(notes), "answer": ans}


def _has(p, text, flags=0):
    return re.search(p, text, flags) is not None


_ABBR = {"jan.", "feb.", "aug.", "sept.", "oct.", "nov.", "dec.", "a.m.", "p.m.", "st.", "no.", "mr.", "ms.",
         "dr.", "ave.", "jr.", "sr.", "inc.", "u.s."}


def _starts_with_numeral(t: str) -> bool:
    """True if some sentence (or line) begins with a digit; periods ending a known abbreviation do not end a
    sentence ("Oct. 23", "9 a.m. 15" are not sentence starts)."""
    for m in re.finditer(r"(?:^|\n)\s*[\"'*>-]*\s*\d|(?<=[.!?])[\"')]?\s+[\"']?\d", t):
        if m.start() == 0 or t[m.start() - 1] == "\n" or t[m.start()] == "\n":
            return True
        prev = t[:m.start() + 1].split()[-1].lower() if t[:m.start() + 1].split() else ""
        if prev.strip("\"')") not in _ABBR:
            return True
    return False


def _slot_ok(name: str, s: dict, t: str) -> bool:
    if name == "date":
        m, d = s["m"], s["d"]
        ap = re.escape(AP_MONTH[m])
        good = _has(rf"(?<![\w.]){ap} {d}(?![\d]|st|nd|rd|th)", t)
        full, abbr3 = calendar.month_name[m], calendar.month_abbr[m]
        bad = [rf"\b{d}(st|nd|rd|th)\b", rf"\b{m}/{d}\b"]
        if AP_MONTH[m].endswith("."):            # AP abbreviates: the full name with a date is wrong
            bad.append(rf"\b{full} {d}\b")
        if abbr3 != full and abbr3 + "." != AP_MONTH[m]:   # e.g. "Mar." / "Sep." are not AP forms
            bad.append(rf"\b{abbr3}\.? {d}\b")
        elif abbr3 != full:                       # AP form needs its period: "Oct 14" is wrong
            bad.append(rf"\b{abbr3} {d}\b")
        return good and not any(_has(b, t) for b in bad)
    if name == "time":
        bad = [r"\b\d{1,2}:00\b", r"\b\d{1,2}(:\d\d)?\s?(AM|PM|A\.M\.|P\.M\.)", r"\b12\s?(a\.m\.|p\.m\.)", r"\b12:00\b"]
        if any(_has(b, t) for b in bad):
            return False
        k = s["kind"]
        if k in ("noon", "midnight"):
            return _has(rf"\b{k}\b", t, re.I)
        ampm = "p\\.m" if s["pm"] else "a\\.m"
        hm = f"{s['h']}" if k == "hour" else f"{s['h']}:30"
        return _has(rf"(?<![\d:]){hm} {ampm}(\.|\b)", t)
    if name == "small":
        return _has(rf"\b{SMALL[s['n']]}\b", t, re.I) and not _has(rf"\b{s['n']}\s+{s['noun']}\b", t)
    if name == "large":
        return _has(rf"(?<![\d,.]){s['n']}(?![\d,])", t) and not _starts_with_numeral(t)
    if name == "age":
        a = s["a"]
        good = _has(rf"\b{a}-year-old\b|\b{a} years old\b|,\s*{a},", t)
        bad = _has(rf"\b{a} year old\b", t) or (a <= 9 and _has(rf"\b{SMALL[a]}[- ]years?[- ]old\b", t, re.I))
        return good and not bad
    if name == "money":
        amt = f"{s['whole']}" + (f".{s['tenth']}" if s["tenth"] else "")
        return _has(rf"\${re.escape(amt)} million\b", t) and not _has(r"\$\d{1,3}(,\d{3}){2,}", t)
    if name == "over":
        return _has(rf"\bmore than {s['q']}\b", t, re.I) and not _has(rf"\bover {s['q']}\b", t, re.I)
    if name == "attrib":
        return _has(r"\bsaid\b", t) and not _has(rf"\b{s['verb']}\b", t, re.I)
    raise ValueError(name)


def grade(text: str, answer: dict) -> tuple[bool, dict | None]:
    """Pass iff every slot of the variant passes. Returns (pass, per-slot results)."""
    if not text.strip():
        return False, None
    per = {k: _slot_ok(k, v, text) for k, v in answer.items()}
    return all(per.values()), per


PLACEBO = """---
name: newsroom-craft
description: General news-writing craft for briefs, stories and headlines. Use when writing or editing news copy.
---

# Newsroom craft guide

Write for the reader in a hurry. Lead with the news, keep it clear, check everything.

## When to use

- Writing news briefs and short stories
- Editing a draft before it goes to the desk
- Turning interview notes or a press release into copy
- Writing headlines and summaries
- Preparing copy for the web and social feeds

## Structure

### The inverted pyramid

| Part | What goes there |
|------|-----------------|
| Lede | The single most important development |
| Second paragraph | The most useful context or a strong quote |
| Body | Supporting facts in order of importance |
| End | Background a reader can skip |

**Why it works:** editors cut from the bottom, and readers stop early. The essential facts survive both.

### Ledes

- Put the news in the first sentence
- Name the actor and the action plainly
- Prefer active verbs: "The board approved" not "Approval was given by the board"
- Keep background out of the lede
- Avoid opening with a quote unless it carries the news

**Weak:** In a long meeting that covered many topics, members discussed several items.

**Better:** The board approved the plan, ending a long dispute.

## Clarity

| Do | Don't |
|----|-------|
| Short sentences | Chains of clauses |
| Concrete nouns | Abstractions |
| Plain words | Jargon and acronyms without explanation |
| One idea per paragraph | Several ideas packed together |
| Specific detail | Vague generalities |

### Cut these

- Throat-clearing openers: "It is important to note"
- Redundant pairs: "each and every," "first and foremost"
- Empty intensifiers: "really," "truly," "very"
- Hedging stacks: "may possibly," "could potentially"
- Clichés: "at the end of the day," "in the wake of"

## Quotes

### Choosing quotes

- Use a quote when the speaker says something better than you could paraphrase it
- Paraphrase routine statements and facts
- Prefer quotes that show emotion, judgment or a distinctive voice
- Never alter a quote's meaning; trim only with care

### Placing quotes

- Put the strongest quote high in the story
- Introduce the speaker clearly so the reader knows who is talking
- Give one speaker per paragraph
- Follow a quote with context when it needs explaining

## Accuracy

### Verification

- Confirm the spelling of every name with the person or an official source
- Check titles and affiliations against an official source
- Confirm places against a map or official listing
- Read every figure back against the source document
- Click every link before publishing

### Fairness and balance

- Seek comment from anyone the story criticizes
- Give people enough time to respond
- Say when someone declined to comment or could not be reached
- Separate fact from opinion
- Avoid loaded words that take sides

## Headlines

| Do | Don't |
|----|-------|
| Lead with the news | Tease without informing |
| Use strong, specific verbs | Use vague verbs like "addresses" |
| Keep it short | Pack in every detail |
| Make it accurate | Overpromise what the story delivers |

**Examples:**

**Weak:** Officials address issue at meeting

**Better:** Board backs longer school day after debate

## Writing for the web

- Front-load keywords readers would search for
- Keep paragraphs short for small screens
- Use subheads to break up longer stories
- Link to source documents and earlier coverage
- Write a summary line that stands on its own

## Tone

- Neutral and direct in news copy
- No first person in straight news
- No exclamation points in news copy
- Let facts carry the weight; drop adjectives that judge
- Match tone to the story: a light feature can be warmer than a crime brief

## Editing your own copy

### Read it twice

1. First pass: does the lede say what happened?
2. Second pass: does every paragraph earn its place?

### Read it aloud

Reading aloud catches clumsy rhythm, missing words and sentences that run too long.

### Cut ten percent

Most drafts improve when trimmed. Remove repetition, background the reader already knows, and any sentence that does not move the story forward.

## Quick reference card

### Before you file

- [ ] Lede states the news
- [ ] Names and titles verified
- [ ] Every fact traced to a source
- [ ] Quotes accurate and fairly placed
- [ ] People criticized were asked to respond
- [ ] No jargon left unexplained
- [ ] Paragraphs short
- [ ] Headline accurate

### Red flags

- A lede that starts with background
- Unnamed sources without a reason
- Opinion words in a news story
- A quote that repeats the paragraph before it
- Passive voice that hides who acted

## Example transformations

**Before (draft):**
> There was a vote by the board on the plan, which many people have discussed for a long time.

**After:**
> The board approved the plan, ending months of debate.

---

**Before:**
> It is important to note that the library will be reopening its doors to the public.

**After:**
> The library is reopening.

---

**Before:**
> Officials say the project could potentially possibly help the neighborhood.

**After:**
> Officials expect the project to help the neighborhood.

## Interviewing

### Before the interview

- Read earlier coverage so you do not ask what is already public
- Write your three most important questions first
- Know what you need from this source that no one else can give you
- Confirm how the person wants to be identified

### During the interview

- Ask open questions: "What happened next?" works better than "Was it bad?"
- Let the source do most of the talking; silence often brings the best answer
- Ask the person to explain any term you do not understand
- Ask the hard question near the end, once trust is built
- Ask who else you should talk to

### After the interview

- Type up notes the same day while memory is fresh
- Mark which notes are direct quotes and which are paraphrase
- Call back to check anything you are unsure of
- Keep your notes; they may be needed if the story is challenged

## Ethics

| Principle | In practice |
|-----------|-------------|
| Seek truth | Use original sources; avoid relying on summaries of summaries |
| Minimize harm | Weigh the public value of a detail against the harm it can cause |
| Act independently | Decline gifts and favors from people you cover |
| Be accountable | Correct errors promptly and say what was wrong |

### Corrections

- Fix the error in the story itself
- Add a note saying what changed and when
- Do not quietly edit a published story to hide a mistake
- Tell the desk so other platforms can be updated

## Working with editors

- Flag anything you could not verify
- Note sources and contact details for the desk
- Say if a story may need legal review
- Ask about house style before writing for a new outlet

---

*General newsroom practice. Check your outlet's own guide for its conventions.*
"""


def all_skills() -> dict:
    real = SKILL_PATH.read_text()
    assert hashlib.sha256(real.encode()).hexdigest() == SKILL_SHA256, "published skill changed"
    return {"real": real, "placebo": PLACEBO}
