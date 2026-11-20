"""Generate run4.py (Haiku) and run4n.py (Nemotron) for live study 4 from run2.py by counted substitutions.

Diff against run2.py: task family (family_s4: the published newsroom-style skill and a placebo), one lineage
(placebo, then the real skill), grid bet, N_max and exam order from the pre-registered power plan
(power_plan_study4.json, written before commit), 20 calibration seeds per arm, ledger kinds study4_ / study4n_,
transcripts4 / transcripts_n4. run4n.py then gets exactly the Nemotron changes used for run2n.py.
  python pilot2/make_study4_runners.py
"""
from pathlib import Path

from make_nemotron_runners import CLIENT

ROOT = Path(__file__).resolve().parent

DOC4 = '''"""Live study 4: the full gate on a real, published third-party skill, with generated hidden variants.
The skill, its pinned source and the task family are in the family module imported below (see its docstring).

  python pilot2/run4.py calibrate   # 20 public-seed variants per skill (real, placebo), both arms; NOT scored; logged
  python pilot2/run4.py commit      # register skills/generator/prompt/plan, commit before a drand round
  python pilot2/run4.py run         # exams in the committed order, e-LOND + lineage levels, stop rules
  python pilot2/run4.py build       # reveal, compute decisions from transcripts, sign the credential
  python pilot2/run4.py verify      # anyone: recompute everything from ledger + transcripts + beacon

Design (committed in theta): one lineage S in a new registry history; exam order and N_max come from the
power plan power_plan_study4.json, fixed after calibration and before commit (default: t=1 placebo, t=2 real skill,
N_max 600). Level min(alpha * gamma_t * (R_{t-1} + 1), alpha * gamma^lin_k), gamma_j = j^-1.6 / zeta(1.6).
Per variant both arms answer the same notes; a pass is every graded AP slot correct; D = pass_with - pass_without.
Betting gate: level-tuned grid mixture bet (Theorem thm:adaptive), delta 0, futility at delta_f 0.05 and alpha_f
0.10 (PrPl futility bet, as in studies 2 and 3). Pairs run in chunks of 6; the decision is the first crossing.
A pair whose API call fails after retries in either arm is reported and excluded.
Budget: hard stop when the summed API cost of study 4 reaches 15 USD.
Model: claude-haiku-5-5, default settings (adaptive thinking, default effort), max_tokens 8000.
"""
'''

R4 = [  # (old, new, expected count)
    ("import family  # noqa: E402", "import {MOD} as family  # noqa: E402", 1),
    ("ALPHA, ALPHA_F, DELTA_F, CAP, NMAX, S = 0.05, 0.10, 0.05, 0.5, 600, 1.6",
     "ALPHA, ALPHA_F, DELTA_F, CAP, S = 0.05, 0.10, 0.05, 0.5, 1.6\n"
     'POWER_FILE = ROOT / "power_plan_study4.json"\n'
     "POWER = json.loads(POWER_FILE.read_text()) if POWER_FILE.exists() else {}\n"
     'NMAX = int(POWER.get("n_max", 600))\nRULE = "grid"\nN_CAL = 20', 1),
    ('EXAMS = [("A", 1, "placebo"), ("B", 1, "leak"), ("C", 1, "harmful"), ("A", 2, "helpful")]',
     'EXAMS = [tuple(e) for e in POWER.get("exams", [["S", 1, "placebo"], ["S", 2, "real"]])]', 1),
    ('TR = ROOT / "transcripts"\n', 'TR = ROOT / "transcripts4"\n', 1),
    ('SYSTEM = "You are a careful operations engineer. Answer the user\'s question about the log they give you."',
     "SYSTEM = family.SYSTEM", 1),
    ('for name in ("helpful", "placebo", "leak", "harmful"):\n        for i in range(5):\n'
     '            seed = hashlib.sha256(f"calibration-{name}-{i}".encode()).digest()',
     'for name in ("real", "placebo"):\n        for i in range(N_CAL):\n'
     '            seed = hashlib.sha256(f"calibration-{family.FAMILY}-{i}".encode()).digest()', 1),
    ('"pairs": 5} for n in ("helpful", "placebo", "leak", "harmful")}',
     '"pairs": N_CAL} for n in ("real", "placebo")}', 1),
    ("gates.betting(np.array([D], float), 0.0, a, ALPHA_F, CAP, delta_f=DELTA_F)",
     "gates.betting(np.array([D], float), 0.0, a, ALPHA_F, CAP, rule=RULE, delta_f=DELTA_F)", 1),
    ('"bets": "PrPl"}', '"bets": RULE, "futility_bets": "PrPl"}', 1),
    ('"pass": "exact match of ip and count in the last JSON line",',
     '"pass": "family.grade (see the family module)",\n'
     '             "power_plan": POWER, "power_plan_sha256": H(POWER_FILE.read_bytes()) if POWER_FILE.exists() else None,', 1),
    ('ROOT / "family.py", ROOT / "run2.py")}',
     'ROOT / "{MOD}.py", ROOT / "run4.py", family.SKILL_PATH)}', 1),
    ('"P9 live study 2"', '"P9 live study 4 (real skill)"', 2),
    ('"study2_', '"study4_', 18),
    ('{"note": "public calibration seeds, not exam variants, not scored",',
     '{"note": "public calibration seeds, not exam variants, not scored", "family": family.FAMILY,', 1),
    ("study 2 already committed", "study 4 already committed", 1),
    ("STUDY2 STOP", "STUDY4 STOP", 1),
    ('"STUDY2 RUNS DONE"', '"STUDY4 RUNS DONE"', 1),
    # release fixes (after the final review of 18 Nov 2026; the committed run4*.py predate them and are unchanged)
    ("    if len(D) == 0:\n        return None, 0\n",
     "    D = list(D)[:NMAX]                                  # never decide on pairs beyond N_max\n"
     "    if len(D) == 0:\n        return None, 0\n", 1),
    ('print("STUDY4 RUNS DONE"', '(TR / "EXAMS_CLOSED").write_text("done"); print("STUDY4 RUNS DONE"', 1),
    ('print("STUDY4 STOP', '(TR / "EXAMS_CLOSED").write_text("budget"); print("STUDY4 STOP', 1),
    ('    rev = append("study4_reveal"',
     '    if not (TR / "EXAMS_CLOSED").exists():\n'
     '        sys.exit("exams not closed; refusing to reveal the committed secret")\n'
     '    rev = append("study4_reveal"', 1),
    ('"LCB_at_decision": lcb', '"LCB_at_decision_prpl_bound": lcb', 1),
]

R4N = [
    ('"""Live study 4: the full gate', '"""Live study 4N (Nemotron arm of live study 4): the full gate', 1),
    ("python pilot2/run4.py", "python pilot2/run4n.py", 5),
    ("Budget: hard stop when the summed API cost of study 4 reaches 15 USD.\n"
     "Model: claude-haiku-5-5, default settings (adaptive thinking, default effort), max_tokens 8000.",
     "Budget: hard stop when the summed API cost of study 4N reaches 25 USD.\n"
     "Model: nvidia/nemotron-3.5-super-vl via OpenRouter, provider pinned to crusoe/bf16 with no fallbacks,\n"
     "API defaults otherwise (default reasoning), max_tokens 16000. Everything else exactly as run4.py.", 1),
    ('MODEL = "claude-haiku-5-5"\nPRICE_IN, PRICE_OUT = 0.10e-6, 0.50e-6          # USD per token, prompts <= 100K tokens\nCAP_USD = 15.0',
     'MODEL = "nvidia/nemotron-3.5-super-vl"\nPROVIDER = {"order": ["crusoe/bf16"], "allow_fallbacks": False}\n'
     'PROVIDER_NAME = "Crusoe"\nMAX_TOKENS = 16000\n'
     'PRICE_IN, PRICE_OUT = 0.10e-6, 0.40e-6          # USD per token, Crusoe list price on OpenRouter 2026-11-17\nCAP_USD = 25.0', 1),
    ('TR = ROOT / "transcripts4"\n', 'TR = ROOT / "transcripts_n4"\n', 1),
    ('def client():\n    import anthropic\n    return anthropic.Anthropic(api_key=env_key(), max_retries=4, timeout=180)\n', CLIENT, 1),
    ("max_tokens=8000", "max_tokens=MAX_TOKENS", 1),
    ('"max_tokens": 8000, "settings": "API defaults"',
     '"max_tokens": MAX_TOKENS, "settings": "API defaults; OpenRouter provider " + json.dumps(PROVIDER)', 1),
    ('ROOT / "run4.py", family.SKILL_PATH)}', 'ROOT / "run4.py", ROOT / "run4n.py", family.SKILL_PATH)}', 1),
    ('"P9 live study 4 (real skill)"', '"P9 live study 4N (real skill, Nemotron)"', 2),
    ('"study4_', '"study4n_', 18),
    ("study 4 already committed", "study 4N already committed", 1),
    ("STUDY4 STOP", "STUDY4N STOP", 1),
    ('"STUDY4 RUNS DONE"', '"STUDY4N RUNS DONE"', 1),
]


def apply(s, subs, src):
    for old, new, n in subs:
        c = s.count(old)
        assert c == n, f"{src}: expected {n} of {old[:60]!r}, found {c}"
        s = s.replace(old, new)
    return s


def write(dst, s):
    out = ROOT / dst
    if out.exists():
        raise SystemExit(f"{dst} exists; not overwriting")
    out.write_text(s)
    print("wrote", dst)


if __name__ == "__main__":
    import sys
    MOD = sys.argv[1]                       # e.g. family_s4 (candidate 1) or family_s4_commit (candidate 2)
    R4 = [(o, n.replace("{MOD}", MOD), c) for o, n, c in R4]
    s = (ROOT / "run2.py").read_text()
    end = s.index('"""\n', 3) + 4
    s = DOC4 + s[end:]
    s4 = apply(s, R4, "run2.py")
    s4n = apply(s4, R4N, "run4.py")
    T = sys.argv[2] if len(sys.argv) > 2 else "4"        # study tag: 4 (primary), 4b, 4c (additional real skills)
    if T != "4":
        def retag(x):
            for a, b in (('"study4n_', f'"study{T}n_'), ('"study4_', f'"study{T}_'),
                         ('"transcripts_n4"', f'"transcripts_n{T}"'), ('"transcripts4"', f'"transcripts{T}"'),
                         ("power_plan_study4.json", f"power_plan_study{T}.json"),
                         ("run4n.py", f"run{T}n.py"), ("run4.py", f"run{T}.py"),
                         ("P9 live study 4N", f"P9 live study {T}N"), ("P9 live study 4", f"P9 live study {T}"),
                         ("study 4N already", f"study {T}N already"), ("study 4 already", f"study {T} already"),
                         ("STUDY4N", f"STUDY{T.upper()}N"), ("STUDY4", f"STUDY{T.upper()}"),
                         ("Live study 4N", f"Live study {T}N"), ("Live study 4", f"Live study {T}")):
                x = x.replace(a, b)
            return x
        s4, s4n = retag(s4), retag(s4n)
    write(f"run{T}.py", s4)
    write(f"run{T}n.py", s4n)
