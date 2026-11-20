"""Generate run5n.py (Nemotron arm of live study 5) from run5.py by counted substitutions, with the same Nemotron
changes used for run2n/run3n/run4n: OpenRouter client pinned to crusoe/bf16 (no fallbacks), max_tokens 16000,
Crusoe prices, budget 40 USD, ledger kinds study5n_*, transcripts_n5, secret names study5n_*. Gate, levels, power
plan (the Haiku study's power_plan_study5.json, unchanged), seeds rule, prompts, skills and graders are untouched.
  python pilot2/make_study5n_runner.py
"""
from pathlib import Path
from make_nemotron_runners import CLIENT
ROOT = Path(__file__).resolve().parent
OLD_CLIENT = '''def client():
    import anthropic
    return anthropic.Anthropic(api_key=env_key(), max_retries=4, timeout=180)
'''
SUBS = [
    ('"""Live study 5 (E25):', '"""Live study 5N (Nemotron arm of live study 5, E25):', 1),
    ('MODEL = "claude-haiku-5-5"\nPRICE_IN, PRICE_OUT = 0.10e-6, 0.50e-6          # USD per token, prompts <= 100K tokens\nCAP_USD = 60.0',
     'MODEL = "nvidia/nemotron-3.5-super-vl"\nPROVIDER = {"order": ["crusoe/bf16"], "allow_fallbacks": False}\n'
     'PROVIDER_NAME = "Crusoe"\nMAX_TOKENS = 16000\n'
     'PRICE_IN, PRICE_OUT = 0.10e-6, 0.40e-6          # USD per token, Crusoe list price on OpenRouter 2026-11-17\nCAP_USD = 40.0', 1),
    ('TR = ROOT / "transcripts5"', 'TR = ROOT / "transcripts_n5"', 1),
    (OLD_CLIENT, CLIENT, 1),
    ("max_tokens=8000, system=sysp", "max_tokens=MAX_TOKENS, system=sysp", 1),
    ('ROOT / "run5.py", ROOT / "requirements_dm_frozen.txt"', 'ROOT / "run5.py", ROOT / "run5n.py", ROOT / "requirements_dm_frozen.txt"', 1),
    ('"study": "P9 live study 5 (external family)", "model": MODEL, "max_tokens": 8000, "settings": "API defaults"',
     '"study": "P9 live study 5N (external family, Nemotron)", "model": MODEL, "max_tokens": MAX_TOKENS, '
     '"settings": "API defaults; OpenRouter provider " + json.dumps(PROVIDER)', 1),
    ('"study5_', '"study5n_', None),
    ("study 4 already committed", "study 5N already committed", 1),
]


def make():
    s = (ROOT / "run5.py").read_text()
    for old, new, n in SUBS:
        c = s.count(old)
        assert c > 0 and (n is None or c == n), (old[:50], c, n)
        s = s.replace(old, new)
    assert "import anthropic" not in s
    (ROOT / "run5n.py").write_text(s)
    print("wrote run5n.py")


if __name__ == "__main__":
    make()
