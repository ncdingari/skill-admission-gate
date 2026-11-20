"""Generate run2n.py / run3n.py (Nemotron arm of live studies 2 and 3) from run2.py / run3.py.

Every change is an exact, counted string substitution, so the diff against the Haiku runners is only:
model + OpenRouter client (pinned to Crusoe bf16, no fallbacks), max_tokens 16000, prices, budget,
ledger kinds (study2n_ / study3n_), transcript folders, secret file names and study names.
Gate, levels, seeds, prompts, skills, graders, chunking and error rule are untouched.
  python pilot2/make_nemotron_runners.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

CLIENT = '''def client():
    """OpenRouter adapter exposing the subset of anthropic.Anthropic().messages.create used by call()."""
    import types
    import urllib.error
    import urllib.request
    k = None
    for line in (Path.home() / ".config/p9_pilot/.env").read_text().splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            k = line.split("=", 1)[1].strip().strip('"')
    if not k:
        raise SystemExit("no OPENROUTER_API_KEY")

    def create(model, max_tokens, system, messages):
        body = {"model": model, "max_tokens": max_tokens, "provider": PROVIDER, "usage": {"include": True},
                "messages": [{"role": "system", "content": system}] + messages}
        req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", method="POST",
                                     data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {k}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                d = json.loads(resp.read())
        except urllib.error.HTTPError as ex:
            raise RuntimeError(f"HTTP {ex.code}: {ex.read()[:200]!r}") from None
        if d.get("error") or not d.get("choices"):
            raise RuntimeError(f"API error: {str(d.get('error'))[:200]}")
        if d.get("provider") != PROVIDER_NAME:
            raise RuntimeError(f"served by {d.get('provider')!r}, not {PROVIDER_NAME}")
        ch, u = d["choices"][0], d.get("usage") or {}
        return types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text=ch["message"].get("content") or "")],
            stop_reason=ch.get("finish_reason"), id=f"{d.get('id')}|{d.get('provider')}|{u.get('cost')}",
            usage=types.SimpleNamespace(input_tokens=u.get("prompt_tokens", 0), output_tokens=u.get("completion_tokens", 0)))

    return types.SimpleNamespace(messages=types.SimpleNamespace(create=create))
'''

R2 = [  # (old, new, expected count)
    ('"""Live study 2: the full gate', '"""Live study 2N (Nemotron arm of live study 2): the full gate', 1),
    ("python pilot2/run2.py", "python pilot2/run2n.py", 5),
    ("Budget: hard stop when the summed API cost of study 2 reaches 15 USD.\n"
     "Model: claude-haiku-5-5, default settings (adaptive thinking, default effort), max_tokens 8000.",
     "Budget: hard stop when the summed API cost of study 2N reaches 40 USD.\n"
     "Model: nvidia/nemotron-3.5-super-vl via OpenRouter, provider pinned to crusoe/bf16 with no fallbacks,\n"
     "API defaults otherwise (default reasoning), max_tokens 16000. Everything else exactly as run2.py.", 1),
    ('MODEL = "claude-haiku-5-5"\nPRICE_IN, PRICE_OUT = 0.10e-6, 0.50e-6          # USD per token, prompts <= 100K tokens\nCAP_USD = 15.0',
     'MODEL = "nvidia/nemotron-3.5-super-vl"\nPROVIDER = {"order": ["crusoe/bf16"], "allow_fallbacks": False}\n'
     'PROVIDER_NAME = "Crusoe"\nMAX_TOKENS = 16000\n'
     'PRICE_IN, PRICE_OUT = 0.10e-6, 0.40e-6          # USD per token, Crusoe list price on OpenRouter 2026-11-17\nCAP_USD = 40.0', 1),
    ('TR = ROOT / "transcripts"\n', 'TR = ROOT / "transcripts_n2"\n', 1),
    ('def client():\n    import anthropic\n    return anthropic.Anthropic(api_key=env_key(), max_retries=4, timeout=180)\n', CLIENT, 1),
    ("max_tokens=8000", "max_tokens=MAX_TOKENS", 1),
    ('"max_tokens": 8000, "settings": "API defaults"',
     '"max_tokens": MAX_TOKENS, "settings": "API defaults; OpenRouter provider " + json.dumps(PROVIDER)', 1),
    ('ROOT / "run2.py")}', 'ROOT / "run2.py", ROOT / "run2n.py")}', 1),
    ('"P9 live study 2"', '"P9 live study 2N (Nemotron)"', 2),
    ('"study2_', '"study2n_', 18),
    ("study 2 already committed", "study 2N already committed", 1),
    ("STUDY2 STOP", "STUDY2N STOP", 1),
    ('"STUDY2 RUNS DONE"', '"STUDY2N RUNS DONE"', 1),
]

R3 = [
    ('"""Live study 3: a second task family', '"""Live study 3N (Nemotron arm of live study 3): a second task family', 1),
    ("python pilot2/run3.py", "python pilot2/run3n.py", 5),
    ("model exactly as in live study 2 (run2.py), budget 15 USD.",
     "model exactly as in live study 2N (run2n.py: Nemotron via OpenRouter, crusoe/bf16, max_tokens 16000), budget 40 USD.", 1),
    ("import run2  # noqa: E402", "import run2n as run2  # noqa: E402", 1),
    ("from run2 import (ALPHA, CAP, CHUNK, DRAND, LEDGER, MODEL, SECRET,",
     "from run2n import (MAX_TOKENS, PROVIDER, ALPHA, CAP, CHUNK, DRAND, LEDGER, MODEL, SECRET,", 1),
    ('TR = ROOT / "transcripts3"', 'TR = ROOT / "transcripts_n3"', 1),
    ("CAP_USD = 15.0", "CAP_USD = 40.0", 1),
    ('P = "study3_"', 'P = "study3n_"', 1),
    ("max_tokens=8000", "max_tokens=MAX_TOKENS", 1),
    ('"max_tokens": 8000, "settings": "API defaults"',
     '"max_tokens": MAX_TOKENS, "settings": "API defaults; OpenRouter provider " + json.dumps(PROVIDER)', 1),
    ('ROOT / "run2.py", ROOT / "run3.py")}', 'ROOT / "run2.py", ROOT / "run3.py", ROOT / "run2n.py", ROOT / "run3n.py")}', 1),
    ('"P9 live study 3"', '"P9 live study 3N (Nemotron)"', 2),
    ("study 3 already committed", "study 3N already committed", 1),
]


def make(src, dst, subs):
    s = (ROOT / src).read_text()
    for old, new, n in subs:
        c = s.count(old)
        assert c == n, f"{src}: expected {n} of {old[:60]!r}, found {c}"
        s = s.replace(old, new)
    out = ROOT / dst
    if out.exists():
        raise SystemExit(f"{dst} exists; not overwriting")
    out.write_text(s)
    print("wrote", dst)


if __name__ == "__main__":
    make("run2.py", "run2n.py", R2)
    make("run3.py", "run3n.py", R3)
