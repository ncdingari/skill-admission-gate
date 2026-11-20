"""E15 third reading: an OpenAI model reads each DO_NOT_INSTALL skill blind (skill files + scanner findings only;
it never sees the first two readings). Writes results/E15/e15_flag_third_reading_openai.csv and fills the
third_reading_openai column of e15_flag_adjudication_v2.csv.

  /opt/anaconda3/bin/python experiments/e15_openai_reading.py MODEL_ID

Needs OPENAI_API_KEY in ~/.config/p9_pilot/.env. Skill files are read from a clean export of the registered
SkillsBench commit (same as e15_skill_security.py). Text files only, each capped at 40k characters."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
MODEL = sys.argv[1]
env = dict(l.split("=", 1) for l in (Path.home() / ".config/p9_pilot/.env").read_text().splitlines() if "=" in l)
client = OpenAI(api_key=env["OPENAI_API_KEY"].strip())

VENDOR = ROOT / "pilot" / "vendor" / "skillsbench"
clean = Path(tempfile.mkdtemp(prefix="e15_openai_"))
arch = subprocess.run(["git", "-C", str(VENDOR), "archive", "HEAD", "tasks"], capture_output=True, check=True).stdout
subprocess.run(["tar", "-x", "-C", str(clean)], input=arch, check=True)

adj = pd.read_csv(ROOT / "results/E15/e15_flag_adjudication_v2.csv")
PROMPT = """A static scanner rated the agent skill below DO_NOT_INSTALL. Read the skill's files and the scanner's
findings, and classify the skill as exactly one of:
- genuine: a real risk a registry should warn users about, even without malice (e.g. uninspectable executables,
  instructions to run offensive or destructive commands);
- false_positive: the flagged content is ordinary and documented;
- coverage: the rating comes mainly from files the scanner could not inspect, and nothing seen is risky;
- mixed: explain.
Answer with one line of JSON: {"verdict": "...", "reason": "<one sentence citing file paths>"}"""
rows = []
for _, r in adj.iterrows():
    sk = clean / "tasks" / r.task / "environment" / "skills" / r.skill
    files = []
    for f in sorted(sk.rglob("*")):
        if f.is_file():
            try:
                files.append(f"=== {f.relative_to(sk)} ===\n{f.read_text()[:40000]}")
            except UnicodeDecodeError:
                files.append(f"=== {f.relative_to(sk)} === (binary, {f.stat().st_size} bytes)")
    raw = json.loads((ROOT / "results/E15/raw" / f"{r.task}.json").read_text())
    rep = next((s for s in (raw.get("skills") or [raw]) if s.get("name") == r.skill), raw)
    findings = [{k: i.get(k) for k in ("id", "severity", "category", "message", "file", "line")} for i in rep.get("issues", [])]
    msg = f"{PROMPT}\n\nSKILL {r.skill} (task {r.task}), score {r.score}\n\nSCANNER FINDINGS:\n{json.dumps(findings)[:30000]}\n\nFILES:\n" + "\n\n".join(files)
    out = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": msg}])
    txt = out.choices[0].message.content or ""
    try:
        d = json.loads(txt[txt.rfind("{"):txt.rfind("}") + 1])
    except ValueError:
        d = {"verdict": "unparsed", "reason": txt[:300]}
    rows.append({"task": r.task, "skill": r.skill, "verdict": d.get("verdict"), "reason": d.get("reason"),
                 "model": out.model, "in_tokens": out.usage.prompt_tokens, "out_tokens": out.usage.completion_tokens})
    print(r.task, d.get("verdict"), flush=True)
t = pd.DataFrame(rows)
t.to_csv(ROOT / "results/E15/e15_flag_third_reading_openai.csv", index=False)
adj["third_reading_openai"] = adj.merge(t, on=["task", "skill"], how="left").verdict.values
adj.to_csv(ROOT / "results/E15/e15_flag_adjudication_v2.csv", index=False)
print(t[["task", "verdict"]].to_string(), "\nmodel:", t.model.iloc[0], "tokens in/out:", t.in_tokens.sum(), t.out_tokens.sum())
