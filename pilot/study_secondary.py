"""Secondary analysis (NOT the registered primary result): run once more the 22 tasks of batches 4 and 6 whose
rerun was blocked when the Anthropic account reached its monthly usage limit at 05:35 on 2026-11-16.
Approved by Chari on 2026-11-16 ("Raise cap, run secondary"). The primary result stays as registered
(errored tasks excluded, not retried). This run uses job names sec04_*/sec06_* so the primary scorer
(b??_*) never reads it; the secondary statistic is computed separately and labelled as such.

  python pilot/study_secondary.py check   # free API check (count_tokens) that the usage limit is lifted
  python pilot/study_secondary.py run     # log study_secondary in the ledger, then run both arms
"""
import json
import os
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from commit import LEDGER, append  # noqa: E402

JOBS = ROOT / "jobs_study"
env = dict(os.environ)
for line in (Path.home() / ".config/p9_pilot/.env").read_text().splitlines():
    if "=" in line and not line.lstrip().startswith("#"):
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"')
env.setdefault("LLM_API_KEY", env.get("ANTHROPIC_API_KEY", ""))
es = [json.loads(l) for l in LEDGER.read_text().splitlines()]
th = [e for e in es if e["kind"] == "study_commit"][-1]["payload"]["theta"]
plan = {b: json.loads((JOBS / f".batch{b}_rerun_done").read_text())["tasks"] for b in ("04", "06")}


def check():
    req = urllib.request.Request("https://api.anthropic.com/v1/messages/count_tokens", method="POST",
                                 data=json.dumps({"model": th["model"].split("/")[-1],
                                                  "messages": [{"role": "user", "content": "hi"}]}).encode(),
                                 headers={"x-api-key": env["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
                                          "content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            print("API OK", r.read().decode()[:80])
            return True
    except urllib.error.HTTPError as e:
        print("API BLOCKED", e.code, e.read().decode()[:200])
        return False


def arm(b, names, skills):
    name = f"sec{b}_" + ("with_skill" if skills else "no_skill")
    cmd = ["bench", "run", str(ROOT / "vendor/skillsbench/tasks"), "--agent", "openhands", "--model", th["model"],
           "--sandbox", "docker", "--include", ",".join(names), "--trials", "3", "-j", "4", "--max-cost-usd", "8",
           "--max-tokens", "15000000", "--jobs-dir", str(JOBS), "--job-name", name, "--quiet"] + (["--task-skills"] if skills else [])
    with open(ROOT / f"logs_study_{name}.txt", "a") as log:
        return subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT).returncode


def run():
    if not check():
        sys.exit("usage limit still in place; not starting")
    if not any(e["kind"] == "study_secondary" for e in es):
        append("study_secondary", {
            "label": "secondary analysis; the registered primary result is unchanged",
            "reason": "rerun of batches 4 and 6 blocked from 05:35 on 2026-11-16 by the account's monthly API usage limit",
            "approved_by": "Chari, 2026-11-16 ('Raise cap, run secondary')", "tasks": plan,
            "job_names": "sec04_*, sec06_*", "settings": "as committed (model, arms, trials, -j, caps)",
            "budget_note": "expected 15-20 USD; total cap 250 USD unchanged",
            "analysis": "D_i and exclusion rule as committed, applied to these runs; reported beside, not merged into, the primary statistic"})
    for b, names in plan.items():
        if (JOBS / f".sec{b}_done").exists():
            print(f"secondary {b}: already done, skipped", flush=True)
            continue
        print(f"secondary {b}: {len(names)} tasks", flush=True)
        with ThreadPoolExecutor(2) as ex:
            rc = list(ex.map(lambda s: arm(b, names, s), (True, False)))
        (JOBS / f".sec{b}_done").write_text(json.dumps({"tasks": names, "rc": rc}))
        print(f"secondary {b} done rc={rc}", flush=True)
        if not check():
            print("SECONDARY STOP: usage limit reached again", flush=True)
            return
    print("SECONDARY DONE", flush=True)


if __name__ == "__main__":
    {"check": check, "run": run}[sys.argv[1]]()
