"""Study step 2b (proctor): rerun batches whose bench job was refused as a whole because ONE task's compose file
mounts host paths (batch 4: drone-planning-control; batch 6: fix-visual-stability, which mounts ~/.config/gcloud).
The refused task is not run (no host access is granted; it is reported as excluded: harness refusal). The other
tasks of the batch run exactly as committed (same model, arms, trials, -j, caps), under the same job names.
Logged in the ledger as study_rerun before any run. Usage: python pilot/study_rerun_refused.py
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from commit import LEDGER, append  # noqa: E402

JOBS = ROOT / "jobs_study"
REFUSED = {"04": "drone-planning-control", "06": "fix-visual-stability"}
es = [json.loads(l) for l in LEDGER.read_text().splitlines()]
th = [e for e in es if e["kind"] == "study_commit"][-1]["payload"]["theta"]
plan = {b: [t for t in json.loads((JOBS / f".batch{b}_done").read_text())["tasks"] if t != bad]
        for b, bad in REFUSED.items()}
if not any(e["kind"] == "study_rerun" for e in es):
    append("study_rerun", {"reason": "bench refused the whole job of batches 4 and 6 at preflight because one task per "
                                     "batch mounts host paths; no run of those batches started and nothing was spent",
                           "refused_tasks_excluded": REFUSED, "rerun_tasks": plan,
                           "unchanged": "model, arms, trials, concurrency, caps, order, D_i, exclusion rule, primary test",
                           "host_access_granted": False})
env = dict(os.environ)
for line in (Path.home() / ".config/p9_pilot/.env").read_text().splitlines():
    if "=" in line and not line.lstrip().startswith("#"):
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"')
env.setdefault("LLM_API_KEY", env.get("ANTHROPIC_API_KEY", ""))


def arm(b, names, skills):
    name = f"b{b}_" + ("with_skill" if skills else "no_skill")
    cmd = ["bench", "run", str(ROOT / "vendor/skillsbench/tasks"), "--agent", "openhands", "--model", th["model"],
           "--sandbox", "docker", "--include", ",".join(names), "--trials", "3", "-j", "4", "--max-cost-usd", "8",
           "--max-tokens", "15000000", "--jobs-dir", str(JOBS), "--job-name", name, "--quiet"] + (["--task-skills"] if skills else [])
    with open(ROOT / f"logs_study_b{b}_rerun_{'with_skill' if skills else 'no_skill'}.txt", "a") as log:
        return subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT).returncode


for b, names in plan.items():
    for c in (["docker", "container", "prune", "-f"], ["docker", "builder", "prune", "-af"], ["docker", "image", "prune", "-f"]):
        subprocess.run(c, capture_output=True)
    print(f"batch {b} rerun: {len(names)} tasks", flush=True)
    with ThreadPoolExecutor(2) as ex:
        rc = list(ex.map(lambda s: arm(b, names, s), (True, False)))
    (JOBS / f".batch{b}_rerun_done").write_text(json.dumps({"tasks": names, "rc": rc}))
    print(f"batch {b} rerun done rc={rc}", flush=True)
print("RERUN DONE", flush=True)
