"""Study step 2 (proctor): derive the committed task order from the beacon, then run both arms batch by
batch, enforcing the budget and disk rules in the study commitment. Rerunning resumes: finished batches
are skipped (bench resumes a job with the same name). Usage: python pilot/study_run.py
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from commit import DRAND, LEDGER, SECRET, H, append  # noqa: E402

JOBS = ROOT / "jobs_study"
BATCH = 12
STOP_USD = None  # replaced 2026-11-16 by the 300 USD ask rule in monitor.needs_approval (was 65, then 210)
ARM_CAP = 8

es = [json.loads(l) for l in LEDGER.read_text().splitlines()]
com = [e for e in es if e["kind"] == "study_commit"][-1]["payload"]
reg = [e for e in es if e["kind"] == "study_register"][-1]["payload"]
th = com["theta"]
try:
    with urllib.request.urlopen(f"{DRAND}/public/{th['beacon_round']}", timeout=20) as r:
        beacon = json.load(r)["randomness"]
except Exception as ex:
    sys.exit(f"beacon round {th['beacon_round']} not available yet: {ex}")
sp = bytes.fromhex((SECRET / "study_sigma_p.hex").read_text())
sigma = hashlib.sha256(sp + bytes.fromhex(beacon)).digest()
order = sorted(reg["tasks"], key=lambda t: H(sigma + t.encode()))
(SECRET / "study_order.json").write_text(json.dumps({"sigma": sigma.hex(), "order": order}))
batches = [order[i:i + BATCH] for i in range(0, len(order), BATCH)]

env = dict(os.environ)
for line in (Path.home() / ".config/p9_pilot/.env").read_text().splitlines():
    if "=" in line and not line.lstrip().startswith("#"):
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"')
env.setdefault("LLM_API_KEY", env.get("ANTHROPIC_API_KEY", ""))


def spent():
    tot = 0.0
    for f in JOBS.glob("openhands/trial-*/b??_*/summary.json"):  # arm summaries only; trial-level copies double-count
        tot += json.loads(f.read_text()).get("total_cost_usd") or 0.0
    return tot


def free_gb():
    return shutil.disk_usage("/").free / 1e9


def arm(b, names, skills):
    name = f"b{b:02d}_" + ("with_skill" if skills else "no_skill")
    cmd = ["bench", "run", str(ROOT / "vendor/skillsbench/tasks"), "--agent", "openhands",
           "--model", th["model"], "--sandbox", "docker", "--include", ",".join(names), "--trials", "3",
           "-j", "4", "--max-cost-usd", str(ARM_CAP), "--max-tokens", "15000000",
           "--jobs-dir", str(JOBS), "--job-name", name, "--quiet"] + (["--task-skills"] if skills else [])
    with open(ROOT / f"logs_study_{name}.txt", "a") as log:
        return subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT).returncode


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


import monitor  # noqa: E402  (Chari's monitoring brief, 2026-11-16)

for b, names in enumerate(batches, 1):
    marker = JOBS / f".batch{b:02d}_done"
    if marker.exists():
        continue
    # Fail fast: tasks the sandbox lockdown has always refused are not launched; they are logged as infra.
    refused = sorted(set(names) & set(monitor.known_refusers()))
    if refused:
        append("study_infra_skip", {"batch": b, "tasks": refused,
                                    "reason": "CloudMetadataPowerError in every earlier attempt; depends on the task image, so not retried"})
        names = [t for t in names if t not in refused]
    # Cost rule (Chari, 2026-11-16): run without asking unless total spend plus this batch's estimate exceeds
    # 300 USD; then hold and ask Chari (approve with P9_APPROVED_OVER_300=<batch>). Replaces the 210 USD stop
    # and the ask-before-every-batch hold.
    over, msg = monitor.needs_approval(len(names))
    if over and os.environ.get("P9_APPROVED_OVER_300") != f"{b:02d}":
        log(f"HOLD: batch {b}: {msg}; ask Chari (set P9_APPROVED_OVER_300={b:02d} to proceed)")
        append("study_hold", {"batch": b, "reason": "estimate above 300 USD", "detail": msg})
        monitor.write_status(f"batch {b} held for Chari: {msg}")
        break
    log(f"batch {b}: {msg}")
    subprocess.run(["docker", "image", "prune", "-f"], capture_output=True)
    subprocess.run(["docker", "container", "prune", "-f"], capture_output=True)
    subprocess.run(["docker", "builder", "prune", "-af"], capture_output=True)
    subprocess.run(["docker", "volume", "prune", "-f"], capture_output=True)
    while free_gb() < 15:
        log(f"PAUSE: free disk {free_gb():.1f} GB < 15 GB")
        time.sleep(300)
    log(f"batch {b}/{len(batches)} start: {len(names)} tasks, spent so far ${spent():.2f}, free {free_gb():.0f} GB")
    with ThreadPoolExecutor(2) as ex:
        rc = list(ex.map(lambda s: arm(b, names, s), (True, False)))
    marker.write_text(json.dumps({"tasks": names, "rc": rc}))
    log(f"batch {b} done rc={rc}, spent ${spent():.2f}")
    monitor.write_status(f"batch {b} finished; {len(refused)} known-refusal tasks skipped")
log(f"STUDY RUNS DONE, reported cost ${spent():.2f}")
