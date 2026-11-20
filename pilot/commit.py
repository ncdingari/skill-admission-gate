"""Pilot step 1: register skill digests, then commit to the exam parameters before the beacon exists.

Writes an append-only, hash-chained log (pilot/ledger.jsonl). The proctor secret goes to
pilot/secret/ and is revealed only after the runs. Usage: python pilot/commit.py
"""
import hashlib
import json
import os
import secrets
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))
from nixigate.credential import jcs  # noqa: E402

TASKS = ROOT / "vendor" / "skillsbench" / "tasks"
LEDGER = ROOT / "ledger.jsonl"
SECRET = ROOT / "secret"
DRAND = "https://api.drand.sh/52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971"


def H(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def append(kind: str, payload: dict) -> dict:
    prev = "0" * 64
    if LEDGER.exists():
        lines = LEDGER.read_text().splitlines()
        if lines:
            prev = json.loads(lines[-1])["entry_hash"]
    entry = {"kind": kind, "payload": payload, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "prev": prev}
    entry["entry_hash"] = H(jcs(entry))
    with LEDGER.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def skill_digest(d: Path) -> str:
    files = sorted(p for p in d.rglob("*") if p.is_file())
    leaves = [H(str(p.relative_to(d)).encode() + b"\0" + p.read_bytes()) for p in files]
    return H("".join(leaves).encode())


def latest_round() -> int:
    with urllib.request.urlopen(DRAND + "/public/latest", timeout=20) as r:
        return int(json.load(r)["round"])


if __name__ == "__main__":
    if LEDGER.exists() and any(json.loads(l)["kind"] == "commit" for l in LEDGER.read_text().splitlines()):
        sys.exit("A commitment is already logged; never commit twice.")
    tasks = sorted(p.name for p in TASKS.iterdir() if (p / "task.md").exists())
    digests = {t: skill_digest(TASKS / t / "environment" / "skills") for t in tasks
               if (TASKS / t / "environment" / "skills").is_dir()}
    reg = append("register", {"skillsbench_commit": "f860ecf02ef53412a030f8661c83d644b739ff01",
                              "n_tasks": len(tasks), "skill_digests": digests})
    theta = {
        "design": "paired with-skill vs no-skill, concurrent arms, same tasks",
        "agent": "openhands", "model": "openai/gpt-5.4-mini", "bench_version": "0.9.2", "sandbox": "docker",
        "n_tasks": 6, "trials_per_arm": 3, "task_pool": "tasks/ (default runnable set)",
        "selection": "bench --limit 6 --sample-seed S, S = first 4 bytes (big-endian) of SHA-256(sigma_p || beacon)",
        "D_i": "(passes with skill - passes without) / 3 per task",
        "test": "betting test, PrPl bets, cap 0.5, alpha 0.05, delta 0; report LCB and estimate",
        "budget_usd_per_arm": 10, "budget_usd_total": 20,
        "exclusion": "a task whose environment fails to build in either arm is reported, not replaced",
        "beacon": "drand quicknet", "beacon_round": None,
    }
    theta["beacon_round"] = latest_round() + 200  # about 10 minutes after the commitment
    SECRET.mkdir(exist_ok=True)
    sigma_p, rho = secrets.token_bytes(32), secrets.token_bytes(16)
    (SECRET / "sigma_p.hex").write_text(sigma_p.hex())
    (SECRET / "rho.hex").write_text(rho.hex())
    os.chmod(SECRET / "sigma_p.hex", 0o600)
    os.chmod(SECRET / "rho.hex", 0o600)
    c = H(sigma_p + rho + jcs(theta))
    com = append("commit", {"commitment": c, "theta": theta, "theta_hash": H(jcs(theta)), "after": reg["entry_hash"]})
    print(json.dumps({"registered_tasks": len(tasks), "skills_hashed": len(digests),
                      "commitment": c, "beacon_round": theta["beacon_round"]}, indent=1))
