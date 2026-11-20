"""Study step 1: commit to the 87-task study plan before its beacon round exists.

Appends to the same hash-chained ledger as the pilot (pilot/ledger.jsonl). The study secret goes to
pilot/secret/study_* and is revealed only after all runs. Approved by Chari on 2026-11-15
("1. Yes 2. Yes", "run the 87-task study"), total spend cap $80 including $4.54 already spent.
Usage: python pilot/study_commit.py
"""
import json
import os
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from commit import H, LEDGER, SECRET, TASKS, append, jcs, latest_round, skill_digest  # noqa: E402

entries = [json.loads(l) for l in LEDGER.read_text().splitlines()]
if any(e["kind"] == "study_commit" for e in entries):
    sys.exit("A study commitment is already logged; never commit twice.")

tasks = sorted(p.name for p in TASKS.iterdir() if (p / "task.md").exists())
digests = {t: skill_digest(TASKS / t / "environment" / "skills") for t in tasks
           if (TASKS / t / "environment" / "skills").is_dir()}
reg = append("study_register", {"skillsbench_commit": "f860ecf02ef53412a030f8661c83d644b739ff01",
                                "n_tasks": len(tasks), "tasks": tasks, "skill_digests": digests})
theta = {
    "study": "P9 live study 1",
    "design": "paired with-skill vs no-skill, concurrent arms, same tasks, every task in the pool",
    "agent": "openhands", "model": "anthropic/claude-haiku-5-5", "bench_version": "0.9.2", "sandbox": "docker",
    "task_pool": "all tasks in tasks/ at the registered commit", "n_tasks": len(tasks), "trials_per_arm": 3,
    "concurrency_per_arm": 4,
    "order": "tasks sorted by SHA-256(sigma || task name), sigma = SHA-256(sigma_p || beacon randomness); "
             "this order is both the run order (batches of 12) and the order of the sequential test",
    "D_i": "(passes with skill - passes without) / 3 per task; a pass is a run with reward 1",
    "exclusion": "a task with any errored run in either arm is reported and excluded, not replaced or retried",
    "primary_test": "betting test on D_i in the committed order, PrPl bets, cap 0.5, alpha 0.05, delta 0; "
                    "report admit or not, the first task index where wealth reaches 1/alpha, final wealth, LCB, mean D",
    "secondary": "mean reward difference per task; per-run cost and tokens; error categories per arm",
    "budget": {"cap_total_usd_including_prior": 80.0, "prior_spend_usd": 4.54,
               "stop_rule": "no new batch starts once reported study cost reaches 65 USD; "
                            "each batch arm runs with --max-cost-usd 8"},
    "disk_rule": "between batches, dangling images and build cache older than 30 minutes are pruned; "
                 "the study pauses if free disk is under 15 GB",
    "beacon": "drand quicknet", "beacon_round": None,
}
theta["beacon_round"] = latest_round() + 200  # about 10 minutes after the commitment
sigma_p, rho = secrets.token_bytes(32), secrets.token_bytes(16)
for name, val in (("study_sigma_p.hex", sigma_p), ("study_rho.hex", rho)):
    (SECRET / name).write_text(val.hex())
    os.chmod(SECRET / name, 0o600)
c = H(sigma_p + rho + jcs(theta))
e = append("study_commit", {"commitment": c, "theta": theta, "theta_hash": H(jcs(theta)), "after": reg["entry_hash"]})
print(json.dumps({"tasks": len(tasks), "skills_hashed": len(digests), "commitment": c,
                  "beacon_round": theta["beacon_round"], "entry": e["entry_hash"]}, indent=1))
