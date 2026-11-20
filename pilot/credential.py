"""Pilot step 4: reveal, build the dual-arm credential, log it, and verify it from public material.

  python pilot/credential.py build    # proctor: reveal secrets, compute statistics, sign, log
  python pilot/credential.py verify   # anyone: recompute everything from ledger.jsonl + jobs/ + beacon

Pre-registered statistic (theta): D_i = (passes with skill - passes without) / 3 per task, where a
pass is a run with reward 1 (BenchFlow's PASS). Tasks whose runs errored in either arm are reported,
not replaced. Test: betting, PrPl bets, cap 0.5, alpha 0.05, delta 0.
"""
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))
sys.path.insert(0, str(ROOT))
from commit import DRAND, LEDGER, SECRET, H, append  # noqa: E402
from nixigate import gates  # noqa: E402
from nixigate.credential import jcs, merkle_root  # noqa: E402

JOBS = ROOT / "jobs" / "openhands"
ARMS = ("with_skill", "no_skill")


def entries():
    return [json.loads(l) for l in LEDGER.read_text().splitlines()]


def chain_ok(es):
    prev = "0" * 64
    for e in es:
        body = {k: e[k] for k in ("kind", "payload", "time_utc", "prev")}
        if e["prev"] != prev or H(jcs(body)) != e["entry_hash"]:
            return False
        prev = e["entry_hash"]
    return True


def beacon(rnd):
    with urllib.request.urlopen(f"{DRAND}/public/{rnd}", timeout=20) as r:
        return json.load(r)["randomness"]


def runs():
    """Per arm: sorted run records (task, trial, reward, error) and the leaf bytes of each results.jsonl line."""
    out = {}
    for arm in ARMS:
        recs, leaves = [], []
        for f in sorted(JOBS.glob(f"trial-0*/{arm}/results.jsonl")):
            for line in f.read_text().splitlines():
                d = json.loads(line)
                leaves.append(line.encode())
                recs.append({"task": d["info"]["task_id"], "trial": f.parts[-3], "reward": d.get("reward"),
                             "error": bool(d.get("error"))})
        out[arm] = (recs, leaves)
    return out


def statistic(R, cap=0.5, alpha=0.05):
    tasks = sorted({r["task"] for arm in ARMS for r in R[arm][0]})
    per, excluded = {}, []
    for t in tasks:
        arm_runs = {arm: [r for r in R[arm][0] if r["task"] == t] for arm in ARMS}
        if any(r["error"] for arm in ARMS for r in arm_runs[arm]):
            excluded.append(t)
            continue
        passes = {arm: sum(1 for r in arm_runs[arm] if r["reward"] is not None and r["reward"] >= 1.0) for arm in ARMS}
        reward = {arm: float(np.mean([r["reward"] for r in arm_runs[arm]])) for arm in ARMS}
        per[t] = {"passes_with": passes["with_skill"], "passes_without": passes["no_skill"],
                  "D": (passes["with_skill"] - passes["no_skill"]) / 3,
                  "mean_reward_with": round(reward["with_skill"], 6), "mean_reward_without": round(reward["no_skill"], 6)}
    D = np.array([[v["D"] for v in per.values()]])
    res = gates.betting(D, 0.0, alpha, None, cap, admit_stop=False)
    lcb = float(gates.lcb_path(D[0], alpha, cap)[-1]) if D.size else None
    return {"tasks_scored": list(per), "tasks_excluded_errors": excluded, "per_task": per,
            "N": int(D.size), "Dbar": float(D.mean()) if D.size else None,
            "final_wealth": float(np.exp(res.log_wealth[0])), "LCB": lcb,
            "decision": "admit" if np.exp(res.log_wealth[0]) >= 1 / alpha else "not admitted"}


def sample_check(S):
    """The committed rule: bench --limit 6 --sample-seed S must select exactly the tasks that were run."""
    out = subprocess.run(["bench", "run", str(ROOT / "vendor/skillsbench/tasks"), "--agent", "openhands",
                          "--model", "anthropic/claude-haiku-5-5", "--limit", "6", "--sample-seed", str(S), "--dry-run"],
                         capture_output=True, text=True).stdout
    return sorted(json.loads(out[out.index("{"):])["tasks"])


def body_from(es, R, S, sigma_hex):
    amend = [e for e in es if e["kind"] == "amend"][-1]
    return {"exam": "P9 pilot", "commitment_entry": amend["entry_hash"], "commitment": amend["payload"]["commitment"],
            "theta_hash": amend["payload"]["theta_hash"], "beacon_round": amend["payload"]["theta"]["beacon_round"],
            "sigma": sigma_hex, "sample_seed": S, "tasks_run": sorted({r["task"] for r in R["with_skill"][0]}),
            "roots": {arm: "sha256:" + merkle_root(R[arm][1]).hex() for arm in ARMS},
            "n_runs": {arm: len(R[arm][1]) for arm in ARMS}, "statistics": statistic(R)}


def build():
    es = entries()
    if any(e["kind"] == "credential" for e in es):
        sys.exit("credential already logged")
    amend = [e for e in es if e["kind"] == "amend"][-1]
    th = amend["payload"]["theta"]
    sp, rho = (bytes.fromhex((SECRET / f).read_text()) for f in ("sigma_p_a1.hex", "rho_a1.hex"))
    sp0, rho0 = (bytes.fromhex((SECRET / f).read_text()) for f in ("sigma_p.hex", "rho.hex"))
    assert H(sp + rho + jcs(th)) == amend["payload"]["commitment"]
    sigma = hashlib.sha256(sp + bytes.fromhex(beacon(th["beacon_round"]))).digest()
    S = int.from_bytes(sigma[:4], "big")
    R = runs()
    assert sample_check(S) == sorted({r["task"] for r in R["with_skill"][0]}), "sample does not match the committed rule"
    rev = append("reveal", {"amendment_1": {"sigma_p": sp.hex(), "rho": rho.hex()},
                            "original_commit": {"sigma_p": sp0.hex(), "rho": rho0.hex()}})
    key = Ed25519PrivateKey.generate()
    (SECRET / "proctor_ed25519.pem").write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                                  serialization.NoEncryption()))
    pk = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    body = body_from(entries(), R, S, sigma.hex())
    chi = H(jcs(body))
    sig = key.sign(bytes.fromhex(chi)).hex()
    append("credential", {"body": body, "chi": chi, "sig": sig, "proctor_pk": pk.hex(), "after_reveal": rev["entry_hash"]})
    print(json.dumps({"chi": chi, "statistics": body["statistics"]}, indent=1))


def verify():
    es = entries()
    checks = {"ledger hash chain intact": chain_ok(es)}
    amend = [e for e in es if e["kind"] == "amend"][-1]
    rev = [e for e in es if e["kind"] == "reveal"][-1]["payload"]
    cred = [e for e in es if e["kind"] == "credential"][-1]["payload"]
    th = amend["payload"]["theta"]
    sp, rho = bytes.fromhex(rev["amendment_1"]["sigma_p"]), bytes.fromhex(rev["amendment_1"]["rho"])
    orig = next(e for e in es if e["kind"] == "commit")["payload"]
    checks["reveal opens amendment commitment"] = H(sp + rho + jcs(th)) == amend["payload"]["commitment"]
    checks["reveal opens original commitment"] = H(bytes.fromhex(rev["original_commit"]["sigma_p"]) + bytes.fromhex(
        rev["original_commit"]["rho"]) + jcs(orig["theta"])) == orig["commitment"]
    sigma = hashlib.sha256(sp + bytes.fromhex(beacon(th["beacon_round"]))).digest()
    S = int.from_bytes(sigma[:4], "big")
    R = runs()
    checks["task sample follows committed rule"] = sample_check(S) == sorted({r["task"] for r in R["with_skill"][0]})
    body = body_from(es, R, S, sigma.hex())
    checks["credential body recomputes from transcripts"] = body == cred["body"]
    checks["digest matches"] = H(jcs(cred["body"])) == cred["chi"]
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(cred["proctor_pk"])).verify(bytes.fromhex(cred["sig"]), bytes.fromhex(cred["chi"]))
        checks["signature valid"] = True
    except Exception:
        checks["signature valid"] = False
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    print("decision:", cred["body"]["statistics"]["decision"])
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    {"build": build, "verify": verify}[sys.argv[1]]()
