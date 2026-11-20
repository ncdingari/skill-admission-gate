"""Study step 3: reveal, compute the pre-registered statistics, sign, log; and verify from public material.

  python pilot/study_credential.py build    # proctor: reveal study secrets, compute statistics, sign, log
  python pilot/study_credential.py verify   # anyone: recompute everything from ledger.jsonl + jobs_study/ + beacon

Pre-registered (ledger entry study_commit): D_i = (passes with skill - passes without)/3 per task; a task with
any errored run in either arm is reported and excluded; betting test on D_i in the committed order
(tasks sorted by SHA-256(sigma || name)), PrPl bets, cap 0.5, alpha 0.05, delta 0.
"""
import calendar
import hashlib
import time
import json
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

JOBS = ROOT / "jobs_study" / "openhands"
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
    """Per arm: run records and the leaf bytes of every results.jsonl line, in a fixed file order."""
    out = {}
    for arm in ARMS:
        recs, leaves, cost, tokens = [], [], 0.0, 0
        for f in sorted(JOBS.glob(f"trial-*/b??_{arm}/results.jsonl")):
            for line in f.read_text().splitlines():
                d = json.loads(line)
                leaves.append(line.encode())
                recs.append({"task": d["info"]["task_id"], "reward": d.get("reward"), "error": bool(d.get("error"))})
        for f in sorted(JOBS.glob(f"trial-*/b??_{arm}/summary.json")):
            s = json.loads(f.read_text())
            cost += s.get("total_cost_usd") or 0.0
            tokens += s.get("total_tokens") or 0
        out[arm] = (recs, leaves, round(cost, 4), tokens)
    return out


def statistic(R, order, cap=0.5, alpha=0.05):
    per, excluded, missing = {}, [], []
    for t in order:
        arm_runs = {arm: [r for r in R[arm][0] if r["task"] == t] for arm in ARMS}
        if any(len(arm_runs[arm]) == 0 for arm in ARMS):
            missing.append(t)
            continue
        if any(r["error"] for arm in ARMS for r in arm_runs[arm]):
            excluded.append(t)
            continue
        passes = {arm: sum(1 for r in arm_runs[arm] if r["reward"] is not None and r["reward"] >= 1.0) for arm in ARMS}
        reward = {arm: float(np.mean([r["reward"] or 0.0 for r in arm_runs[arm]])) for arm in ARMS}
        per[t] = {"n_runs": {arm: len(arm_runs[arm]) for arm in ARMS},
                  "passes_with": passes["with_skill"], "passes_without": passes["no_skill"],
                  "D": (passes["with_skill"] - passes["no_skill"]) / 3,
                  "mean_reward_with": round(reward["with_skill"], 6), "mean_reward_without": round(reward["no_skill"], 6)}
    D = np.array([[v["D"] for v in per.values()]])
    out = {"order_scored": list(per), "tasks_excluded_errors": excluded, "tasks_missing": missing, "per_task": per,
           "N": int(D.size)}
    if D.size:
        stop = gates.betting(D, 0.0, alpha, None, cap, admit_stop=True)
        full = gates.betting(D, 0.0, alpha, None, cap, admit_stop=False)
        out.update({"Dbar": float(D.mean()), "n_positive": int((D > 0).sum()), "n_negative": int((D < 0).sum()),
                    "mean_reward_diff": float(np.mean([v["mean_reward_with"] - v["mean_reward_without"] for v in per.values()])),
                    "admitted": bool(stop.admit[0]), "crossing_index": int(stop.n[0]) if stop.admit[0] else None,
                    "final_wealth": float(np.exp(full.log_wealth[0])),
                    "LCB_final": float(gates.lcb_path(D[0], alpha, cap)[-1]),
                    "decision": "admit" if stop.admit[0] else "not admitted"})
    out["cost_usd"] = {arm: R[arm][2] for arm in ARMS}
    out["tokens"] = {arm: R[arm][3] for arm in ARMS}
    return out


def order_from(es, sp):
    com = [e for e in es if e["kind"] == "study_commit"][-1]["payload"]
    reg = [e for e in es if e["kind"] == "study_register"][-1]["payload"]
    sigma = hashlib.sha256(sp + bytes.fromhex(beacon(com["theta"]["beacon_round"]))).digest()
    return sigma, sorted(reg["tasks"], key=lambda t: H(sigma + t.encode())), com


def body_from(es, R, sigma, order, com_entry):
    return {"exam": "P9 live study 1", "commitment_entry": com_entry["entry_hash"],
            "commitment": com_entry["payload"]["commitment"], "theta_hash": com_entry["payload"]["theta_hash"],
            "beacon_round": com_entry["payload"]["theta"]["beacon_round"], "sigma": sigma.hex(), "order": order,
            "roots": {arm: "sha256:" + merkle_root(R[arm][1]).hex() for arm in ARMS},
            "n_runs": {arm: len(R[arm][1]) for arm in ARMS}, "statistics": statistic(R, order)}


def build():
    es = entries()
    if any(e["kind"] == "study_credential" for e in es):
        sys.exit("study credential already logged")
    com_entry = [e for e in es if e["kind"] == "study_commit"][-1]
    th = com_entry["payload"]["theta"]
    sp, rho = (bytes.fromhex((SECRET / f).read_text()) for f in ("study_sigma_p.hex", "study_rho.hex"))
    assert H(sp + rho + jcs(th)) == com_entry["payload"]["commitment"]
    sigma, order, _ = order_from(es, sp)
    R = runs()
    rev = append("study_reveal", {"sigma_p": sp.hex(), "rho": rho.hex()})
    key = Ed25519PrivateKey.generate()
    (SECRET / "study_proctor_ed25519.pem").write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    pk = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    body = body_from(entries(), R, sigma, order, com_entry)
    chi = H(jcs(body))
    append("study_credential", {"body": body, "chi": chi, "sig": key.sign(bytes.fromhex(chi)).hex(),
                                "proctor_pk": pk.hex(), "after_reveal": rev["entry_hash"]})
    s = body["statistics"]
    print(json.dumps({"chi": chi, **{k: s.get(k) for k in ("N", "Dbar", "n_positive", "n_negative", "admitted",
                      "crossing_index", "final_wealth", "LCB_final", "mean_reward_diff", "cost_usd")},
                      "excluded": len(s["tasks_excluded_errors"]), "missing": len(s["tasks_missing"])}, indent=1))


def verify():
    es = entries()
    checks = {"ledger hash chain intact": chain_ok(es)}
    com_entry = [e for e in es if e["kind"] == "study_commit"][-1]
    rev = [e for e in es if e["kind"] == "study_reveal"][-1]["payload"]
    cred = [e for e in es if e["kind"] == "study_credential"][-1]["payload"]
    sp, rho = bytes.fromhex(rev["sigma_p"]), bytes.fromhex(rev["rho"])
    checks["reveal opens study commitment"] = H(sp + rho + jcs(com_entry["payload"]["theta"])) == com_entry["payload"]["commitment"]
    with urllib.request.urlopen(f"{DRAND}/info", timeout=20) as r:
        info = json.load(r)
    rnd = com_entry["payload"]["theta"]["beacon_round"]
    t_round = info["genesis_time"] + (rnd - 1) * info["period"]
    t_commit = calendar.timegm(time.strptime(com_entry["time_utc"], "%Y-%m-%dT%H:%M:%SZ"))
    checks["commitment logged before beacon round existed"] = t_commit < t_round
    sigma, order, _ = order_from(es, sp)
    R = runs()
    body = body_from(es, R, sigma, order, com_entry)
    checks["task order follows committed rule"] = body["order"] == cred["body"]["order"]
    checks["credential body recomputes from transcripts"] = body == cred["body"]
    checks["digest matches"] = H(jcs(cred["body"])) == cred["chi"]
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(cred["proctor_pk"])).verify(bytes.fromhex(cred["sig"]), bytes.fromhex(cred["chi"]))
        checks["signature valid"] = True
    except Exception:
        checks["signature valid"] = False
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    print("decision:", cred["body"]["statistics"].get("decision"))
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    {"build": build, "verify": verify}[sys.argv[1]]()
