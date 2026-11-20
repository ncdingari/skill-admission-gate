"""Live study 3N (Nemotron arm of live study 3): a second task family with a moderate baseline, and a live resubmission attack.

  python pilot2/run3n.py calibrate   # 5 public-seed variants per family-B skill, both arms; NOT scored; logged
  python pilot2/run3n.py commit      # register skills/generators/prompts/plan, commit before a drand round
  python pilot2/run3n.py run         # exams in the committed order
  python pilot2/run3n.py build       # reveal, recompute decisions from transcripts, sign
  python pilot2/run3n.py verify      # anyone: recompute everything from ledger + transcripts + beacon

A new registry history (t restarts at 1). Committed exam order:
  t=1 lineage D attempt 1: family B placebo
  t=2 lineage E attempt 1: family B harmful
  t=3 lineage D attempt 2: family B helpful
  t=4..8 lineage G attempts 1..5: family A (edge-logs@1) placebo, resubmitted unchanged five times
Levels, gate, chunking, pass rule, error rule and model exactly as in live study 2N (run2n.py: Nemotron via OpenRouter, crusoe/bf16, max_tokens 16000), budget 40 USD.
For the resubmission lineage we also report, as a counterfactual computed from the same transcripts, the
decision each attempt would get at a per-exam level alpha with no lineage spending.
"""
import calendar
import hashlib
import json
import os
import secrets
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import family as fam_a  # noqa: E402
import family_b as fam_b  # noqa: E402
import run2n as run2  # noqa: E402
from run2n import (MAX_TOKENS, PROVIDER, ALPHA, CAP, CHUNK, DRAND, LEDGER, MODEL, SECRET, SYSTEM, H, append, cost, client,  # noqa: E402
                  decide, gates, jcs, latest_round, level, merkle_root, seed, sigma_from)

FAM = {"A": fam_a, "B": fam_b}
EXAMS = [("D", 1, "B", "placebo"), ("E", 1, "B", "harmful"), ("D", 2, "B", "helpful")] + \
        [("G", k, "A", "placebo") for k in range(1, 6)]
TR = ROOT / "transcripts_n3"
CAP_USD = 40.0
P = "study3n_"


def skills():
    return {"A": fam_a.all_skills(), "B": fam_b.all_skills()}


def prompt(f, v):
    if f == "A":
        return fam_a.TASK.replace("{log}", v["log"])
    return fam_b.TASK.format(region=v["region"], csv=v["csv"])


def call(cl, skill_text, user):
    sysp = run2.system_for(skill_text)
    req_hash = H((sysp + "\x00" + user).encode())
    err = "not attempted"
    for attempt in range(3):
        try:
            r = cl.messages.create(model=MODEL, max_tokens=MAX_TOKENS, system=sysp,
                                   messages=[{"role": "user", "content": user}])
            text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            return {"ok": True, "text": text, "stop_reason": r.stop_reason, "in": r.usage.input_tokens,
                    "out": r.usage.output_tokens, "req": req_hash, "id": r.id}
        except Exception as ex:  # noqa: BLE001
            err = f"{type(ex).__name__}: {str(ex)[:200]}"
            if "credit balance" in err:
                break
            time.sleep(5 * (attempt + 1))
    return {"ok": False, "error": err, "req": req_hash, "in": 0, "out": 0}


def pair(cl, f, skill_text, sd):
    v = FAM[f].variant(sd)
    u = prompt(f, v)
    with ThreadPoolExecutor(2) as ex:
        w, n = ex.map(lambda s: call(cl, s, u), (skill_text, None))
    out = {}
    for arm, rec in (("with_skill", w), ("no_skill", n)):
        ok, parsed = FAM[f].grade(rec.get("text", ""), v["answer"]) if rec["ok"] else (False, None)
        out[arm] = {**rec, "pass": bool(ok), "parsed": parsed}
    return v, out


def spent():
    tot = 0.0
    for fl in TR.glob("*.jsonl"):
        for l in fl.read_text().splitlines():
            d = json.loads(l)
            tot += sum(cost(d[a]) for a in ("with_skill", "no_skill"))
    return tot


def entries():
    return [json.loads(l) for l in LEDGER.read_text().splitlines()]


def calibrate():
    cl, sk = client(), skills()["B"]
    TR.mkdir(exist_ok=True)
    f = TR / "calibration.jsonl"
    summ = {}
    for name in ("helpful", "placebo", "harmful"):
        w = n = ok = 0
        for i in range(5):
            sd = hashlib.sha256(f"calibration-B-{name}-{i}".encode()).digest()
            v, o = pair(cl, "B", sk[name], sd)
            with f.open("a") as fh:
                fh.write(json.dumps({"skill": name, "i": i, "seed": sd.hex(), "refund_matters": v["refund_matters"], **o}) + "\n")
            w += o["with_skill"]["pass"]; n += o["no_skill"]["pass"]; ok += o["with_skill"]["ok"] and o["no_skill"]["ok"]
            print(name, i, "with", o["with_skill"]["pass"], "without", o["no_skill"]["pass"], "refund matters", v["refund_matters"], f"${spent():.3f}", flush=True)
        summ[name] = {"with": w, "without": n, "ok_pairs": ok, "pairs": 5}
    append(P + "calibration", {"note": "family B, public calibration seeds, not exam variants, not scored",
                               "summary": summ, "transcript_sha256": H(f.read_bytes()), "cost_usd": round(spent(), 4)})
    print(json.dumps(summ, indent=1))


def commit():
    if any(e["kind"] == P + "commit" for e in entries()):
        sys.exit("study 3N already committed")
    sk = skills()
    theta = {"study": "P9 live study 3N (Nemotron)", "model": MODEL, "max_tokens": MAX_TOKENS, "settings": "API defaults; OpenRouter provider " + json.dumps(PROVIDER),
             "families": {"A": fam_a.FAMILY, "B": fam_b.FAMILY},
             "source_sha256": {p.name: H(p.read_bytes()) for p in (ROOT / "family.py", ROOT / "family_b.py", ROOT / "run2.py", ROOT / "run3.py", ROOT / "run2n.py", ROOT / "run3n.py")},
             "skills_sha256": {f: {k: H(v.encode()) for k, v in d.items()} for f, d in sk.items()},
             "system_prompt_sha256": H(SYSTEM.encode()),
             "task_template_sha256": {"A": H(fam_a.TASK.encode()), "B": H(fam_b.TASK.encode())},
             "exams": [{"t": t + 1, "lineage": L, "attempt": k, "family": f, "skill": s} for t, (L, k, f, s) in enumerate(EXAMS)],
             "levels": "as study 2; new history, t restarts at 1",
             "gate": "as study 2 (PrPl, cap 0.5, delta 0, futility delta_f 0.05 alpha_f 0.10, N_max 600)",
             "counterfactual": "lineage G also reported at per-exam level alpha without lineage spending",
             "seeds": "variant j of exam t uses SHA-256(sigma || t || j), sigma = SHA-256(sigma_p || beacon)",
             "errors": "a pair with a failed call in either arm after retries is reported and excluded",
             "chunking": f"chunks of {CHUNK}; decision at the first crossing", "budget_usd": CAP_USD,
             "beacon": "drand quicknet", "beacon_round": latest_round() + 40}
    sp, rho = secrets.token_bytes(32), secrets.token_bytes(16)
    for name, val in ((P + "sigma_p.hex", sp), (P + "rho.hex", rho)):
        (SECRET / name).write_text(val.hex())
        os.chmod(SECRET / name, 0o600)
    e = append(P + "commit", {"commitment": H(sp + rho + jcs(theta)), "theta": theta, "theta_hash": H(jcs(theta))})
    print(json.dumps({"beacon_round": theta["beacon_round"], "entry": e["entry_hash"]}, indent=1))


def exam_pairs(t):
    fl = TR / f"exam{t}.jsonl"
    return [json.loads(l) for l in fl.read_text().splitlines()] if fl.exists() else []


D_of = run2.D_of


def run():
    th = [e for e in entries() if e["kind"] == P + "commit"][-1]["payload"]["theta"]
    sigma = sigma_from(bytes.fromhex((SECRET / (P + "sigma_p.hex")).read_text()), th)
    sk = skills()
    assert {f: {k: H(v.encode()) for k, v in d.items()} for f, d in sk.items()} == th["skills_sha256"]
    cl, R = client(), 0
    for t, (L, k, f, name) in enumerate(EXAMS, 1):
        a = level(t, k, R)
        recs = exam_pairs(t)
        j = len(recs)
        dec, n = decide(D_of(recs), a)
        print(f"exam {t} ({L}{k} {f}:{name}) level {a:.5f} start at pair {j}", flush=True)
        while dec is None:
            if spent() >= CAP_USD:
                append(P + "stop", {"reason": "budget", "spent_usd": round(spent(), 4), "exam": t})
                print("STUDY3 STOP budget", flush=True)
                return
            with ThreadPoolExecutor(CHUNK) as ex:
                outs = list(ex.map(lambda jj: (jj, *pair(cl, f, sk[f][name], seed(sigma, t, jj))), range(j, j + CHUNK)))
            if all(not o["with_skill"]["ok"] and "credit balance" in o["with_skill"].get("error", "") for _, _, o in outs):
                append(P + "stop", {"reason": "provider rejected requests: credit balance", "exam": t, "spent_usd": round(spent(), 4)})
                print("STUDY3 STOP credit", flush=True)
                return
            with (TR / f"exam{t}.jsonl").open("a") as fh:
                for jj, v, o in outs:
                    fh.write(json.dumps({"t": t, "j": jj, "seed": seed(sigma, t, jj).hex(), "answer": v["answer"], **o}) + "\n")
            j += CHUNK
            recs = exam_pairs(t)
            dec, n = decide(D_of(recs), a)
            Ds = D_of(recs)
            print(f"  t={t} pairs={len(recs)} valid={len(Ds)} Dbar={np.mean(Ds):.3f} spent=${spent():.2f}", flush=True)
        R += dec == "admit"
        print(f"exam {t} decision {dec} at valid pair {n}", flush=True)
    print("STUDY3 RUNS DONE", f"${spent():.2f}", flush=True)


def body_from(sigma, th, com):
    exams, R = [], 0
    for t, (L, k, f, name) in enumerate(EXAMS, 1):
        recs = exam_pairs(t)
        for r in recs:
            assert r["seed"] == seed(sigma, t, r["j"]).hex()
            ans = FAM[f].variant(bytes.fromhex(r["seed"]))["answer"]
            for arm in ("with_skill", "no_skill"):
                ok = FAM[f].grade(r[arm].get("text", ""), ans)[0] if r[arm]["ok"] else False
                assert ok == r[arm]["pass"]
        a = level(t, k, R)
        D = D_of(recs)
        dec, n = decide(D, a)
        lcb = float(gates.lcb_path(np.array(D[:n], float), a, CAP)[-1]) if n else None
        cf = decide(D, ALPHA) if L == "G" else (None, None)
        exams.append({"t": t, "lineage": L, "attempt": k, "family": f, "skill": name, "level": a, "decision": dec,
                      "decision_pair": n, "pairs_run": len(recs), "pairs_excluded": len(recs) - len(D),
                      "Dbar_at_decision": float(np.mean(D[:n])) if n else None, "LCB_at_decision": lcb,
                      "pass_with": sum(r["with_skill"]["pass"] for r in recs),
                      "pass_without": sum(r["no_skill"]["pass"] for r in recs),
                      "per_exam_alpha_decision": cf[0], "per_exam_alpha_pair": cf[1],
                      "root": "sha256:" + merkle_root([json.dumps(r, sort_keys=True).encode() for r in recs]).hex()})
        R += dec == "admit"
    return {"exam": "P9 live study 3N (Nemotron)", "commitment": com["payload"]["commitment"], "theta_hash": com["payload"]["theta_hash"],
            "beacon_round": th["beacon_round"], "sigma": sigma.hex(), "exams": exams, "cost_usd": round(spent(), 4)}


def build():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    es = entries()
    if any(e["kind"] == P + "credential" for e in es):
        sys.exit("already built")
    com = [e for e in es if e["kind"] == P + "commit"][-1]
    th = com["payload"]["theta"]
    sp, rho = (bytes.fromhex((SECRET / (P + x)).read_text()) for x in ("sigma_p.hex", "rho.hex"))
    assert H(sp + rho + jcs(th)) == com["payload"]["commitment"]
    rev = append(P + "reveal", {"sigma_p": sp.hex(), "rho": rho.hex()})
    body = body_from(sigma_from(sp, th), th, com)
    key = Ed25519PrivateKey.generate()
    (SECRET / (P + "proctor_ed25519.pem")).write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    pk = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    chi = H(jcs(body))
    append(P + "credential", {"body": body, "chi": chi, "sig": key.sign(bytes.fromhex(chi)).hex(),
                              "proctor_pk": pk.hex(), "after_reveal": rev["entry_hash"]})
    print(json.dumps(body["exams"], indent=1), "\ncost", body["cost_usd"])


def verify():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    es = entries()
    prev, ok_chain = "0" * 64, True
    for e in es:
        b = {k: e[k] for k in ("kind", "payload", "time_utc", "prev")}
        ok_chain &= e["prev"] == prev and H(jcs(b)) == e["entry_hash"]
        prev = e["entry_hash"]
    com = [e for e in es if e["kind"] == P + "commit"][-1]
    th = com["payload"]["theta"]
    rev = [e for e in es if e["kind"] == P + "reveal"][-1]["payload"]
    cred = [e for e in es if e["kind"] == P + "credential"][-1]["payload"]
    sp, rho = bytes.fromhex(rev["sigma_p"]), bytes.fromhex(rev["rho"])
    with urllib.request.urlopen(f"{DRAND}/info", timeout=20) as r:
        info = json.load(r)
    t_round = info["genesis_time"] + (th["beacon_round"] - 1) * info["period"]
    t_commit = calendar.timegm(time.strptime(com["time_utc"], "%Y-%m-%dT%H:%M:%SZ"))
    checks = {"ledger hash chain intact": ok_chain,
              "reveal opens commitment": H(sp + rho + jcs(th)) == com["payload"]["commitment"],
              "commitment logged before beacon round existed": t_commit < t_round,
              "skills match registration": {f: {k: H(v.encode()) for k, v in d.items()} for f, d in skills().items()} == th["skills_sha256"]}
    try:
        checks["every grade and decision recomputes from transcripts"] = body_from(sigma_from(sp, th), th, com) == cred["body"]
    except AssertionError:
        checks["every grade and decision recomputes from transcripts"] = False
    checks["digest matches"] = H(jcs(cred["body"])) == cred["chi"]
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(cred["proctor_pk"])).verify(bytes.fromhex(cred["sig"]), bytes.fromhex(cred["chi"]))
        checks["signature valid"] = True
    except Exception:  # noqa: BLE001
        checks["signature valid"] = False
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    for e in cred["body"]["exams"]:
        cf = f" (per-exam alpha: {e['per_exam_alpha_decision']})" if e["lineage"] == "G" else ""
        print(f"exam {e['t']} {e['lineage']}{e['attempt']} {e['family']}:{e['skill']}: {e['decision']} at pair {e['decision_pair']}{cf}")
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    {"calibrate": calibrate, "commit": commit, "run": run, "build": build, "verify": verify}[sys.argv[1]]()
