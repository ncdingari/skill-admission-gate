"""Live study 4c: the full gate on a real, published third-party skill, with generated hidden variants.
The skill, its pinned source and the task family are in the family module imported below (see its docstring).

  python pilot2/run4c.py calibrate   # 20 public-seed variants per skill (real, placebo), both arms; NOT scored; logged
  python pilot2/run4c.py commit      # register skills/generator/prompt/plan, commit before a drand round
  python pilot2/run4c.py run         # exams in the committed order, e-LOND + lineage levels, stop rules
  python pilot2/run4c.py build       # reveal, compute decisions from transcripts, sign the credential
  python pilot2/run4c.py verify      # anyone: recompute everything from ledger + transcripts + beacon

Design (committed in theta): one lineage S in a new registry history; exam order and N_max come from the
power plan power_plan_study4c.json, fixed after calibration and before commit (default: t=1 placebo, t=2 real skill,
N_max 600). Level min(alpha * gamma_t * (R_{t-1} + 1), alpha * gamma^lin_k), gamma_j = j^-1.6 / zeta(1.6).
Per variant both arms answer the same notes; a pass is every graded AP slot correct; D = pass_with - pass_without.
Betting gate: level-tuned grid mixture bet (Theorem thm:adaptive), delta 0, futility at delta_f 0.05 and alpha_f
0.10 (PrPl futility bet, as in studies 2 and 3). Pairs run in chunks of 6; the decision is the first crossing.
A pair whose API call fails after retries in either arm is reported and excluded.
Budget: hard stop when the summed API cost of study 4 reaches 15 USD.
Model: claude-haiku-5-5, default settings (adaptive thinking, default effort), max_tokens 8000.
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
P9 = ROOT.parent
sys.path.insert(0, str(P9 / "src"))
sys.path.insert(0, str(P9 / "pilot"))
from commit import DRAND, LEDGER, SECRET, H, append, jcs, latest_round  # noqa: E402
from nixigate import fdr, gates  # noqa: E402
from nixigate.credential import merkle_root  # noqa: E402

sys.path.insert(0, str(ROOT))
import family_s4_commit as family  # noqa: E402

MODEL = "claude-haiku-5-5"
PRICE_IN, PRICE_OUT = 0.10e-6, 0.50e-6          # USD per token, prompts <= 100K tokens
CAP_USD = 15.0
ALPHA, ALPHA_F, DELTA_F, CAP, S = 0.05, 0.10, 0.05, 0.5, 1.6
POWER_FILE = ROOT / "power_plan_study4c.json"
POWER = json.loads(POWER_FILE.read_text()) if POWER_FILE.exists() else {}
NMAX = int(POWER.get("n_max", 600))
RULE = "grid"
N_CAL = 20
CHUNK = 6
EXAMS = [tuple(e) for e in POWER.get("exams", [["S", 1, "placebo"], ["S", 2, "real"]])]
TR = ROOT / "transcripts4c"
SYSTEM = family.SYSTEM


def env_key():
    for line in (Path.home() / ".config/p9_pilot/.env").read_text().splitlines():
        if line.startswith("ANTHROPIC_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("no key")


def client():
    import anthropic
    return anthropic.Anthropic(api_key=env_key(), max_retries=4, timeout=180)


def system_for(skill_text):
    if skill_text is None:
        return SYSTEM
    return SYSTEM + "\n\nThe following skill is installed. Use it when relevant.\n\n<skill>\n" + skill_text + "\n</skill>"


def call(cl, skill_text, log):
    user = family.TASK.replace("{log}", log)
    sysp = system_for(skill_text)
    req_hash = H((sysp + "\x00" + user).encode())
    err = "not attempted"
    for attempt in range(3):
        try:
            r = cl.messages.create(model=MODEL, max_tokens=8000, system=sysp,
                                   messages=[{"role": "user", "content": user}])
            text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            u = r.usage
            return {"ok": True, "text": text, "stop_reason": r.stop_reason, "in": u.input_tokens,
                    "out": u.output_tokens, "req": req_hash, "id": r.id}
        except Exception as ex:  # noqa: BLE001
            err = f"{type(ex).__name__}: {str(ex)[:200]}"
            time.sleep(5 * (attempt + 1))
    return {"ok": False, "error": err, "req": req_hash, "in": 0, "out": 0}


def cost(rec):
    return rec["in"] * PRICE_IN + rec["out"] * PRICE_OUT


def pair(cl, skill_text, seed):
    v = family.variant(seed)
    with ThreadPoolExecutor(2) as ex:
        w, n = ex.map(lambda s: call(cl, s, v["log"]), (skill_text, None))
    out = {}
    for arm, rec in (("with_skill", w), ("no_skill", n)):
        ok, parsed = family.grade(rec.get("text", ""), v["answer"]) if rec["ok"] else (False, None)
        out[arm] = {**rec, "pass": bool(ok), "parsed": parsed}
    return v, out


def spent():
    tot = 0.0
    for f in TR.glob("*.jsonl"):
        for l in f.read_text().splitlines():
            d = json.loads(l)
            tot += sum(cost(d[a]) for a in ("with_skill", "no_skill"))
    return tot


def entries():
    return [json.loads(l) for l in LEDGER.read_text().splitlines()]


# ------------------------------------------------------------------ calibrate (not scored)
def calibrate():
    cl = client()
    sk = family.all_skills()
    TR.mkdir(exist_ok=True)
    rows, f = [], TR / "calibration.jsonl"
    for name in ("real", "placebo"):
        for i in range(N_CAL):
            seed = hashlib.sha256(f"calibration-{family.FAMILY}-{i}".encode()).digest()
            v, o = pair(cl, sk[name], seed)
            rec = {"skill": name, "i": i, "seed": seed.hex(), **o}
            with f.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            rows.append((name, o["with_skill"]["pass"], o["no_skill"]["pass"], o["with_skill"]["ok"] and o["no_skill"]["ok"]))
            print(name, i, "with", o["with_skill"]["pass"], "without", o["no_skill"]["pass"],
                  "stop", o["with_skill"].get("stop_reason"), f"${spent():.3f}", flush=True)
    summ = {n: {"with": sum(r[1] for r in rows if r[0] == n), "without": sum(r[2] for r in rows if r[0] == n),
                "ok_pairs": sum(r[3] for r in rows if r[0] == n), "pairs": N_CAL} for n in ("real", "placebo")}
    append("study4c_calibration", {"note": "public calibration seeds, not exam variants, not scored", "family": family.FAMILY,
                                  "summary": summ, "transcript_sha256": H(f.read_bytes()), "cost_usd": round(spent(), 4)})
    print(json.dumps(summ, indent=1))


# ------------------------------------------------------------------ commit
def commit():
    es = entries()
    if any(e["kind"] == "study4c_commit" for e in es):
        sys.exit("study 4c already committed")
    sk = family.all_skills()
    src = {p.name: H(p.read_bytes()) for p in (ROOT / "family_s4_commit.py", ROOT / "run4c.py", family.SKILL_PATH)}
    theta = {"study": "P9 live study 4c (real skill)", "model": MODEL, "max_tokens": 8000, "settings": "API defaults",
             "family": family.FAMILY, "source_sha256": src,
             "skills_sha256": {k: H(v.encode()) for k, v in sk.items()},
             "system_prompt_sha256": H(SYSTEM.encode()), "task_template_sha256": H(family.TASK.encode()),
             "exams": [{"t": t + 1, "lineage": L, "attempt": k, "skill": s} for t, (L, k, s) in enumerate(EXAMS)],
             "levels": "min(alpha*gamma_t*(R_{t-1}+1), alpha*gamma_k^lin), gamma_j = j^-1.6/zeta(1.6)",
             "gate": {"alpha": ALPHA, "alpha_futility": ALPHA_F, "futility_margin": DELTA_F, "cap": CAP,
                      "n_max": NMAX, "delta": 0.0, "bets": RULE, "futility_bets": "PrPl"},
             "seeds": "variant j of exam t uses seed SHA-256(sigma || t || j), sigma = SHA-256(sigma_p || beacon)",
             "pass": "family.grade (see the family module)",
             "power_plan": POWER, "power_plan_sha256": H(POWER_FILE.read_bytes()) if POWER_FILE.exists() else None,
             "errors": "a pair with a failed call in either arm after retries is reported and excluded",
             "chunking": f"pairs run in chunks of {CHUNK}; decision at the first crossing",
             "budget_usd": CAP_USD, "beacon": "drand quicknet", "beacon_round": None}
    theta["beacon_round"] = latest_round() + 40       # about two minutes ahead
    sp, rho = secrets.token_bytes(32), secrets.token_bytes(16)
    for name, val in (("study4c_sigma_p.hex", sp), ("study4c_rho.hex", rho)):
        (SECRET / name).write_text(val.hex())
        os.chmod(SECRET / name, 0o600)
    e = append("study4c_commit", {"commitment": H(sp + rho + jcs(theta)), "theta": theta, "theta_hash": H(jcs(theta))})
    print(json.dumps({"beacon_round": theta["beacon_round"], "entry": e["entry_hash"]}, indent=1))


def beacon(rnd):
    with urllib.request.urlopen(f"{DRAND}/public/{rnd}", timeout=20) as r:
        return json.load(r)["randomness"]


def sigma_from(sp, th):
    return hashlib.sha256(sp + bytes.fromhex(beacon(th["beacon_round"]))).digest()


def seed(sigma, t, j):
    return hashlib.sha256(sigma + t.to_bytes(2, "big") + j.to_bytes(4, "big")).digest()


def level(t, k, R):
    g = fdr.gamma_seq(max(t, k), S)
    return float(min(ALPHA * g[t - 1] * (R + 1), ALPHA * g[k - 1]))


def decide(D, a):
    """Gate decision on the pairs so far: ('admit'|'futility'|'nmax'|None, index)."""
    if len(D) == 0:
        return None, 0
    res = gates.betting(np.array([D], float), 0.0, a, ALPHA_F, CAP, rule=RULE, delta_f=DELTA_F)
    n = int(res.n[0])
    if res.admit[0]:
        return "admit", n
    if res.futility[0]:
        return "futility", n
    if len(D) >= NMAX:
        return "nmax", NMAX
    return None, n


def exam_pairs(t):
    f = TR / f"exam{t}.jsonl"
    return [json.loads(l) for l in f.read_text().splitlines()] if f.exists() else []


def D_of(recs):
    return [int(r["with_skill"]["pass"]) - int(r["no_skill"]["pass"]) for r in recs
            if r["with_skill"]["ok"] and r["no_skill"]["ok"]]


# ------------------------------------------------------------------ run
def run():
    es = entries()
    th = [e for e in es if e["kind"] == "study4c_commit"][-1]["payload"]["theta"]
    sp = bytes.fromhex((SECRET / "study4c_sigma_p.hex").read_text())
    sigma = sigma_from(sp, th)
    sk = family.all_skills()
    assert {k: H(v.encode()) for k, v in sk.items()} == th["skills_sha256"], "skills changed since commit"
    cl = client()
    R = 0
    for t, (L, k, name) in enumerate(EXAMS, 1):
        a = level(t, k, R)
        recs = exam_pairs(t)
        j = len(recs)
        dec, n = decide(D_of(recs), a)
        print(f"exam {t} ({L}{k} {name}) level {a:.5f} start at pair {j}", flush=True)
        while dec is None:
            if spent() >= CAP_USD:
                append("study4c_stop", {"reason": "budget", "spent_usd": round(spent(), 4), "exam": t})
                print("STUDY4C STOP budget", flush=True)
                return
            with ThreadPoolExecutor(CHUNK) as ex:
                outs = list(ex.map(lambda jj: (jj, *pair(cl, sk[name], seed(sigma, t, jj))), range(j, j + CHUNK)))
            with (TR / f"exam{t}.jsonl").open("a") as fh:
                for jj, v, o in outs:
                    fh.write(json.dumps({"t": t, "j": jj, "seed": seed(sigma, t, jj).hex(),
                                         "answer": v["answer"], **o}) + "\n")
            j += CHUNK
            recs = exam_pairs(t)
            dec, n = decide(D_of(recs), a)
            Ds = D_of(recs)
            print(f"  t={t} pairs={len(recs)} valid={len(Ds)} Dbar={np.mean(Ds):.3f} spent=${spent():.2f}", flush=True)
        R += dec == "admit"
        print(f"exam {t} decision {dec} at valid pair {n}", flush=True)
    print("STUDY4C RUNS DONE", f"${spent():.2f}", flush=True)


# ------------------------------------------------------------------ build / verify
def body_from(es, sigma, th, com_entry):
    exams, R = [], 0
    for t, (L, k, name) in enumerate(EXAMS, 1):
        recs = exam_pairs(t)
        for r in recs:                                   # recompute every grade from the regenerated variant
            assert r["seed"] == seed(sigma, t, r["j"]).hex()
            ans = family.variant(bytes.fromhex(r["seed"]))["answer"]
            for arm in ("with_skill", "no_skill"):
                ok = family.grade(r[arm].get("text", ""), ans)[0] if r[arm]["ok"] else False
                assert ok == r[arm]["pass"]
        a = level(t, k, R)
        D = D_of(recs)
        dec, n = decide(D, a)
        lcb = float(gates.lcb_path(np.array(D[:n], float), a, CAP)[-1]) if n else None
        exams.append({"t": t, "lineage": L, "attempt": k, "skill": name, "level": a, "decision": dec,
                      "decision_pair": n, "pairs_run": len(recs), "pairs_excluded": len(recs) - len(D),
                      "Dbar_at_decision": float(np.mean(D[:n])) if n else None, "LCB_at_decision": lcb,
                      "pass_with": sum(r["with_skill"]["pass"] for r in recs),
                      "pass_without": sum(r["no_skill"]["pass"] for r in recs),
                      "root": "sha256:" + merkle_root([json.dumps(r, sort_keys=True).encode() for r in recs]).hex()})
        R += dec == "admit"
    return {"exam": "P9 live study 4c (real skill)", "commitment": com_entry["payload"]["commitment"],
            "theta_hash": com_entry["payload"]["theta_hash"], "beacon_round": th["beacon_round"],
            "sigma": sigma.hex(), "exams": exams, "cost_usd": round(spent(), 4)}


def build():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    es = entries()
    if any(e["kind"] == "study4c_credential" for e in es):
        sys.exit("already built")
    com = [e for e in es if e["kind"] == "study4c_commit"][-1]
    th = com["payload"]["theta"]
    sp, rho = (bytes.fromhex((SECRET / f).read_text()) for f in ("study4c_sigma_p.hex", "study4c_rho.hex"))
    assert H(sp + rho + jcs(th)) == com["payload"]["commitment"]
    rev = append("study4c_reveal", {"sigma_p": sp.hex(), "rho": rho.hex()})
    body = body_from(entries(), sigma_from(sp, th), th, com)
    key = Ed25519PrivateKey.generate()
    (SECRET / "study4c_proctor_ed25519.pem").write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    pk = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    chi = H(jcs(body))
    append("study4c_credential", {"body": body, "chi": chi, "sig": key.sign(bytes.fromhex(chi)).hex(),
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
    com = [e for e in es if e["kind"] == "study4c_commit"][-1]
    th = com["payload"]["theta"]
    rev = [e for e in es if e["kind"] == "study4c_reveal"][-1]["payload"]
    cred = [e for e in es if e["kind"] == "study4c_credential"][-1]["payload"]
    sp, rho = bytes.fromhex(rev["sigma_p"]), bytes.fromhex(rev["rho"])
    with urllib.request.urlopen(f"{DRAND}/info", timeout=20) as r:
        info = json.load(r)
    t_round = info["genesis_time"] + (th["beacon_round"] - 1) * info["period"]
    t_commit = calendar.timegm(time.strptime(com["time_utc"], "%Y-%m-%dT%H:%M:%SZ"))
    checks = {"ledger hash chain intact": ok_chain,
              "reveal opens commitment": H(sp + rho + jcs(th)) == com["payload"]["commitment"],
              "commitment logged before beacon round existed": t_commit < t_round,
              "skills match registration": {k: H(v.encode()) for k, v in family.all_skills().items()} == th["skills_sha256"]}
    try:
        body = body_from(es, sigma_from(sp, th), th, com)
        checks["every grade and decision recomputes from transcripts"] = body == cred["body"]
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
        print(f"exam {e['t']} {e['lineage']}{e['attempt']} {e['skill']}: {e['decision']} at pair {e['decision_pair']}")
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    {"calibrate": calibrate, "commit": commit, "run": run, "build": build, "verify": verify}[sys.argv[1]]()
