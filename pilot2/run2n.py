"""Live study 2N (Nemotron arm of live study 2): the full gate on a live model, with generated hidden variants and four registered skills.

  python pilot2/run2n.py calibrate   # 5 public-seed variants per skill, both arms; NOT scored; logged
  python pilot2/run2n.py commit      # register skills/generator/prompt/plan, commit before a drand round
  python pilot2/run2n.py run         # exams in the committed order, e-LOND + lineage levels, stop rules
  python pilot2/run2n.py build       # reveal, compute decisions from transcripts, sign the credential
  python pilot2/run2n.py verify      # anyone: recompute everything from ledger + transcripts + beacon

Design (committed in theta): four exams in this order, each on fresh variants derived from the beacon:
  t=1 lineage A attempt 1: placebo     t=2 lineage B attempt 1: leak
  t=3 lineage C attempt 1: harmful     t=4 lineage A attempt 2: helpful
Exam t runs at level min(alpha * gamma_t * (R_{t-1} + 1), alpha * gamma^lin_k) with gamma_j = j^-1.6 / zeta(1.6).
Per variant, both arms (with and without the skill) answer the same generated log concurrently; a pass is an
exact match of ip and count; D = pass_with - pass_without. Betting gate: PrPl bets, cap 0.5, delta 0,
futility at delta_f 0.05 and alpha_f 0.10, N_max 600. Pairs run in chunks of 6; the decision is the first
crossing; pairs past it are reported but do not change the decision. A pair whose API call fails after
retries in either arm is reported and excluded (the exam moves to the next variant).
Budget: hard stop when the summed API cost of study 2N reaches 40 USD.
Model: nvidia/nemotron-3.5-super-vl via OpenRouter, provider pinned to crusoe/bf16 with no fallbacks,
API defaults otherwise (default reasoning), max_tokens 16000. Everything else exactly as run2.py.
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
import family  # noqa: E402

MODEL = "nvidia/nemotron-3.5-super-vl"
PROVIDER = {"order": ["crusoe/bf16"], "allow_fallbacks": False}
PROVIDER_NAME = "Crusoe"
MAX_TOKENS = 16000
PRICE_IN, PRICE_OUT = 0.10e-6, 0.40e-6          # USD per token, Crusoe list price on OpenRouter 2026-11-17
CAP_USD = 40.0
ALPHA, ALPHA_F, DELTA_F, CAP, NMAX, S = 0.05, 0.10, 0.05, 0.5, 600, 1.6
CHUNK = 6
EXAMS = [("A", 1, "placebo"), ("B", 1, "leak"), ("C", 1, "harmful"), ("A", 2, "helpful")]
TR = ROOT / "transcripts_n2"
SYSTEM = "You are a careful operations engineer. Answer the user's question about the log they give you."


def env_key():
    for line in (Path.home() / ".config/p9_pilot/.env").read_text().splitlines():
        if line.startswith("ANTHROPIC_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("no key")


def client():
    """OpenRouter adapter exposing the subset of anthropic.Anthropic().messages.create used by call()."""
    import types
    import urllib.error
    import urllib.request
    k = None
    for line in (Path.home() / ".config/p9_pilot/.env").read_text().splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            k = line.split("=", 1)[1].strip().strip('"')
    if not k:
        raise SystemExit("no OPENROUTER_API_KEY")

    def create(model, max_tokens, system, messages):
        body = {"model": model, "max_tokens": max_tokens, "provider": PROVIDER, "usage": {"include": True},
                "messages": [{"role": "system", "content": system}] + messages}
        req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", method="POST",
                                     data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {k}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                d = json.loads(resp.read())
        except urllib.error.HTTPError as ex:
            raise RuntimeError(f"HTTP {ex.code}: {ex.read()[:200]!r}") from None
        if d.get("error") or not d.get("choices"):
            raise RuntimeError(f"API error: {str(d.get('error'))[:200]}")
        if d.get("provider") != PROVIDER_NAME:
            raise RuntimeError(f"served by {d.get('provider')!r}, not {PROVIDER_NAME}")
        ch, u = d["choices"][0], d.get("usage") or {}
        return types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text=ch["message"].get("content") or "")],
            stop_reason=ch.get("finish_reason"), id=f"{d.get('id')}|{d.get('provider')}|{u.get('cost')}",
            usage=types.SimpleNamespace(input_tokens=u.get("prompt_tokens", 0), output_tokens=u.get("completion_tokens", 0)))

    return types.SimpleNamespace(messages=types.SimpleNamespace(create=create))


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
            r = cl.messages.create(model=MODEL, max_tokens=MAX_TOKENS, system=sysp,
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
    for name in ("helpful", "placebo", "leak", "harmful"):
        for i in range(5):
            seed = hashlib.sha256(f"calibration-{name}-{i}".encode()).digest()
            v, o = pair(cl, sk[name], seed)
            rec = {"skill": name, "i": i, "seed": seed.hex(), **o}
            with f.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            rows.append((name, o["with_skill"]["pass"], o["no_skill"]["pass"], o["with_skill"]["ok"] and o["no_skill"]["ok"]))
            print(name, i, "with", o["with_skill"]["pass"], "without", o["no_skill"]["pass"],
                  "stop", o["with_skill"].get("stop_reason"), f"${spent():.3f}", flush=True)
    summ = {n: {"with": sum(r[1] for r in rows if r[0] == n), "without": sum(r[2] for r in rows if r[0] == n),
                "ok_pairs": sum(r[3] for r in rows if r[0] == n), "pairs": 5} for n in ("helpful", "placebo", "leak", "harmful")}
    append("study2n_calibration", {"note": "public calibration seeds, not exam variants, not scored",
                                  "summary": summ, "transcript_sha256": H(f.read_bytes()), "cost_usd": round(spent(), 4)})
    print(json.dumps(summ, indent=1))


# ------------------------------------------------------------------ commit
def commit():
    es = entries()
    if any(e["kind"] == "study2n_commit" for e in es):
        sys.exit("study 2N already committed")
    sk = family.all_skills()
    src = {p.name: H(p.read_bytes()) for p in (ROOT / "family.py", ROOT / "run2.py", ROOT / "run2n.py")}
    theta = {"study": "P9 live study 2N (Nemotron)", "model": MODEL, "max_tokens": MAX_TOKENS, "settings": "API defaults; OpenRouter provider " + json.dumps(PROVIDER),
             "family": family.FAMILY, "source_sha256": src,
             "skills_sha256": {k: H(v.encode()) for k, v in sk.items()},
             "system_prompt_sha256": H(SYSTEM.encode()), "task_template_sha256": H(family.TASK.encode()),
             "exams": [{"t": t + 1, "lineage": L, "attempt": k, "skill": s} for t, (L, k, s) in enumerate(EXAMS)],
             "levels": "min(alpha*gamma_t*(R_{t-1}+1), alpha*gamma_k^lin), gamma_j = j^-1.6/zeta(1.6)",
             "gate": {"alpha": ALPHA, "alpha_futility": ALPHA_F, "futility_margin": DELTA_F, "cap": CAP,
                      "n_max": NMAX, "delta": 0.0, "bets": "PrPl"},
             "seeds": "variant j of exam t uses seed SHA-256(sigma || t || j), sigma = SHA-256(sigma_p || beacon)",
             "pass": "exact match of ip and count in the last JSON line",
             "errors": "a pair with a failed call in either arm after retries is reported and excluded",
             "chunking": f"pairs run in chunks of {CHUNK}; decision at the first crossing",
             "budget_usd": CAP_USD, "beacon": "drand quicknet", "beacon_round": None}
    theta["beacon_round"] = latest_round() + 40       # about two minutes ahead
    sp, rho = secrets.token_bytes(32), secrets.token_bytes(16)
    for name, val in (("study2n_sigma_p.hex", sp), ("study2n_rho.hex", rho)):
        (SECRET / name).write_text(val.hex())
        os.chmod(SECRET / name, 0o600)
    e = append("study2n_commit", {"commitment": H(sp + rho + jcs(theta)), "theta": theta, "theta_hash": H(jcs(theta))})
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
    res = gates.betting(np.array([D], float), 0.0, a, ALPHA_F, CAP, delta_f=DELTA_F)
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
    th = [e for e in es if e["kind"] == "study2n_commit"][-1]["payload"]["theta"]
    sp = bytes.fromhex((SECRET / "study2n_sigma_p.hex").read_text())
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
                append("study2n_stop", {"reason": "budget", "spent_usd": round(spent(), 4), "exam": t})
                print("STUDY2N STOP budget", flush=True)
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
    print("STUDY2N RUNS DONE", f"${spent():.2f}", flush=True)


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
    return {"exam": "P9 live study 2N (Nemotron)", "commitment": com_entry["payload"]["commitment"],
            "theta_hash": com_entry["payload"]["theta_hash"], "beacon_round": th["beacon_round"],
            "sigma": sigma.hex(), "exams": exams, "cost_usd": round(spent(), 4)}


def build():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    es = entries()
    if any(e["kind"] == "study2n_credential" for e in es):
        sys.exit("already built")
    com = [e for e in es if e["kind"] == "study2n_commit"][-1]
    th = com["payload"]["theta"]
    sp, rho = (bytes.fromhex((SECRET / f).read_text()) for f in ("study2n_sigma_p.hex", "study2n_rho.hex"))
    assert H(sp + rho + jcs(th)) == com["payload"]["commitment"]
    rev = append("study2n_reveal", {"sigma_p": sp.hex(), "rho": rho.hex()})
    body = body_from(entries(), sigma_from(sp, th), th, com)
    key = Ed25519PrivateKey.generate()
    (SECRET / "study2n_proctor_ed25519.pem").write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    pk = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    chi = H(jcs(body))
    append("study2n_credential", {"body": body, "chi": chi, "sig": key.sign(bytes.fromhex(chi)).hex(),
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
    com = [e for e in es if e["kind"] == "study2n_commit"][-1]
    th = com["payload"]["theta"]
    rev = [e for e in es if e["kind"] == "study2n_reveal"][-1]["payload"]
    cred = [e for e in es if e["kind"] == "study2n_credential"][-1]["payload"]
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
