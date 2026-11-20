"""Smoke test: Nemotron 3.5 Super VL on the live-study-2 task family, two API routes (request of 2026-11-17).

  python pilot2/smoke_open_models.py            # runs both routes
  python pilot2/smoke_open_models.py hf         # one route only (hf | openrouter | nvidia)

Items: the 10 public CALIBRATION seeds of live study 2 for the helpful and placebo skills (5 each), both arms
(with / without skill). These are not exam variants and touch nothing committed. Same system prompt, task prompt,
skill text, max_tokens and grader as run2.py; only the model and API change. Haiku 5.5 results on the same seeds
are read from transcripts/calibration.jsonl for comparison.

Rules from the request: hard stop at 10 USD across routes; no retries; stop a route after 3 consecutive infra
errors (5xx, timeouts, connection errors); keys only from the environment or ~/.config/p9_pilot/.env, never printed.
Output: pilot2/smoke_open/<route>.jsonl and pilot2/smoke_open/summary.md. Nothing is written to the ledger.
"""
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import family  # noqa: E402

SYSTEM = "You are a careful operations engineer. Answer the user's question about the log they give you."
MAX_TOKENS = 8000
CAP_USD = 10.0
OUT = ROOT / "smoke_open"
ROUTES = {
    # price: USD per token (in, out) used only when the API does not return a cost; the HF value is the highest
    # listed provider price for this model on 2026-11-17 (baseten 0.20 / 0.50 per 1M), i.e. conservative.
    "hf": {"url": "https://router.huggingface.co/v1/chat/completions", "key": ("HF_TOKEN",),
           "model": "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16:cheapest", "price": (0.20e-6, 0.50e-6)},
    "openrouter": {"url": "https://openrouter.ai/api/v1/chat/completions", "key": ("OPENROUTER_API_KEY",),
                   "model": "nvidia/nemotron-3.5-super-vl", "price": (0.15e-6, 0.50e-6)},
    # NVIDIA API catalog (build.nvidia.com), paid from Inception credits; price is a placeholder for the cap only.
    "nvidia": {"url": "https://integrate.api.nvidia.com/v1/chat/completions", "key": ("NVIDIA_API_KEY",),
               "model": "nvidia/nemotron-3.5-super-vl-120b-a12b", "price": (0.20e-6, 0.50e-6)},
}


def key(names):
    for n in names:
        if os.environ.get(n):
            return os.environ[n]
    f = Path.home() / ".config/p9_pilot/.env"
    if f.exists():
        for line in f.read_text().splitlines():
            for n in names:
                if line.startswith(n + "="):
                    return line.split("=", 1)[1].strip().strip('"')
    return None


def system_for(skill_text):  # identical to run2.system_for
    if skill_text is None:
        return SYSTEM
    return SYSTEM + "\n\nThe following skill is installed. Use it when relevant.\n\n<skill>\n" + skill_text + "\n</skill>"


def call(route, k, skill_text, log):
    r = ROUTES[route]
    body = {"model": r["model"], "max_tokens": MAX_TOKENS,
            "messages": [{"role": "system", "content": system_for(skill_text)},
                         {"role": "user", "content": family.TASK.replace("{log}", log)}]}
    if route == "openrouter":
        body["usage"] = {"include": True}
    req = urllib.request.Request(r["url"], data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {k}", "Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            d = json.loads(resp.read())
            prov = resp.headers.get("x-inference-provider") or d.get("provider")
    except urllib.error.HTTPError as ex:
        infra = ex.code >= 500 or ex.code == 429
        return {"ok": False, "infra": infra, "error": f"HTTP {ex.code}: {ex.read()[:200]!r}",
                "latency_s": round(time.time() - t0, 2), "in": 0, "out": 0, "cost": 0.0}
    except Exception as ex:  # noqa: BLE001  timeouts, connection errors
        return {"ok": False, "infra": True, "error": f"{type(ex).__name__}: {str(ex)[:200]}",
                "latency_s": round(time.time() - t0, 2), "in": 0, "out": 0, "cost": 0.0}
    lat = round(time.time() - t0, 2)
    if "error" in d or not d.get("choices"):
        return {"ok": False, "infra": True, "error": str(d.get("error"))[:200], "latency_s": lat,
                "in": 0, "out": 0, "cost": 0.0}
    u = d.get("usage") or {}
    tin, tout = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
    c = u.get("cost")
    c = float(c) if c is not None else tin * r["price"][0] + tout * r["price"][1]
    ch = d["choices"][0]
    return {"ok": True, "text": ch["message"].get("content") or "", "finish": ch.get("finish_reason"),
            "provider": prov, "in": tin, "out": tout, "cost": c, "latency_s": lat, "id": d.get("id")}


def items():
    rows = [json.loads(l) for l in (ROOT / "transcripts/calibration.jsonl").read_text().splitlines()]
    return [r for r in rows if r["skill"] in ("helpful", "placebo")]


def spent_all():
    tot = 0.0
    for f in OUT.glob("*.jsonl"):
        for l in f.read_text().splitlines():
            d = json.loads(l)
            tot += d["with_skill"]["cost"] + d["no_skill"]["cost"]
    return tot


def run(route):
    k = key(ROUTES[route]["key"])
    if not k:
        print(f"[{route}] no key ({'/'.join(ROUTES[route]['key'])}); skipped", flush=True)
        return
    sk = family.all_skills()
    f = OUT / f"{route}.jsonl"
    if f.exists():
        raise SystemExit(f"{f} exists; refusing to overwrite (rename it first)")
    consec = 0
    for it in items():
        if spent_all() >= CAP_USD:
            print(f"[{route}] cost cap reached; stop", flush=True)
            return
        v = family.variant(bytes.fromhex(it["seed"]))
        with ThreadPoolExecutor(2) as ex:
            w, n = ex.map(lambda s: call(route, k, s, v["log"]), (sk[it["skill"]], None))
        rec = {"route": route, "skill": it["skill"], "i": it["i"], "seed": it["seed"]}
        for arm, x in (("with_skill", w), ("no_skill", n)):
            ok, parsed = family.grade(x.get("text", ""), v["answer"]) if x["ok"] else (False, None)
            rec[arm] = {**x, "pass": bool(ok), "parsed": parsed}
            consec = consec + 1 if (not x["ok"] and x.get("infra")) else 0 if x["ok"] else consec
        with f.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[{route}] {it['skill']} {it['i']} with={rec['with_skill']['pass']} without={rec['no_skill']['pass']} "
              f"ok={w['ok']}/{n['ok']} lat={w['latency_s']}/{n['latency_s']}s total=${spent_all():.4f}", flush=True)
        if consec >= 3:
            print(f"[{route}] 3 consecutive infra errors; stop route. last: {w.get('error') or n.get('error')}", flush=True)
            return


def summarize():
    import statistics as st
    lines = ["| Model / route | Pairs | Helpful: with / without | Placebo: with / without | Failed calls | Cut off at limit | "
             "Median latency (s) | Cost (USD) | Providers |", "|:--|--:|:--|:--|--:|--:|--:|--:|:--|"]

    def row(name, recs, lat=True):
        def cnt(s, a):
            r = [x for x in recs if x["skill"] == s]
            return f"{sum(x[a]['pass'] for x in r)}/{len(r)}"
        calls = [x[a] for x in recs for a in ("with_skill", "no_skill")]
        fail = sum(not c["ok"] for c in calls)
        trunc = sum(c.get("finish") == "length" for c in calls)
        lats = [c["latency_s"] for c in calls if c.get("ok") and "latency_s" in c]
        cost = sum(c.get("cost", 0.0) for c in calls)
        prov = sorted({str(c.get("provider")) for c in calls if c.get("provider")}) or ["-"]
        lines.append(f"| {name} | {len(recs)} | {cnt('helpful','with_skill')} / {cnt('helpful','no_skill')} | "
                     f"{cnt('placebo','with_skill')} / {cnt('placebo','no_skill')} | {fail}/{len(calls)} | {trunc}/{len(calls)} | "
                     f"{round(st.median(lats), 1) if lats else '-'} | {cost:.4f} | {', '.join(prov)} |")

    hk = items()
    for x in hk:
        for a in ("with_skill", "no_skill"):
            x[a]["cost"] = x[a]["in"] * 0.10e-6 + x[a]["out"] * 0.50e-6
    row("Claude Haiku 5.5 (study 2 calibration, Nov 16)", hk)
    for route in ROUTES:
        f = OUT / f"{route}.jsonl"
        if f.exists():
            row(f"Nemotron 3.5 Super VL via {route} ({ROUTES[route]['model']})",
                [json.loads(l) for l in f.read_text().splitlines()])
    text = "# Open-model smoke test (2026-11-17)\n\n" + "\n".join(lines) + "\n"
    (OUT / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for r in (sys.argv[1:] or list(ROUTES)):
        run(r)
    summarize()
