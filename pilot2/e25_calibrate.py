"""E25 calibration, step 1 (not scored, public seeds): no-skill pass rate of claude-haiku-5-5 on every
order-independent module and regime of DeepMind's generator, 16 variants each. Used only to pick the
family by the rule fixed in E25_PLAN.md (modules whose no-skill pass rate lies in [0.30, 0.60]).
  python pilot2/e25_calibrate.py
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import family_dm as fam  # noqa: E402
from run4 import MODEL, PRICE_IN, PRICE_OUT, client  # noqa: E402

N = 16
CAP_USD = 5.0
OUT = ROOT / "transcripts5" / "calibration_noskill.jsonl"


def one(cl, v):
    user = fam.TASK.replace("{question}", v["question"])
    for attempt in range(3):
        try:
            r = cl.messages.create(model=MODEL, max_tokens=8000, system=fam.SYSTEM,
                                   messages=[{"role": "user", "content": user}])
            text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            ok, got = fam.grade(text, v["answer"])
            return {**v, "ok": True, "pass": ok, "got": got, "text": text, "stop": r.stop_reason,
                    "in": r.usage.input_tokens, "out": r.usage.output_tokens, "id": r.id}
        except Exception as ex:  # noqa: BLE001
            err = f"{type(ex).__name__}: {str(ex)[:200]}"
            time.sleep(5 * (attempt + 1))
    return {**v, "ok": False, "error": err, "in": 0, "out": 0}


def main():
    OUT.parent.mkdir(exist_ok=True)
    done = set()
    spent = 0.0
    if OUT.exists():
        for l in OUT.read_text().splitlines():
            d = json.loads(l)
            done.add((d["module"], d["regime"], d["seed"]))
            spent += d["in"] * PRICE_IN + d["out"] * PRICE_OUT
    mods = fam.modules()
    items = [(m, r, fam.seed_for("calibration-public", i)) for r in ("train_hard", "extrapolate")
             for m in mods[r] for i in range(N)]
    items = [it for it in items if it not in done]
    variants = fam.generate(items)
    cl = client()
    with ThreadPoolExecutor(8) as ex, OUT.open("a") as fh:
        for k in range(0, len(variants), 64):
            if spent >= CAP_USD:
                print("cost cap reached", spent); break
            for rec in ex.map(lambda v: one(cl, v), variants[k:k + 64]):
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                spent += rec["in"] * PRICE_IN + rec["out"] * PRICE_OUT
            print(f"{k + 64}/{len(variants)} ${spent:.3f}", flush=True)


if __name__ == "__main__":
    main()
