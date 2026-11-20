"""Read-only monitor for the P9 BenchFlow study runs (Chari's monitoring brief, 2026-11-16).

  python pilot/monitor.py report            # per-job infra/result counts, spend, refusal causes; exit 3 if gate fails
  python pilot/monitor.py gate [JOB_GLOB]   # exit 0 if the infra-failure rate of the given jobs is <= 25%, else 3

A rollout is an INFRA failure (not a result) when its sandbox never started for the agent:
  CloudMetadataPowerError (BenchFlow lockdown refused the sandbox), other sandbox-startup errors, image
  pull/build failures, or a provider-side rejection before the agent produced a step. Infra failures are
  excluded from scoring by the committed rule; they are counted here so a run can be paused.
Refusal kind: 'delegate_power' (image ships setuid/sudo or file capabilities: deterministic, a retry fails
again) or 'audit_unavailable' (privilege audit did not finish: load-dependent, may pass on retry)."""
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
JOBS = ROOT / "jobs_study" / "openhands"
THRESH = None  # infra-failure pause removed by Chari, 2026-11-16 (was 0.25 -> 0.40 -> 0.75); infra failures are still counted and reported
CAP = 210.0


def classify(r: dict) -> tuple[str, str]:
    err = str(r.get("error") or "")
    cat = str(r.get("error_category") or "")
    if "CloudMetadataPowerError" in err or "sandbox startup refused" in err:
        kind = "delegate_power" if "may acquire root" in err or "firewall tool" in err or "own process" in err else "other_refusal"
        if "did not finish" in err and "may acquire root" not in err:
            kind = "audit_unavailable"
        return "infra", "CloudMetadataPowerError:" + kind
    if r.get("sandbox_startup_info") or "sandbox" in cat.lower() or "docker" in err.lower() or "pull" in err.lower():
        return "infra", "sandbox_or_image"
    if r.get("usage_limit_info") or "credit balance" in err or "usage limit" in err.lower():
        return "infra", "provider_rejected"
    if err or r.get("api_error_info") or r.get("transport_error_info"):
        return "error", (cat or err.split(":")[0])[:60]
    return "result", "ok"


def rollouts(job_glob: str):
    for res in glob.glob(str(JOBS / "trial-*" / job_glob / "*" / "result.json")):
        try:
            r = json.loads(Path(res).read_text())
        except (OSError, ValueError):
            continue
        p = Path(res)
        yield p.parent.parent.name, p.parent.parent.parent.name, r


def spend() -> float:
    tot = 0.0
    for f in glob.glob(str(JOBS / "trial-*" / "*" / "summary.json")):
        try:
            d = json.loads(Path(f).read_text())
        except (OSError, ValueError):
            continue
        tot += float(d.get("total_cost_usd") or d.get("cost_usd") or 0.0)
    return tot


def stats(job_glob: str):
    per_job = defaultdict(Counter)
    causes = Counter()
    task_kind = defaultdict(Counter)
    for job, trial, r in rollouts(job_glob):
        cls, why = classify(r)
        per_job[job][cls] += 1
        if cls != "result":
            causes[why] += 1
        task_kind[r.get("task_name")][cls] += 1
    return per_job, causes, task_kind


def rate(per_job) -> tuple[int, int]:
    n = sum(sum(c.values()) for c in per_job.values())
    inf = sum(c["infra"] for c in per_job.values())
    return inf, n


def report():
    per_job, causes, task_kind = stats("*")
    print(f"{'job':28s} {'results':>7s} {'errors':>6s} {'infra':>6s} {'infra%':>6s}")
    for job in sorted(per_job):
        c = per_job[job]
        n = sum(c.values())
        print(f"{job:28s} {c['result']:7d} {c['error']:6d} {c['infra']:6d} {100 * c['infra'] / max(n, 1):5.0f}%")
    inf, n = rate(per_job)
    print(f"\nall jobs: {inf}/{n} infra ({100 * inf / max(n, 1):.0f}%)   spend ${spend():.2f} of ${CAP:.0f} cap")
    print("causes:", dict(causes.most_common()))
    always = sorted(t for t, c in task_kind.items() if c["infra"] and not c["result"] and not c["error"])
    print(f"tasks whose sandbox never started in any trial ({len(always)}):", ", ".join(always))
    return per_job, causes, always


def gate(job_glob: str) -> int:
    per_job, _, _ = stats(job_glob)
    inf, n = rate(per_job)
    r = inf / max(n, 1)
    ok = THRESH is None or r <= THRESH
    lim = "no limit" if THRESH is None else f"limit {100 * THRESH:.0f}%"
    print(f"gate {job_glob}: infra {inf}/{n} = {100 * r:.0f}% ({lim}), BenchFlow spend ${spend():.2f} -> {'OK' if ok else 'PAUSE'}")
    return 0 if ok else 3


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "report":
        report()
    elif cmd == "gate":
        sys.exit(gate(sys.argv[2] if len(sys.argv) > 2 else "*"))


def known_refusers() -> list[str]:
    """Tasks whose sandbox was refused by the BenchFlow lockdown (CloudMetadataPowerError) and that never got
    past sandbox start in any attempt. The refusal depends on the task image (setuid/sudo, file capabilities,
    missing setpriv or /proc), so a retry fails again: skip these up front and log them as infra."""
    seen = defaultdict(Counter)
    for _, _, r in rollouts("*"):
        cls, why = classify(r)
        seen[r.get("task_name")]["refused" if why.startswith("CloudMetadataPowerError") else ("other_infra" if cls == "infra" else "started")] += 1
    return sorted(t for t, c in seen.items() if c["refused"] and not c["started"])


def write_status(note: str):
    """Append a short dated entry to pilot/monitor_status.md: spend, failures, what changed."""
    import time
    per_job, causes, _ = stats("*")
    inf, n = rate(per_job)
    lines = [f"\n## {time.strftime('%Y-%m-%d %H:%M %Z')}",
             f"- All API spend on this paper: ${total_spend()["total"]:.2f} (BenchFlow study ${spend():.2f}); ask rule: estimate above ${ASK_ABOVE_USD:.0f}.",
             f"- Rollouts so far: {n}; infra failures (not results): {inf} ({100 * inf / max(n, 1):.0f}%).",
             f"- Top causes: " + ", ".join(f"{k} {v}" for k, v in causes.most_common(5)) + ".",
             f"- Tasks skipped up front as known sandbox refusals: {len(known_refusers())}.",
             f"- Note: {note}"]
    with (ROOT / "monitor_status.md").open("a") as f:
        f.write("\n".join(lines) + "\n")


# ------------------------------------------------ cost rule (Chari, 2026-11-16): ask only if the estimate exceeds 300 USD
ASK_ABOVE_USD = 300.0
HAIKU_IN, HAIKU_OUT = 0.10e-6, 0.50e-6


def total_spend() -> dict:
    """All API spend on this paper: BenchFlow study runs, BenchFlow pilot/smoke runs, direct-API studies 2 and 3."""
    def bench(root):
        t = 0.0
        for f in glob.glob(str(root / "openhands" / "trial-*" / "*" / "summary.json")) + glob.glob(str(root / "*" / "summary.json")):
            if "/trial-" in f and f.count("/trial-") and Path(f).parent.parent.name.startswith("trial-") is False:
                continue
            try:
                t += float(json.loads(Path(f).read_text()).get("total_cost_usd") or 0.0)
            except (OSError, ValueError):
                pass
        return t
    direct = 0.0
    for d in (ROOT.parent / "pilot2" / "transcripts", ROOT.parent / "pilot2" / "transcripts3"):
        for f in d.glob("*.jsonl"):
            for l in f.read_text().splitlines():
                r = json.loads(l)
                for a in ("with_skill", "no_skill"):
                    if a in r:
                        direct += r[a].get("in", 0) * HAIKU_IN + r[a].get("out", 0) * HAIKU_OUT
    out = {"benchflow_study": spend(), "benchflow_pilot": bench(ROOT / "jobs") + bench(ROOT / "jobs_smoke"),
           "direct_api_studies": direct}
    out["total"] = sum(out.values())
    return out


def cost_per_run() -> float:
    """Mean reported cost per BenchFlow run that got past sandbox start (results and agent errors)."""
    n = 0
    for _, _, r in rollouts("*"):
        if classify(r)[0] != "infra":
            n += 1
    return spend() / max(n, 1)


def batch_estimate(n_tasks: int, trials: int = 3, arms: int = 2) -> float:
    """Upper-side estimate: every run starts and costs the mean so far."""
    return cost_per_run() * n_tasks * trials * arms


def needs_approval(n_tasks: int) -> tuple[bool, str]:
    tot = total_spend()["total"]
    est = batch_estimate(n_tasks)
    msg = f"spent ${tot:.2f} + batch estimate ${est:.2f} = ${tot + est:.2f} (ask above ${ASK_ABOVE_USD:.0f})"
    return tot + est > ASK_ABOVE_USD, msg
