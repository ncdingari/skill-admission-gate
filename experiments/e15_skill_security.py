"""E15: static security screen of every SkillsBench skill (NVIDIA SkillSpector, --no-llm), joined with the
released per-task lift. Descriptive only; prior art in planning/Prior_Art_E6_SkillSecurity_v1.md
(SkillTester, SkillSafetyBench and others measure utility and security together already).

For each of the 87 tasks at the registered SkillsBench commit, scan environment/skills/ recursively, offline,
with no LLM. Record per skill: risk score (0-100), severity, recommendation, coverage, issue rules and
categories. Per task: maximum score over its skills. Join with the released lift (mean D over model
configurations) and report how risk and lift relate.

Raw scanner JSON is kept in results/E15/raw/ so every number can be rechecked.
Requires the skillspector CLI on PATH (version recorded)."""
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from _common import Run
from _realdata import skillsbench

run = Run("E15")
VENDOR = Path(__file__).resolve().parent.parent / "pilot" / "vendor" / "skillsbench"
# Scan a clean export of the committed tree, not the working copy: running task scripts locally leaves
# git-ignored __pycache__/*.pyc files inside skills (found 2026-11-16 in court-form-filling/pdf), which the
# scanner rates as uninspectable executables. The export contains exactly the files at the registered commit.
import tempfile
_CLEAN = Path(tempfile.mkdtemp(prefix="e15_clean_"))
COMMIT = subprocess.run(["git", "-C", str(VENDOR), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
_arch = subprocess.run(["git", "-C", str(VENDOR), "archive", COMMIT, "tasks"], capture_output=True, check=True).stdout
subprocess.run(["tar", "-x", "-C", str(_CLEAN)], input=_arch, check=True)
TASKS = _CLEAN / "tasks"
RAW = run.dir / "raw"
RAW.mkdir(exist_ok=True)
version = subprocess.run(["skillspector", "--version"], capture_output=True, text=True).stdout.strip()

skill_rows, task_rows = [], []
for t in sorted(p.name for p in TASKS.iterdir() if (p / "task.md").exists()):
    sk = TASKS / t / "environment" / "skills"
    if not sk.is_dir():
        task_rows.append({"task": t, "n_skills": 0, "max_risk": np.nan, "scan_ok": False})
        continue
    out = RAW / f"{t}.json"
    if not out.exists():
        r = subprocess.run(["skillspector", "scan", str(sk), "--recursive", "--no-llm", "--format", "json",
                            "--output", str(out)], capture_output=True, text=True, timeout=900)
        if r.returncode == 2 or not out.exists():
            task_rows.append({"task": t, "n_skills": np.nan, "max_risk": np.nan, "scan_ok": False,
                              "err": (r.stderr or r.stdout)[-300:]})
            continue
    d = json.loads(out.read_text())
    skills = d.get("skills") or [d]          # single-skill reports are not wrapped
    for s in skills:
        ra = s.get("risk_assessment", {})
        iss = s.get("issues", [])
        cov = [i for i in iss if "coverage" in (i.get("tags") or [])]       # scanner could not inspect something
        sub = [i for i in iss if "coverage" not in (i.get("tags") or [])]   # substantive security findings
        skill_rows.append({
            "task": t, "skill": s.get("name"), "risk": ra.get("score", s.get("risk_score")),
            "severity": ra.get("severity", s.get("risk_severity")), "recommendation": ra.get("recommendation"),
            "n_issues": len(iss), "has_scripts": s.get("metadata", {}).get("has_executable_scripts"),
            "n_coverage_issues": len(cov), "n_substantive": len(sub),
            "n_substantive_high_crit": sum(1 for i in sub if i.get("severity") in ("HIGH", "CRITICAL")),
            "substantive_rules": ";".join(sorted({str(i.get("id")) for i in sub})),
            "substantive_categories": ";".join(sorted({str(i.get("category")) for i in sub})),
            "coverage_rules": ";".join(sorted({str(i.get("id")) for i in cov}))})
    comp = d.get("analysis_completeness", {})
    task_rows.append({"task": t, "n_skills": d.get("skill_count", len(skills)), "max_risk": d.get("max_risk_score"),
                      "task_severity": d.get("risk_severity"), "task_recommendation": d.get("risk_recommendation"),
                      "coverage_percent": comp.get("coverage_percent"), "scan_ok": bool(d.get("execution_successful"))})

sk = pd.DataFrame(skill_rows)
tk = pd.DataFrame(task_rows)
lift = skillsbench().groupby("task").agg(D_mean=("D", "mean"), n_configs=("D", "size")).reset_index()
tk = tk.merge(lift, on="task", how="left")
run.save(sk, "e15_skills")
run.save(tk, "e15_tasks")

ok = tk.dropna(subset=["max_risk", "D_mean"])
rho, p_rho = stats.spearmanr(ok.max_risk, ok.D_mean) if len(ok) > 2 else (np.nan, np.nan)
flag = ok.max_risk > 50                     # HIGH or CRITICAL: scanner says do not install
summary = {
    "skillspector_version": version, "n_tasks": int(len(tk)), "n_tasks_scanned": int(tk.scan_ok.sum()),
    "n_skills": int(len(sk)),
    "skills_by_recommendation": sk.recommendation.value_counts().to_dict(),
    "skills_by_severity": sk.severity.value_counts().to_dict(),
    "tasks_flagged_do_not_install": int(flag.sum()),
    "tasks_flagged_and_helpful": int((flag & (ok.D_mean > 1e-9)).sum()),
    "tasks_helpful": int((ok.D_mean > 1e-9).sum()),
    "skills_with_substantive_high_crit": int((sk.n_substantive_high_crit > 0).sum()),
    "skills_with_only_coverage_findings": int(((sk.n_substantive == 0) & (sk.n_coverage_issues > 0)).sum()),
    "substantive_categories": sk.substantive_categories.str.split(";").explode().replace("", np.nan).dropna().value_counts().to_dict(),
    "spearman_risk_vs_lift": float(rho), "spearman_p": float(p_rho), "n_joined": int(len(ok)),
    "median_risk_helpful": float(ok[ok.D_mean > 1e-9].max_risk.median()),
    "median_risk_not_helpful": float(ok[ok.D_mean <= 1e-9].max_risk.median()),
}
run.save(pd.DataFrame([summary]).astype(str), "e15_summary")
run.done(skillsbench_commit=COMMIT, **{k: v for k, v in summary.items() if not isinstance(v, dict)})
print(json.dumps(summary, indent=1))
