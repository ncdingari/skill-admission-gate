#!/bin/bash
# Full runs. Usage: bash experiments/run_all.sh   (NIXI_QUICK=1 for smoke runs)
set -e
cd "$(dirname "$0")"
PY=${PY:-$HOME/.venvs/p9_nixi/bin/python}
for e in e1_false_admission e2_power_cost e3_library_fdr e4_estimation e5_library_scenarios e6_leakage e7_verification e8_swe_skills_reanalysis e9_skillsbench_reanalysis; do
  echo "== $e $(date +%H:%M:%S)"; $PY $e.py > /dev/null
done
echo "ALL DONE $(date +%H:%M:%S)"
