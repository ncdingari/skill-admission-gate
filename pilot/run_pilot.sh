#!/bin/bash
# Pilot step 3: run both arms concurrently on the committed task sample, each arm capped at $10.
# Needs an Anthropic key in ~/.config/p9_pilot/.env (line: ANTHROPIC_API_KEY=sk-ant-...). Amendment 1.
set -euo pipefail
cd "$(dirname "$0")"
ENVF="$HOME/.config/p9_pilot/.env"
[ -f "$ENVF" ] || { echo "missing $ENVF"; exit 1; }
set -a; source "$ENVF"; LLM_API_KEY="${LLM_API_KEY:-$ANTHROPIC_API_KEY}"; set +a
S=$($HOME/.venvs/p9_nixi/bin/python seed.py)
COMMON=(vendor/skillsbench/tasks --agent openhands --model anthropic/claude-haiku-5-5 --sandbox docker
        --limit 6 --sample-seed "$S" --trials 3 --max-cost-usd 10 --max-tokens 15000000 --jobs-dir jobs --quiet)
bench run "${COMMON[@]}" --task-skills --job-name with_skill  > logs_with.txt 2>&1 &
P1=$!
bench run "${COMMON[@]}"                --job-name no_skill    > logs_without.txt 2>&1 &
P2=$!
wait $P1 || echo "with-skill arm exited non-zero"
wait $P2 || echo "no-skill arm exited non-zero"
echo "PILOT RUNS DONE $(date +%H:%M:%S)"
