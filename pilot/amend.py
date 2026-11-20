"""Pilot step 1b: amend the commitment before any run or seed derivation.

Amendment 1 (2026-11-14): model changed from openai/gpt-5.4-mini to anthropic/claude-haiku-5-5
because the OpenAI key was rejected. No run was made and seed.py was never run under the first
commitment. Its beacon round passed while the proctor held sigma_p, so the amendment draws fresh
secrets and a new future beacon round instead of reusing the old ones. The old secrets are kept
and revealed with the new ones. Usage: python pilot/amend.py
"""
import json
import os
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from commit import H, LEDGER, SECRET, append, jcs, latest_round  # noqa: E402

entries = [json.loads(l) for l in LEDGER.read_text().splitlines()]
if any(e["kind"] == "amend" for e in entries):
    sys.exit("Amendment 1 is already logged.")
old = next(e for e in entries if e["kind"] == "commit")
if (SECRET / "exam_seed.json").exists():
    sys.exit("A seed was already derived; an amendment is no longer clean.")

theta = dict(old["payload"]["theta"])
theta["model"] = "anthropic/claude-haiku-5-5"
theta["beacon_round"] = latest_round() + 200  # about 10 minutes from now
sigma_p, rho = secrets.token_bytes(32), secrets.token_bytes(16)
for name, val in (("sigma_p_a1.hex", sigma_p), ("rho_a1.hex", rho)):
    (SECRET / name).write_text(val.hex())
    os.chmod(SECRET / name, 0o600)
c = H(sigma_p + rho + jcs(theta))
e = append("amend", {
    "amendment": 1, "supersedes": old["entry_hash"], "commitment": c, "theta": theta, "theta_hash": H(jcs(theta)),
    "changed": {"model": [old["payload"]["theta"]["model"], theta["model"]],
                "beacon_round": [old["payload"]["theta"]["beacon_round"], theta["beacon_round"]],
                "secrets": "fresh"},
    "reason": "OpenAI key rejected; no runs made and no seed derived under the first commitment",
})
print(json.dumps({"commitment": c, "beacon_round": theta["beacon_round"], "entry": e["entry_hash"]}, indent=1))
