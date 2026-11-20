"""Pilot step 2 (proctor, private): derive the exam seed from sigma_p and the drand beacon round
fixed in the commitment, and print the bench sample seed. Fails if the beacon round does not exist yet."""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DRAND = "https://api.drand.sh/52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971"

# the latest commitment wins: an amendment supersedes the original commit
com = [json.loads(l) for l in (ROOT / "ledger.jsonl").read_text().splitlines() if json.loads(l)["kind"] in ("commit", "amend")][-1]
SP = "sigma_p_a1.hex" if com["kind"] == "amend" else "sigma_p.hex"
rnd = com["payload"]["theta"]["beacon_round"]
try:
    with urllib.request.urlopen(f"{DRAND}/public/{rnd}", timeout=20) as r:
        beacon = json.load(r)
except Exception as e:  # round not produced yet
    sys.exit(f"beacon round {rnd} not available yet: {e}")
sigma_p = bytes.fromhex((ROOT / "secret" / SP).read_text())
sigma = hashlib.sha256(sigma_p + bytes.fromhex(beacon["randomness"])).digest()
S = int.from_bytes(sigma[:4], "big")
(ROOT / "secret" / "exam_seed.json").write_text(json.dumps({"round": rnd, "beacon_randomness": beacon["randomness"],
                                                            "sigma": sigma.hex(), "sample_seed": S}))
print(S)
