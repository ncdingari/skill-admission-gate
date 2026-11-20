"""E7: time to run the proctor bookkeeping and to verify a credential, versus N (RQ7)."""
import time

import numpy as np
import pandas as pd
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from _common import P, QUICK, Run, rng_for, v
from nixigate import credential as C

run = Run("E7")
kb = int(v(P, "transcript_kb"))
rows = []
for N in ([50, 100] if QUICK else [50, 100, 200, 400, 600]):
    for rep in range(1 if QUICK else 3):
        ledger, store, key = C.Ledger(), C.Store(), Ed25519PrivateKey.generate()
        th = C.theta_block(alpha=1e-12, alpha_f=1e-12, n_max=N)   # never stops early: exactly N variants
        skill = {"SKILL.md": b"# skill\n" * 50}
        t0 = time.perf_counter()
        chi = C.run_exam(ledger, store, key, {"model": "m@1", "harness": "h@1"}, skill, th,
                         lambda i, a: 0.35 + 0.1 * a, rng_for(f"E7{N}{rep}"), transcript_bytes=kb * 1000)
        t1 = time.perf_counter()
        ok, why = C.verify(chi, ledger, store, C.proctor_pk(key), skill)
        t2 = time.perf_counter()
        assert ok, why
        onchain = sum(len(C.jcs(e["payload"])) for e in ledger.events)
        rows.append({"n_variants": N, "rep": rep, "transcripts": 2 * N, "transcript_mb": 2 * N * kb / 1000,
                     "proctor_s": t1 - t0, "verify_s": t2 - t1, "ledger_bytes": onchain})
df = pd.DataFrame(rows).groupby("n_variants").median(numeric_only=True).reset_index()
run.save(df, "e7_verification")
run.done()
