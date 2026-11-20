"""Dual-arm credential: canonical JSON, Merkle roots, commit-reveal on an append-only log, verification.

This is a reference implementation of the credential of Section 4.4 and Algorithm 1 of the paper, with the ledger
abstracted to any append-only log that exposes block heights and a public beacon (a NEAR contract,
a Certificate-Transparency-style log plus drand, ...). Here the log is simulated in memory.

Hashes are SHA-256, written "sha256:<hex>". Canonical JSON follows RFC 8785 (sorted keys, no
whitespace, ECMAScript number formatting) for the value types used here. Merkle trees follow
RFC 6962 (leaf prefix 0x00, node prefix 0x01, split at the largest power of two below n).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from decimal import Decimal

import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives import serialization

from . import gates


# ------------------------------------------------------------------ hashing and canonical JSON
def h(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def hx(b: bytes) -> str:
    return "sha256:" + b.hex()


def _num(x) -> str:
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x)
    if not math.isfinite(x):
        raise ValueError("non-finite number in canonical JSON")
    if x == 0:
        return "0"
    sign, digits, exp = Decimal(repr(float(x))).normalize().as_tuple()
    ds = "".join(map(str, digits))
    k = len(ds)
    n = exp + k                              # value = 0.ds * 10^n
    s = "-" if sign else ""
    if k <= n <= 21:
        return s + ds + "0" * (n - k)
    if 0 < n <= 21:
        return s + ds[:n] + "." + ds[n:]
    if -6 < n <= 0:
        return s + "0." + "0" * (-n) + ds
    e = n - 1
    mant = ds[0] + ("." + ds[1:] if k > 1 else "")
    return s + mant + "e" + ("+" if e > 0 else "-") + str(abs(e))


def jcs(obj) -> bytes:
    def enc(o):
        if o is None:
            return "null"
        if isinstance(o, (bool, int, float, np.integer, np.floating)):
            return _num(o.item() if hasattr(o, "item") else o)
        if isinstance(o, str):
            return json.dumps(o, ensure_ascii=False)
        if isinstance(o, (list, tuple)):
            return "[" + ",".join(enc(v) for v in o) + "]"
        if isinstance(o, dict):
            items = sorted(o.items(), key=lambda kv: kv[0].encode("utf-16-be"))
            return "{" + ",".join(json.dumps(k, ensure_ascii=False) + ":" + enc(v) for k, v in items) + "}"
        raise TypeError(type(o))
    return enc(obj).encode("utf-8")


# ------------------------------------------------------------------ RFC 6962 Merkle tree
def _split(n: int) -> int:
    k = 1
    while k * 2 < n:
        k *= 2
    return k


def merkle_root(leaves: list[bytes]) -> bytes:
    if not leaves:
        return h(b"")
    if len(leaves) == 1:
        return h(b"\x00" + leaves[0])
    k = _split(len(leaves))
    return h(b"\x01" + merkle_root(leaves[:k]) + merkle_root(leaves[k:]))


def inclusion_proof(leaves: list[bytes], i: int) -> list[bytes]:
    if len(leaves) <= 1:
        return []
    k = _split(len(leaves))
    if i < k:
        return inclusion_proof(leaves[:k], i) + [merkle_root(leaves[k:])]
    return inclusion_proof(leaves[k:], i - k) + [merkle_root(leaves[:k])]


def verify_inclusion(leaf: bytes, i: int, n: int, proof: list[bytes], root: bytes) -> bool:
    def rec(i, n, proof):
        if n == 1:
            return h(b"\x00" + leaf), proof
        k = _split(n)
        if i < k:
            left, rest = rec(i, k, proof[:-1])
            return h(b"\x01" + left + proof[-1]), rest
        right, rest = rec(i - k, n - k, proof[:-1])
        return h(b"\x01" + proof[-1] + right), rest
    if len(proof) != _depth(n, i):
        return False
    got, _ = rec(i, n, proof)
    return got == root


def _depth(n: int, i: int) -> int:
    if n <= 1:
        return 0
    k = _split(n)
    return 1 + (_depth(k, i) if i < k else _depth(n - k, i - k))


# ------------------------------------------------------------------ append-only log with beacon
@dataclass
class Ledger:
    """In-memory append-only log. Heights increase by one per block; the beacon of a height is
    unknown until that height exists (simulating NEAR block randomness or a drand round)."""
    chain_secret: bytes = b"simulated-chain"
    height: int = 0
    events: list = field(default_factory=list)

    def mine(self, k: int = 1) -> int:
        self.height += k
        return self.height

    def append(self, kind: str, payload: dict) -> int:
        self.mine()
        self.events.append({"height": self.height, "kind": kind, "payload": payload})
        return self.height

    def beacon(self, height: int) -> bytes:
        if height > self.height:
            raise RuntimeError("beacon not yet available")
        return h(self.chain_secret + height.to_bytes(8, "big"))

    def find(self, kind: str, **match) -> dict | None:
        for e in self.events:
            if e["kind"] == kind and all(e["payload"].get(k) == v for k, v in match.items()):
                return e
        return None


# ------------------------------------------------------------------ a parametric task family
def variant(seed: bytes, i: int) -> dict:
    """Generator G_exam(seed, i): a log-forensics instance with a hidden answer.
    The instance is derived from H(seed || i), so it is unpredictable without the seed."""
    d = h(seed + i.to_bytes(8, "big"))
    ip = f"10.{d[0]}.{d[1]}.{d[2]}"
    code = 400 + d[3] % 200
    count = 3 + d[4] % 97
    return {"index": i, "instruction": "Find the IP with the most failed requests, its status code and count.",
            "logs_digest": hx(d), "answer": {"ip": ip, "code": code, "count": count}}


def verifier(final_state: dict, inst: dict) -> float:
    """Deterministic verifier V: 1.0 if the submitted answer equals the hidden answer."""
    return 1.0 if final_state.get("answer") == inst["answer"] else 0.0


# ------------------------------------------------------------------ exam and credential
@dataclass
class Store:
    """Content-addressed store for transcripts, skill files and credential bodies."""
    blobs: dict = field(default_factory=dict)

    def put(self, b: bytes) -> str:
        k = hx(h(b))
        self.blobs[k] = b
        return k

    def get(self, k: str) -> bytes:
        return self.blobs[k]


def theta_block(alpha=0.05, alpha_f=0.10, delta=0.0, r=1, n_max=600, cap=0.5) -> dict:
    return {"alpha": alpha, "alpha_futility": alpha_f, "delta": delta, "attempts_per_arm": r,
            "n_max": n_max, "bet_rule": "prpl", "bet_cap": cap, "primary_metric": "verifier_pass",
            "generator": "logs@1"}


def skill_digest(files: dict[str, bytes]) -> str:
    leaves = [jcs({"path": p, "sha256": hashlib.sha256(files[p]).hexdigest()}) for p in sorted(files)]
    return hx(merkle_root(leaves))


def run_exam(ledger: Ledger, store: Store, key: Ed25519PrivateKey, agent: dict, skill_files: dict,
             theta: dict, p_arm, rng: np.random.Generator, transcript_bytes: int = 0,
             sigma_p: bytes | None = None, salt: bytes | None = None) -> str:
    """Proctor side . `p_arm(i, arm) -> pass probability` stands in for the
    agent. Returns the credential digest chi. Transcripts are padded to `transcript_bytes`."""
    s_hash = skill_digest(skill_files)
    agent_id = hx(h(jcs(agent)))
    b0 = ledger.append("register", {"agent_id": agent_id, "skill_hash": s_hash})
    sigma_p = sigma_p or rng.bytes(32)
    salt = salt or rng.bytes(16)
    th_hash = hx(h(jcs(theta)))
    c = hx(h(sigma_p + salt + jcs(theta)))
    b1 = ledger.append("commit", {"commitment": c, "params_hash": th_hash, "reg_height": b0,
                                  "beacon_height": ledger.height + 2})
    b2 = ledger.height + 1
    ledger.mine(2)
    seed = h(sigma_p + ledger.beacon(b2))

    m0 = (1 + theta["delta"]) / 2
    alpha, alpha_f, cap = theta["alpha"], theta["alpha_futility"], theta["bet_cap"]
    leaves = {0: [], 1: []}
    X = []
    decision = "reject_max_n"
    pad = "x" * max(0, transcript_bytes - 400)
    for i in range(1, theta["n_max"] + 1):
        inst = variant(seed, i)
        order = [0, 1] if rng.random() < 0.5 else [1, 0]
        ys = {0: [], 1: []}
        u = rng.random(theta["attempts_per_arm"])
        for arm in order:
            for ell in range(theta["attempts_per_arm"]):
                ok = u[ell] < p_arm(i, arm)
                final = {"answer": inst["answer"] if ok else {"ip": "0.0.0.0", "code": 0, "count": 0}}
                y = verifier(final, inst)
                tr = {"variant": i, "variant_digest": hx(h(jcs(inst))), "arm": arm, "attempt": ell,
                      "model": agent["model"], "harness": agent["harness"], "messages": pad,
                      "final_state": final, "score": y}
                leaves[arm].append(jcs(tr))
                store.put(jcs(tr))
                ys[arm].append(y)
        D = float(np.mean(ys[1]) - np.mean(ys[0]))
        X.append((1 + D) / 2)
        Xa = np.array(X)[None, :]
        logK = float(np.sum(np.log1p(gates.prpl_bets(Xa, alpha, cap)[0] * (Xa[0] - m0))))
        logKf = float(np.sum(np.log1p(-gates.prpl_bets(Xa, alpha_f, cap)[0] * (Xa[0] - m0))))
        if logK >= np.log(1 / alpha):
            decision = "admit"
            break
        if logKf >= np.log(1 / alpha_f):
            decision = "reject_futility"
            break
    N = len(X)
    Dv = 2 * np.array(X) - 1
    body = {
        "version": "nixi-cred/0.2", "status": "active",
        "agent": {**agent, "agent_id": agent_id}, "skill": {"skill_hash": s_hash},
        "comparator": {"kind": "no_skill"},
        "exam": {"commitment": c, "params": theta, "params_hash": th_hash, "reg_height": b0,
                 "commit_height": b1, "beacon_height": b2,
                 "reveal": {"sigma_p": sigma_p.hex(), "salt": salt.hex()}},
        "arms": {"without": {"merkle_root": hx(merkle_root(leaves[0])), "n_leaves": len(leaves[0]),
                             "leaves": [hx(h(x)) for x in leaves[0]]},
                 "with": {"merkle_root": hx(merkle_root(leaves[1])), "n_leaves": len(leaves[1]),
                          "leaves": [hx(h(x)) for x in leaves[1]]}},
        "statistics": {"n": N, "delta_hat": float(Dv.mean()), "decision": decision,
                       "lcb": float(gates.lcb_path(Dv, alpha, cap)[-1]), "test": "betting_prpl"},
    }
    chi = hx(h(jcs(body)))
    sig = key.sign(chi.encode()).hex()
    store.put(jcs(body))
    b3 = ledger.append("reveal", {"commitment": c, "sigma_p": sigma_p.hex(), "salt": salt.hex()})
    ledger.append("credential", {"chi": chi, "commitment": c, "signature": sig, "status": "active",
                                 "reveal_height": b3})
    return chi


def proctor_pk(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def verify(chi: str, ledger: Ledger, store: Store, pk: bytes, skill_files: dict) -> tuple[bool, str]:
    """Algorithm 1 (without TEE). Returns (valid, reason)."""
    anc = ledger.find("credential", chi=chi)
    if anc is None or anc["payload"]["status"] == "revoked":
        return False, "not anchored"
    body_b = store.blobs.get(chi)
    if body_b is None:
        body_b = next((b for b in store.blobs.values() if hx(h(b)) == chi), None)
    if body_b is None or hx(h(body_b)) != chi:
        return False, "body hash mismatch"
    B = json.loads(body_b)
    try:
        Ed25519PublicKey.from_public_bytes(pk).verify(bytes.fromhex(anc["payload"]["signature"]), chi.encode())
    except Exception:
        return False, "bad signature"
    ex = B["exam"]
    reg = ledger.find("register", agent_id=B["agent"]["agent_id"], skill_hash=B["skill"]["skill_hash"])
    com = ledger.find("commit", commitment=ex["commitment"])
    rev = ledger.find("reveal", commitment=ex["commitment"])
    if not (reg and com and rev):
        return False, "missing ledger events"
    b0, b1, b2, b3 = reg["height"], com["height"], ex["beacon_height"], rev["height"]
    if not (b0 < b1 < b2 < b3) or com["payload"]["beacon_height"] != b2:
        return False, "bad block order"
    sp, salt = bytes.fromhex(ex["reveal"]["sigma_p"]), bytes.fromhex(ex["reveal"]["salt"])
    if hx(h(sp + salt + jcs(ex["params"]))) != ex["commitment"]:
        return False, "reveal does not open commitment"
    if hx(h(jcs(ex["params"]))) != ex["params_hash"]:
        return False, "params hash mismatch"
    if skill_digest(skill_files) != B["skill"]["skill_hash"]:
        return False, "skill hash mismatch"
    seed = h(sp + ledger.beacon(b2))
    th = ex["params"]
    trs = {}
    for arm, name in ((0, "without"), (1, "with")):
        lv = B["arms"][name]["leaves"]
        blobs = [store.blobs.get(k) for k in lv]
        if any(b is None for b in blobs) or hx(merkle_root(blobs)) != B["arms"][name]["merkle_root"]:
            return False, f"merkle root mismatch ({name})"
        trs[arm] = [json.loads(b) for b in blobs]
    r = th["attempts_per_arm"]
    N = B["statistics"]["n"]
    if len(trs[0]) != N * r or len(trs[1]) != N * r:
        return False, "transcript count mismatch"
    D = []
    for i in range(1, N + 1):
        inst = variant(seed, i)
        vd = hx(h(jcs(inst)))
        ys = {}
        for arm in (0, 1):
            rows = trs[arm][(i - 1) * r: i * r]
            for ell, t in enumerate(rows):
                if t["variant"] != i or t["attempt"] != ell or t["variant_digest"] != vd:
                    return False, f"variant mismatch at {i}"
                if t["model"] != B["agent"]["model"] or t["harness"] != B["agent"]["harness"]:
                    return False, "model or harness mismatch"
                if verifier(t["final_state"], inst) != t["score"]:
                    return False, f"verifier score mismatch at {i}"
            ys[arm] = np.mean([t["score"] for t in rows])
        D.append(ys[1] - ys[0])
    D = np.array(D)
    res = gates.betting(D[None, :], th["delta"], th["alpha"], th["alpha_futility"], th["bet_cap"])
    dec = "admit" if res.admit[0] else ("reject_futility" if res.futility[0] else "reject_max_n")
    if res.n[0] != N or (dec == "reject_max_n" and N != th["n_max"]):
        return False, "stopping time inconsistent with theta"
    if dec != B["statistics"]["decision"]:
        return False, "decision mismatch"
    if abs(float(D.mean()) - B["statistics"]["delta_hat"]) > 1e-12:
        return False, "estimate mismatch"
    return True, "valid"
