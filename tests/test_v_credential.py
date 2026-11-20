"""V7: canonical JSON, Merkle proofs, commit-reveal ordering and tamper detection."""
import copy
import json

import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from nixigate import credential as C


def test_jcs_rfc8785_numbers_and_order():
    assert C.jcs({"b": 1, "a": [1.0, 0.5, 1e-7, 1e21, -0.0, 123456789012]}) == \
        b'{"a":[1,0.5,1e-7,1e+21,0,123456789012],"b":1}'
    assert C.jcs({"x": 1e16}) == b'{"x":10000000000000000}'
    assert C.jcs({"a": 1, "b": 2}) == C.jcs({"b": 2, "a": 1})


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 13])
def test_merkle_inclusion(n):
    leaves = [bytes([i]) * 7 for i in range(n)]
    root = C.merkle_root(leaves)
    for i in range(n):
        assert C.verify_inclusion(leaves[i], i, n, C.inclusion_proof(leaves, i), root)
    if n > 1:
        assert not C.verify_inclusion(b"forged", 0, n, C.inclusion_proof(leaves, 0), root)


@pytest.fixture()
def world():
    ledger, store = C.Ledger(), C.Store()
    key = Ed25519PrivateKey.generate()
    agent = {"model": "provider/model@2026-11-01", "harness": "harness@1.2", "controller_pk": "pk1",
             "config_digest": "sha256:" + "0" * 64}
    skill = {"SKILL.md": b"# logs skill\nUse awk.\n"}
    theta = C.theta_block(n_max=200)
    rng = np.random.default_rng(11)
    p = lambda i, arm: 0.35 + 0.35 * arm
    chi = C.run_exam(ledger, store, key, agent, skill, theta, p, rng)
    return ledger, store, key, skill, chi


def test_valid_credential_verifies(world):
    ledger, store, key, skill, chi = world
    ok, why = C.verify(chi, ledger, store, C.proctor_pk(key), skill)
    assert ok, why


def _tamper_body(store, chi, fn):
    B = json.loads(store.blobs[chi])
    fn(B)
    nb = C.jcs(B)
    return store.put(nb), B


def test_tamper_transcript_byte(world):
    ledger, store, key, skill, chi = world
    B = json.loads(store.blobs[chi])
    k = B["arms"]["with"]["leaves"][0]
    store.blobs[k] = store.blobs[k].replace(b'"score":', b'"score" :')
    ok, why = C.verify(chi, ledger, store, C.proctor_pk(key), skill)
    assert not ok and "merkle" in why


def test_tamper_skill(world):
    ledger, store, key, skill, chi = world
    ok, why = C.verify(chi, ledger, store, C.proctor_pk(key), {"SKILL.md": b"# other"})
    assert not ok and "skill" in why


def test_tamper_decision_breaks_hash_and_signature(world):
    ledger, store, key, skill, chi = world
    new_chi, _ = _tamper_body(store, chi, lambda B: B["statistics"].update(decision="admit", n=5))
    assert new_chi != chi
    ok, why = C.verify(new_chi, ledger, store, C.proctor_pk(key), skill)
    assert not ok and "anchored" in why


def test_wrong_key(world):
    ledger, store, key, skill, chi = world
    ok, why = C.verify(chi, ledger, store, C.proctor_pk(Ed25519PrivateKey.generate()), skill)
    assert not ok and "signature" in why


def test_dishonest_proctor_hides_early_stop():
    """A proctor that keeps running past an admit/futility crossing and re-signs is caught."""
    ledger, store = C.Ledger(), C.Store()
    key = Ed25519PrivateKey.generate()
    agent = {"model": "m@1", "harness": "h@1"}
    skill = {"SKILL.md": b"s"}
    theta = C.theta_block(n_max=150)
    chi = C.run_exam(ledger, store, key, agent, skill, theta, lambda i, a: 0.3 + 0.5 * a,
                     np.random.default_rng(3))
    B = json.loads(store.blobs[chi])
    # claim the exam ran one variant fewer than it did (drop the last transcript pair)
    for arm in ("with", "without"):
        B["arms"][arm]["leaves"] = B["arms"][arm]["leaves"][:-1]
        lv = [store.blobs[k] for k in B["arms"][arm]["leaves"]]
        B["arms"][arm]["merkle_root"] = C.hx(C.merkle_root(lv))
        B["arms"][arm]["n_leaves"] = len(lv)
    B["statistics"]["n"] -= 1
    nb = C.jcs(B)
    chi2 = store.put(nb)
    ledger.append("credential", {"chi": chi2, "commitment": B["exam"]["commitment"],
                                 "signature": key.sign(chi2.encode()).hex(), "status": "active"})
    ok, why = C.verify(chi2, ledger, store, C.proctor_pk(key), skill)
    assert not ok and ("stopping" in why or "decision" in why or "estimate" in why)


def test_bad_reveal(world):
    ledger, store, key, skill, chi = world
    ev = ledger.find("reveal")
    B = json.loads(store.blobs[chi])
    B2 = copy.deepcopy(B)
    B2["exam"]["reveal"]["salt"] = "00" * 16
    chi2 = store.put(C.jcs(B2))
    ledger.append("credential", {"chi": chi2, "commitment": B["exam"]["commitment"],
                                 "signature": key.sign(chi2.encode()).hex(), "status": "active"})
    ok, why = C.verify(chi2, ledger, store, C.proctor_pk(key), skill)
    assert not ok and "commitment" in why
    assert ev is not None


def test_beacon_unavailable_before_height():
    L = C.Ledger()
    with pytest.raises(RuntimeError):
        L.beacon(5)
