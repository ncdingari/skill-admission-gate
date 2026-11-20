"""Bridge to Google DeepMind's mathematics_dataset generator (Apache-2.0, v1.0.1, unmodified).

Runs inside the pinned Python 3.11 venv of requirements_dm_frozen.txt (sympy 1.4, numpy 1.26.4), with
PYTHONHASHSEED=0. Reads JSON {"items": [[module, regime, seed_hex], ...]} on stdin and writes a JSON list of
{"module", "regime", "seed", "question", "answer"} on stdout.

A variant is fully determined by (module, regime, seed): Python's `random` and NumPy's legacy global RNG are
seeded from SHA-256 of those three strings immediately before one call of the generator's own
`sample_from_module`. Only modules whose draws do not depend on earlier draws in the same process are allowed
(checked forward and reverse on 2026-11-20; 35 of 56 training modules pass). The only shim restores NumPy
aliases removed after the generator was written (np.object etc.), which are plain synonyms of builtins.
regime: "train_hard" = hardest third of the training entropy range (_make_entropy_fn(2, 3));
        "extrapolate" = the generator's own extrapolation test modules.
"""
import hashlib
import json
import random
import sys

import numpy as np

for _a, _t in (("object", object), ("int", int), ("bool", bool), ("float", float)):
    if not hasattr(np, _a):
        setattr(np, _a, _t)

from absl import flags  # noqa: E402
from mathematics_dataset import generate as g  # noqa: E402
from mathematics_dataset.modules import modules  # noqa: E402

flags.FLAGS(["dm_generate"])

ORDER_INDEPENDENT = {
    "algebra__linear_2d", "algebra__sequence_next_term", "algebra__sequence_nth_term", "arithmetic__add_or_sub",
    "arithmetic__add_or_sub_in_base", "arithmetic__add_sub_multiple", "arithmetic__div", "arithmetic__mixed",
    "arithmetic__mul", "arithmetic__mul_div_multiple", "arithmetic__nearest_integer_root",
    "arithmetic__simplify_surd", "comparison__closest", "comparison__kth_biggest", "comparison__pair",
    "comparison__sort", "measurement__conversion", "measurement__time", "numbers__base_conversion",
    "numbers__div_remainder", "numbers__gcd", "numbers__is_factor", "numbers__is_prime", "numbers__is_prime_composed",
    "numbers__lcm", "numbers__lcm_composed", "numbers__list_prime_factors", "numbers__list_prime_factors_composed",
    "numbers__place_value", "numbers__round_number", "polynomials__coefficient_named", "polynomials__collect",
    "polynomials__simplify_power", "probability__swr_p_level_set", "probability__swr_p_sequence"}


EXTRA_ORDER_DEPENDENT = {"algebra__polynomial_roots_big"}  # checked forward/reverse, 2026-11-20


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, prefix + k + "__"))
        else:
            out[prefix + k] = v
    return out


TRAIN = _flat(modules.train(g._make_entropy_fn(2, 3)))
EXTRA = _flat(modules.test_extra())


def draw(module: str, regime: str, seed_hex: str) -> dict:
    if regime == "train_hard":
        if module not in ORDER_INDEPENDENT:
            raise ValueError(f"module {module} is not order-independent")
        fn = TRAIN[module]
    elif regime == "extrapolate":
        if module in EXTRA_ORDER_DEPENDENT:
            raise ValueError(f"module {module} is not order-independent")
        fn = EXTRA[module]
    else:
        raise ValueError(regime)
    h = int.from_bytes(hashlib.sha256(f"{module}|{regime}|{seed_hex}".encode()).digest()[:8], "big")
    random.seed(h)
    np.random.seed(h % 2**32)
    p, _ = g.sample_from_module(fn)
    return {"module": module, "regime": regime, "seed": seed_hex, "question": str(p.question),
            "answer": str(p.answer)}


if __name__ == "__main__":
    req = json.load(sys.stdin)
    if req.get("list"):
        print(json.dumps({"train_hard": sorted(ORDER_INDEPENDENT), "extrapolate": sorted(set(EXTRA) - EXTRA_ORDER_DEPENDENT)}))
    else:
        print(json.dumps([draw(m, r, s) for m, r, s in req["items"]]))
