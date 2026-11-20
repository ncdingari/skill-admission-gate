import numpy as np
from nixigate import leakage as L


def test_detectors_by_kind():
    rng = np.random.default_rng(0)
    keys = np.arange(50)
    sample = np.arange(200)
    expect = {"plain": (1, 1, 1), "spaced": (0, 1, 1), "split": (0, 1, 1),
              "base64": (0, 0, 1), "hex": (0, 0, 1), "computed": (0, 0, 0)}
    for kind, flags in expect.items():
        s = L.leaky_skill(rng, keys, kind)
        for det, f in zip(("literal", "normalized", "decoded"), flags):
            cov = L.coverage(s, sample, det, 0.5)
            assert (cov > 0.2) == bool(f), (kind, det, cov)
