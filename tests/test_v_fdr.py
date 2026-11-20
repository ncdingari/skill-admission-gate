"""V3: multiple-testing rules against hand-worked examples."""
import numpy as np
from nixigate import fdr


def test_bh_example():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205])
    # alpha 0.05, m 8: thresholds 0.00625, 0.0125, ..., k* = 2
    assert fdr.bh(p, 0.05).tolist() == [True, True] + [False] * 6


def test_ebh_example():
    e = np.array([100.0, 45.0, 30.0, 1.0, 0.5])
    # m/(alpha k) with m 5, alpha .1: 50, 25, 16.7, 12.5, 10 -> k* = 3
    assert fdr.ebh(e, 0.1).tolist() == [True, True, True, False, False]


def test_ebh_stopped_at_one_over_alpha_needs_all():
    # every candidate stopped exactly at 1/alpha: e-BH admits only if all m crossed
    assert fdr.ebh(np.array([20.0, 20.0, 1.0]), 0.05).sum() == 0
    assert fdr.ebh(np.array([20.0, 20.0, 20.0]), 0.05).sum() == 3


def test_elond_and_gamma():
    g = fdr.gamma_seq(10_000_000)
    assert 0.99 < g.sum() <= 1.0
    lv = fdr.elond_levels(0, 1, 0.05)
    assert np.isclose(lv, 0.05 / __import__("scipy.special").special.zeta(1.6))
    e = np.array([1 / lv + 1, 1.0, 1.0])
    out = fdr.elond(e, 0.05)
    assert out[0] and not out[1:].any()


def test_lineage_levels_sum_below_alpha():
    assert fdr.lineage_levels(1000, 0.05).sum() < 0.05
