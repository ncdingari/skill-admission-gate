"""The 'grid' bet in gates.betting: valid under the null, powered under a lift, same weights as E13."""
import numpy as np

from nixigate import gates


def test_grid_weights_sum_below_one():
    j = np.arange(10**5)
    assert gates.C_U * np.sum(1 / ((j + 2) * np.log(j + 2) ** 2)) <= 1.0


def test_grid_null_and_power():
    rng = np.random.default_rng(7)
    Z = rng.choice([-1.0, 0.0, 1.0], p=[.2, .6, .2], size=(3000, 400))
    assert gates.betting(Z, 0.0, 0.05, 0.10, 0.5, rule="grid", delta_f=0.05).admit.mean() <= 0.05
    D = rng.choice([-1.0, 0.0, 1.0], p=[.1, .6, .3], size=(300, 400))
    assert gates.betting(D, 0.0, 0.05, 0.10, 0.5, rule="grid", delta_f=0.05).admit.mean() > 0.95


def test_grid_accepts_per_row_levels():
    rng = np.random.default_rng(11)
    D = rng.choice([-1.0, 0.0, 1.0], p=[.1, .6, .3], size=(6, 300))
    lv = np.array([0.05, 0.01, 0.002, 0.05, 0.01, 0.002])
    joint = gates.betting(D, 0.0, lv, 0.10, 0.5, rule="grid", delta_f=0.05)
    for i in range(6):
        one = gates.betting(D[i:i + 1], 0.0, float(lv[i]), 0.10, 0.5, rule="grid", delta_f=0.05)
        assert joint.admit[i] == one.admit[0] and joint.n[i] == one.n[0]


def test_common_rule_suffix(monkeypatch):
    import importlib, sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "experiments"))
    monkeypatch.setenv("NIXI_RULE", "grid")
    import _common
    importlib.reload(_common)
    assert _common.RULE == "grid" and _common.out_name("e3_window_fdr") == "e3_window_fdr_grid"
    monkeypatch.setenv("NIXI_RULE", "prpl")
    importlib.reload(_common)
    assert _common.out_name("e3_window_fdr") == "e3_window_fdr"
