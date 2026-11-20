"""Revision B: grid bet available as a primary gate in suite.run_all (G7_grid, G7_grid_futm)."""
import numpy as np

from nixigate import dgp, gates, suite


def _design():
    return suite.Design(alpha=0.05, alpha_f=0.10, n_max=600, lift1=0.10, power=0.8, n_min=10, every=10, z=1.96,
                        beta=0.2, tau=0.1, cap=0.5, cap_alt=0.75, delta_f=0.05)


def test_run_all_has_grid_gates_matching_direct_call():
    d = _design()
    m = dgp.Model.for_lift(0.10, 0.4, 0.15, 0.05, 0.5, 1)
    D = dgp.simulate(np.random.default_rng(3), m, 200, 600)
    res = suite.run_all(D, d, {"n_fixed": 150, "sd1": 0.5, "p1": np.nan}, 1)
    assert {"G7_grid", "G7_grid_futm"} <= set(res) and {"G7_grid", "G7_grid_futm"} <= set(suite.GATES)
    ref = gates.betting(D[:, :600], 0.0, 0.05, 0.10, rule="grid", delta_f=0.05)
    assert np.array_equal(res["G7_grid_futm"].admit, ref.admit) and np.array_equal(res["G7_grid_futm"].n, ref.n)


def test_grid_gate_valid_at_null():
    d = _design()
    m = dgp.Model.for_lift(0.0, 0.4, 0.15, 0.05, 0.5, 1)
    D = dgp.simulate(np.random.default_rng(4), m, 4000, 600)
    res = suite.run_all(D, d, {"n_fixed": 150, "sd1": 0.5, "p1": np.nan}, 1)
    assert res["G7_grid_futm"].admit.mean() <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / 4000)
