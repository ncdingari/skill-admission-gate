"""E19: stopped e-BH (Wang, Dandapanthula, Ramdas 2025) and e-BH on adjusted suprema (Tavyrikov et al. 2025)."""
import numpy as np

from nixigate import fdr, gates


def _paths(rng, R, m, n, p):
    D = rng.choice([-1.0, 0.0, 1.0], p=p, size=(R * m, n))
    X = (1 + D) / 2
    logK = np.cumsum(np.log1p(gates.prpl_bets(X, 0.05, 0.5) * (X - 0.5)), axis=1)
    return np.exp(logK).reshape(R, m, n)


def test_ebh_path_one_column_equals_ebh():
    e = np.array([[30.0], [2.0], [500.0], [0.5]])
    assert (fdr.ebh_path(e, 0.05)[:, 0] == fdr.ebh(e[:, 0], 0.05)).all()


def test_adjuster_sqrt():
    assert np.allclose(fdr.adjust_sqrt(np.array([1.0, 4.0, 100.0])), [0.0, 1.0, 9.0])


def test_stopped_ebh_controls_fdr_under_null():
    rng = np.random.default_rng(3)
    E = _paths(rng, 2000, 20, 200, [.2, .6, .2])
    done = np.zeros((20, 200), bool)
    fdp = [fdr.stopped_ebh(E[r], 0.05, done)[0].any() for r in range(2000)]
    assert np.mean(fdp) <= 0.05 + 0.01


def test_ebh_suprema_controls_fdr_under_null():
    rng = np.random.default_rng(4)
    E = _paths(rng, 2000, 20, 200, [.2, .6, .2])
    done = np.zeros((20, 200), bool)
    fdp = [fdr.ebh_suprema(E[r], 0.05, done)[0].any() for r in range(2000)]
    assert np.mean(fdp) <= 0.05 + 0.01


def test_stopped_ebh_has_power_and_stops_early():
    rng = np.random.default_rng(5)
    E = _paths(rng, 50, 20, 400, [.1, .6, .3])
    done = np.zeros((20, 400), bool)
    res = [fdr.stopped_ebh(E[r], 0.05, done) for r in range(50)]
    assert np.mean([r[0].mean() for r in res]) > 0.9
    assert np.mean([r[1].mean() for r in res]) < 300
