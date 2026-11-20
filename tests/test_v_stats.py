"""V1, V2, V4, V5, V6: statistical validity of the gate and the outcome model."""
import numpy as np
import pytest

from nixigate import dgp, gates

ALPHA = 0.05


def se(p, n):
    return np.sqrt(p * (1 - p) / n)


@pytest.fixture(scope="module")
def null_model():
    return dgp.Model.for_lift(0.0, 0.339, sd_theta=1.5, sd_lift=0.5, rho_crn=0.5)


def test_v6_lift_solver_exact():
    for lift in (-0.2, -0.05, 0.0, 0.05, 0.166, 0.3):
        m = dgp.Model.for_lift(lift, 0.339, sd_theta=1.5, sd_lift=0.5)
        assert abs(m.lift - lift) < 1e-8
        assert abs(dgp.mean_sigmoid(m.b0, 1.5) - 0.339) < 1e-10


def test_v6_lift_solver_matches_mc():
    m = dgp.Model.for_lift(0.10, 0.339)
    D = dgp.simulate(np.random.default_rng(1), m, 200, 2000)
    assert abs(D.mean() - 0.10) < 3 * D.std() / np.sqrt(D.size) + 1e-3


def test_v1_wealth_supermartingale(null_model):
    rng = np.random.default_rng(2)
    D = dgp.simulate(rng, null_model, 20000, 500)
    X = (1 + D) / 2
    lam = gates.prpl_bets(X, ALPHA, 0.5)
    K = np.exp(np.cumsum(np.log1p(lam * (X - 0.5)), axis=1))
    for n in (10, 100, 500):
        k = K[:, n - 1]
        assert k.mean() <= 1 + 3 * k.std() / np.sqrt(len(k))


@pytest.mark.parametrize("kind", ["model", "two_point"])
@pytest.mark.parametrize("rule", ["prpl", "cap_m0", "agrapa"])
def test_v2_ville_type_one(null_model, kind, rule):
    rng = np.random.default_rng(3)
    R = 20000
    D = dgp.simulate(rng, null_model, R, 600) if kind == "model" else dgp.simulate_two_point(rng, 0.0, R, 600)
    res = gates.betting(D, 0.0, ALPHA, alpha_f=None, rule=rule)
    rate = res.admit.mean()
    assert rate <= ALPHA + 3 * se(ALPHA, R)


def test_v4_cs_coverage():
    m = dgp.Model.for_lift(0.12, 0.339)
    rng = np.random.default_rng(4)
    D = dgp.simulate(rng, m, 1500, 300)
    miss = sum((gates.lcb_path(D[j], ALPHA) > m.lift + 1e-9).any() for j in range(len(D)))
    assert miss / len(D) <= ALPHA + 3 * se(ALPHA, len(D))


def test_lcb_consistent_with_gate():
    m = dgp.Model.for_lift(0.15, 0.339)
    D = dgp.simulate(np.random.default_rng(5), m, 200, 400)
    res = gates.betting(D, 0.0, ALPHA, alpha_f=None)
    for j in range(len(D)):
        path = gates.lcb_path(D[j], ALPHA, grid=4001)
        crossed = (path > 1e-3).any()
        if res.admit[j]:
            assert path[res.n[j] - 1] >= -1e-3
        if (path > 2e-3).any():
            assert res.admit[j]
        _ = crossed


def test_v5_fixed_n_power():
    m1 = dgp.Model.for_lift(0.10, 0.339)
    sd = dgp.sd_D(m1)
    n = gates.sample_size(sd, 0.10, 0.0, ALPHA, 0.80)
    D = dgp.simulate(np.random.default_rng(6), m1, 4000, n)
    pw = gates.fixed_t(D, n, 0.0, ALPHA).admit.mean()
    assert abs(pw - 0.80) < 0.03


def test_fixed_t_type_one(null_model):
    D = dgp.simulate(np.random.default_rng(7), null_model, 20000, 250)
    rate = gates.fixed_t(D, 250, 0.0, ALPHA).admit.mean()
    assert abs(rate - ALPHA) < 4 * se(ALPHA, 20000) + 0.005


def test_peeking_inflates(null_model):
    D = dgp.simulate(np.random.default_rng(8), null_model, 5000, 600)
    assert gates.peek_t(D, 0.0, ALPHA, 10).admit.mean() > 2 * ALPHA


def test_sprt_alt_and_null():
    m0 = dgp.Model.for_lift(0.0, 0.339)
    p1 = dgp.discordant_share(dgp.Model.for_lift(0.10, 0.339))
    assert 0.5 < p1 < 1
    assert abs(dgp.discordant_share(m0) - 0.5) < 0.02
