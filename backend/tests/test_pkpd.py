import numpy as np

from neuroforge.loop.pkpd import (
    PKPDParams,
    Regimen,
    effect_curve,
    recommend_regimen,
    simulate_regimen,
    steady_state_efficacy,
)


def test_concentration_curve_shape_and_positivity():
    t, c = simulate_regimen(Regimen(n_doses=3, interval_h=12.0), PKPDParams())
    assert len(t) == len(c) > 0
    assert np.all(c >= -1e-9)
    assert c.max() > 0  # drug is absorbed


def test_accumulation_across_doses():
    # Average concentration in the last interval should exceed the first (accumulation).
    reg = Regimen(dose_mg=100, n_doses=5, interval_h=12.0)
    params = PKPDParams()
    t, c = simulate_regimen(reg, params)
    first = c[t < reg.interval_h].mean()
    last = c[t >= (t[-1] - reg.interval_h)].mean()
    assert last > first


def test_emax_saturates():
    params = PKPDParams(emax=0.7, ec50=1.0)
    e = effect_curve(np.array([0.0, 1.0, 1000.0]), params)
    assert e[0] == 0.0
    assert abs(e[1] - 0.35) < 1e-6  # half of Emax at C=EC50
    assert e[2] < params.emax and e[2] > 0.69  # approaches Emax


def test_potency_monotonic():
    # Higher predicted pKi -> higher steady-state efficacy.
    _, _, eff_low, _ = recommend_regimen(5.0)
    _, _, eff_high, _ = recommend_regimen(8.5)
    assert eff_high > eff_low
    assert 0.0 <= eff_low <= 0.7
    assert 0.0 <= eff_high <= 0.7


def test_steady_state_efficacy_bounds():
    eff = steady_state_efficacy(Regimen(), PKPDParams())
    assert 0.0 <= eff <= 0.7
