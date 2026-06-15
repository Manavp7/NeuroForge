import numpy as np

from neuroforge.config import EEG_BANDS
from neuroforge.data.eeg import EEGSimulator


def test_simulate_shapes_and_features():
    sim = EEGSimulator(seed=1)
    raw, feat = sim.simulate({}, duration_s=4.0, fs=128.0)
    assert raw.shape[0] == 4
    assert raw.shape[1] == int(4.0 * 128.0)
    # Relative powers sum to ~1 across bands.
    assert set(feat.relative_power) == set(EEG_BANDS)
    assert abs(sum(feat.relative_power.values()) - 1.0) < 1e-6
    assert 0.0 <= feat.artifact_ratio <= 1.0


def test_determinism_same_seed():
    a = EEGSimulator(seed=42).simulate({"neuroinflammation": 0.8}, session=0)[1]
    b = EEGSimulator(seed=42).simulate({"neuroinflammation": 0.8}, session=0)[1]
    assert a.relative_power == b.relative_power
    assert a.frontal_alpha_asymmetry == b.frontal_alpha_asymmetry


def test_latent_state_shifts_spectrum():
    sim = EEGSimulator(seed=3)
    # Neuroinflammation -> slowing: more delta relative power than a healthy control.
    healthy = sim.simulate({c: 0.0 for c in ["neuroinflammation"]})[1]
    inflamed = sim.simulate({"neuroinflammation": 1.2})[1]
    assert inflamed.relative_power["delta"] > healthy.relative_power["delta"]
    # Seizure risk -> elevated gamma.
    seiz = sim.simulate({"seizure_risk": 1.2})[1]
    assert seiz.relative_power["gamma"] > healthy.relative_power["gamma"]


def test_artifacts_increase_with_level():
    sim = EEGSimulator(seed=9)
    low = np.mean([sim.simulate({}, artifact_level=0.0)[1].artifact_ratio for _ in range(3)])
    high = np.mean([sim.simulate({}, artifact_level=1.0)[1].artifact_ratio for _ in range(3)])
    assert high >= low
