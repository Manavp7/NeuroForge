import numpy as np

from neuroforge.config import EEG_BANDS
from neuroforge.data.eeg import EEGSimulator
from neuroforge.data.eeg_io import features_from_array, mne_available


def test_features_from_array_matches_extractor():
    sim = EEGSimulator(seed=3)
    raw, feat = sim.simulate({"neuroinflammation": 0.8})
    # Re-extract from the same raw via the public array API; should match the simulator.
    re = features_from_array(raw, fs=128.0)
    assert set(re.relative_power) == set(EEG_BANDS)
    for b in EEG_BANDS:
        assert abs(re.relative_power[b] - feat.relative_power[b]) < 1e-9


def test_features_from_single_channel():
    rng = np.random.default_rng(0)
    sig = rng.standard_normal(1024)
    feat = features_from_array(sig, fs=128.0)
    assert abs(sum(feat.relative_power.values()) - 1.0) < 1e-6


def test_mne_available_returns_bool():
    assert isinstance(mne_available(), bool)
