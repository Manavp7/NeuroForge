"""Real EEG ingestion: load EDF/BDF recordings (via MNE) or raw arrays into EEGFeatures.

MNE is optional (`pip install -e ".[bio]"`). The array-based path needs only scipy, so the same
feature extractor used by the simulator works on real recordings too — demonstrating that the
pipeline can consume genuine BCI data, not just synthetic signals.
"""

from __future__ import annotations

import numpy as np

from ..models import EEGFeatures
from .eeg import extract_band_features


def mne_available() -> bool:
    try:
        import mne  # noqa: F401

        return True
    except Exception:
        return False


def features_from_array(data: np.ndarray, fs: float, artifact_ratio: float = 0.0) -> EEGFeatures:
    """Extract features from a (n_channels, n_samples) array at sampling rate ``fs``."""
    data = np.asarray(data, dtype=float)
    return extract_band_features(data, fs, artifact_ratio)


def features_from_edf(path: str, max_seconds: float = 60.0) -> EEGFeatures:
    """Load an EDF/BDF file via MNE and extract band-power features."""
    try:
        import mne
    except Exception as exc:  # pragma: no cover - exercised only without MNE
        raise RuntimeError(
            "Real EEG ingestion requires MNE. Install with: pip install -e '.[bio]'"
        ) from exc

    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    raw.pick("eeg")
    fs = float(raw.info["sfreq"])
    data = raw.get_data()  # (n_channels, n_samples), volts
    if max_seconds:
        data = data[:, : int(max_seconds * fs)]
    # Scale to microvolt-ish range so PSD magnitudes are comparable to the simulator.
    data = data * 1e6
    return extract_band_features(data, fs)
