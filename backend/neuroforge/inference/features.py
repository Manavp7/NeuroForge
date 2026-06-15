"""Flatten a :class:`PatientProfile` into a deterministic named feature vector.

Only *observable* modalities are used — never the hidden ``latent_state``.
"""

from __future__ import annotations

import numpy as np

from ..models import PatientProfile


def feature_dict(profile: PatientProfile) -> dict[str, float]:
    """Return an ordered mapping of feature name -> value (deterministic key order)."""
    feats: dict[str, float] = {}
    for k in sorted(profile.proteomics.markers):
        feats[f"proteo_{k}"] = profile.proteomics.markers[k]
    for k in sorted(profile.genomics.pathway_risk):
        feats[f"geno_{k}"] = profile.genomics.pathway_risk[k]
    feats["wear_hrv"] = profile.wearables.hrv_ms
    feats["wear_rhr"] = profile.wearables.resting_hr
    feats["wear_sleep"] = profile.wearables.sleep_efficiency
    feats["wear_activity"] = profile.wearables.activity_index
    for k in sorted(profile.labs.values):
        feats[f"lab_{k}"] = profile.labs.values[k]
    for k in sorted(profile.eeg.relative_power):
        feats[f"eeg_{k}"] = profile.eeg.relative_power[k]
    feats["eeg_faa"] = profile.eeg.frontal_alpha_asymmetry
    feats["eeg_snr"] = profile.eeg.snr_db
    return feats


def feature_names(profile: PatientProfile) -> list[str]:
    return list(feature_dict(profile).keys())


def feature_vector(profile: PatientProfile, names: list[str]) -> np.ndarray:
    """Vectorize a profile against a fixed feature-name ordering."""
    d = feature_dict(profile)
    return np.array([d.get(n, 0.0) for n in names], dtype=float)
