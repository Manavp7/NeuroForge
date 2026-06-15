"""Robustness / adversarial perturbations for stress-testing the inference pipeline.

Provides perturbations (EEG quality degradation, omics noise, adversarial feature shifts) and a
helper to measure how the inferred state + its uncertainty respond. A robust system should remain
bounded under small noise and *inflate* uncertainty as signal quality drops.
"""

from __future__ import annotations

import numpy as np

from .models import PatientProfile, PatientState


def degrade_eeg(
    profile: PatientProfile, artifact_ratio: float = 0.5, snr_drop_db: float = 6.0
) -> PatientProfile:
    p = profile.model_copy(deep=True)
    p.eeg.artifact_ratio = float(min(1.0, p.eeg.artifact_ratio + artifact_ratio))
    p.eeg.snr_db = float(p.eeg.snr_db - snr_drop_db)
    return p


def add_omics_noise(profile: PatientProfile, sigma: float = 0.2, seed: int = 0) -> PatientProfile:
    rng = np.random.default_rng(seed)
    p = profile.model_copy(deep=True)
    for k in p.proteomics.markers:
        p.proteomics.markers[k] = float(
            max(0.0, p.proteomics.markers[k] + sigma * rng.standard_normal())
        )
    for k in p.labs.values:
        p.labs.values[k] = float(max(0.0, p.labs.values[k] + sigma * rng.standard_normal()))
    return p


def adversarial_shift(profile: PatientProfile, scale: float = 0.5) -> PatientProfile:
    """Push inflammatory markers up to try to fool the estimator."""
    p = profile.model_copy(deep=True)
    for k in ("IL6", "TNFa", "CRP"):
        if k in p.proteomics.markers:
            p.proteomics.markers[k] += scale
    return p


def mean_uncertainty(state: PatientState) -> float:
    if not state.constructs:
        return 0.0
    return float(np.mean([u.std for u in state.constructs.values()]))
