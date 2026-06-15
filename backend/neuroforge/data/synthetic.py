"""Synthetic multi-omics / wearable / lab patient generator (sensing layer).

A hidden *latent state* (per-construct abnormality severity, 0 = healthy) drives correlated,
noisy observations across modalities. Inference later tries to recover this latent state from
the observations alone — it never sees the ground truth.
"""

from __future__ import annotations

import numpy as np

from ..config import SETTINGS, STATE_CONSTRUCTS
from ..models import (
    Genomics,
    Labs,
    PatientProfile,
    Proteomics,
    Wearables,
)
from .eeg import EEGSimulator

# Per-condition mean latent severities (0 = healthy baseline, ~1 = strongly abnormal).
_CONDITION_LATENT: dict[str, dict[str, float]] = {
    "neuroinflammatory": {"neuroinflammation": 1.0, "pain_index": 0.5, "mood_index": 0.3},
    "parkinsonian": {"dopaminergic_deficit": 1.0, "neuroinflammation": 0.3},
    "mood_disorder": {"serotonergic_deficit": 0.9, "mood_index": 0.8},
    "epileptiform": {"seizure_risk": 1.0, "neuroinflammation": 0.2},
    "healthy_control": {},
}

# Biomarker base levels (healthy) and per-construct sensitivities.
_PROTEO_BASE = {
    "IL6": 1.0,
    "TNFa": 1.0,
    "CRP": 1.0,
    "BDNF": 1.0,
    "dopamine": 1.0,
    "serotonin": 1.0,
    "alpha_synuclein": 1.0,
    "neurofilament": 1.0,
}
_PROTEO_SENS: dict[str, dict[str, float]] = {
    "neuroinflammation": {"IL6": 1.2, "TNFa": 1.0, "CRP": 1.1, "neurofilament": 0.6, "BDNF": -0.4},
    "dopaminergic_deficit": {"dopamine": -0.7, "alpha_synuclein": 1.0, "neurofilament": 0.4},
    "serotonergic_deficit": {"serotonin": -0.7, "BDNF": -0.5},
    "pain_index": {"IL6": 0.4, "CRP": 0.3},
    "mood_index": {"BDNF": -0.4, "serotonin": -0.2},
    "seizure_risk": {"neurofilament": 0.5},
}

_LAB_BASE = {"crp_mgL": 1.0, "glucose": 1.0, "wbc": 1.0, "vitamin_d": 1.0}
_LAB_SENS: dict[str, dict[str, float]] = {
    "neuroinflammation": {"crp_mgL": 1.2, "wbc": 0.6, "vitamin_d": -0.3},
    "pain_index": {"crp_mgL": 0.4},
    "mood_index": {"vitamin_d": -0.3},
}


class SyntheticPatientGenerator:
    """Generate :class:`PatientProfile` instances for the supported conditions."""

    def __init__(self, seed: int | None = None):
        self.seed = SETTINGS.default_seed if seed is None else seed
        self._counter = 0

    # ------------------------------------------------------------------ #
    def generate(
        self, condition: str, patient_id: str | None = None, session: int = 0
    ) -> PatientProfile:
        if condition not in _CONDITION_LATENT:
            raise ValueError(f"Unknown condition {condition!r}; valid: {list(_CONDITION_LATENT)}")
        rng = np.random.default_rng(self.seed + 7919 * self._counter)
        self._counter += 1
        pid = patient_id or f"{condition[:4]}-{rng.integers(10000, 99999)}"

        latent = self._sample_latent(condition, rng)
        genomics = self._genomics(condition, latent, rng)
        proteomics = self._proteomics(latent, rng)
        wearables = self._wearables(latent, rng)
        labs = self._labs(latent, rng)

        _, eeg = EEGSimulator(seed=self.seed + self._counter).simulate(latent, session=session)

        return PatientProfile(
            id=pid,
            condition=condition,
            genomics=genomics,
            proteomics=proteomics,
            wearables=wearables,
            labs=labs,
            eeg=eeg,
            latent_state=latent,
        )

    # ------------------------------------------------------------------ #
    def _sample_latent(self, condition: str, rng: np.random.Generator) -> dict[str, float]:
        means = _CONDITION_LATENT[condition]
        latent: dict[str, float] = {}
        for c in STATE_CONSTRUCTS:
            mu = means.get(c, 0.0)
            val = mu + 0.12 * rng.standard_normal()
            latent[c] = float(np.clip(val, 0.0, 1.5))
        return latent

    def _genomics(
        self, condition: str, latent: dict[str, float], rng: np.random.Generator
    ) -> Genomics:
        # Polygenic risk loosely correlated with (but noisier than) the latent severity.
        pathways = {
            "inflammatory": latent["neuroinflammation"],
            "dopaminergic": latent["dopaminergic_deficit"],
            "serotonergic": latent["serotonergic_deficit"],
            "excitability": latent["seizure_risk"],
        }
        risk = {k: float(np.clip(v + 0.3 * rng.standard_normal(), 0.0, 2.0)) for k, v in pathways.items()}
        return Genomics(pathway_risk=risk)

    def _proteomics(self, latent: dict[str, float], rng: np.random.Generator) -> Proteomics:
        markers = {}
        for m, base in _PROTEO_BASE.items():
            level = base
            for construct, sens in _PROTEO_SENS.items():
                level += sens.get(m, 0.0) * latent[construct]
            level += 0.08 * rng.standard_normal()
            markers[m] = float(max(0.0, level))
        return Proteomics(markers=markers)

    def _wearables(self, latent: dict[str, float], rng: np.random.Generator) -> Wearables:
        burden = latent["neuroinflammation"] + latent["mood_index"] + 0.5 * latent["pain_index"]
        return Wearables(
            hrv_ms=float(max(5.0, 60.0 - 18.0 * burden + 4.0 * rng.standard_normal())),
            resting_hr=float(60.0 + 8.0 * burden + 3.0 * rng.standard_normal()),
            sleep_efficiency=float(np.clip(0.92 - 0.18 * latent["mood_index"] + 0.03 * rng.standard_normal(), 0.4, 1.0)),
            activity_index=float(np.clip(1.0 - 0.4 * latent["dopaminergic_deficit"] - 0.2 * latent["mood_index"] + 0.05 * rng.standard_normal(), 0.0, 1.5)),
        )

    def _labs(self, latent: dict[str, float], rng: np.random.Generator) -> Labs:
        values = {}
        for lab, base in _LAB_BASE.items():
            level = base
            for construct, sens in _LAB_SENS.items():
                level += sens.get(lab, 0.0) * latent[construct]
            level += 0.06 * rng.standard_normal()
            values[lab] = float(max(0.0, level))
        return Labs(values=values)
