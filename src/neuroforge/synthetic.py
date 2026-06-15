"""Synthetic patient, biomarker, and neural-signal generation."""

from __future__ import annotations

import hashlib
import math

import numpy as np

from neuroforge.schemas import (
    BiomarkerSnapshot,
    NeuralSignalWindow,
    PatientProfile,
    Sex,
)


CHANNEL_NAMES = ["Fp1", "Fp2", "C3", "C4", "Pz"]
BAND_RANGES_HZ = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _stable_patient_id(seed: int | None) -> str:
    raw = f"neuroforge-synthetic-{seed if seed is not None else 'random'}".encode()
    digest = hashlib.sha256(raw).hexdigest()[:10]
    return f"syn-{digest}"


class SyntheticPatientGenerator:
    """Factory for deterministic synthetic NeuroForge input streams.

    The generated values are intentionally stylized and bounded. They are useful
    for testing orchestration, safety gates, and UI behavior, not biological
    inference or clinical modeling.
    """

    def __init__(
        self,
        sample_rate_hz: float = 128.0,
        window_seconds: float = 2.0,
        channel_names: list[str] | None = None,
    ) -> None:
        self.sample_rate_hz = sample_rate_hz
        self.window_seconds = window_seconds
        self.channel_names = channel_names or CHANNEL_NAMES

    def rng(self, seed: int | None, step: int = 0) -> np.random.Generator:
        """Return a deterministic NumPy generator for a seed/step pair."""

        base = 0 if seed is None else int(seed)
        return np.random.default_rng(base + (step * 10_007))

    def generate_profile(self, seed: int | None = None) -> PatientProfile:
        rng = self.rng(seed)
        sex = rng.choice([Sex.FEMALE, Sex.MALE, Sex.OTHER], p=[0.49, 0.49, 0.02])
        inflammation_risk = _clip01(rng.beta(2.5, 3.0))
        seizure_risk = _clip01(rng.beta(1.6, 5.0))
        mood_risk = _clip01(rng.beta(2.2, 3.4))

        return PatientProfile(
            patient_id=_stable_patient_id(seed),
            age=int(rng.integers(24, 83)),
            sex=sex,
            baseline_neuroinflammation_risk=inflammation_risk,
            baseline_seizure_susceptibility=seizure_risk,
            baseline_mood_instability=mood_risk,
            genomic_markers={
                "apoe_neurodegeneration_proxy": _clip01(
                    0.55 * inflammation_risk + 0.45 * rng.random()
                ),
                "ion_channel_excitability_proxy": _clip01(
                    0.65 * seizure_risk + 0.35 * rng.random()
                ),
                "stress_axis_polygenic_proxy": _clip01(
                    0.60 * mood_risk + 0.40 * rng.random()
                ),
            },
            proteomic_markers={
                "crp_inflammation_proxy": _clip01(
                    0.70 * inflammation_risk + 0.30 * rng.random()
                ),
                "bdnf_resilience_proxy": _clip01(1.0 - mood_risk + 0.10 * rng.normal()),
                "synaptic_stress_proxy": _clip01(
                    0.45 * seizure_risk + 0.35 * mood_risk + 0.20 * rng.random()
                ),
            },
            safety_notes=[
                "Synthetic profile generated for simulator testing only.",
                "No real patient identifiers or clinical facts are present.",
            ],
        )

    def generate_biomarkers(
        self,
        profile: PatientProfile,
        step: int,
        rng: np.random.Generator | None = None,
    ) -> BiomarkerSnapshot:
        rng = rng or self.rng(_seed_from_patient_id(profile.patient_id), step)
        circadian = 0.5 + 0.5 * math.sin(step / 3.0)
        perturbation = 0.08 * math.sin(step * 1.7)

        inflammation = _clip01(
            0.55 * profile.baseline_neuroinflammation_risk
            + 0.25 * circadian
            + perturbation
            + rng.normal(0, 0.035)
        )
        stress = _clip01(
            0.50 * profile.baseline_mood_instability
            + 0.25 * (1 - circadian)
            + 0.20 * rng.random()
            + rng.normal(0, 0.035)
        )
        sleep_recovery = _clip01(
            0.75
            - 0.35 * stress
            - 0.15 * profile.baseline_mood_instability
            + rng.normal(0, 0.04)
        )
        hrv = _clip01(0.78 - 0.42 * stress - 0.14 * inflammation + rng.normal(0, 0.04))
        glutamate_proxy = _clip01(
            0.35
            + 0.40 * profile.baseline_seizure_susceptibility
            + 0.18 * stress
            + rng.normal(0, 0.035)
        )
        gaba_proxy = _clip01(0.68 - 0.30 * stress - 0.20 * glutamate_proxy + rng.normal(0, 0.035))
        serotonin_proxy = _clip01(
            0.65 - 0.25 * stress + 0.18 * sleep_recovery + rng.normal(0, 0.035)
        )

        return BiomarkerSnapshot(
            step=step,
            inflammation=inflammation,
            stress=stress,
            sleep_recovery=sleep_recovery,
            hrv=hrv,
            glutamate_proxy=glutamate_proxy,
            gaba_proxy=gaba_proxy,
            serotonin_proxy=serotonin_proxy,
            wearable_context={
                "resting_heart_rate_proxy": _clip01(0.35 + 0.50 * stress),
                "activity_recovery_proxy": sleep_recovery,
                "skin_temperature_deviation_proxy": _clip01(0.25 + 0.50 * inflammation),
            },
        )

    def generate_neural_window(
        self,
        profile: PatientProfile,
        biomarkers: BiomarkerSnapshot,
        step: int,
        rng: np.random.Generator | None = None,
    ) -> NeuralSignalWindow:
        rng = rng or self.rng(_seed_from_patient_id(profile.patient_id), step + 1_000)
        sample_count = int(round(self.sample_rate_hz * self.window_seconds))
        timestamps = np.arange(sample_count, dtype=float) / self.sample_rate_hz

        theta_amp = 0.35 + 0.35 * profile.baseline_seizure_susceptibility
        alpha_amp = 0.75 - 0.25 * biomarkers.stress + 0.15 * biomarkers.sleep_recovery
        beta_amp = 0.30 + 0.45 * biomarkers.stress + 0.18 * biomarkers.inflammation
        gamma_amp = 0.18 + 0.38 * biomarkers.inflammation + 0.25 * biomarkers.glutamate_proxy
        artifact_score = _clip01(
            0.08 + 0.20 * biomarkers.stress + 0.08 * rng.random() + 0.04 * step / 24
        )

        signals: list[list[float]] = []
        for channel_idx, _channel in enumerate(self.channel_names):
            phase = channel_idx * np.pi / 7.0
            channel_scale = 0.90 + 0.08 * channel_idx
            signal = (
                theta_amp * np.sin(2 * np.pi * 6.0 * timestamps + phase)
                + alpha_amp * np.sin(2 * np.pi * 10.0 * timestamps + phase / 2)
                + beta_amp * np.sin(2 * np.pi * 20.0 * timestamps + phase / 3)
                + gamma_amp * np.sin(2 * np.pi * 38.0 * timestamps + phase / 4)
            )
            drift = 0.05 * np.sin(2 * np.pi * 0.5 * timestamps + phase)
            noise = rng.normal(0.0, 0.08 + artifact_score * 0.10, sample_count)
            artifact = np.zeros(sample_count)
            if artifact_score > 0.22:
                artifact_start = int(rng.integers(0, max(1, sample_count - 8)))
                artifact[artifact_start : artifact_start + 8] = rng.normal(
                    0.0, artifact_score * 1.2, 8
                )
            signals.append(((signal + drift + noise + artifact) * channel_scale).tolist())

        band_powers = self._extract_band_powers(np.asarray(signals), self.sample_rate_hz)
        return NeuralSignalWindow(
            sample_rate_hz=self.sample_rate_hz,
            timestamps=timestamps.tolist(),
            channel_names=self.channel_names,
            signals=signals,
            band_powers=band_powers,
            artifact_score=artifact_score,
        )

    @staticmethod
    def _extract_band_powers(signals: np.ndarray, sample_rate_hz: float) -> dict[str, float]:
        freqs = np.fft.rfftfreq(signals.shape[1], d=1.0 / sample_rate_hz)
        spectrum = np.abs(np.fft.rfft(signals, axis=1)) ** 2
        mean_power = spectrum.mean(axis=0)
        total_power = float(mean_power[(freqs >= 1.0) & (freqs <= 45.0)].sum())
        if total_power <= 0:
            return {band: 0.0 for band in BAND_RANGES_HZ}
        band_powers = {}
        for band, (low, high) in BAND_RANGES_HZ.items():
            mask = (freqs >= low) & (freqs < high)
            band_powers[band] = _clip01(float(mean_power[mask].sum() / total_power))
        return band_powers


def _seed_from_patient_id(patient_id: str) -> int:
    digest = hashlib.sha256(patient_id.encode()).hexdigest()[:8]
    return int(digest, 16)
