"""Transparent heuristic state inference for synthetic NeuroForge streams."""

from __future__ import annotations

from neuroforge.schemas import BiomarkerSnapshot, NeuralSignalWindow, PatientProfile, PatientState


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class StateInferenceEngine:
    """Infer synthetic patient states from multimodal simulator features.

    This intentionally uses simple, inspectable scoring rather than opaque model
    weights so safety and UI flows can be evaluated before any real model exists.
    """

    def infer(
        self,
        profile: PatientProfile,
        biomarkers: BiomarkerSnapshot,
        signal_window: NeuralSignalWindow,
    ) -> PatientState:
        bands = signal_window.band_powers
        theta = bands.get("theta", 0.0)
        alpha = bands.get("alpha", 0.0)
        beta = bands.get("beta", 0.0)
        gamma = bands.get("gamma", 0.0)

        neuroinflammation = _clip01(
            0.42 * biomarkers.inflammation
            + 0.24 * gamma
            + 0.16 * beta
            + 0.18 * profile.baseline_neuroinflammation_risk
        )
        mood_instability = _clip01(
            0.32 * biomarkers.stress
            + 0.24 * (1.0 - biomarkers.sleep_recovery)
            + 0.22 * (1.0 - biomarkers.hrv)
            + 0.14 * profile.baseline_mood_instability
            + 0.08 * (1.0 - alpha)
        )
        seizure_risk = _clip01(
            0.28 * theta
            + 0.22 * gamma
            + 0.22 * biomarkers.glutamate_proxy
            + 0.16 * (1.0 - biomarkers.gaba_proxy)
            + 0.12 * profile.baseline_seizure_susceptibility
        )
        pain_risk = _clip01(
            0.36 * biomarkers.inflammation
            + 0.25 * biomarkers.stress
            + 0.18 * (1.0 - biomarkers.hrv)
            + 0.12 * beta
            + 0.09 * profile.baseline_neuroinflammation_risk
        )

        state_scores = {
            "neuroinflammation": neuroinflammation,
            "pain_risk": pain_risk,
            "seizure_risk": seizure_risk,
            "mood_instability": mood_instability,
        }
        dominant_state = max(state_scores, key=state_scores.get)
        signal_quality = 1.0 - signal_window.artifact_score
        evidence_strength = max(state_scores.values())
        confidence = _clip01(0.25 + 0.45 * evidence_strength + 0.30 * signal_quality)

        feature_contributions = {
            "inflammation": biomarkers.inflammation,
            "stress": biomarkers.stress,
            "sleep_recovery_deficit": 1.0 - biomarkers.sleep_recovery,
            "hrv_deficit": 1.0 - biomarkers.hrv,
            "theta_power": theta,
            "beta_power": beta,
            "gamma_power": gamma,
            "artifact_penalty": -signal_window.artifact_score,
        }
        explanation = [
            (
                f"Dominant synthetic state is {dominant_state} "
                f"(score={state_scores[dominant_state]:.2f})."
            ),
            (
                "Neuroinflammation proxy combines inflammation marker, gamma/beta "
                "band power, and baseline risk."
            ),
            (
                "Mood instability proxy combines stress, sleep recovery deficit, "
                "HRV deficit, alpha suppression, and baseline risk."
            ),
            (
                "Seizure-risk proxy combines theta/gamma band power, glutamate/GABA "
                "balance, and baseline susceptibility."
            ),
            (
                f"Signal artifact score {signal_window.artifact_score:.2f} "
                f"contributes to confidence {confidence:.2f}."
            ),
        ]

        return PatientState(
            neuroinflammation=neuroinflammation,
            pain_risk=pain_risk,
            seizure_risk=seizure_risk,
            mood_instability=mood_instability,
            confidence=confidence,
            dominant_state=dominant_state,
            feature_contributions=feature_contributions,
            explanation=explanation,
        )
