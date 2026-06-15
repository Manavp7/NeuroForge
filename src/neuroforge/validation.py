"""Surrogate validation and safety gating for toy candidates."""

from __future__ import annotations

from dataclasses import dataclass

from neuroforge.generation import TEMPLATE_LIBRARY
from neuroforge.schemas import MoleculeCandidate, PatientProfile, PatientState, ValidationResult


@dataclass(frozen=True)
class ValidationThresholds:
    """Configurable thresholds for the synthetic validation gate."""

    min_efficacy: float = 0.45
    max_toxicity: float = 0.42
    max_off_target: float = 0.42
    max_uncertainty: float = 0.55


class SurrogateValidator:
    """Evaluate controlled candidates with transparent toy heuristics."""

    def __init__(self, thresholds: ValidationThresholds | None = None) -> None:
        self.thresholds = thresholds or ValidationThresholds()

    def validate(
        self,
        candidate: MoleculeCandidate,
        state: PatientState,
        profile: PatientProfile,
    ) -> ValidationResult:
        expected_pathways = {
            template.target_pathway
            for templates in TEMPLATE_LIBRARY.values()
            for template in templates
            if template.template_id == candidate.template_id
        }
        known_template = bool(expected_pathways)
        pathway_matches_template = candidate.target_pathway in expected_pathways
        pathway_matches_state = candidate.target_pathway in {
            template.target_pathway
            for template in TEMPLATE_LIBRARY.get(state.dominant_state, [])
        }

        alignment = 0.95 if pathway_matches_state else 0.35
        if not pathway_matches_template:
            alignment -= 0.20
        if not candidate.controlled_template:
            alignment -= 0.25
        binding_score = _clip01(alignment)

        dominant_score = _state_score(state, state.dominant_state)
        efficacy_score = _clip01(binding_score * (0.58 + 0.42 * state.confidence) * dominant_score)

        descriptors = candidate.descriptors
        length = descriptors.get("length", float(len(candidate.smiles)))
        hetero = descriptors.get("hetero_atom_count", 0.0)
        complexity = descriptors.get("synthetic_complexity_proxy", 0.0)
        fragments = descriptors.get("fragment_count", 1.0)
        polar_proxy = descriptors.get("polar_token_proxy", 0.0)

        toxicity_risk = _clip01(
            0.08
            + 0.010 * max(0.0, length - 18.0)
            + 0.045 * max(0.0, hetero - 5.0)
            + 0.055 * max(0.0, fragments - 1.0)
            + 0.10 * max(0.0, complexity - 0.75)
        )
        off_target_risk = _clip01(
            0.10
            + (0.04 if pathway_matches_state else 0.25)
            + 0.08 * profile.baseline_seizure_susceptibility
            + 0.05 * profile.baseline_mood_instability
            + (0.20 if not known_template else 0.0)
            + (0.14 if not candidate.controlled_template else 0.0)
        )
        admet_score = _clip01(
            0.82
            - 0.010 * abs(length - 16.0)
            - 0.035 * max(0.0, polar_proxy - 5.0)
            - 0.08 * max(0.0, fragments - 1.0)
            - 0.08 * max(0.0, complexity - 0.9)
        )
        uncertainty = _clip01(
            0.16
            + 0.30 * (1.0 - state.confidence)
            + 0.16 * max(0.0, dominant_score - 0.72)
            + 0.16 * max(0.0, profile.baseline_seizure_susceptibility - 0.55)
            + (0.22 if not known_template else 0.0)
            + (0.16 if not pathway_matches_template else 0.0)
            + (0.14 if not candidate.controlled_template else 0.0)
        )

        flags = {
            "efficacy_above_minimum": efficacy_score >= self.thresholds.min_efficacy,
            "toxicity_below_maximum": toxicity_risk <= self.thresholds.max_toxicity,
            "off_target_below_maximum": off_target_risk <= self.thresholds.max_off_target,
            "uncertainty_below_maximum": uncertainty <= self.thresholds.max_uncertainty,
            "controlled_template": candidate.controlled_template and known_template,
        }
        passed = all(flags.values())
        warnings = _warnings(flags, self.thresholds)

        return ValidationResult(
            binding_score=binding_score,
            efficacy_score=efficacy_score,
            toxicity_risk=toxicity_risk,
            off_target_risk=off_target_risk,
            admet_score=admet_score,
            uncertainty=uncertainty,
            passed=passed,
            threshold_flags=flags,
            warnings=warnings,
        )


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _state_score(state: PatientState, name: str) -> float:
    return {
        "neuroinflammation": state.neuroinflammation,
        "pain_risk": state.pain_risk,
        "seizure_risk": state.seizure_risk,
        "mood_instability": state.mood_instability,
    }.get(name, 0.0)


def _warnings(flags: dict[str, bool], thresholds: ValidationThresholds) -> list[str]:
    warnings: list[str] = []
    if not flags["efficacy_above_minimum"]:
        warnings.append(f"Surrogate efficacy is below minimum {thresholds.min_efficacy:.2f}.")
    if not flags["toxicity_below_maximum"]:
        warnings.append(f"Toxicity risk exceeds maximum {thresholds.max_toxicity:.2f}.")
    if not flags["off_target_below_maximum"]:
        warnings.append(f"Off-target risk exceeds maximum {thresholds.max_off_target:.2f}.")
    if not flags["uncertainty_below_maximum"]:
        warnings.append(f"Uncertainty exceeds maximum {thresholds.max_uncertainty:.2f}.")
    if not flags["controlled_template"]:
        warnings.append("Candidate is not recognized as a controlled toy template.")
    if not warnings:
        warnings.append("All surrogate thresholds passed for synthetic research demo only.")
    return warnings
