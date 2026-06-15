"""Domain schemas for the NeuroForge synthetic simulator."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SAFETY_DISCLAIMER = (
    "NeuroForge MVP is a synthetic research simulator. Outputs are toy surrogate "
    "artifacts, not medical advice, clinical decision support, dosing guidance, "
    "or validated molecular designs."
)


class Sex(StrEnum):
    """Synthetic sex metadata used only for simulator stratification."""

    FEMALE = "female"
    MALE = "male"
    OTHER = "other"


class ApprovalStatus(StrEnum):
    """Closed-loop candidate approval state."""

    BLOCKED_PENDING_REVIEW = "blocked_pending_review"
    BLOCKED_VALIDATION_FAILED = "blocked_validation_failed"
    APPROVED_FOR_SIMULATED_DELIVERY = "approved_for_simulated_delivery"


class PatientProfile(BaseModel):
    """Synthetic patient baseline profile.

    The omics and risk fields are normalized proxies in ``[0, 1]``. They are not
    linked to real patient data or validated biology.
    """

    model_config = ConfigDict(extra="forbid")

    patient_id: str = Field(min_length=3)
    synthetic: bool = True
    age: int = Field(ge=18, le=100)
    sex: Sex
    baseline_neuroinflammation_risk: float = Field(ge=0, le=1)
    baseline_seizure_susceptibility: float = Field(ge=0, le=1)
    baseline_mood_instability: float = Field(ge=0, le=1)
    genomic_markers: dict[str, float] = Field(default_factory=dict)
    proteomic_markers: dict[str, float] = Field(default_factory=dict)
    safety_notes: list[str] = Field(default_factory=list)

    @field_validator("synthetic")
    @classmethod
    def must_be_synthetic(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError("MVP only supports synthetic patient profiles")
        return value

    @field_validator("genomic_markers", "proteomic_markers")
    @classmethod
    def normalized_marker_values(cls, value: dict[str, float]) -> dict[str, float]:
        for marker, score in value.items():
            if not 0 <= score <= 1:
                raise ValueError(f"{marker} must be in [0, 1]")
        return value


class NeuralSignalWindow(BaseModel):
    """Synthetic BCI-like signal window with extracted frequency-band proxies."""

    model_config = ConfigDict(extra="forbid")

    sample_rate_hz: float = Field(gt=0)
    timestamps: list[float] = Field(min_length=2)
    channel_names: list[str] = Field(min_length=1)
    signals: list[list[float]] = Field(min_length=1)
    band_powers: dict[str, float] = Field(default_factory=dict)
    artifact_score: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_signal_shape(self) -> "NeuralSignalWindow":
        if len(self.signals) != len(self.channel_names):
            raise ValueError("signals must contain one row per channel")
        expected_samples = len(self.timestamps)
        for row in self.signals:
            if len(row) != expected_samples:
                raise ValueError("every signal row must match timestamp count")
        for band, power in self.band_powers.items():
            if power < 0:
                raise ValueError(f"band power {band} must be non-negative")
        return self


class BiomarkerSnapshot(BaseModel):
    """Synthetic multimodal biomarker snapshot for one loop step."""

    model_config = ConfigDict(extra="forbid")

    step: int = Field(ge=0)
    inflammation: float = Field(ge=0, le=1)
    stress: float = Field(ge=0, le=1)
    sleep_recovery: float = Field(ge=0, le=1)
    hrv: float = Field(ge=0, le=1)
    glutamate_proxy: float = Field(ge=0, le=1)
    gaba_proxy: float = Field(ge=0, le=1)
    serotonin_proxy: float = Field(ge=0, le=1)
    wearable_context: dict[str, float] = Field(default_factory=dict)

    @field_validator("wearable_context")
    @classmethod
    def normalized_wearable_context(cls, value: dict[str, float]) -> dict[str, float]:
        for key, score in value.items():
            if not 0 <= score <= 1:
                raise ValueError(f"{key} must be in [0, 1]")
        return value


class PatientState(BaseModel):
    """Inferred synthetic patient state produced by transparent heuristics."""

    model_config = ConfigDict(extra="forbid")

    neuroinflammation: float = Field(ge=0, le=1)
    pain_risk: float = Field(ge=0, le=1)
    seizure_risk: float = Field(ge=0, le=1)
    mood_instability: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    dominant_state: str = Field(min_length=1)
    feature_contributions: dict[str, float] = Field(default_factory=dict)
    explanation: list[str] = Field(default_factory=list)

    @field_validator("feature_contributions")
    @classmethod
    def finite_feature_contributions(cls, value: dict[str, float]) -> dict[str, float]:
        for key, score in value.items():
            if not -1 <= score <= 1:
                raise ValueError(f"{key} contribution must be in [-1, 1]")
        return value


class MoleculeCandidate(BaseModel):
    """Controlled toy candidate from a template library.

    ``smiles`` is a SMILES-like string used for surrogate descriptor extraction;
    it is not intended as a real validated chemical design.
    """

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(min_length=3)
    smiles: str = Field(min_length=1)
    target_pathway: str = Field(min_length=1)
    template_id: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    descriptors: dict[str, float] = Field(default_factory=dict)
    controlled_template: bool = True

    @field_validator("descriptors")
    @classmethod
    def non_negative_descriptors(cls, value: dict[str, float]) -> dict[str, float]:
        for descriptor, score in value.items():
            if score < 0:
                raise ValueError(f"{descriptor} must be non-negative")
        return value


class ValidationResult(BaseModel):
    """Surrogate validation summary for a toy candidate."""

    model_config = ConfigDict(extra="forbid")

    binding_score: float = Field(ge=0, le=1)
    efficacy_score: float = Field(ge=0, le=1)
    toxicity_risk: float = Field(ge=0, le=1)
    off_target_risk: float = Field(ge=0, le=1)
    admet_score: float = Field(ge=0, le=1)
    uncertainty: float = Field(ge=0, le=1)
    passed: bool
    threshold_flags: dict[str, bool] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ClosedLoopIteration(BaseModel):
    """One complete synthetic closed-loop iteration."""

    model_config = ConfigDict(extra="forbid")

    step: int = Field(ge=0)
    patient: PatientProfile
    biomarkers: BiomarkerSnapshot
    signal_window: NeuralSignalWindow
    inferred_state: PatientState
    candidate: MoleculeCandidate
    validation: ValidationResult
    doctor_approved: bool = False
    approval_status: ApprovalStatus
    deliverable: bool = False
    audit_notes: list[str] = Field(default_factory=list)
    safety_disclaimer: str = SAFETY_DISCLAIMER

    @model_validator(mode="after")
    def enforce_delivery_gate(self) -> "ClosedLoopIteration":
        allowed = (
            self.validation.passed
            and self.doctor_approved
            and self.approval_status
            == ApprovalStatus.APPROVED_FOR_SIMULATED_DELIVERY
        )
        if self.deliverable != allowed:
            raise ValueError("deliverable must reflect validation and doctor approval gates")
        return self


class SimulationRequest(BaseModel):
    """Request for running one or more synthetic loop iterations."""

    model_config = ConfigDict(extra="forbid")

    seed: int = Field(default=7, ge=0)
    step: int = Field(default=0, ge=0)
    steps: int = Field(default=3, ge=1, le=24)
    doctor_approved: bool = False
    require_approval: bool = True


class SimulationResponse(BaseModel):
    """API response wrapper for closed-loop simulations."""

    model_config = ConfigDict(extra="forbid")

    iterations: list[ClosedLoopIteration]
    safety_disclaimer: str = SAFETY_DISCLAIMER
    metadata: dict[str, Any] = Field(default_factory=dict)
