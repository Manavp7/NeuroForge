"""Pydantic data models shared across NeuroForge modules and the API.

All numeric "clinical-ish" quantities are synthetic and illustrative only.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from . import DISCLAIMER


class Uncertain(BaseModel):
    """A scalar estimate with an associated (1-sigma) uncertainty."""

    value: float
    std: float = 0.0

    def as_tuple(self) -> tuple[float, float]:
        return self.value, self.std


# --------------------------------------------------------------------------- #
# Sensing layer
# --------------------------------------------------------------------------- #
class Genomics(BaseModel):
    """Per-pathway polygenic risk scores derived from (synthetic) SNP panels."""

    pathway_risk: dict[str, float] = Field(default_factory=dict)


class Proteomics(BaseModel):
    """Plasma/CSF biomarker levels (arbitrary normalized units)."""

    markers: dict[str, float] = Field(default_factory=dict)


class Wearables(BaseModel):
    """Summarized continuous wearable features."""

    hrv_ms: float = 0.0
    resting_hr: float = 0.0
    sleep_efficiency: float = 0.0
    activity_index: float = 0.0


class Labs(BaseModel):
    """Periodic lab panel values (arbitrary normalized units)."""

    values: dict[str, float] = Field(default_factory=dict)


class EEGFeatures(BaseModel):
    """Band-power features + quality flags extracted from simulated EEG."""

    relative_power: dict[str, float] = Field(default_factory=dict)
    frontal_alpha_asymmetry: float = 0.0
    snr_db: float = 0.0
    artifact_ratio: float = 0.0


class PatientProfile(BaseModel):
    """A synthetic patient. ``latent_state`` is ground truth used ONLY by the simulator."""

    id: str
    condition: str
    genomics: Genomics
    proteomics: Proteomics
    wearables: Wearables
    labs: Labs
    eeg: EEGFeatures
    # Hidden ground-truth latent abnormality drivers (never exposed by inference).
    latent_state: dict[str, float] = Field(default_factory=dict, repr=False)


# --------------------------------------------------------------------------- #
# Inference layer
# --------------------------------------------------------------------------- #
class PatientState(BaseModel):
    """Inferred patient state: each construct with uncertainty + overall confidence."""

    constructs: dict[str, Uncertain] = Field(default_factory=dict)
    confidence: float = 0.0
    explanations: dict[str, list[tuple[str, float]]] = Field(default_factory=dict)

    def abnormality(self) -> float:
        """Severity of the *most* abnormal construct (0 = healthy baseline).

        Using the max (rather than the mean) avoids a single severe abnormality being
        diluted by many healthy constructs, which gives the closed loop meaningful
        multi-step dynamics.
        """
        if not self.constructs:
            return 0.0
        return max(abs(u.value) for u in self.constructs.values())


# --------------------------------------------------------------------------- #
# Design layer
# --------------------------------------------------------------------------- #
class TargetProfile(BaseModel):
    """Desired molecular property windows + intended (mock) protein target."""

    target_id: str
    target_name: str
    rationale: str = ""
    # Desired property windows: name -> (low, high)
    property_windows: dict[str, tuple[float, float]] = Field(default_factory=dict)
    # Importance weights per state construct driving this target.
    driving_constructs: dict[str, float] = Field(default_factory=dict)


class ADMET(BaseModel):
    """Computed physicochemical / ADMET-style descriptors (illustrative)."""

    mol_weight: float = 0.0
    logp: float = 0.0
    tpsa: float = 0.0
    hbd: int = 0
    hba: int = 0
    rotatable_bonds: int = 0
    qed: float = 0.0
    sa_score: float = 0.0
    lipinski_violations: int = 0
    tox_flags: list[str] = Field(default_factory=list)


class Candidate(BaseModel):
    """A proposed therapeutic molecule with validation results."""

    id: str
    smiles: str
    admet: ADMET
    binding: Uncertain = Field(default_factory=lambda: Uncertain(value=0.0, std=0.0))
    score: float = 0.0
    risk_adjusted_score: float = 0.0  # score penalized by binding uncertainty
    safe: bool = True
    safety_notes: list[str] = Field(default_factory=list)
    rationale: str = ""
    predicted_effect: float = 0.0  # PK/PD steady-state efficacy fraction (0..Emax)
    pkpd: dict[str, Any] = Field(default_factory=dict)  # regimen + concentration summary
    provenance: dict[str, Any] = Field(default_factory=dict)
    svg: str | None = None


# --------------------------------------------------------------------------- #
# Closed loop
# --------------------------------------------------------------------------- #
class LoopEvent(BaseModel):
    """A single typed event emitted during a closed-loop run."""

    iteration: int
    phase: (
        str  # sense | infer | plan | design | validate | critique | gate | deliver | monitor | done
    )
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)


class Iteration(BaseModel):
    """One full pass of the loop."""

    index: int
    state: PatientState
    target: TargetProfile | None = None
    candidates: list[Candidate] = Field(default_factory=list)
    chosen: Candidate | None = None
    approved: bool | None = None
    abnormality_before: float = 0.0
    abnormality_after: float | None = None


class LoopRun(BaseModel):
    """A complete (or in-progress) closed-loop run for one patient."""

    id: str
    patient_id: str
    status: str = (
        "created"  # created | awaiting_approval | running | stabilized | exhausted | rejected
    )
    iterations: list[Iteration] = Field(default_factory=list)
    events: list[LoopEvent] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER
