import pytest
from pydantic import ValidationError

from neuroforge.schemas import (
    ApprovalStatus,
    BiomarkerSnapshot,
    ClosedLoopIteration,
    MoleculeCandidate,
    NeuralSignalWindow,
    PatientProfile,
    PatientState,
    Sex,
    ValidationResult,
)


def example_profile() -> PatientProfile:
    return PatientProfile(
        patient_id="syn-001",
        age=52,
        sex=Sex.FEMALE,
        baseline_neuroinflammation_risk=0.4,
        baseline_seizure_susceptibility=0.2,
        baseline_mood_instability=0.5,
        genomic_markers={"APOE_proxy": 0.7},
        proteomic_markers={"CRP_proxy": 0.3},
    )


def test_neural_window_validates_shape_and_sample_rate() -> None:
    window = NeuralSignalWindow(
        sample_rate_hz=128,
        timestamps=[0.0, 0.01, 0.02],
        channel_names=["Fp1", "Fp2"],
        signals=[[0.1, 0.2, 0.3], [0.2, 0.1, 0.0]],
        band_powers={"alpha": 0.3},
        artifact_score=0.1,
    )

    assert window.sample_rate_hz == 128

    with pytest.raises(ValidationError):
        NeuralSignalWindow(
            sample_rate_hz=-1,
            timestamps=[0.0, 0.01],
            channel_names=["Fp1"],
            signals=[[0.1, 0.2]],
            band_powers={},
            artifact_score=0.1,
        )

    with pytest.raises(ValidationError):
        NeuralSignalWindow(
            sample_rate_hz=128,
            timestamps=[0.0, 0.01],
            channel_names=["Fp1", "Fp2"],
            signals=[[0.1, 0.2]],
            band_powers={},
            artifact_score=0.1,
        )


def test_profile_requires_synthetic_normalized_markers() -> None:
    profile = example_profile()
    assert profile.synthetic is True

    with pytest.raises(ValidationError):
        PatientProfile(
            patient_id="real-001",
            synthetic=False,
            age=45,
            sex=Sex.MALE,
            baseline_neuroinflammation_risk=0.1,
            baseline_seizure_susceptibility=0.1,
            baseline_mood_instability=0.1,
            genomic_markers={},
            proteomic_markers={},
        )

    with pytest.raises(ValidationError):
        PatientProfile(
            patient_id="syn-002",
            age=45,
            sex=Sex.MALE,
            baseline_neuroinflammation_risk=0.1,
            baseline_seizure_susceptibility=0.1,
            baseline_mood_instability=0.1,
            genomic_markers={"bad": 1.5},
            proteomic_markers={},
        )


def test_closed_loop_serializes_and_enforces_delivery_gate() -> None:
    iteration = ClosedLoopIteration(
        step=0,
        patient=example_profile(),
        biomarkers=BiomarkerSnapshot(
            step=0,
            inflammation=0.4,
            stress=0.5,
            sleep_recovery=0.6,
            hrv=0.7,
            glutamate_proxy=0.5,
            gaba_proxy=0.5,
            serotonin_proxy=0.5,
        ),
        signal_window=NeuralSignalWindow(
            sample_rate_hz=128,
            timestamps=[0.0, 0.01],
            channel_names=["C3"],
            signals=[[0.1, 0.2]],
            band_powers={"alpha": 0.2},
            artifact_score=0.1,
        ),
        inferred_state=PatientState(
            neuroinflammation=0.4,
            pain_risk=0.3,
            seizure_risk=0.2,
            mood_instability=0.5,
            confidence=0.7,
            dominant_state="mood_instability",
            feature_contributions={"stress": 0.5},
        ),
        candidate=MoleculeCandidate(
            candidate_id="toy-001",
            smiles="CCN(O)C",
            target_pathway="stress_modulation",
            template_id="stress-1",
            rationale="Synthetic demonstration candidate.",
            descriptors={"length": 7},
        ),
        validation=ValidationResult(
            binding_score=0.8,
            efficacy_score=0.7,
            toxicity_risk=0.2,
            off_target_risk=0.2,
            admet_score=0.8,
            uncertainty=0.2,
            passed=True,
            threshold_flags={"toxicity": True},
        ),
        doctor_approved=True,
        approval_status=ApprovalStatus.APPROVED_FOR_SIMULATED_DELIVERY,
        deliverable=True,
    )

    payload = iteration.model_dump(mode="json")

    assert payload["patient"]["patient_id"] == "syn-001"
    assert "synthetic research simulator" in payload["safety_disclaimer"]

    with pytest.raises(ValidationError):
        ClosedLoopIteration(
            **{**payload, "doctor_approved": False, "deliverable": True}
        )
