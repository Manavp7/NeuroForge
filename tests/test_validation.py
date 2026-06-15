from neuroforge.generation import CandidateGenerator
from neuroforge.schemas import MoleculeCandidate, PatientProfile, PatientState, Sex
from neuroforge.validation import SurrogateValidator


def high_confidence_state() -> PatientState:
    return PatientState(
        neuroinflammation=0.88,
        pain_risk=0.4,
        seizure_risk=0.2,
        mood_instability=0.3,
        confidence=0.9,
        dominant_state="neuroinflammation",
        feature_contributions={"inflammation": 0.9, "gamma_power": 0.6},
        explanation=[],
    )


def profile() -> PatientProfile:
    return PatientProfile(
        patient_id="syn-val",
        age=57,
        sex=Sex.FEMALE,
        baseline_neuroinflammation_risk=0.6,
        baseline_seizure_susceptibility=0.15,
        baseline_mood_instability=0.2,
        genomic_markers={},
        proteomic_markers={},
    )


def test_plausible_controlled_candidate_passes_surrogate_gate() -> None:
    state = high_confidence_state()
    candidate = CandidateGenerator().generate(state, profile())

    result = SurrogateValidator().validate(candidate, state, profile())

    assert result.passed is True
    assert result.threshold_flags["controlled_template"] is True
    assert result.efficacy_score >= 0.45
    assert result.toxicity_risk <= 0.42
    assert "synthetic research demo only" in result.warnings[0]


def test_unsafe_unknown_candidate_fails_with_warnings() -> None:
    bad_candidate = MoleculeCandidate(
        candidate_id="toy-bad",
        smiles="CCCCCCCCCCCCCCCCCCCCNNNNNNNN.O.O.O",
        target_pathway="unknown_autonomous_pathway",
        template_id="not-library",
        rationale="Bad synthetic test candidate.",
        descriptors={
            "length": 36.0,
            "hetero_atom_count": 8.0,
            "polar_token_proxy": 8.0,
            "fragment_count": 4.0,
            "synthetic_complexity_proxy": 2.5,
        },
        controlled_template=False,
    )

    result = SurrogateValidator().validate(bad_candidate, high_confidence_state(), profile())

    assert result.passed is False
    assert result.threshold_flags["controlled_template"] is False
    assert result.toxicity_risk > 0.42
    assert any("Candidate is not recognized" in warning for warning in result.warnings)
