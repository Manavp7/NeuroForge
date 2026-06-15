import numpy as np

from neuroforge.generation import CandidateGenerator, TEMPLATE_LIBRARY, compute_descriptors
from neuroforge.schemas import PatientProfile, PatientState, Sex


def patient() -> PatientProfile:
    return PatientProfile(
        patient_id="syn-gen",
        age=44,
        sex=Sex.OTHER,
        baseline_neuroinflammation_risk=0.2,
        baseline_seizure_susceptibility=0.7,
        baseline_mood_instability=0.3,
        genomic_markers={},
        proteomic_markers={},
    )


def test_generation_selects_pathway_for_dominant_state() -> None:
    state = PatientState(
        neuroinflammation=0.2,
        pain_risk=0.3,
        seizure_risk=0.9,
        mood_instability=0.4,
        confidence=0.8,
        dominant_state="seizure_risk",
        feature_contributions={"theta_power": 0.8, "gamma_power": 0.6},
        explanation=[],
    )

    candidate = CandidateGenerator().generate(state, patient(), np.random.default_rng(9))
    allowed_pathways = {template.target_pathway for template in TEMPLATE_LIBRARY["seizure_risk"]}

    assert candidate.target_pathway in allowed_pathways
    assert candidate.template_id.startswith("nf-")
    assert candidate.controlled_template is True
    assert "theta_power" in candidate.rationale
    assert candidate.descriptors["length"] > 0


def test_descriptor_extraction_works_without_rdkit() -> None:
    descriptors = compute_descriptors("NCC(O)CN1CCOCC1.CO")

    assert descriptors["hetero_atom_count"] >= 4
    assert descriptors["fragment_count"] == 2
    assert descriptors["synthetic_complexity_proxy"] > 0
    assert all(value >= 0 for value in descriptors.values())
