from neuroforge.inference import StateInferenceEngine
from neuroforge.schemas import BiomarkerSnapshot, NeuralSignalWindow, PatientProfile, Sex
from neuroforge.synthetic import SyntheticPatientGenerator


def test_inference_is_deterministic_and_explainable() -> None:
    generator = SyntheticPatientGenerator()
    profile = generator.generate_profile(seed=11)
    rng = generator.rng(11, step=1)
    biomarkers = generator.generate_biomarkers(profile, step=1, rng=rng)
    window = generator.generate_neural_window(profile, biomarkers, step=1, rng=rng)
    engine = StateInferenceEngine()

    state_a = engine.infer(profile, biomarkers, window)
    state_b = engine.infer(profile, biomarkers, window)

    assert state_a == state_b
    assert 0 <= state_a.confidence <= 1
    assert state_a.dominant_state in {
        "neuroinflammation",
        "pain_risk",
        "seizure_risk",
        "mood_instability",
    }
    assert any("gamma/beta" in line for line in state_a.explanation)
    assert "artifact_penalty" in state_a.feature_contributions


def test_high_inflammation_signal_drives_neuroinflammation_score() -> None:
    profile = PatientProfile(
        patient_id="syn-high",
        age=61,
        sex=Sex.FEMALE,
        baseline_neuroinflammation_risk=0.9,
        baseline_seizure_susceptibility=0.1,
        baseline_mood_instability=0.2,
        genomic_markers={},
        proteomic_markers={},
    )
    biomarkers = BiomarkerSnapshot(
        step=0,
        inflammation=0.95,
        stress=0.2,
        sleep_recovery=0.8,
        hrv=0.8,
        glutamate_proxy=0.3,
        gaba_proxy=0.7,
        serotonin_proxy=0.7,
    )
    window = NeuralSignalWindow(
        sample_rate_hz=128,
        timestamps=[0.0, 0.01, 0.02],
        channel_names=["C3"],
        signals=[[0.0, 0.1, 0.0]],
        band_powers={"theta": 0.05, "alpha": 0.1, "beta": 0.35, "gamma": 0.45},
        artifact_score=0.05,
    )

    state = StateInferenceEngine().infer(profile, biomarkers, window)

    assert state.neuroinflammation > state.mood_instability
    assert state.neuroinflammation > state.seizure_risk
    assert state.dominant_state == "neuroinflammation"
