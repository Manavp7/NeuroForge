import numpy as np
import pytest

from neuroforge.data.synthetic import SyntheticPatientGenerator
from neuroforge.explain.explain import summarize_state, top_factors
from neuroforge.inference.state import StateEstimator


@pytest.fixture(scope="module")
def estimator():
    return StateEstimator(seed=3, n_train=180, ensemble=10)


def test_recovers_dominant_construct(estimator):
    gen = SyntheticPatientGenerator(seed=55)
    # Inflammatory patients should score highest on neuroinflammation.
    correct = 0
    for _ in range(10):
        p = gen.generate("neuroinflammatory")
        st = estimator.estimate(p)
        top = max(st.constructs.items(), key=lambda kv: kv[1].value)[0]
        if top == "neuroinflammation":
            correct += 1
    assert correct >= 7  # robust majority


def test_uncertainty_increases_with_artifacts(estimator):
    gen = SyntheticPatientGenerator(seed=7)
    p = gen.generate("parkinsonian")
    clean = estimator.estimate(p)
    # Corrupt the EEG quality and re-estimate.
    p.eeg.artifact_ratio = 0.6
    p.eeg.snr_db = -3.0
    noisy = estimator.estimate(p)
    clean_std = np.mean([u.std for u in clean.constructs.values()])
    noisy_std = np.mean([u.std for u in noisy.constructs.values()])
    assert noisy_std > clean_std
    assert noisy.confidence < clean.confidence


def test_explanations_present(estimator):
    gen = SyntheticPatientGenerator(seed=8)
    st = estimator.estimate(gen.generate("mood_disorder"))
    assert st.explanations
    factors = top_factors(st, "serotonergic_deficit")
    assert isinstance(factors, list) and factors
    assert isinstance(summarize_state(st), str)


def test_healthy_low_abnormality(estimator):
    gen = SyntheticPatientGenerator(seed=99)
    abns = [estimator.estimate(gen.generate("healthy_control")).abnormality() for _ in range(8)]
    infl = [estimator.estimate(gen.generate("neuroinflammatory")).abnormality() for _ in range(8)]
    assert np.mean(abns) < np.mean(infl)
