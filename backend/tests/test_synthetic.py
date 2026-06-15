import numpy as np

from neuroforge.config import CONDITIONS, STATE_CONSTRUCTS
from neuroforge.data.synthetic import SyntheticPatientGenerator


def test_generate_all_conditions():
    gen = SyntheticPatientGenerator(seed=5)
    for cond in CONDITIONS:
        p = gen.generate(cond)
        assert p.condition == cond
        assert set(p.latent_state) == set(STATE_CONSTRUCTS)
        assert p.proteomics.markers  # non-empty
        assert p.eeg.relative_power


def test_unknown_condition_raises():
    gen = SyntheticPatientGenerator(seed=5)
    try:
        gen.generate("not_a_condition")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_determinism_same_seed():
    a = SyntheticPatientGenerator(seed=11).generate("parkinsonian", patient_id="x")
    b = SyntheticPatientGenerator(seed=11).generate("parkinsonian", patient_id="x")
    assert a.proteomics.markers == b.proteomics.markers
    assert a.latent_state == b.latent_state


def test_condition_signatures_separable():
    """Inflammatory patients should have higher inflammatory markers than healthy controls."""
    gen = SyntheticPatientGenerator(seed=21)
    infl = [gen.generate("neuroinflammatory") for _ in range(12)]
    healthy = [gen.generate("healthy_control") for _ in range(12)]
    infl_il6 = np.mean([p.proteomics.markers["IL6"] for p in infl])
    healthy_il6 = np.mean([p.proteomics.markers["IL6"] for p in healthy])
    assert infl_il6 > healthy_il6

    park = [gen.generate("parkinsonian") for _ in range(12)]
    park_da = np.mean([p.proteomics.markers["dopamine"] for p in park])
    healthy_da = np.mean([p.proteomics.markers["dopamine"] for p in healthy])
    assert park_da < healthy_da  # dopaminergic deficit lowers dopamine
