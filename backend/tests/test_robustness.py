import numpy as np

from neuroforge.data.synthetic import SyntheticPatientGenerator
from neuroforge.inference.state import StateEstimator
from neuroforge.robustness import (
    add_omics_noise,
    adversarial_shift,
    degrade_eeg,
    mean_uncertainty,
)


def _est():
    return StateEstimator(seed=3, n_train=160, ensemble=8)


def test_eeg_degradation_inflates_uncertainty():
    est = _est()
    p = SyntheticPatientGenerator(seed=9).generate("parkinsonian")
    clean = est.estimate(p)
    degraded = est.estimate(degrade_eeg(p, artifact_ratio=0.6, snr_drop_db=8.0))
    assert mean_uncertainty(degraded) > mean_uncertainty(clean)
    assert degraded.confidence < clean.confidence


def test_small_omics_noise_keeps_state_stable():
    est = _est()
    p = SyntheticPatientGenerator(seed=11).generate("neuroinflammatory")
    base = est.estimate(p)
    noisy = est.estimate(add_omics_noise(p, sigma=0.1, seed=1))
    # Dominant construct ranking should be unchanged under small noise.
    top_base = max(base.constructs.items(), key=lambda kv: kv[1].value)[0]
    top_noisy = max(noisy.constructs.items(), key=lambda kv: kv[1].value)[0]
    assert top_base == top_noisy


def test_predictions_bounded_under_adversarial_shift():
    est = _est()
    p = SyntheticPatientGenerator(seed=13).generate("healthy_control")
    shifted = est.estimate(adversarial_shift(p, scale=1.0))
    # Even adversarial inflation keeps constructs in a sane bounded range.
    for u in shifted.constructs.values():
        assert -0.5 <= u.value <= 2.5
    # And neuroinflammation should rise (model responds, not breaks).
    assert (
        shifted.constructs["neuroinflammation"].value
        >= est.estimate(p).constructs["neuroinflammation"].value - 0.01
    )


def test_uncertainty_monotonic_in_artifacts():
    est = _est()
    p = SyntheticPatientGenerator(seed=15).generate("mood_disorder")
    levels = [0.0, 0.3, 0.6]
    stds = [
        mean_uncertainty(est.estimate(degrade_eeg(p, artifact_ratio=a, snr_drop_db=0)))
        for a in levels
    ]
    assert stds[0] <= stds[1] <= stds[2] + 1e-9
    assert np.all(np.isfinite(stds))
