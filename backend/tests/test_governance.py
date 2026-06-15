import numpy as np
import pytest

from neuroforge.governance.audit import history, record_decision, verify
from neuroforge.governance.privacy import add_dp_noise, clip_l2, gaussian_sigma
from neuroforge.persistence import Database


def test_gaussian_sigma_scaling():
    # Smaller epsilon (stronger privacy) -> more noise.
    assert gaussian_sigma(0.5) > gaussian_sigma(2.0)
    with pytest.raises(ValueError):
        gaussian_sigma(0.0)


def test_add_dp_noise_and_clip():
    rng = np.random.default_rng(0)
    x = np.ones((5, 3))
    noisy = add_dp_noise(x, sigma=0.1, rng=rng)
    assert noisy.shape == x.shape
    assert not np.allclose(noisy, x)
    # sigma=0 is a no-op.
    assert np.allclose(add_dp_noise(x, 0.0), x)
    clipped = clip_l2(np.array([3.0, 4.0]), max_norm=1.0)
    assert abs(np.linalg.norm(clipped) - 1.0) < 1e-9


def test_audit_via_governance_helpers():
    db = Database(":memory:")
    record_decision(db, "run1", True, "cand-1")
    record_decision(db, "run1", False, "cand-2")
    assert len(history(db, "run1")) == 2
    assert verify(db) is True


def test_dp_epsilon_in_federated():
    from neuroforge.federated.fedavg import run_federated

    rep = run_federated(n_sites=3, per_site=40, rounds=10, seed=3, dp_epsilon=2.0)
    assert rep.dp_sigma > 0.0
    assert rep.federated_mse >= 0.0
