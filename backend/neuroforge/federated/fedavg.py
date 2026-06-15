"""Federated averaging (FedAvg) for the linear state estimator across simulated sites.

Each "site" (e.g., a hospital) holds its own synthetic patients and never shares raw data — only
model weight updates are averaged on a server. This demonstrates privacy-preserving training:
the federated model approaches centralized performance without pooling patient data. Optional
differential-privacy noise can be added to each site's update (see ``dp_sigma``).

Pure NumPy, no heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import CONDITIONS, STATE_CONSTRUCTS
from ..data.synthetic import SyntheticPatientGenerator
from ..inference.features import feature_dict, feature_vector


def make_site_data(seed: int, n: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate one site's (X, Y, feature_names) from synthetic patients."""
    gen = SyntheticPatientGenerator(seed=seed)
    profiles = [gen.generate(CONDITIONS[i % len(CONDITIONS)]) for i in range(n)]
    names = list(feature_dict(profiles[0]).keys())
    X = np.array([feature_vector(p, names) for p in profiles])
    Y = np.array([[p.latent_state[c] for c in STATE_CONSTRUCTS] for p in profiles])
    return X, Y, names


def _standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def _mse(W: np.ndarray, b: np.ndarray, X: np.ndarray, Y: np.ndarray) -> float:
    return float(np.mean((X @ W + b - Y) ** 2))


def _local_train(
    W: np.ndarray, b: np.ndarray, X: np.ndarray, Y: np.ndarray, epochs: int, lr: float
) -> tuple[np.ndarray, np.ndarray]:
    W, b = W.copy(), b.copy()
    n = len(X)
    for _ in range(epochs):
        err = X @ W + b - Y
        W -= lr * (X.T @ err) / n
        b -= lr * err.mean(axis=0)
    return W, b


@dataclass
class FederatedReport:
    rounds: int
    n_sites: int
    centralized_mse: float
    federated_mse: float
    federated_history: list[float] = field(default_factory=list)
    dp_sigma: float = 0.0


def run_federated(
    n_sites: int = 4,
    per_site: int = 60,
    rounds: int = 25,
    local_epochs: int = 5,
    lr: float = 0.05,
    seed: int = 3,
    dp_sigma: float = 0.0,
    dp_epsilon: float | None = None,
    dp_delta: float = 1e-5,
) -> FederatedReport:
    """Train via FedAvg and compare to a centralized model trained on pooled data.

    If ``dp_epsilon`` is given, the per-update noise ``dp_sigma`` is derived from the Gaussian
    mechanism for (epsilon, delta)-DP.
    """
    if dp_epsilon is not None:
        from ..governance.privacy import gaussian_sigma

        dp_sigma = gaussian_sigma(dp_epsilon, dp_delta, sensitivity=1.0) * 0.01
    sites = [make_site_data(seed + 100 * (i + 1), per_site) for i in range(n_sites)]
    names = sites[0][2]
    Xtest, Ytest, _ = make_site_data(seed + 9999, 80)

    # Public standardization stats (shared, derived from a reference distribution).
    pooled_X = np.vstack([s[0] for s in sites])
    mean = pooled_X.mean(axis=0)
    std = pooled_X.std(axis=0) + 1e-8
    sites_std = [
        (_standardize(X, mean, std), Y, n) for X, Y, n in [(s[0], s[1], len(s[0])) for s in sites]
    ]
    Xtest_s = _standardize(Xtest, mean, std)

    d, k = len(names), len(STATE_CONSTRUCTS)
    rng = np.random.default_rng(seed)

    # ---- Federated ----
    W = np.zeros((d, k))
    b = np.zeros(k)
    history: list[float] = []
    for _ in range(rounds):
        updates = []
        total = 0
        for X, Y, n in sites_std:
            lW, lb = _local_train(W, b, X, Y, local_epochs, lr)
            if dp_sigma > 0:  # differential-privacy noise on the shared update
                lW = lW + rng.normal(0, dp_sigma, size=lW.shape)
                lb = lb + rng.normal(0, dp_sigma, size=lb.shape)
            updates.append((lW * n, lb * n, n))
            total += n
        W = sum(u[0] for u in updates) / total
        b = sum(u[1] for u in updates) / total
        history.append(_mse(W, b, Xtest_s, Ytest))
    fed_mse = history[-1]

    # ---- Centralized (pooled), comparable compute ----
    Xall = np.vstack([s[0] for s in sites_std])
    Yall = np.vstack([s[1] for s in sites_std])
    Wc = np.zeros((d, k))
    bc = np.zeros(k)
    Wc, bc = _local_train(Wc, bc, Xall, Yall, rounds * local_epochs, lr)
    cen_mse = _mse(Wc, bc, Xtest_s, Ytest)

    return FederatedReport(
        rounds=rounds,
        n_sites=n_sites,
        centralized_mse=round(cen_mse, 5),
        federated_mse=round(fed_mse, 5),
        federated_history=[round(h, 5) for h in history],
        dp_sigma=dp_sigma,
    )
