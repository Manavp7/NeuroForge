"""Differential privacy utilities (Gaussian mechanism).

Used to add calibrated noise to shared model updates in federated training. The Gaussian
mechanism provides (epsilon, delta)-DP for a query of given L2 sensitivity with
``sigma >= sqrt(2 ln(1.25/delta)) * sensitivity / epsilon``.

This is a teaching implementation, not a certified DP accountant.
"""

from __future__ import annotations

import numpy as np


def gaussian_sigma(epsilon: float, delta: float = 1e-5, sensitivity: float = 1.0) -> float:
    """Noise standard deviation for (epsilon, delta)-DP via the Gaussian mechanism."""
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    return float(np.sqrt(2.0 * np.log(1.25 / delta)) * sensitivity / epsilon)


def add_dp_noise(
    array: np.ndarray, sigma: float, rng: np.random.Generator | None = None
) -> np.ndarray:
    if sigma <= 0:
        return array
    rng = rng or np.random.default_rng()
    return array + rng.normal(0.0, sigma, size=np.shape(array))


def clip_l2(array: np.ndarray, max_norm: float) -> np.ndarray:
    """Clip an update to a maximum L2 norm (bounds sensitivity before adding noise)."""
    norm = float(np.linalg.norm(array))
    if norm > max_norm and norm > 0:
        return array * (max_norm / norm)
    return array
