"""Multimodal patient-state estimation with uncertainty quantification.

The estimator is a small, *transparent* bagged-Ridge ensemble. It is trained at construction
time on synthetic patients (whose latent state is known) so that, at inference, it recovers the
hidden state from observable features alone. Uncertainty comes from two sources:

1. **Epistemic** — disagreement across the bootstrap ensemble.
2. **Signal quality** — EEG artifact ratio / low SNR inflate the reported uncertainty,
   mirroring how a noisy BCI session should reduce confidence.

Explainability is first-class: each construct exposes the top contributing features
(standardized value × averaged ensemble coefficient).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from ..config import CONDITIONS, SETTINGS, STATE_CONSTRUCTS
from ..data.synthetic import SyntheticPatientGenerator
from ..models import PatientProfile, PatientState, Uncertain
from .features import feature_dict, feature_vector


class StateEstimator:
    def __init__(
        self,
        seed: int | None = None,
        n_train: int = 240,
        ensemble: int | None = None,
        alpha: float = 1.0,
    ):
        self.seed = SETTINGS.default_seed if seed is None else seed
        self.ensemble = SETTINGS.inference_ensemble if ensemble is None else ensemble
        self.alpha = alpha
        self.feature_names: list[str] = []
        self.scaler = StandardScaler()
        self.models: list[Ridge] = []  # one multi-output Ridge per bootstrap
        self._fit(n_train)

    # ------------------------------------------------------------------ #
    def _fit(self, n_train: int) -> None:
        gen = SyntheticPatientGenerator(seed=self.seed + 101)
        profiles: list[PatientProfile] = []
        for i in range(n_train):
            cond = CONDITIONS[i % len(CONDITIONS)]
            profiles.append(gen.generate(cond))

        self.feature_names = list(feature_dict(profiles[0]).keys())
        X = np.array([feature_vector(p, self.feature_names) for p in profiles])
        Y = np.array([[p.latent_state[c] for c in STATE_CONSTRUCTS] for p in profiles])

        Xs = self.scaler.fit_transform(X)
        rng = np.random.default_rng(self.seed + 202)
        n = len(Xs)
        for _ in range(self.ensemble):
            idx = rng.integers(0, n, size=n)  # bootstrap resample
            model = Ridge(alpha=self.alpha)
            model.fit(Xs[idx], Y[idx])
            self.models.append(model)

    # ------------------------------------------------------------------ #
    def predict_array(self, X_raw: np.ndarray) -> np.ndarray:
        """Mean ensemble prediction for raw (unscaled) feature rows -> (n, n_constructs)."""
        if X_raw.ndim == 1:
            X_raw = X_raw[None, :]
        Xs = self.scaler.transform(X_raw)
        preds = np.array([m.predict(Xs) for m in self.models])  # (ensemble, n, k)
        return preds.mean(axis=0)

    def baseline(self) -> np.ndarray:
        """Reference feature vector (training means) used as the explanation background."""
        return np.asarray(self.scaler.mean_, dtype=float)

    # ------------------------------------------------------------------ #
    def estimate(self, profile: PatientProfile) -> PatientState:
        x = feature_vector(profile, self.feature_names).reshape(1, -1)
        xs = self.scaler.transform(x)
        preds = np.array([m.predict(xs)[0] for m in self.models])  # (ensemble, n_constructs)
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)

        # Inflate uncertainty for poor signal quality.
        quality_penalty = 1.0 + 2.0 * float(profile.eeg.artifact_ratio)
        if profile.eeg.snr_db < 0:
            quality_penalty *= 1.3
        std = std * quality_penalty

        constructs: dict[str, Uncertain] = {}
        for i, c in enumerate(STATE_CONSTRUCTS):
            constructs[c] = Uncertain(value=float(max(0.0, mean[i])), std=float(std[i]))

        mean_std = float(np.mean(std)) if len(std) else 0.0
        confidence = float(1.0 / (1.0 + 4.0 * mean_std))

        explanations = self._explain(xs[0])
        return PatientState(constructs=constructs, confidence=confidence, explanations=explanations)

    # ------------------------------------------------------------------ #
    def _explain(self, xs_row: np.ndarray, top_k: int = 4) -> dict[str, list[tuple[str, float]]]:
        # Average coefficients across the ensemble: shape (n_constructs, n_features).
        coefs = np.mean([m.coef_ for m in self.models], axis=0)
        out: dict[str, list[tuple[str, float]]] = {}
        for i, c in enumerate(STATE_CONSTRUCTS):
            contrib = coefs[i] * xs_row  # contribution of each standardized feature
            order = np.argsort(np.abs(contrib))[::-1][:top_k]
            out[c] = [(self.feature_names[j], float(contrib[j])) for j in order]
        return out


_DEFAULT: StateEstimator | None = None


def get_default_estimator() -> StateEstimator:
    """Lazily build and cache a default estimator (training is mildly expensive)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = StateEstimator()
    return _DEFAULT
