"""Model-agnostic local attributions for the state estimator.

Uses SHAP when installed (`pip install -e ".[explain]"`), otherwise falls back to an
occlusion-based local attribution: replace each feature with its training-mean value and measure
the change in the predicted construct. Both return per-construct ranked (feature, value) factors.
"""

from __future__ import annotations

import numpy as np

from ..config import STATE_CONSTRUCTS
from ..inference.features import feature_vector
from ..inference.state import StateEstimator
from ..models import PatientProfile
from .explain import label


def shap_available() -> bool:
    try:
        import shap  # noqa: F401

        return True
    except Exception:
        return False


def _occlusion(estimator: StateEstimator, x: np.ndarray) -> np.ndarray:
    """Return attribution matrix (n_constructs, n_features) via mean-imputation occlusion."""
    base = estimator.baseline()
    full = estimator.predict_array(x)[0]  # (k,)
    n_features = len(x)
    attribs = np.zeros((len(STATE_CONSTRUCTS), n_features))
    for j in range(n_features):
        x2 = x.copy()
        x2[j] = base[j]
        pred_j = estimator.predict_array(x2)[0]
        attribs[:, j] = full - pred_j  # contribution of feature j
    return attribs


def _shap(estimator: StateEstimator, x: np.ndarray) -> np.ndarray | None:
    try:
        import shap

        background = estimator.baseline()[None, :]
        explainer = shap.Explainer(estimator.predict_array, background)
        values = explainer(x[None, :]).values  # (1, n_features, n_constructs) or (1, n_features)
        arr = np.asarray(values)[0]
        if arr.ndim == 1:  # single output broadcast
            arr = np.tile(arr[:, None], (1, len(STATE_CONSTRUCTS)))
        return arr.T  # -> (n_constructs, n_features)
    except Exception:
        return None


def explain_state(
    estimator: StateEstimator, profile: PatientProfile, method: str = "auto", top_k: int = 5
) -> dict:
    x = feature_vector(profile, estimator.feature_names)
    use_shap = method == "shap" or (method == "auto" and shap_available())
    attribs = _shap(estimator, x) if use_shap else None
    used = "shap"
    if attribs is None:
        attribs = _occlusion(estimator, x)
        used = "occlusion"

    out: dict[str, list[dict]] = {}
    for i, construct in enumerate(STATE_CONSTRUCTS):
        order = np.argsort(np.abs(attribs[i]))[::-1][:top_k]
        out[construct] = [
            {
                "feature": estimator.feature_names[j],
                "label": label(estimator.feature_names[j]),
                "attribution": float(attribs[i, j]),
            }
            for j in order
        ]
    return {"method": used, "factors": out}
