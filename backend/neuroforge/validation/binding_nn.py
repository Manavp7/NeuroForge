"""Neural binding surrogates with ensemble uncertainty.

A learned alternative to the analytic :class:`~neuroforge.validation.binding.BindingPredictor`.
The sklearn `MLPRegressor` ensemble is always available; an optional torch model is used when
``kind="torch"`` and torch is installed (otherwise it transparently falls back to the MLP).

Both are trained to reproduce the analytic teacher over the seed-library distribution, so they
plug into the same pipeline while demonstrating a "learned surrogate" upgrade path toward real
GNN/AlphaFold-style models. Predictors are cached per (target, seed, kind).
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from sklearn.neural_network import MLPRegressor

from ..chem import PHARM_DIM, pharmacophore_vector
from ..config import SETTINGS
from ..models import Uncertain
from .binding import _LIB_VECTORS, BindingPredictor


def _training_set(target_id: str, seed: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample pharmacophore vectors around the library + label with the analytic teacher."""
    teacher = BindingPredictor(target_id, seed=seed)
    rng = np.random.default_rng(seed + 17)
    base = _LIB_VECTORS
    idx = rng.integers(0, len(base), size=n)
    noise = 0.15 * rng.standard_normal((n, PHARM_DIM))
    X = np.clip(base[idx] + noise, 0.0, None)
    y = np.array([teacher.predict_vector(x).value for x in X])
    return X, y


class MLPBindingPredictor:
    kind = "mlp"

    def __init__(
        self, target_id: str, seed: int | None = None, ensemble: int = 5, n_train: int = 400
    ):
        self.target_id = target_id
        seed = SETTINGS.default_seed if seed is None else seed
        X, y = _training_set(target_id, seed, n_train)
        self.models: list[MLPRegressor] = []
        rng = np.random.default_rng(seed + 91)
        for k in range(ensemble):
            bs = rng.integers(0, len(X), size=len(X))  # bootstrap
            m = MLPRegressor(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                max_iter=400,
                random_state=seed + k,
            )
            m.fit(X[bs], y[bs])
            self.models.append(m)

    def predict(self, mol: Chem.Mol) -> Uncertain:
        vec = pharmacophore_vector(mol).reshape(1, -1)
        preds = np.array([m.predict(vec)[0] for m in self.models])
        mean = float(np.clip(preds.mean(), 4.0, 9.0))
        std = float(preds.std())
        ood = float(np.min(np.linalg.norm(_LIB_VECTORS - vec[0], axis=1)))
        std = (std + 0.05) * (1.0 + ood)  # floor + OOD inflation
        return Uncertain(value=round(mean, 3), std=round(std, 3))


def _make_torch_predictor(target_id: str, seed: int):  # pragma: no cover - optional dep
    try:
        import torch  # noqa: F401
    except Exception:
        return None
    from .binding_torch import TorchBindingPredictor

    return TorchBindingPredictor(target_id, seed=seed)


_CACHE: dict[tuple, object] = {}


def get_nn_predictor(target_id: str, seed: int | None = None, kind: str = "mlp"):
    seed = SETTINGS.default_seed if seed is None else seed
    key = (target_id, seed, kind)
    if key in _CACHE:
        return _CACHE[key]
    predictor: object
    if kind == "torch":
        predictor = _make_torch_predictor(target_id, seed) or MLPBindingPredictor(target_id, seed)
    else:
        predictor = MLPBindingPredictor(target_id, seed)
    _CACHE[key] = predictor
    return predictor
