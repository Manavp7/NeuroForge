"""Surrogate binding-affinity predictor with ensemble uncertainty.

This is a *stand-in* for physics-based docking / molecular dynamics (GROMACS/OpenMM) and for
structure prediction (AlphaFold3). It estimates a pseudo-pKi from how closely a molecule's
pharmacophore vector matches a (mock) target, using an ensemble of perturbed scoring functions
to produce epistemic uncertainty. Out-of-distribution molecules (far from the seed library)
get inflated uncertainty.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem

from ..chem import PHARM_DIM, pharmacophore_vector
from ..config import SETTINGS
from ..design.library import SEED_SMILES, TARGETS
from ..models import Uncertain

_LIB_VECTORS = np.array(
    [pharmacophore_vector(m) for m in (Chem.MolFromSmiles(s) for s in SEED_SMILES) if m is not None]
)


class BindingPredictor:
    def __init__(self, target_id: str, seed: int | None = None, ensemble: int | None = None):
        if target_id not in TARGETS:
            raise ValueError(f"Unknown target {target_id!r}")
        self.target = TARGETS[target_id]
        self.ensemble = SETTINGS.binding_ensemble if ensemble is None else ensemble
        rng = np.random.default_rng((SETTINGS.default_seed if seed is None else seed) + 313)
        # Each ensemble member weights the pharmacophore dimensions slightly differently.
        self.weights = 1.0 + 0.18 * rng.standard_normal((self.ensemble, PHARM_DIM))

    def predict(self, mol: Chem.Mol) -> Uncertain:
        return self.predict_vector(pharmacophore_vector(mol))

    def predict_vector(self, vec: np.ndarray) -> Uncertain:
        diff = vec - self.target.ideal_pharmacophore
        # Per-member similarity -> pseudo pKi in ~[4, 9].
        dists = np.linalg.norm(self.weights * diff, axis=1)
        sims = np.exp(-dists)
        affinities = 4.0 + 5.0 * sims
        mean = float(affinities.mean())
        std = float(affinities.std())
        # Out-of-distribution inflation: distance to nearest seed-library molecule.
        ood = float(np.min(np.linalg.norm(_LIB_VECTORS - vec, axis=1)))
        std *= 1.0 + ood
        return Uncertain(value=round(mean, 3), std=round(std, 3))


def make_predictor(target_id: str, seed: int | None = None, kind: str | None = None):
    """Factory selecting the binding model: 'heuristic' (default), 'mlp', or 'torch'."""
    from ..config import SETTINGS

    kind = kind or SETTINGS.binding_model
    if kind == "heuristic":
        return BindingPredictor(target_id, seed=seed)
    # Imported lazily to avoid a hard dependency when unused.
    from .binding_nn import get_nn_predictor

    return get_nn_predictor(target_id, seed=seed, kind=kind)
