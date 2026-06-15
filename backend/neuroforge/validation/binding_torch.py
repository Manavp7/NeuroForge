"""Optional torch-based binding surrogate (used only when torch is installed).

A tiny MLP with Monte-Carlo dropout for uncertainty, trained to reproduce the analytic teacher.
Selected via ``binding_model="torch"``; otherwise the project never imports torch.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem

from ..chem import PHARM_DIM, pharmacophore_vector
from ..config import SETTINGS
from ..models import Uncertain
from .binding import _LIB_VECTORS
from .binding_nn import _training_set


class TorchBindingPredictor:  # pragma: no cover - exercised only when torch is present
    kind = "torch"

    def __init__(
        self, target_id: str, seed: int | None = None, epochs: int = 300, n_train: int = 600
    ):
        import torch
        import torch.nn as nn

        seed = SETTINGS.default_seed if seed is None else seed
        torch.manual_seed(seed)
        self.torch = torch
        self.target_id = target_id

        X, y = _training_set(target_id, seed, n_train)
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        self.model = nn.Sequential(
            nn.Linear(PHARM_DIM, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-2)
        loss_fn = nn.MSELoss()
        self.model.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss = loss_fn(self.model(Xt), yt)
            loss.backward()
            opt.step()

    def predict(self, mol: Chem.Mol, mc_samples: int = 30) -> Uncertain:
        torch = self.torch
        vec = pharmacophore_vector(mol)
        xt = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
        self.model.train()  # keep dropout active for MC uncertainty
        with torch.no_grad():
            preds = np.array([float(self.model(xt).item()) for _ in range(mc_samples)])
        mean = float(np.clip(preds.mean(), 4.0, 9.0))
        std = float(preds.std())
        ood = float(np.min(np.linalg.norm(_LIB_VECTORS - vec, axis=1)))
        std = (std + 0.05) * (1.0 + ood)
        return Uncertain(value=round(mean, 3), std=round(std, 3))
