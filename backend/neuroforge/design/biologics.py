"""Illustrative biologic / CRISPR-guide design STUBS.

.. warning::
    These are **non-functional placeholders** for demo completeness. They do **not** perform real
    sequence design, real off-target analysis, or any genomics. Never use for any real purpose.
    They are not part of the default closed loop.
"""

from __future__ import annotations

import numpy as np

_AA = "ACDEFGHIKLMNPQRSTVWY"
_NT = "ACGT"


class BiologicDesigner:
    """Toy peptide-sequence proposer (random drug-like-length peptides)."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def propose(self, target_name: str, length: int = 12) -> dict:
        seq = "".join(self.rng.choice(list(_AA), size=length))
        return {
            "type": "peptide",
            "target": target_name,
            "sequence": seq,
            "note": "ILLUSTRATIVE STUB — not a real biologic design.",
        }


class CRISPRGuideDesigner:
    """Toy gRNA proposer with a fake on-/off-target heuristic."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def propose(self, gene: str, n: int = 3) -> list[dict]:
        out = []
        for _ in range(n):
            guide = "".join(self.rng.choice(list(_NT), size=20))
            gc = (guide.count("G") + guide.count("C")) / len(guide)
            on_target = float(np.clip(1.0 - abs(gc - 0.55) * 2.0, 0.0, 1.0))
            off_target = float(self.rng.uniform(0.0, 0.3))
            out.append(
                {
                    "gene": gene,
                    "guide": guide + "NGG",
                    "on_target_score": round(on_target, 3),
                    "off_target_risk": round(off_target, 3),
                    "note": "ILLUSTRATIVE STUB — not a real gRNA design.",
                }
            )
        return out
