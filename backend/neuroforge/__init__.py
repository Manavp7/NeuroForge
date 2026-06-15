"""NeuroForge — software-only, fully-simulated adaptive closed-loop molecular therapy demo.

.. warning::
    **RESEARCH / SIMULATION ONLY.** NeuroForge uses synthetic data and surrogate models.
    It is **not a medical device**, is **not validated**, and must **not** be used for any
    clinical, diagnostic, or treatment decision. Nothing here designs real drugs or guides
    real therapy.
"""

from __future__ import annotations

__version__ = "0.1.0"

DISCLAIMER = (
    "RESEARCH/SIMULATION ONLY — NeuroForge uses synthetic data and surrogate models. "
    "It is NOT a medical device, is NOT validated, and must NOT be used for any clinical, "
    "diagnostic, or treatment decision."
)

__all__ = ["__version__", "DISCLAIMER"]
