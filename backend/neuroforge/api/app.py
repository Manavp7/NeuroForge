"""FastAPI service for NeuroForge. (Expanded in Phase 6 with the full closed-loop API.)"""

from __future__ import annotations

from fastapi import FastAPI

from .. import DISCLAIMER, __version__

app = FastAPI(
    title="NeuroForge API",
    version=__version__,
    description="Software-only, fully-simulated adaptive closed-loop molecular therapy demo. "
    "RESEARCH/SIMULATION ONLY — not a medical device.",
)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "version": __version__, "disclaimer": DISCLAIMER}
