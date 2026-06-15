"""FastAPI surface for the NeuroForge synthetic simulator."""

from __future__ import annotations

from fastapi import FastAPI

from neuroforge import __version__
from neuroforge.orchestrator import ClosedLoopOrchestrator
from neuroforge.schemas import (
    ClosedLoopIteration,
    SAFETY_DISCLAIMER,
    SimulationRequest,
    SimulationResponse,
)
from neuroforge.validation import ValidationThresholds


def create_app() -> FastAPI:
    """Create a FastAPI app for local NeuroForge simulation."""

    app = FastAPI(
        title="NeuroForge Synthetic Simulator",
        version=__version__,
        description=SAFETY_DISCLAIMER,
    )
    orchestrator = ClosedLoopOrchestrator()
    thresholds = ValidationThresholds()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "version": __version__,
            "mode": "synthetic-research-simulator",
        }

    @app.get("/safety")
    def safety() -> dict[str, object]:
        return {
            "disclaimer": SAFETY_DISCLAIMER,
            "clinical_use": "not_allowed",
            "data_scope": "synthetic_only",
            "thresholds": {
                "min_efficacy": thresholds.min_efficacy,
                "max_toxicity": thresholds.max_toxicity,
                "max_off_target": thresholds.max_off_target,
                "max_uncertainty": thresholds.max_uncertainty,
            },
            "guardrails": [
                "No real patient data ingestion in this MVP.",
                "Candidate generation is restricted to a toy template library.",
                "Simulated delivery requires validation pass and approval gate.",
            ],
        }

    @app.post("/simulate/iteration", response_model=ClosedLoopIteration)
    def simulate_iteration(request: SimulationRequest) -> ClosedLoopIteration:
        return orchestrator.run_iteration(
            seed=request.seed,
            step=request.step,
            doctor_approved=request.doctor_approved,
            require_approval=request.require_approval,
        )

    @app.post("/simulate/session", response_model=SimulationResponse)
    def simulate_session(request: SimulationRequest) -> SimulationResponse:
        iterations = orchestrator.run_session(
            seed=request.seed,
            steps=request.steps,
            doctor_approved=request.doctor_approved,
            require_approval=request.require_approval,
        )
        return SimulationResponse(
            iterations=iterations,
            metadata={
                "seed": request.seed,
                "steps": request.steps,
                "approval_required": request.require_approval,
            },
        )

    return app


app = create_app()
