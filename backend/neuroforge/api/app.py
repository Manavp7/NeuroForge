"""FastAPI service for NeuroForge — the full closed-loop API.

RESEARCH/SIMULATION ONLY — not a medical device. Patient ``latent_state`` (ground truth) is
never exposed; clients only ever see inferred state.
"""

from __future__ import annotations

import json
import random
import tempfile

from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .. import DISCLAIMER, __version__
from ..config import CONDITIONS
from ..data.synthetic import SyntheticPatientGenerator
from ..loop.orchestrator import ClosedLoopController
from ..models import PatientProfile
from ..render.molecule_svg import molecule_to_svg
from ..store import STORE, RunSession

app = FastAPI(
    title="NeuroForge API",
    version=__version__,
    description="Software-only, fully-simulated adaptive closed-loop molecular therapy demo. "
    "RESEARCH/SIMULATION ONLY — not a medical device.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_CONTROLLER: ClosedLoopController | None = None


def controller() -> ClosedLoopController:
    global _CONTROLLER
    if _CONTROLLER is None:
        _CONTROLLER = ClosedLoopController()
    return _CONTROLLER


def public_profile(profile: PatientProfile) -> dict:
    """Serialize a profile WITHOUT the hidden ground-truth latent state."""
    return profile.model_dump(exclude={"latent_state"})


# --------------------------------------------------------------------------- #
class CreatePatientRequest(BaseModel):
    condition: str = "neuroinflammatory"
    seed: int | None = None


class CreateRunRequest(BaseModel):
    patient_id: str
    max_iter: int | None = None


class ApproveRequest(BaseModel):
    candidate_id: str | None = None


# --------------------------------------------------------------------------- #
@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "version": __version__, "disclaimer": DISCLAIMER}


@app.get("/conditions")
def conditions() -> dict:
    return {"conditions": list(CONDITIONS), "disclaimer": DISCLAIMER}


@app.get("/modelcards")
def modelcards() -> dict:
    from ..cards import list_cards

    return {"modelcards": list_cards(), "disclaimer": DISCLAIMER}


@app.get("/modelcards/{card_id}")
def modelcard(card_id: str) -> dict:
    from ..cards import get_card

    content = get_card(card_id)
    if content is None:
        raise HTTPException(404, "model card not found")
    return {"id": card_id, "content": content, "disclaimer": DISCLAIMER}


@app.get("/rag")
def rag(query: str = Query(...), k: int = 2) -> dict:
    from ..agent.rag import cite

    return {"query": query, "citations": cite(query, k=k), "disclaimer": DISCLAIMER}


@app.post("/patients")
def create_patient(req: CreatePatientRequest) -> dict:
    if req.condition not in CONDITIONS:
        raise HTTPException(400, f"Unknown condition {req.condition!r}")
    seed = req.seed if req.seed is not None else random.randint(1, 2**31 - 1)
    profile = SyntheticPatientGenerator(seed=seed).generate(req.condition)
    STORE.add_patient(profile)
    return {"patient": public_profile(profile), "disclaimer": DISCLAIMER}


@app.get("/patients/{pid}")
def get_patient(pid: str) -> dict:
    profile = STORE.get_patient(pid)
    if profile is None:
        raise HTTPException(404, "patient not found")
    return {"patient": public_profile(profile), "disclaimer": DISCLAIMER}


@app.get("/patients/{pid}/state")
def get_patient_state(pid: str) -> dict:
    profile = STORE.get_patient(pid)
    if profile is None:
        raise HTTPException(404, "patient not found")
    state = controller().infer(profile)
    return {"state": state.model_dump(), "disclaimer": DISCLAIMER}


@app.get("/patients/{pid}/explain")
def patient_explain(pid: str, method: str = "auto") -> dict:
    """Per-construct feature attributions for the inferred state (SHAP or occlusion fallback)."""
    from ..explain.shap_explain import explain_state

    profile = STORE.get_patient(pid)
    if profile is None:
        raise HTTPException(404, "patient not found")
    result = explain_state(controller().estimator, profile, method=method)
    return {**result, "disclaimer": DISCLAIMER}


@app.post("/patients/{pid}/combination")
def patient_combination(pid: str, max_targets: int = 2, threshold: float = 0.4) -> dict:
    """Propose a polypharmacology combination regimen for the patient's current state."""
    from ..loop.combination import design_combination

    profile = STORE.get_patient(pid)
    if profile is None:
        raise HTTPException(404, "patient not found")
    state = controller().infer(profile)
    items = design_combination(
        state, seed=controller().seed, max_targets=max_targets, threshold=threshold
    )
    serialized = [
        {
            "target": it["target"].model_dump(),
            "candidate": it["candidate"].model_dump() if it["candidate"] else None,
        }
        for it in items
    ]
    return {"state": state.model_dump(), "combination": serialized, "disclaimer": DISCLAIMER}


@app.post("/runs")
def create_run(req: CreateRunRequest) -> dict:
    profile = STORE.get_patient(req.patient_id)
    if profile is None:
        raise HTTPException(404, "patient not found")
    session = RunSession(profile, controller(), max_iter=req.max_iter)
    STORE.add_session(session)
    return {"run_id": session.id, "status": session.run.status, "disclaimer": DISCLAIMER}


@app.get("/runs")
def list_runs() -> dict:
    return {"runs": STORE.list_runs(), "disclaimer": DISCLAIMER}


@app.get("/patients")
def list_patients() -> dict:
    return {"patients": STORE.list_patients(), "disclaimer": DISCLAIMER}


@app.get("/runs/{rid}")
def get_run(rid: str) -> dict:
    snapshot = STORE.get_run_snapshot(rid)
    if snapshot is None:
        raise HTTPException(404, "run not found")
    return {"run": snapshot, "disclaimer": DISCLAIMER}


@app.post("/runs/{rid}/step")
def step_run(rid: str) -> dict:
    session = STORE.get_session(rid)
    if session is None:
        raise HTTPException(404, "run not found")
    result = session.step()
    STORE.persist_session(session)
    payload = {k: (v.model_dump() if hasattr(v, "model_dump") else v) for k, v in result.items()}
    payload["disclaimer"] = DISCLAIMER
    return payload


@app.post("/runs/{rid}/approve")
def approve_run(rid: str, req: ApproveRequest) -> dict:
    return _decide(rid, True, req.candidate_id)


@app.post("/runs/{rid}/reject")
def reject_run(rid: str) -> dict:
    return _decide(rid, False, None)


def _decide(rid: str, approved: bool, candidate_id: str | None) -> dict:
    session = STORE.get_session(rid)
    if session is None:
        raise HTTPException(404, "run not found")
    if session.pending is None:
        raise HTTPException(409, "no pending iteration to decide on")
    result = session.decide(approved, candidate_id)
    if STORE.db is not None:
        it = result.get("iteration")
        cid = candidate_id or (it.chosen.id if it is not None and it.chosen else None)
        STORE.db.append_audit(
            rid,
            actor="clinician",
            action="approve" if approved else "reject",
            candidate_id=cid,
            detail=f"decision on run {rid}",
        )
    STORE.persist_session(session)
    payload = {k: (v.model_dump() if hasattr(v, "model_dump") else v) for k, v in result.items()}
    payload["disclaimer"] = DISCLAIMER
    return payload


@app.get("/runs/{rid}/stream")
def stream_run(rid: str) -> StreamingResponse:
    """Autonomously drive the loop (auto-approving safe candidates) and stream events as SSE."""
    session = STORE.get_session(rid)
    if session is None:
        raise HTTPException(404, "run not found")

    def gen():
        emitted = 0
        terminal = {"stabilized", "exhausted", "rejected"}
        for _ in range(session.max_iter * 2 + 2):
            result = session.step()
            while emitted < len(session.run.events):
                ev = session.run.events[emitted]
                emitted += 1
                yield f"data: {ev.model_dump_json()}\n\n"
            if session.run.status in terminal:
                break
            if result.get("status") == "awaiting_approval" and session.pending is not None:
                approved = session.pending.chosen is not None and session.pending.chosen.safe
                session.decide(approved)
                while emitted < len(session.run.events):
                    ev = session.run.events[emitted]
                    emitted += 1
                    yield f"data: {ev.model_dump_json()}\n\n"
        STORE.persist_session(session)
        yield f"data: {json.dumps({'phase': 'eof', 'status': session.run.status})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/runs/{rid}/audit")
def run_audit(rid: str) -> dict:
    if STORE.db is None:
        return {"audit": [], "verified": True, "disclaimer": DISCLAIMER}
    return {
        "audit": STORE.db.list_audit(rid),
        "verified": STORE.db.verify_audit(),
        "disclaimer": DISCLAIMER,
    }


@app.get("/eeg/available")
def eeg_available() -> dict:
    from ..data.eeg_io import mne_available

    return {"mne_available": mne_available(), "disclaimer": DISCLAIMER}


@app.post("/eeg/features")
async def eeg_features(file: UploadFile = File(...)) -> dict:
    """Upload an EDF/BDF recording and extract band-power EEG features (requires MNE)."""
    from ..data.eeg_io import features_from_edf

    suffix = "." + (file.filename or "rec.edf").split(".")[-1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(await file.read())
        tmp.flush()
        try:
            feats = features_from_edf(tmp.name)
        except RuntimeError as exc:
            raise HTTPException(501, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(422, f"could not parse EEG: {exc}") from exc
    return {"eeg": feats.model_dump(), "disclaimer": DISCLAIMER}


@app.get("/molecule/svg")
def molecule_svg(smiles: str = Query(...)) -> Response:
    return Response(content=molecule_to_svg(smiles), media_type="image/svg+xml")


@app.get("/molecule/molblock")
def molecule_molblock(smiles: str = Query(...), seed: int = 7) -> dict:
    from ..chem3d import mol_to_molblock, shape_profile

    block = mol_to_molblock(smiles, seed=seed)
    if block is None:
        raise HTTPException(422, "could not embed 3D conformer")
    return {"molblock": block, "shape": shape_profile(smiles, seed=seed), "disclaimer": DISCLAIMER}
