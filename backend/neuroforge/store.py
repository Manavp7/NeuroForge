"""In-memory stores + stateful closed-loop session for the API.

A :class:`RunSession` drives the loop one approval gate at a time, so the API can pause for
doctor-in-the-loop decisions. (Persistence beyond process lifetime is out of scope for the MVP.)
"""

from __future__ import annotations

import os
import uuid

from .config import SETTINGS
from .loop.orchestrator import ClosedLoopController
from .models import Iteration, LoopEvent, LoopRun, PatientProfile, PatientState
from .persistence import Database


class RunSession:
    def __init__(
        self,
        profile: PatientProfile,
        controller: ClosedLoopController,
        max_iter: int | None = None,
    ):
        self.id = str(uuid.uuid4())
        self.controller = controller
        self.profile = profile  # current (mutated as therapy is applied)
        self.max_iter = max_iter or SETTINGS.max_iterations
        self.index = 0
        self.run = LoopRun(id=self.id, patient_id=profile.id, status="created")
        self.pending: Iteration | None = None
        self.plan_text: str = ""
        self.critique_text: str = ""

    # ------------------------------------------------------------------ #
    def _emit(self, phase: str, message: str, **payload) -> LoopEvent:
        ev = LoopEvent(iteration=self.index, phase=phase, message=message, payload=payload)
        self.run.events.append(ev)
        return ev

    def current_state(self) -> PatientState:
        return self.controller.infer(self.profile)

    def step(self) -> dict:
        """Advance to the next approval gate (or finish). Returns a status payload."""
        if self.pending is not None:
            return {"status": "awaiting_approval", "pending": self.pending}
        if self.index >= self.max_iter:
            self.run.status = "exhausted"
            return {"status": self.run.status}

        state = self.current_state()
        abn = state.abnormality()
        self._emit("sense", f"Sensed multimodal data; signal confidence {state.confidence:.2f}.")
        self._emit("infer", f"Inferred state; abnormality index {abn:.3f}.", abnormality=abn)

        if abn < SETTINGS.state_target_threshold:
            self.run.status = "stabilized"
            self._emit("done", f"Patient state stabilized (abnormality {abn:.3f}).")
            return {"status": self.run.status}

        iteration, plan_text, critique_text = self.controller.build_iteration(
            self.profile, state, self.index
        )
        self.plan_text, self.critique_text = plan_text, critique_text
        self._emit("plan", plan_text, target=iteration.target.target_id)
        self._emit(
            "design",
            f"Generated {len(iteration.candidates)} candidate(s) for {iteration.target.target_name}.",
        )
        self._emit(
            "validate",
            f"{sum(c.safe for c in iteration.candidates)} candidate(s) cleared the safety gate.",
        )
        self._emit("critique", critique_text)

        self.pending = iteration
        self.run.status = "awaiting_approval"
        return {
            "status": self.run.status,
            "pending": iteration,
            "plan": plan_text,
            "critique": critique_text,
        }

    def decide(self, approved: bool, candidate_id: str | None = None) -> dict:
        """Approve/reject the pending iteration; apply therapy if approved."""
        if self.pending is None:
            return {"status": self.run.status, "error": "no pending iteration"}
        iteration = self.pending

        if approved and candidate_id is not None:
            override = next((c for c in iteration.candidates if c.id == candidate_id), None)
            if override is not None:
                iteration.chosen = override

        chosen = iteration.chosen
        self._emit(
            "gate",
            f"Doctor-in-the-loop {'APPROVED' if approved else 'REJECTED'} "
            f"{chosen.id if chosen else 'candidate'}.",
            candidate_id=chosen.id if chosen else None,
            approved=approved,
        )

        self.profile, abn_after = self.controller.apply_decision(self.profile, iteration, approved)
        self.run.iterations.append(iteration)
        if approved and abn_after is not None:
            self._emit(
                "deliver",
                f"Therapy delivered (sim); abnormality {iteration.abnormality_before:.3f} "
                f"→ {abn_after:.3f}.",
                abnormality_after=abn_after,
            )
        else:
            self._emit("monitor", "Therapy not delivered; continuing observation.")

        self.pending = None
        self.index += 1
        self.run.status = "running"
        return {"status": self.run.status, "iteration": iteration}


class Store:
    def __init__(self, db: Database | None = None):
        self.patients: dict[str, PatientProfile] = {}
        self.sessions: dict[str, RunSession] = {}
        self.db = db

    def add_patient(self, profile: PatientProfile) -> None:
        self.patients[profile.id] = profile
        if self.db is not None:
            self.db.save_patient(profile)

    def get_patient(self, pid: str) -> PatientProfile | None:
        return self.patients.get(pid)

    def add_session(self, session: RunSession) -> None:
        self.sessions[session.id] = session
        self.persist_session(session)

    def get_session(self, sid: str) -> RunSession | None:
        return self.sessions.get(sid)

    def persist_session(self, session: RunSession) -> None:
        if self.db is not None:
            self.db.save_run(session.run)

    def list_runs(self) -> list[dict]:
        return (
            self.db.list_runs()
            if self.db is not None
            else [
                {"id": s.id, "patient_id": s.run.patient_id, "status": s.run.status}
                for s in self.sessions.values()
            ]
        )

    def get_run_snapshot(self, rid: str) -> dict | None:
        session = self.sessions.get(rid)
        if session is not None:
            return session.run.model_dump()
        return self.db.get_run(rid) if self.db is not None else None

    def list_patients(self) -> list[dict]:
        if self.db is not None:
            return self.db.list_patients()
        return [{"id": p.id, "condition": p.condition} for p in self.patients.values()]


def _default_db() -> Database | None:
    path = os.getenv("NEUROFORGE_DB", ":memory:")
    try:
        return Database(path)
    except Exception:
        return None


STORE = Store(db=_default_db())
