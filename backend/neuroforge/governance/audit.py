"""Thin governance wrapper around the tamper-evident audit log in :mod:`neuroforge.persistence`.

The audit log is a hash chain: each record's hash covers the previous hash, so any tampering
breaks the chain (detectable via :func:`verify`). Decisions (approve/reject) are recorded by the
API; this module exposes a stable governance-facing surface.
"""

from __future__ import annotations

from ..persistence import Database


def record_decision(
    db: Database, run_id: str, approved: bool, candidate_id: str | None, actor: str = "clinician"
) -> str:
    return db.append_audit(
        run_id,
        actor=actor,
        action="approve" if approved else "reject",
        candidate_id=candidate_id,
        detail=f"decision on run {run_id}",
    )


def history(db: Database, run_id: str | None = None) -> list[dict]:
    return db.list_audit(run_id)


def verify(db: Database) -> bool:
    return db.verify_audit()
