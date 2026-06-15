"""SQLite persistence for patients, runs, and the governance audit log.

Snapshots are stored as JSON so the schema stays migration-free. The in-memory
:class:`~neuroforge.store.Store` keeps *live* sessions; this layer persists snapshots so run
history survives restarts and can be queried/compared. Thread-safe for FastAPI's worker threads.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time

from .models import LoopRun, PatientProfile

_SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id TEXT PRIMARY KEY,
    condition TEXT,
    profile_json TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    patient_id TEXT,
    status TEXT,
    run_json TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS audit (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    ts REAL NOT NULL,
    actor TEXT,
    action TEXT,
    candidate_id TEXT,
    detail TEXT,
    prev_hash TEXT,
    hash TEXT NOT NULL
);
"""


class Database:
    def __init__(self, path: str = ":memory:"):
        self.path = path
        self._lock = threading.Lock()
        # check_same_thread=False: guarded by our own lock for cross-thread use.
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self._lock:
            self.conn.executescript(_SCHEMA)
            self.conn.commit()

    # ------------------------------------------------------------------ #
    def save_patient(self, profile: PatientProfile) -> None:
        # Persist WITHOUT the hidden ground-truth latent state.
        payload = profile.model_dump(exclude={"latent_state"})
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO patients (id, condition, profile_json, created_at) "
                "VALUES (?, ?, ?, ?)",
                (profile.id, profile.condition, json.dumps(payload), time.time()),
            )
            self.conn.commit()

    def get_patient(self, pid: str) -> dict | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT profile_json FROM patients WHERE id = ?", (pid,)
            ).fetchone()
        return json.loads(row["profile_json"]) if row else None

    def list_patients(self) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, condition, created_at FROM patients ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    def save_run(self, run: LoopRun) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO runs (id, patient_id, status, run_json, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (run.id, run.patient_id, run.status, run.model_dump_json(), time.time()),
            )
            self.conn.commit()

    def get_run(self, rid: str) -> dict | None:
        with self._lock:
            row = self.conn.execute("SELECT run_json FROM runs WHERE id = ?", (rid,)).fetchone()
        return json.loads(row["run_json"]) if row else None

    def list_runs(self) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, patient_id, status, updated_at FROM runs ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    def append_audit(
        self,
        run_id: str,
        actor: str,
        action: str,
        candidate_id: str | None = None,
        detail: str = "",
    ) -> str:
        """Append a tamper-evident audit record (hash-chained). Returns the new hash."""
        ts = time.time()
        with self._lock:
            prev = self.conn.execute("SELECT hash FROM audit ORDER BY seq DESC LIMIT 1").fetchone()
            prev_hash = prev["hash"] if prev else ""
            body = f"{prev_hash}|{run_id}|{ts}|{actor}|{action}|{candidate_id}|{detail}"
            h = hashlib.sha256(body.encode()).hexdigest()
            self.conn.execute(
                "INSERT INTO audit (run_id, ts, actor, action, candidate_id, detail, prev_hash, hash) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, ts, actor, action, candidate_id, detail, prev_hash, h),
            )
            self.conn.commit()
        return h

    def list_audit(self, run_id: str | None = None) -> list[dict]:
        with self._lock:
            if run_id:
                rows = self.conn.execute(
                    "SELECT * FROM audit WHERE run_id = ? ORDER BY seq", (run_id,)
                ).fetchall()
            else:
                rows = self.conn.execute("SELECT * FROM audit ORDER BY seq").fetchall()
        return [dict(r) for r in rows]

    def verify_audit(self) -> bool:
        """Re-derive the hash chain and confirm it is intact."""
        prev_hash = ""
        for r in self.list_audit():
            body = f"{prev_hash}|{r['run_id']}|{r['ts']}|{r['actor']}|{r['action']}|{r['candidate_id']}|{r['detail']}"
            if hashlib.sha256(body.encode()).hexdigest() != r["hash"]:
                return False
            prev_hash = r["hash"]
        return True
