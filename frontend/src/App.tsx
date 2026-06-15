import { useEffect, useState } from "react";
import {
  approveRun,
  bestSafeCandidate,
  createPatient,
  createRun,
  getConditions,
  getPatientState,
  getRun,
  rejectRun,
  stepRun,
  streamRun,
} from "./api/client";
import type { Candidate, Iteration, LoopEvent, PatientProfile, PatientState } from "./types";
import Disclaimer from "./components/Disclaimer";
import PatientPanel from "./components/PatientPanel";
import EEGChart from "./components/EEGChart";
import StateRadar from "./components/StateRadar";
import MoleculeCard from "./components/MoleculeCard";
import ValidationPanel from "./components/ValidationPanel";
import LoopTimeline from "./components/LoopTimeline";
import ApprovalControls from "./components/ApprovalControls";

export default function App() {
  const [conditions, setConditions] = useState<string[]>([]);
  const [condition, setCondition] = useState("neuroinflammatory");
  const [patient, setPatient] = useState<PatientProfile | null>(null);
  const [state, setState] = useState<PatientState | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [events, setEvents] = useState<LoopEvent[]>([]);
  const [pending, setPending] = useState<Iteration | null>(null);
  const [selected, setSelected] = useState<Candidate | null>(null);
  const [status, setStatus] = useState<string>("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getConditions()
      .then(setConditions)
      .catch((e) => setError(String(e)));
  }, []);

  async function guard<T>(fn: () => Promise<T>): Promise<T | undefined> {
    setBusy(true);
    setError(null);
    try {
      return await fn();
    } catch (e) {
      setError(String(e));
      return undefined;
    } finally {
      setBusy(false);
    }
  }

  async function onCreatePatient() {
    await guard(async () => {
      const p = await createPatient(condition);
      setPatient(p);
      setState(await getPatientState(p.id));
      setRunId(null);
      setEvents([]);
      setPending(null);
      setSelected(null);
      setStatus("");
    });
  }

  async function refreshRun(id: string) {
    const run = await getRun(id);
    setEvents([...run.events]);
    setStatus(run.status);
  }

  async function onStartRun() {
    if (!patient) return;
    await guard(async () => {
      const id = await createRun(patient.id, 6);
      setRunId(id);
      setEvents([]);
      setPending(null);
      await onStep(id);
    });
  }

  async function onStep(id: string) {
    const res = await stepRun(id);
    await refreshRun(id);
    if (res.status === "awaiting_approval" && res.pending) {
      setPending(res.pending);
      setSelected(bestSafeCandidate(res.pending));
    } else {
      setPending(null);
    }
    setStatus(res.status);
  }

  async function onApprove() {
    if (!runId || !pending) return;
    await guard(async () => {
      await approveRun(runId, selected?.id);
      setPending(null);
      await onStep(runId);
    });
  }

  async function onReject() {
    if (!runId || !pending) return;
    await guard(async () => {
      await rejectRun(runId);
      setPending(null);
      await onStep(runId);
    });
  }

  function onAutoRun() {
    if (!patient) return;
    void guard(async () => {
      const id = await createRun(patient.id, 6);
      setRunId(id);
      setEvents([]);
      setPending(null);
      setStatus("running");
      const collected: LoopEvent[] = [];
      streamRun(
        id,
        (ev) => {
          collected.push(ev);
          setEvents([...collected]);
        },
        () => void refreshRun(id),
      );
    });
  }

  const terminal = ["stabilized", "exhausted", "rejected"].includes(status);

  return (
    <div className="app">
      <Disclaimer />
      <header>
        <h1>NeuroForge</h1>
        <p className="muted">
          Adaptive closed-loop molecular therapy — fully-simulated research demo
        </p>
      </header>

      <div className="controls card">
        <label>
          Condition&nbsp;
          <select value={condition} onChange={(e) => setCondition(e.target.value)}>
            {conditions.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </label>
        <button className="btn" disabled={busy} onClick={onCreatePatient}>
          Generate patient
        </button>
        <button className="btn" disabled={busy || !patient} onClick={onStartRun}>
          Start run (manual)
        </button>
        <button className="btn" disabled={busy || !patient} onClick={onAutoRun}>
          ▶ Auto-run (stream)
        </button>
        {status && <span className={`status status-${status}`}>status: {status}</span>}
      </div>

      {error && <div className="error">{error}</div>}

      {patient && state && (
        <div className="grid">
          <PatientPanel patient={patient} />
          <EEGChart eeg={patient.eeg} />
          <StateRadar state={state} />
        </div>
      )}

      {pending && pending.target && (
        <div className="card">
          <h3>
            Iteration {pending.index}: targeting {pending.target.target_name}
          </h3>
          <p className="small muted">{pending.target.rationale}</p>
          <div className="molecule-grid">
            {pending.candidates.map((c) => (
              <MoleculeCard
                key={c.id}
                candidate={c}
                selected={selected?.id === c.id}
                onSelect={setSelected}
              />
            ))}
          </div>
          <ApprovalControls
            disabled={busy || !selected || !selected.safe}
            onApprove={onApprove}
            onReject={onReject}
          />
        </div>
      )}

      <div className="grid">
        {selected && <ValidationPanel candidate={selected} />}
        {(runId || events.length > 0) && <LoopTimeline events={events} />}
      </div>

      {terminal && (
        <div className="card">
          <h3>Run complete — {status}</h3>
          <p className="muted small">Generate a new patient to run another adaptive loop.</p>
        </div>
      )}
    </div>
  );
}
