import type { Candidate, Iteration, LoopEvent, PatientProfile, PatientState } from "../types";

export const API_BASE =
  (import.meta as { env?: Record<string, string> }).env?.VITE_API_BASE ?? "http://localhost:8000";

async function jsonFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "content-type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    throw new Error(`API ${path} failed: ${res.status}`);
  }
  return (await res.json()) as T;
}

export async function getConditions(): Promise<string[]> {
  const data = await jsonFetch<{ conditions: string[] }>("/conditions");
  return data.conditions;
}

export async function createPatient(condition: string, seed?: number): Promise<PatientProfile> {
  const data = await jsonFetch<{ patient: PatientProfile }>("/patients", {
    method: "POST",
    body: JSON.stringify({ condition, seed }),
  });
  return data.patient;
}

export async function getPatientState(patientId: string): Promise<PatientState> {
  const data = await jsonFetch<{ state: PatientState }>(`/patients/${patientId}/state`);
  return data.state;
}

export async function createRun(patientId: string, maxIter?: number): Promise<string> {
  const data = await jsonFetch<{ run_id: string }>("/runs", {
    method: "POST",
    body: JSON.stringify({ patient_id: patientId, max_iter: maxIter }),
  });
  return data.run_id;
}

export interface RunData {
  id: string;
  patient_id: string;
  status: string;
  iterations: Iteration[];
  events: LoopEvent[];
}

export async function getRun(runId: string): Promise<RunData> {
  const data = await jsonFetch<{ run: RunData }>(`/runs/${runId}`);
  return data.run;
}

export interface StepResult {
  status: string;
  pending?: Iteration;
  plan?: string;
  critique?: string;
}

export async function stepRun(runId: string): Promise<StepResult> {
  return jsonFetch<StepResult>(`/runs/${runId}/step`, { method: "POST" });
}

export async function approveRun(runId: string, candidateId?: string): Promise<unknown> {
  return jsonFetch(`/runs/${runId}/approve`, {
    method: "POST",
    body: JSON.stringify({ candidate_id: candidateId }),
  });
}

export async function rejectRun(runId: string): Promise<unknown> {
  return jsonFetch(`/runs/${runId}/reject`, { method: "POST" });
}

export function moleculeSvgUrl(smiles: string): string {
  return `${API_BASE}/molecule/svg?smiles=${encodeURIComponent(smiles)}`;
}

/** Stream loop events via SSE. Returns an unsubscribe function. */
export function streamRun(
  runId: string,
  onEvent: (ev: LoopEvent) => void,
  onDone?: () => void,
): () => void {
  const source = new EventSource(`${API_BASE}/runs/${runId}/stream`);
  source.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);
      if (data.phase === "eof") {
        source.close();
        onDone?.();
        return;
      }
      onEvent(data as LoopEvent);
    } catch {
      /* ignore malformed chunks */
    }
  };
  source.onerror = () => {
    source.close();
    onDone?.();
  };
  return () => source.close();
}

export function bestSafeCandidate(it: Iteration): Candidate | null {
  const safe = it.candidates.filter((c) => c.safe);
  if (safe.length === 0) return null;
  return safe.reduce((a, b) => (b.score > a.score ? b : a));
}
