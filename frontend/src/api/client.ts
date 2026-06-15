import type { Candidate, Iteration, LoopEvent, PatientProfile, PatientState } from "../types";

export const API_BASE =
  (import.meta as { env?: Record<string, string> }).env?.VITE_API_BASE ?? "http://localhost:8000";

let authRole = "clinician";
let authToken: string | null = null;

export function setRole(role: string) {
  authRole = role;
}
export function getRole(): string {
  return authRole;
}
export function setToken(token: string | null) {
  authToken = token;
}

async function jsonFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const headers: Record<string, string> = {
    "content-type": "application/json",
    "X-Role": authRole,
    ...((init?.headers as Record<string, string>) ?? {}),
  };
  if (authToken) headers["Authorization"] = `Bearer ${authToken}`;
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers,
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

export interface TargetInfo {
  id: string;
  name: string;
}

export async function getTargets(): Promise<TargetInfo[]> {
  const data = await jsonFetch<{ targets: TargetInfo[] }>("/targets");
  return data.targets;
}

export async function evaluateMolecule(smiles: string, targetId: string): Promise<Candidate> {
  const data = await jsonFetch<{ candidate: Candidate }>("/molecule/evaluate", {
    method: "POST",
    body: JSON.stringify({ smiles, target_id: targetId }),
  });
  return data.candidate;
}

export interface CohortPatient {
  patient_id: string;
  initial_abnormality: number;
  final_abnormality: number;
  reduction: number;
  stabilized: boolean;
  iterations: number;
}

export interface CohortResult {
  summary: {
    condition: string;
    n: number;
    stabilized_rate: number;
    mean_reduction: number;
    std_reduction: number;
    mean_iterations: number;
  };
  patients: CohortPatient[];
}

export async function runCohort(condition: string, n: number): Promise<CohortResult> {
  return jsonFetch<CohortResult>("/cohort", {
    method: "POST",
    body: JSON.stringify({ condition, n }),
  });
}

export interface RunSummary {
  id: string;
  patient_id: string;
  status: string;
}

export async function listRuns(): Promise<RunSummary[]> {
  const data = await jsonFetch<{ runs: RunSummary[] }>("/runs");
  return data.runs;
}

export async function getMolblock(smiles: string): Promise<string> {
  const data = await jsonFetch<{ molblock: string }>(
    `/molecule/molblock?smiles=${encodeURIComponent(smiles)}`,
  );
  return data.molblock;
}

/** Trajectory of abnormality across a run's iterations (for charts). */
export function trajectory(run: { iterations: Iteration[] }): { before: number[]; after: number[] } {
  const before = run.iterations.map((it) => it.abnormality_before);
  const after = run.iterations.map((it) =>
    it.abnormality_after == null ? it.abnormality_before : it.abnormality_after,
  );
  return { before, after };
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
