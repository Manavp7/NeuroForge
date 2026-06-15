export interface Uncertain {
  value: number;
  std: number;
}

export interface EEGFeatures {
  relative_power: Record<string, number>;
  frontal_alpha_asymmetry: number;
  snr_db: number;
  artifact_ratio: number;
}

export interface PatientProfile {
  id: string;
  condition: string;
  genomics: { pathway_risk: Record<string, number> };
  proteomics: { markers: Record<string, number> };
  wearables: {
    hrv_ms: number;
    resting_hr: number;
    sleep_efficiency: number;
    activity_index: number;
  };
  labs: { values: Record<string, number> };
  eeg: EEGFeatures;
}

export interface PatientState {
  constructs: Record<string, Uncertain>;
  confidence: number;
  explanations: Record<string, [string, number][]>;
}

export interface ADMET {
  mol_weight: number;
  logp: number;
  tpsa: number;
  hbd: number;
  hba: number;
  rotatable_bonds: number;
  qed: number;
  sa_score: number;
  lipinski_violations: number;
  tox_flags: string[];
}

export interface Candidate {
  id: string;
  smiles: string;
  admet: ADMET;
  binding: Uncertain;
  score: number;
  safe: boolean;
  safety_notes: string[];
  rationale: string;
}

export interface TargetProfile {
  target_id: string;
  target_name: string;
  rationale: string;
  driving_constructs: Record<string, number>;
}

export interface Iteration {
  index: number;
  state: PatientState;
  target: TargetProfile | null;
  candidates: Candidate[];
  chosen: Candidate | null;
  approved: boolean | null;
  abnormality_before: number;
  abnormality_after: number | null;
}

export interface LoopEvent {
  iteration: number;
  phase: string;
  message: string;
  payload: Record<string, unknown>;
}
