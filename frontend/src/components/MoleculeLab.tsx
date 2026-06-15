import { useEffect, useState } from "react";
import { evaluateMolecule, getTargets, type TargetInfo } from "../api/client";
import type { Candidate } from "../types";
import MoleculeCard from "./MoleculeCard";
import ValidationPanel from "./ValidationPanel";
import Molecule3D from "./Molecule3D";

const EXAMPLES = [
  "CC(=O)Oc1ccccc1C(=O)O",
  "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
  "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
];

export default function MoleculeLab() {
  const [targets, setTargets] = useState<TargetInfo[]>([]);
  const [targetId, setTargetId] = useState("TNF_alpha");
  const [smiles, setSmiles] = useState(EXAMPLES[0]);
  const [candidate, setCandidate] = useState<Candidate | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    getTargets()
      .then(setTargets)
      .catch((e) => setError(String(e)));
  }, []);

  async function evaluate() {
    setBusy(true);
    setError(null);
    try {
      setCandidate(await evaluateMolecule(smiles, targetId));
    } catch (e) {
      setError("Could not evaluate (invalid SMILES?)");
      setCandidate(null);
      void e;
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <div className="card">
        <h3>Molecule lab — edit &amp; validate any molecule</h3>
        <div className="controls" style={{ marginBottom: 8 }}>
          <label>
            Target&nbsp;
            <select value={targetId} onChange={(e) => setTargetId(e.target.value)}>
              {targets.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.name}
                </option>
              ))}
            </select>
          </label>
          {EXAMPLES.map((ex) => (
            <button key={ex} className="btn" onClick={() => setSmiles(ex)}>
              example
            </button>
          ))}
        </div>
        <input
          className="mono"
          style={{ width: "100%", padding: 8, background: "#0e1117", color: "#cdd3e0", border: "1px solid #232a3a", borderRadius: 6 }}
          value={smiles}
          onChange={(e) => setSmiles(e.target.value)}
          placeholder="enter SMILES, e.g. CCO"
        />
        <div style={{ marginTop: 8 }}>
          <button className="btn" disabled={busy || !smiles} onClick={evaluate}>
            {busy ? "Validating…" : "Validate molecule"}
          </button>
        </div>
        {error && <div className="error" style={{ marginTop: 8 }}>{error}</div>}
      </div>

      {candidate && (
        <div className="grid">
          <MoleculeCard candidate={candidate} selected />
          <ValidationPanel candidate={candidate} />
          <Molecule3D smiles={candidate.smiles} />
        </div>
      )}
    </div>
  );
}
