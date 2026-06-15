import type { Candidate } from "../types";

export default function ValidationPanel({ candidate }: { candidate: Candidate }) {
  const a = candidate.admet;
  const rows: [string, string][] = [
    ["Molecular weight", a.mol_weight.toFixed(1)],
    ["logP", a.logp.toFixed(2)],
    ["TPSA", a.tpsa.toFixed(1)],
    ["H-bond donors / acceptors", `${a.hbd} / ${a.hba}`],
    ["Rotatable bonds", `${a.rotatable_bonds}`],
    ["QED (drug-likeness)", a.qed.toFixed(3)],
    ["Synthetic accessibility", a.sa_score.toFixed(2)],
    ["Lipinski violations", `${a.lipinski_violations}`],
    ["Predicted binding (pKi)", `${candidate.binding.value.toFixed(2)} ± ${candidate.binding.std.toFixed(2)}`],
    ["Structural alerts", a.tox_flags.length ? a.tox_flags.join(", ") : "none"],
  ];
  return (
    <div className="card">
      <h3>In-silico validation — {candidate.id}</h3>
      <table className="kv">
        <tbody>
          {rows.map(([k, v]) => (
            <tr key={k}>
              <td className="muted">{k}</td>
              <td className="mono">{v}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {candidate.rationale && <p className="small muted">{candidate.rationale}</p>}
    </div>
  );
}
