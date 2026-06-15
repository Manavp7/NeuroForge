import { moleculeSvgUrl } from "../api/client";
import type { Candidate } from "../types";

export default function MoleculeCard({
  candidate,
  selected,
  onSelect,
}: {
  candidate: Candidate;
  selected?: boolean;
  onSelect?: (c: Candidate) => void;
}) {
  return (
    <div
      className={`molecule-card ${selected ? "selected" : ""} ${candidate.safe ? "" : "unsafe"}`}
      onClick={() => onSelect?.(candidate)}
    >
      <img src={moleculeSvgUrl(candidate.smiles)} alt={candidate.smiles} loading="lazy" />
      <div className="mol-meta">
        <div className="mono small ellipsis" title={candidate.smiles}>
          {candidate.smiles}
        </div>
        <div className="badges">
          <span className="badge">score {candidate.score.toFixed(2)}</span>
          <span className="badge">
            pKi {candidate.binding.value.toFixed(1)}±{candidate.binding.std.toFixed(1)}
          </span>
          <span className="badge">QED {candidate.admet.qed.toFixed(2)}</span>
          <span className={`badge ${candidate.safe ? "ok" : "danger"}`}>
            {candidate.safe ? "safe" : "flagged"}
          </span>
        </div>
        {!candidate.safe && candidate.safety_notes.length > 0 && (
          <div className="small danger">{candidate.safety_notes.join("; ")}</div>
        )}
      </div>
    </div>
  );
}
