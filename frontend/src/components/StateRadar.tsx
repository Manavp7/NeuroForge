import Plot from "./Plot";
import type { PatientState } from "../types";

export default function StateRadar({ state }: { state: PatientState }) {
  const constructs = Object.keys(state.constructs);
  const values = constructs.map((c) => state.constructs[c].value);
  const labels = constructs.map((c) => c.replace(/_/g, " "));
  return (
    <div className="card">
      <h3>Inferred patient state (confidence {state.confidence.toFixed(2)})</h3>
      <Plot
        data={[
          {
            type: "scatterpolar",
            r: [...values, values[0]],
            theta: [...labels, labels[0]],
            fill: "toself",
            line: { color: "#ff7eb6" },
          },
        ]}
        layout={{
          height: 280,
          margin: { l: 40, r: 40, t: 20, b: 20 },
          paper_bgcolor: "transparent",
          font: { color: "#cdd3e0", size: 10 },
          polar: { radialaxis: { range: [0, 1.2], visible: true }, bgcolor: "transparent" },
          showlegend: false,
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%" }}
      />
    </div>
  );
}
