import { useState } from "react";
import Plot from "./Plot";
import { runCohort, type CohortResult } from "../api/client";

export default function CohortView({ conditions }: { conditions: string[] }) {
  const [condition, setCondition] = useState("neuroinflammatory");
  const [n, setN] = useState(6);
  const [result, setResult] = useState<CohortResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    setBusy(true);
    setError(null);
    try {
      setResult(await runCohort(condition, n));
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
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
        <label>
          Patients&nbsp;
          <input
            type="number"
            min={1}
            max={12}
            value={n}
            onChange={(e) => setN(Number(e.target.value))}
            style={{ width: 60 }}
          />
        </label>
        <button className="btn" disabled={busy} onClick={run}>
          {busy ? "Running…" : "Run cohort"}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      {result && (
        <div className="grid">
          <div className="card">
            <h3>Cohort outcome — {result.summary.condition}</h3>
            <div className="metrics">
              <div>
                <div className="metric-value">{(result.summary.stabilized_rate * 100).toFixed(0)}%</div>
                <div className="muted small">stabilized</div>
              </div>
              <div>
                <div className="metric-value">{result.summary.mean_reduction.toFixed(2)}</div>
                <div className="muted small">mean Δabnormality</div>
              </div>
              <div>
                <div className="metric-value">{result.summary.mean_iterations.toFixed(1)}</div>
                <div className="muted small">mean iterations</div>
              </div>
              <div>
                <div className="metric-value">{result.summary.n}</div>
                <div className="muted small">patients</div>
              </div>
            </div>
          </div>
          <div className="card">
            <h3>Abnormality reduction distribution</h3>
            <Plot
              data={[
                {
                  type: "bar",
                  x: result.patients.map((p) => p.patient_id),
                  y: result.patients.map((p) => p.reduction),
                  marker: {
                    color: result.patients.map((p) => (p.stabilized ? "#3fb950" : "#ff6b6b")),
                  },
                },
              ]}
              layout={{
                height: 260,
                margin: { l: 40, r: 10, t: 10, b: 80 },
                paper_bgcolor: "transparent",
                plot_bgcolor: "transparent",
                font: { color: "#cdd3e0", size: 10 },
                yaxis: { title: { text: "Δabnormality" } },
                xaxis: { tickangle: -45 },
              }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: "100%" }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
