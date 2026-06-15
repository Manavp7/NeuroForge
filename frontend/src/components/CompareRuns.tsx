import { useEffect, useState } from "react";
import Plot from "./Plot";
import { getRun, listRuns, trajectory, type RunSummary } from "../api/client";
import type { LoopEvent } from "../types";

function abnSeries(events: LoopEvent[]): { x: number[]; y: number[] } {
  const pts = events
    .filter((e) => e.phase === "infer" && typeof e.payload?.abnormality === "number")
    .map((e) => ({ x: e.iteration, y: e.payload.abnormality as number }));
  return { x: pts.map((p) => p.x), y: pts.map((p) => p.y) };
}

export default function CompareRuns() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [series, setSeries] = useState<{ id: string; x: number[]; y: number[] }[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listRuns()
      .then(setRuns)
      .catch((e) => setError(String(e)));
  }, []);

  function toggle(id: string) {
    setSelected((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id]));
  }

  async function compare() {
    setError(null);
    try {
      const out = [];
      for (const id of selected) {
        const run = await getRun(id);
        const s = abnSeries(run.events);
        // Fall back to iteration-derived trajectory if events were not persisted.
        if (s.x.length === 0 && run.iterations?.length) {
          const t = trajectory(run);
          out.push({ id, x: run.iterations.map((_, i) => i), y: t.before });
        } else {
          out.push({ id, ...s });
        }
      }
      setSeries(out);
    } catch (e) {
      setError(String(e));
    }
  }

  return (
    <div>
      <div className="card">
        <h3>Compare runs</h3>
        {runs.length === 0 && <div className="muted small">No runs yet — start some in the Closed Loop tab.</div>}
        <div className="run-list">
          {runs.map((r) => (
            <label key={r.id} className="run-item">
              <input
                type="checkbox"
                checked={selected.includes(r.id)}
                onChange={() => toggle(r.id)}
              />
              <span className="mono small">{r.id.slice(0, 8)}</span>
              <span className={`status status-${r.status}`}>{r.status}</span>
            </label>
          ))}
        </div>
        <button className="btn" disabled={selected.length === 0} onClick={compare}>
          Compare {selected.length > 0 ? `(${selected.length})` : ""}
        </button>
        {error && <div className="error" style={{ marginTop: 8 }}>{error}</div>}
      </div>

      {series.length > 0 && (
        <div className="card">
          <h3>Abnormality trajectories</h3>
          <Plot
            data={series.map((s) => ({
              type: "scatter",
              mode: "lines+markers",
              x: s.x,
              y: s.y,
              name: s.id.slice(0, 8),
            }))}
            layout={{
              height: 300,
              margin: { l: 40, r: 10, t: 10, b: 40 },
              paper_bgcolor: "transparent",
              plot_bgcolor: "transparent",
              font: { color: "#cdd3e0" },
              xaxis: { title: { text: "iteration" }, dtick: 1 },
              yaxis: { title: { text: "abnormality" }, rangemode: "tozero" },
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: "100%" }}
          />
        </div>
      )}
    </div>
  );
}
