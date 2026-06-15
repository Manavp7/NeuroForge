import Plot from "./Plot";
import type { LoopEvent } from "../types";

/** Plot the abnormality index across iterations, parsed from streamed `infer` events. */
export default function TrajectoryChart({ events }: { events: LoopEvent[] }) {
  const points = events
    .filter((e) => e.phase === "infer" && typeof e.payload?.abnormality === "number")
    .map((e) => ({ iter: e.iteration, abn: e.payload.abnormality as number }));

  if (points.length < 1) return null;

  return (
    <div className="card">
      <h3>Abnormality trajectory</h3>
      <Plot
        data={[
          {
            type: "scatter",
            mode: "lines+markers",
            x: points.map((p) => p.iter),
            y: points.map((p) => p.abn),
            line: { color: "#3fb950", width: 2 },
            marker: { size: 8 },
            name: "abnormality",
          },
          {
            type: "scatter",
            mode: "lines",
            x: points.map((p) => p.iter),
            y: points.map(() => 0.35),
            line: { color: "#ff7eb6", dash: "dash", width: 1 },
            name: "stabilization threshold",
          },
        ]}
        layout={{
          height: 240,
          margin: { l: 40, r: 10, t: 10, b: 36 },
          paper_bgcolor: "transparent",
          plot_bgcolor: "transparent",
          font: { color: "#cdd3e0" },
          xaxis: { title: { text: "iteration" }, dtick: 1 },
          yaxis: { title: { text: "abnormality" }, rangemode: "tozero" },
          legend: { orientation: "h", y: -0.3 },
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%" }}
      />
    </div>
  );
}
