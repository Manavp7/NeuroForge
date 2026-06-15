import Plot from "./Plot";
import type { EEGFeatures } from "../types";

export default function EEGChart({ eeg }: { eeg: EEGFeatures }) {
  const bands = Object.keys(eeg.relative_power);
  const values = bands.map((b) => eeg.relative_power[b]);
  return (
    <div className="card">
      <h3>EEG band power (relative)</h3>
      <Plot
        data={[{ type: "bar", x: bands, y: values, marker: { color: "#6c8cff" } }]}
        layout={{
          height: 220,
          margin: { l: 40, r: 10, t: 10, b: 30 },
          paper_bgcolor: "transparent",
          plot_bgcolor: "transparent",
          font: { color: "#cdd3e0" },
          yaxis: { range: [0, Math.max(0.5, ...values) * 1.2] },
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%" }}
      />
      <div className="muted small">
        SNR {eeg.snr_db.toFixed(1)} dB · artifacts {(eeg.artifact_ratio * 100).toFixed(0)}% · FAA{" "}
        {eeg.frontal_alpha_asymmetry.toFixed(2)}
      </div>
    </div>
  );
}
