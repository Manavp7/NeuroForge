import type { PatientProfile } from "../types";

export default function PatientPanel({ patient }: { patient: PatientProfile }) {
  const w = patient.wearables;
  const topMarkers = Object.entries(patient.proteomics.markers)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 4);
  return (
    <div className="card">
      <h3>
        Patient <span className="mono">{patient.id}</span>
      </h3>
      <div className="muted small">condition: {patient.condition}</div>
      <div className="metrics">
        <div>
          <div className="metric-value">{w.hrv_ms.toFixed(0)}</div>
          <div className="muted small">HRV (ms)</div>
        </div>
        <div>
          <div className="metric-value">{w.resting_hr.toFixed(0)}</div>
          <div className="muted small">resting HR</div>
        </div>
        <div>
          <div className="metric-value">{(w.sleep_efficiency * 100).toFixed(0)}%</div>
          <div className="muted small">sleep eff.</div>
        </div>
        <div>
          <div className="metric-value">{w.activity_index.toFixed(2)}</div>
          <div className="muted small">activity</div>
        </div>
      </div>
      <div className="small muted" style={{ marginTop: 8 }}>
        top biomarkers:{" "}
        {topMarkers.map(([k, v]) => `${k} ${v.toFixed(2)}`).join(" · ")}
      </div>
    </div>
  );
}
