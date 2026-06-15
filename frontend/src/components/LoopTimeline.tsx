import type { LoopEvent } from "../types";

const PHASE_ICON: Record<string, string> = {
  sense: "📡",
  infer: "🧠",
  plan: "🗒️",
  design: "⚗️",
  validate: "🔬",
  critique: "🧐",
  gate: "🩺",
  deliver: "💉",
  monitor: "👀",
  done: "✅",
};

export default function LoopTimeline({ events }: { events: LoopEvent[] }) {
  return (
    <div className="card timeline">
      <h3>Closed-loop timeline</h3>
      <ul>
        {events.map((ev, i) => (
          <li key={i} className={`phase-${ev.phase}`}>
            <span className="phase-icon">{PHASE_ICON[ev.phase] ?? "•"}</span>
            <span className="phase-tag">
              [{ev.iteration}] {ev.phase}
            </span>
            <span className="phase-msg">{ev.message}</span>
          </li>
        ))}
        {events.length === 0 && <li className="muted">No events yet — start a run.</li>}
      </ul>
    </div>
  );
}
