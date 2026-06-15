import { useEffect, useRef, useState } from "react";
import { getMolblock } from "../api/client";

// 3Dmol.js is loaded lazily to keep the main bundle lean.
export default function Molecule3D({ smiles }: { smiles: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let viewer: { clear: () => void } | null = null;
    let cancelled = false;
    (async () => {
      try {
        const molblock = await getMolblock(smiles);
        const $3Dmol = await import("3dmol");
        if (cancelled || !ref.current) return;
        ref.current.innerHTML = "";
        const v = $3Dmol.createViewer(ref.current, { backgroundColor: "#0e1117" });
        viewer = v as unknown as { clear: () => void };
        v.addModel(molblock, "mol");
        v.setStyle({}, { stick: { radius: 0.15 }, sphere: { scale: 0.25 } });
        v.zoomTo();
        v.render();
      } catch (e) {
        if (!cancelled) setError(String(e));
      }
    })();
    return () => {
      cancelled = true;
      try {
        viewer?.clear();
      } catch {
        /* ignore */
      }
    };
  }, [smiles]);

  return (
    <div className="card">
      <h3>3D conformer</h3>
      {error && <div className="small danger">{error}</div>}
      <div
        ref={ref}
        style={{ position: "relative", width: "100%", height: 280, borderRadius: 8 }}
      />
      <div className="small muted">ETKDG + MMFF (illustrative). Drag to rotate.</div>
    </div>
  );
}
