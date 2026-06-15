import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import Disclaimer from "../components/Disclaimer";
import ApprovalControls from "../components/ApprovalControls";
import MoleculeCard from "../components/MoleculeCard";
import ValidationPanel from "../components/ValidationPanel";
import type { Candidate } from "../types";

const candidate: Candidate = {
  id: "cand-1",
  smiles: "CCO",
  admet: {
    mol_weight: 46.07,
    logp: -0.0014,
    tpsa: 20.23,
    hbd: 1,
    hba: 1,
    rotatable_bonds: 0,
    qed: 0.41,
    sa_score: 1.5,
    lipinski_violations: 0,
    tox_flags: [],
  },
  binding: { value: 6.2, std: 0.3 },
  score: 0.55,
  safe: true,
  safety_notes: [],
  rationale: "test rationale",
};

describe("components", () => {
  it("renders disclaimer text", () => {
    render(<Disclaimer />);
    expect(screen.getByRole("alert").textContent).toMatch(/NOT a medical device/i);
  });

  it("approval buttons fire callbacks and respect disabled", () => {
    const onApprove = vi.fn();
    const onReject = vi.fn();
    const { rerender } = render(
      <ApprovalControls disabled={false} onApprove={onApprove} onReject={onReject} />,
    );
    fireEvent.click(screen.getByText(/Approve/));
    fireEvent.click(screen.getByText(/Reject/));
    expect(onApprove).toHaveBeenCalledOnce();
    expect(onReject).toHaveBeenCalledOnce();

    rerender(<ApprovalControls disabled onApprove={onApprove} onReject={onReject} />);
    expect(screen.getByText(/Approve/)).toBeDisabled();
  });

  it("molecule card shows score + selects on click", () => {
    const onSelect = vi.fn();
    render(<MoleculeCard candidate={candidate} onSelect={onSelect} />);
    expect(screen.getByText(/score 0.55/)).toBeInTheDocument();
    fireEvent.click(screen.getByText(/score 0.55/));
    expect(onSelect).toHaveBeenCalledWith(candidate);
  });

  it("validation panel lists admet rows", () => {
    render(<ValidationPanel candidate={candidate} />);
    expect(screen.getByText("Molecular weight")).toBeInTheDocument();
    expect(screen.getByText("test rationale")).toBeInTheDocument();
  });
});
