import { afterEach, describe, expect, it, vi } from "vitest";
import {
  bestSafeCandidate,
  createPatient,
  getConditions,
  moleculeSvgUrl,
} from "../api/client";
import type { Candidate, Iteration } from "../types";

function mockFetchOnce(body: unknown, ok = true, status = 200) {
  vi.spyOn(globalThis, "fetch").mockResolvedValueOnce({
    ok,
    status,
    json: async () => body,
  } as Response);
}

afterEach(() => vi.restoreAllMocks());

describe("api client", () => {
  it("parses conditions", async () => {
    mockFetchOnce({ conditions: ["a", "b"] });
    expect(await getConditions()).toEqual(["a", "b"]);
  });

  it("unwraps created patient", async () => {
    mockFetchOnce({ patient: { id: "p1", condition: "x" } });
    const p = await createPatient("x");
    expect(p.id).toBe("p1");
  });

  it("throws on non-ok response", async () => {
    mockFetchOnce({}, false, 500);
    await expect(getConditions()).rejects.toThrow();
  });

  it("builds molecule svg url with encoded smiles", () => {
    const url = moleculeSvgUrl("C#N");
    expect(url).toContain("smiles=C%23N");
  });
});

describe("bestSafeCandidate", () => {
  const mk = (id: string, score: number, safe: boolean): Candidate => ({
    id,
    smiles: "C",
    admet: {
      mol_weight: 0,
      logp: 0,
      tpsa: 0,
      hbd: 0,
      hba: 0,
      rotatable_bonds: 0,
      qed: 0,
      sa_score: 0,
      lipinski_violations: 0,
      tox_flags: [],
    },
    binding: { value: 0, std: 0 },
    score,
    safe,
    safety_notes: [],
    rationale: "",
  });

  it("returns highest-scoring safe candidate", () => {
    const it = {
      candidates: [mk("a", 0.5, true), mk("b", 0.9, false), mk("c", 0.7, true)],
    } as unknown as Iteration;
    expect(bestSafeCandidate(it)?.id).toBe("c");
  });

  it("returns null if none safe", () => {
    const it = { candidates: [mk("b", 0.9, false)] } as unknown as Iteration;
    expect(bestSafeCandidate(it)).toBeNull();
  });
});
