"""Pluggable LLM client for the agentic orchestration layer.

Default is a deterministic :class:`MockLLM` that builds plausible planning / critique / rationale
text from structured inputs — no network, no API key, fully reproducible. :class:`OpenAILLM` is an
optional drop-in used only when ``OPENAI_API_KEY`` is set and the ``openai`` package is installed.
"""

from __future__ import annotations

import os
from typing import Protocol

from ..explain.explain import summarize_state, top_factors
from ..models import Candidate, PatientState, TargetProfile


class LLMClient(Protocol):
    def plan(self, state: PatientState, target: TargetProfile) -> str: ...
    def rationale(self, candidate: Candidate, target: TargetProfile) -> str: ...
    def critique(self, candidates: list[Candidate], target: TargetProfile) -> str: ...


class MockLLM:
    """Deterministic, template-based agent reasoning."""

    name = "mock"

    def plan(self, state: PatientState, target: TargetProfile) -> str:
        summary = summarize_state(state)
        construct = next(iter(target.driving_constructs), "the dominant abnormality")
        drivers = top_factors(state, construct)
        driver_txt = ("; key drivers: " + ", ".join(drivers)) if drivers else ""
        return (
            f"{summary} Plan: design a molecule modulating {target.target_name} to reduce "
            f"{construct.replace('_', ' ')}{driver_txt}. Optimize for drug-likeness and a "
            f"high-confidence predicted binding within safe ADMET limits."
        )

    def rationale(self, candidate: Candidate, target: TargetProfile) -> str:
        verdict = "passes safety screen" if candidate.safe else "FLAGGED by safety screen"
        return (
            f"Candidate {candidate.id} targets {target.target_name}: predicted pKi "
            f"{candidate.binding.value:.2f}±{candidate.binding.std:.2f}, QED "
            f"{candidate.admet.qed:.2f}, {candidate.admet.lipinski_violations} Lipinski violation(s); "
            f"{verdict}."
        )

    def critique(self, candidates: list[Candidate], target: TargetProfile) -> str:
        if not candidates:
            return "No candidates were produced; widen the search or relax constraints."
        safe = [c for c in candidates if c.safe]
        best = max(candidates, key=lambda c: c.score)
        if not safe:
            return (
                f"All {len(candidates)} candidates were flagged by the safety gate; recommend "
                f"re-running design with stricter property windows. Best (unsafe) was {best.id}."
            )
        best_safe = max(safe, key=lambda c: c.score)
        return (
            f"{len(safe)}/{len(candidates)} candidates cleared the safety gate. Recommending "
            f"{best_safe.id} (score {best_safe.score:.2f}, binding {best_safe.binding.value:.2f}"
            f"±{best_safe.binding.std:.2f}) for {target.target_name}."
        )


class OpenAILLM:  # pragma: no cover - exercised only when configured
    """Optional OpenAI-backed client. Falls back to MockLLM phrasing on any error."""

    name = "openai"

    def __init__(self, model: str | None = None):
        from openai import OpenAI  # imported lazily

        self.client = OpenAI()
        self.model = model or os.getenv("NEUROFORGE_OPENAI_MODEL", "gpt-4o-mini")
        self._fallback = MockLLM()

    def _chat(self, system: str, user: str, fallback: str) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.2,
            )
            return resp.choices[0].message.content or fallback
        except Exception:
            return fallback

    def plan(self, state: PatientState, target: TargetProfile) -> str:
        fb = self._fallback.plan(state, target)
        return self._chat(
            "You are a cautious computational-therapeutics planning assistant for a SIMULATION.",
            fb,
            fb,
        )

    def rationale(self, candidate: Candidate, target: TargetProfile) -> str:
        return self._fallback.rationale(candidate, target)

    def critique(self, candidates: list[Candidate], target: TargetProfile) -> str:
        fb = self._fallback.critique(candidates, target)
        return self._chat("You critique molecule candidates in a SIMULATION.", fb, fb)


def get_llm() -> LLMClient:
    """Return OpenAILLM if configured, else the deterministic MockLLM."""
    if os.getenv("OPENAI_API_KEY"):
        try:
            return OpenAILLM()
        except Exception:
            pass
    return MockLLM()
