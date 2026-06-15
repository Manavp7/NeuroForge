"""Design constraints derived from validation feedback, enabling a critique→redesign pass.

After an initial design+validation round, the agent inspects the results and, if quality is poor
(no safe candidate, or the best score is below a bar), derives tightened constraints (property
windows, synthetic-accessibility ceiling, toxicity exclusion, minimum binding) that steer a second
design pass. This is the "agentic critique → redesign" loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..models import Candidate, TargetProfile


@dataclass
class DesignConstraints:
    property_windows: dict[str, tuple[float, float]] = field(default_factory=dict)
    max_sa: float | None = None
    exclude_tox: bool = True
    min_binding: float | None = None
    note: str = ""


def derive_constraints(candidates: list[Candidate], target: TargetProfile) -> DesignConstraints:
    """Inspect a round of candidates and produce tighter constraints for a redesign."""
    reasons: list[str] = []
    bindings = [c.binding.value for c in candidates] or [6.0]
    safe = [c for c in candidates if c.safe]

    # Raise the binding bar toward the better end of what we just saw.
    min_binding = float(max(6.0, np.percentile(bindings, 60)))
    reasons.append(f"require predicted pKi ≥ {min_binding:.1f}")

    exclude_tox = True
    if any(c.admet.tox_flags for c in candidates):
        reasons.append("exclude structural alerts")

    # If drug-likeness was weak, tighten the QED window.
    windows = dict(target.property_windows)
    if safe:
        mean_qed = float(np.mean([c.admet.qed for c in safe]))
    else:
        mean_qed = float(np.mean([c.admet.qed for c in candidates])) if candidates else 0.0
    if mean_qed < 0.6:
        windows["qed"] = (0.55, 1.0)
        reasons.append("tighten QED ≥ 0.55")

    return DesignConstraints(
        property_windows=windows,
        max_sa=5.0,
        exclude_tox=exclude_tox,
        min_binding=min_binding,
        note="; ".join(reasons),
    )


def passes_constraints(candidate: Candidate, constraints: DesignConstraints) -> bool:
    if constraints.exclude_tox and candidate.admet.tox_flags:
        return False
    if constraints.min_binding is not None and candidate.binding.value < constraints.min_binding:
        return False
    if constraints.max_sa is not None and candidate.admet.sa_score > constraints.max_sa:
        return False
    return True
