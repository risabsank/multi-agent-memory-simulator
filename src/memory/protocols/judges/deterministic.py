from __future__ import annotations

from typing import ClassVar

from .types import ConflictDecision
from ...model import Artifact, CoherenceState


class DeterministicConflictJudge:
    """Confidence-based deterministic policy used for current parity behavior."""

    PROFILE_THRESHOLDS: ClassVar[dict[str, float]] = {
        "permissive": -0.05,
        "balanced": 0.0,
        "strict": 0.05,
    }

    def __init__(self, profile: str = "balanced") -> None:
        if profile not in self.PROFILE_THRESHOLDS:
            allowed = ", ".join(sorted(self.PROFILE_THRESHOLDS))
            raise ValueError(
                f"unsupported deterministic judge profile: {profile} (allowed: {allowed})"
            )
        self.profile = profile
        self.min_confidence_delta = self.PROFILE_THRESHOLDS[profile]

    def judge(
        self,
        *,
        previous: Artifact | None,
        candidate_confidence: float,
        candidate_payload: dict,
    ) -> ConflictDecision:
        previous_confidence = previous.confidence if previous else None
        delta = (
            None
            if previous_confidence is None
            else candidate_confidence - previous_confidence
        )

        if previous and delta is not None and delta < self.min_confidence_delta:
            reason_code = "below_admission_profile_threshold"
            if self.min_confidence_delta == 0.0:
                reason_code = "lower_confidence_than_accepted"
            return ConflictDecision(
                coherence_state=CoherenceState.CONTESTED,
                reason_codes=[reason_code],
                confidence_delta=delta,
            )

        return ConflictDecision(
            coherence_state=CoherenceState.ACCEPTED,
            reason_codes=["no_contradiction"],
            confidence_delta=delta,
        )
