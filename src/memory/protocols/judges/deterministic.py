from __future__ import annotations

from .types import ConflictDecision
from ...model import Artifact, CoherenceState


class DeterministicConflictJudge:
    """Confidence-based deterministic policy used for current parity behavior."""

    def judge(
        self,
        *,
        previous: Artifact | None,
        candidate_confidence: float,
        candidate_payload: dict,
    ) -> ConflictDecision:
        previous_confidence = previous.confidence if previous else None
        delta = None if previous_confidence is None else candidate_confidence - previous_confidence

        if previous and candidate_confidence < previous.confidence:
            return ConflictDecision(
                coherence_state=CoherenceState.CONTESTED,
                reason_codes=["lower_confidence_than_accepted"],
                confidence_delta=delta,
            )

        return ConflictDecision(
            coherence_state=CoherenceState.ACCEPTED,
            reason_codes=["no_contradiction"],
            confidence_delta=delta,
        )
