from __future__ import annotations

from dataclasses import dataclass

from .model import Artifact, CoherenceState


@dataclass(frozen=True)
class SemanticDecision:
    state: CoherenceState
    reason_codes: tuple[str, ...]


class SemanticCoherenceEngine:
    """Deterministic contradiction checks + explicit state transitions."""

    def decide_write_state(
        self,
        *,
        previous: Artifact | None,
        candidate_confidence: float,
        candidate_payload: dict,
        default_state: CoherenceState,
    ) -> SemanticDecision:
        reasons: list[str] = []
        state = default_state

        if previous is None:
            reasons.append("new_artifact")
            return SemanticDecision(state=state, reason_codes=tuple(reasons))

        if bool(candidate_payload.get("contradiction", False)):
            state = CoherenceState.CONTESTED
            reasons.append("explicit_contradiction")

        supersedes_version = candidate_payload.get("supersedes_version")
        if isinstance(supersedes_version, int) and supersedes_version < previous.version_id:
            state = CoherenceState.CONTESTED
            reasons.append("stale_supersession")

        if candidate_confidence < 0.35:
            state = CoherenceState.PROVISIONAL
            reasons.append("low_confidence")

        if previous.coherence_state == CoherenceState.DEPRECATED and candidate_confidence >= previous.confidence:
            state = CoherenceState.ACCEPTED
            reasons.append("rehydrated_from_deprecated")

        if previous.coherence_state == CoherenceState.ACCEPTED and candidate_confidence + 0.2 < previous.confidence:
            state = CoherenceState.PROVISIONAL
            reasons.append("confidence_regression")

        if not reasons:
            reasons.append("stable_transition")
        return SemanticDecision(state=state, reason_codes=tuple(reasons))
