from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ...model import Artifact, CoherenceState


@dataclass
class ConflictDecision:
    coherence_state: CoherenceState
    reason_codes: list[str] = field(default_factory=list)
    confidence_delta: float | None = None # candidate_confidence - previous_confidence
    provider: str = "deterministic"
    model: str = "rules"
    prompt_version: str = "deterministic"
    prompt_hash: str | None = None
    fallback_used: bool = False
    warning: str | None = None

    # serializes the decision into a dictionary
    def to_metadata(self) -> dict[str, object]:
        return {
            "coherence_state": self.coherence_state.value,
            "reason_codes": list(self.reason_codes),
            "confidence_delta": self.confidence_delta,
            "judge_provider": self.provider,
            "judge_model": self.model,
            "judge_prompt_version": self.prompt_version,
            "judge_prompt_hash": self.prompt_hash,
            "judge_fallback_used": self.fallback_used,
            "judge_warning": self.warning,
        }


class ConflictJudge(Protocol):
    def judge(
        self,
        *,
        previous: Artifact | None,
        candidate_confidence: float,
        candidate_payload: dict,
    ) -> ConflictDecision:
        ...
