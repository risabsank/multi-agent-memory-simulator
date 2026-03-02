from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ...model import Artifact, CoherenceState


@dataclass
class ConflictDecision:
    coherence_state: CoherenceState
    reason_codes: list[str] = field(default_factory=list)
    confidence_delta: float | None = None
    provider: str = "deterministic"
    model: str = "rule_engine_v1"
    prompt_version: str = "deterministic_v1"
    prompt_hash: str | None = None
    fallback_used: bool = False
    warning: str | None = None
    judge_latency_ms: float | None = None

    def to_metadata(self) -> dict[str, object]:
        return {
            "coherence_state": self.coherence_state.value,
            "reason_codes": list(self.reason_codes),
            "confidence_delta": self.confidence_delta,
            # Primary audit keys.
            "provider": self.provider,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "prompt_hash": self.prompt_hash,
            "fallback_used": self.fallback_used,
            "warning": self.warning,
            # Backwards-compatible aliases kept for existing trace consumers/tests.
            "judge_provider": self.provider,
            "judge_model": self.model,
            "judge_prompt_version": self.prompt_version,
            "judge_prompt_hash": self.prompt_hash,
            "judge_fallback_used": self.fallback_used,
            "judge_warning": self.warning,
            "judge_latency_ms": self.judge_latency_ms,
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