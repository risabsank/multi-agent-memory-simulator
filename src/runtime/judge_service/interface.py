from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from runtime.schemas import CoherenceState


@dataclass(frozen=True)
class JudgeDecision:
    """Outcome of a semantic or policy write evaluation."""

    state: CoherenceState
    reason_codes: tuple[str, ...]
    advisory: bool = False


class JudgeService(Protocol):
    """Evaluates candidate writes.

    Deterministic logic is authoritative. Any LLM-based signal, if present,
    must be advisory-only and optional.
    """

    def evaluate(
        self,
        *,
        tenant_id: str,
        key: str,
        previous_value: dict | None,
        candidate_value: dict,
    ) -> JudgeDecision:
        """Return a decision for applying a candidate write."""
