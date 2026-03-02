from __future__ import annotations

from typing import Callable

from .deterministic import DeterministicConflictJudge
from .llm import LLMConflictJudge
from .types import ConflictDecision, ConflictJudge


def build_conflict_judge(
    *,
    judge_mode: str = "deterministic",
    llm_inference_fn: Callable[[str], str] | None = None,
    llm_provider: str = "llm",
    llm_model: str = "unknown",
    llm_timeout_s: float = 0.25,
) -> ConflictJudge:
    if judge_mode == "deterministic":
        return DeterministicConflictJudge()
    if judge_mode == "llm":
        if llm_inference_fn is None:
            raise ValueError("llm_inference_fn is required when judge_mode='llm'")
        return LLMConflictJudge(
            inference_fn=llm_inference_fn,
            provider=llm_provider,
            model=llm_model,
            timeout_s=llm_timeout_s,
        )
    raise ValueError(f"unsupported judge_mode: {judge_mode}")


__all__ = [
    "ConflictDecision",
    "ConflictJudge",
    "DeterministicConflictJudge",
    "LLMConflictJudge",
    "build_conflict_judge",
]
