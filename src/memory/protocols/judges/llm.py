from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter
from typing import Callable

from ...model import Artifact, CoherenceState
from .deterministic import DeterministicConflictJudge
from .types import ConflictDecision


@dataclass(frozen=True)
class LLMPromptSpec:
    version: str = "llm_conflict_v1"
    template: str = """You are a conflict judge for a multi-agent memory system.
Compare candidate write confidence against existing committed artifact confidence and context.
Return ONLY JSON with this exact schema:
{{"coherence_state":"accepted|contested|deprecated","reason_codes":["code"],"confidence_delta":number|null}}
Input:
previous_confidence={previous_confidence}
candidate_confidence={candidate_confidence}
candidate_payload={candidate_payload}
"""


def build_openai_inference_fn(model: str = "gpt-4.1-mini") -> Callable[[str], str]:

    from openai import OpenAI
    
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def inference_fn(prompt: str) -> str:
        response = client.responses.create(
            model=model,
            input=prompt,
            store=False,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "conflict_decision",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "coherence_state": {
                                "type": "string",
                                "enum": ["accepted", "contested", "deprecated"],
                            },
                            "reason_codes": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "confidence_delta": {
                                "type": ["number", "null"],
                            },
                        },
                        "required": [
                            "coherence_state",
                            "reason_codes",
                            "confidence_delta",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
        )
        return response.output_text

    return inference_fn


class LLMConflictJudge:
    """LLM-backed judge with strict schema validation and deterministic fallback."""

    def __init__(
        self,
        *,
        inference_fn: Callable[[str], str],
        provider: str = "openai",
        model: str = "gpt-4.1-mini",
        timeout_s: float = 1.5,
        prompt_spec: LLMPromptSpec | None = None,
        fallback_judge: DeterministicConflictJudge | None = None,
    ) -> None:
        self.inference_fn = inference_fn
        self.provider = provider
        self.model = model
        self.timeout_s = max(0.01, timeout_s)
        self.prompt_spec = prompt_spec or LLMPromptSpec()
        self.fallback_judge = fallback_judge or DeterministicConflictJudge()

    def judge(
        self,
        *,
        previous: Artifact | None,
        candidate_confidence: float,
        candidate_payload: dict,
    ) -> ConflictDecision:
        previous_confidence = previous.confidence if previous else None
        prompt = self.prompt_spec.template.format(
            previous_confidence=previous_confidence,
            candidate_confidence=candidate_confidence,
            candidate_payload=candidate_payload,
        )
        prompt_hash = sha256(prompt.encode("utf-8")).hexdigest()
        started = perf_counter()

        try:
            raw = self._run_with_timeout(prompt)
            payload = self._parse_response(raw)
            elapsed_ms = (perf_counter() - started) * 1000.0
            return ConflictDecision(
                coherence_state=CoherenceState(payload["coherence_state"]),
                reason_codes=list(payload["reason_codes"]),
                confidence_delta=payload.get("confidence_delta"),
                provider=self.provider,
                model=self.model,
                prompt_version=self.prompt_spec.version,
                prompt_hash=prompt_hash,
                judge_latency_ms=elapsed_ms,
            )
        except Exception as exc:
            fallback = self.fallback_judge.judge(
                previous=previous,
                candidate_confidence=candidate_confidence,
                candidate_payload=candidate_payload,
            )
            fallback.provider = self.provider
            fallback.model = self.model
            fallback.prompt_version = self.prompt_spec.version
            fallback.prompt_hash = prompt_hash
            fallback.fallback_used = True
            fallback.warning = self._warning_code(exc)
            fallback.judge_latency_ms = (perf_counter() - started) * 1000.0
            return fallback

    def _run_with_timeout(self, prompt: str) -> str:
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(self.inference_fn, prompt)
            try:
                return fut.result(timeout=self.timeout_s)
            except TimeoutError as exc:
                fut.cancel()
                raise TimeoutError("llm_timeout") from exc

    @staticmethod
    def _parse_response(raw: str) -> dict:
        payload = json.loads(raw)
        required = {"coherence_state", "reason_codes", "confidence_delta"}
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"schema_missing:{','.join(sorted(missing))}")

        state = payload["coherence_state"]
        if state not in {"accepted", "contested", "deprecated"}:
            raise ValueError(f"invalid_state:{state}")

        reasons = payload["reason_codes"]
        if not isinstance(reasons, list) or not all(isinstance(v, str) for v in reasons):
            raise ValueError("invalid_reason_codes")

        delta = payload["confidence_delta"]
        if delta is not None and not isinstance(delta, (int, float)):
            raise ValueError("invalid_confidence_delta")
        payload["confidence_delta"] = None if delta is None else float(delta)
        return payload

    @staticmethod
    def _warning_code(exc: Exception) -> str:
        msg = str(exc)
        if isinstance(exc, TimeoutError) or "llm_timeout" in msg:
            return "llm_timeout"
        if msg.startswith("schema_missing"):
            return "llm_schema_missing"
        if msg.startswith("invalid_state"):
            return "llm_invalid_state"
        if msg == "invalid_reason_codes":
            return "llm_invalid_reason_codes"
        if msg == "invalid_confidence_delta":
            return "llm_invalid_confidence_delta"
        if isinstance(exc, json.JSONDecodeError):
            return "llm_invalid_json"
        return "llm_runtime_error"