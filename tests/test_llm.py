import json
import time

from memory.model import Agent, Artifact, ArtifactScope, GlobalMemory
from memory.protocols import EventualProtocol, LLMConflictJudge, WriteThroughStrongProtocol
from memory.simulator import Simulator


def test_llm_judge_success_populates_prompt_metadata() -> None:
    def fake_llm(_prompt: str) -> str:
        return json.dumps(
            {
                "coherence_state": "contested",
                "reason_codes": ["llm_detected_conflict"],
                "confidence_delta": -0.2,
            }
        )

    artifact_id = ("T1", "shared")
    sim = Simulator(
        agents=[Agent("A")],
        global_memory=GlobalMemory(
            latency=1,
            store={artifact_id: Artifact(artifact_id=artifact_id, version_id=1, size=1, scope=ArtifactScope.TASK)},
        ),
        protocol=WriteThroughStrongProtocol(
            judge_mode="llm",
            llm_inference_fn=fake_llm,
            llm_provider="openai",
            llm_model="gpt-sim",
        ),
    )
    sim.schedule_write(0, "A", artifact_id, 5)
    result = sim.run()
    conflict_event = next(line for line in result.trace if line.event == "EV_CONFLICT_CHECK")

    assert conflict_event.metadata["judge_provider"] == "openai"
    assert conflict_event.metadata["judge_model"] == "gpt-sim"
    assert conflict_event.metadata["judge_prompt_version"] == "llm_conflict_v1"
    assert isinstance(conflict_event.metadata["judge_prompt_hash"], str)
    assert len(conflict_event.metadata["judge_prompt_hash"]) == 64
    assert conflict_event.metadata["judge_fallback_used"] is False


def test_llm_judge_failure_falls_back_with_warning_and_report_metrics() -> None:
    def slow_bad_llm(_prompt: str) -> str:
        time.sleep(0.05)
        return "{bad json"

    artifact_id = ("T1", "shared")
    sim = Simulator(
        agents=[Agent("A")],
        global_memory=GlobalMemory(
            latency=1,
            store={
                artifact_id: Artifact(
                    artifact_id=artifact_id,
                    version_id=1,
                    size=1,
                    scope=ArtifactScope.TASK,
                    confidence=0.95,
                )
            },
        ),
        protocol=EventualProtocol(
            propagation_delay=1,
            judge_mode="llm",
            llm_inference_fn=slow_bad_llm,
            llm_provider="openai",
            llm_model="gpt-sim",
            llm_timeout_s=0.001,
        ),
    )
    sim.schedule_write(0, "A", artifact_id, 5)
    result = sim.run()
    conflict_event = next(line for line in result.trace if line.event == "EV_CONFLICT_CHECK")

    assert conflict_event.metadata["judge_fallback_used"] is True
    assert conflict_event.metadata["judge_warning"] == "llm_timeout"
    assert conflict_event.metadata["reason_codes"] == ["lower_confidence_than_accepted"]
    assert isinstance(conflict_event.metadata["judge_latency_ms"], float)

    report = sim.build_report()
    assert report.judge_provider_breakdown == {"openai": 1}
    assert report.fallback_count == 1
    assert report.llm_failure_categories == {"llm_timeout": 1}
    assert report.reason_code_counts["lower_confidence_than_accepted"] == 1
    assert report.avg_judge_latency is not None


def test_llm_conflict_judge_direct_parse_error_fallback_code() -> None:
    judge = LLMConflictJudge(inference_fn=lambda _p: "{}", provider="p", model="m", timeout_s=0.1)
    decision = judge.judge(previous=None, candidate_confidence=0.7, candidate_payload={})
    assert decision.fallback_used is True
    assert decision.warning == "llm_schema_missing"
