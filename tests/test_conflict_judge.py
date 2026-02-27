from memory.model import Agent, Artifact, ArtifactScope, GlobalMemory
from memory.protocols import DeterministicConflictJudge, EventualProtocol, WriteThroughStrongProtocol
from memory.simulator import Simulator


def test_deterministic_conflict_judge_parity_states() -> None:
    judge = DeterministicConflictJudge()
    previous = Artifact(
        artifact_id=("T1", "shared"),
        version_id=1,
        size=10,
        scope=ArtifactScope.TASK,
        confidence=0.9,
    )

    contested = judge.judge(previous=previous, candidate_confidence=0.5, candidate_payload={})
    assert contested.coherence_state.value == "contested"
    assert contested.reason_codes == ["lower_confidence_than_accepted"]
    assert contested.confidence_delta is not None
    assert abs(contested.confidence_delta + 0.4) < 1e-9

    accepted = judge.judge(previous=previous, candidate_confidence=0.95, candidate_payload={})
    assert accepted.coherence_state.value == "accepted"
    assert accepted.reason_codes == ["no_contradiction"]
    assert accepted.confidence_delta is not None
    assert abs(accepted.confidence_delta - 0.05) < 1e-9


def test_strong_protocol_conflict_check_has_judge_audit_metadata() -> None:
    artifact_id = ("T1", "shared")
    global_memory = GlobalMemory(
        latency=1,
        store={
            artifact_id: Artifact(
                artifact_id=artifact_id,
                version_id=1,
                size=10,
                scope=ArtifactScope.TASK,
                confidence=0.95,
            )
        },
    )

    sim = Simulator(
        agents=[Agent("A")],
        global_memory=global_memory,
        protocol=WriteThroughStrongProtocol(),
    )
    sim.schedule_write(t=0, agent_id="A", artifact_id=artifact_id, size=20)
    result = sim.run()

    conflict_event = next(line for line in result.trace if line.event == "EV_CONFLICT_CHECK")
    assert conflict_event.metadata["judge_provider"] == "deterministic"
    assert conflict_event.metadata["judge_model"] == "rule_engine_v1"
    assert isinstance(conflict_event.metadata["reason_codes"], list)


def test_eventual_protocol_uses_judge_for_sync_decision() -> None:
    artifact_id = ("T1", "shared")
    global_memory = GlobalMemory(
        latency=1,
        store={
            artifact_id: Artifact(
                artifact_id=artifact_id,
                version_id=1,
                size=10,
                scope=ArtifactScope.TASK,
                confidence=0.9,
            )
        },
    )

    sim = Simulator(
        agents=[Agent("A")],
        global_memory=global_memory,
        protocol=EventualProtocol(propagation_delay=1),
    )
    sim.schedule_write(t=0, agent_id="A", artifact_id=artifact_id, size=20)
    result = sim.run()

    conflict_event = next(line for line in result.trace if line.event == "EV_CONFLICT_CHECK")
    assert conflict_event.metadata["judge_provider"] == "deterministic"
    assert conflict_event.metadata["judge_prompt_version"] == "deterministic_v1"
    assert conflict_event.metadata["reason_codes"] in (["no_contradiction"], ["lower_confidence_than_accepted"])

    committed = result.global_memory.store[artifact_id]
    assert committed.coherence_state.value in {"accepted", "contested", "deprecated"}
