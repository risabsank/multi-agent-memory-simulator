from memory.model import Agent, Artifact, ArtifactScope, CoherenceState, GlobalMemory, Memory
from memory.protocols import ConflictDecision, DeterministicConflictJudge, EventualProtocol, WriteThroughStrongProtocol
from memory.simulator import Simulator
from memory.lib import human2bytes


class DeprecatedJudge:
    def judge(self, *, previous, candidate_confidence: float, candidate_payload: dict) -> ConflictDecision:
        return ConflictDecision(
            coherence_state=CoherenceState.DEPRECATED,
            reason_codes=["manual_override"],
            confidence_delta=0.0,
            provider="test",
            model="override",
            prompt_version="test_v1",
            prompt_hash="abc123",
        )


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
    artifact = Artifact(
                artifact_id=artifact_id,
                version_id=1,
                size=10,
                scope=ArtifactScope.TASK,
                confidence=0.95,
            )

    global_memory = GlobalMemory(
        read_latency=1,
        write_latency=1,
        swap_latency=100,
        total_size=human2bytes("4 gb"),
        block_size=4096,
    )
    global_memory.store_artifact(artifact)

    sim = Simulator(
        agents=[Agent("A", Memory(1, 1, human2bytes("1 gb"), 4096))],
        global_memory=global_memory,
        protocol=WriteThroughStrongProtocol(),
    )
    sim.schedule_write(t=0, agent_id="A", artifact_id=artifact_id, size=20)
    result = sim.run()

    conflict_event = next(line for line in result.trace if line.event == "EV_CONFLICT_CHECK")
    assert conflict_event.metadata["judge_provider"] == "deterministic"
    assert conflict_event.metadata["judge_model"] == "rule_engine_v1"
    assert conflict_event.metadata["provider"] == "deterministic"
    assert isinstance(conflict_event.metadata["reason_codes"], list)


def test_eventual_protocol_uses_judge_for_sync_decision() -> None:
    artifact_id = ("T1", "shared")
    artifact = Artifact(
                artifact_id=artifact_id,
                version_id=1,
                size=10,
                scope=ArtifactScope.TASK,
                confidence=0.9,
            )

    global_memory = GlobalMemory(
        read_latency=1,
        write_latency=1,
        swap_latency=100,
        total_size=human2bytes("4 gb"),
        block_size=4096,
    )
    global_memory.store_artifact(artifact)

    sim = Simulator(
        agents=[Agent("A", Memory(1, 1, human2bytes("1 gb"), 4096))],
        global_memory=global_memory,
        protocol=EventualProtocol(propagation_delay=1, conflict_judge=DeprecatedJudge()),
    )
    sim.schedule_write(t=0, agent_id="A", artifact_id=artifact_id, size=20)
    result = sim.run()

    conflict_event = next(line for line in result.trace if line.event == "EV_CONFLICT_CHECK")
    assert conflict_event.metadata["judge_provider"] == "test"
    assert conflict_event.metadata["judge_prompt_version"] == "test_v1"
    assert conflict_event.metadata["reason_codes"] == ["manual_override"]

    assert result.global_memory is not None
    committed = result.global_memory.get_artifact(artifact_id)
    assert committed.coherence_state == CoherenceState.DEPRECATED
