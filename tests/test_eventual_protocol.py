from memory.model import Agent, Artifact, ArtifactScope, CoherenceState, GlobalMemory
from memory.protocols import EventualProtocol
from memory.simulator import Simulator


def test_eventual_protocol_delays_global_visibility() -> None:
    artifact_id = ("T1", "shared_plan")
    global_memory = GlobalMemory(
        latency=2,
        store={
            artifact_id: Artifact(
                artifact_id=artifact_id,
                version_id=1,
                size=64,
                scope=ArtifactScope.TASK,
                confidence=1.0,
            )
        },
    )

    sim = Simulator(
        agents=[Agent("A"), Agent("B")],
        global_memory=global_memory,
        protocol=EventualProtocol(propagation_delay=2),
    )

    # Writer updates local cache first. Global commit is deferred.
    sim.schedule_write(t=0, agent_id="A", artifact_id=artifact_id, size=80)

    # Reader B reads before delayed propagation has committed.
    sim.schedule_read(t=1, agent_id="B", artifact_id=artifact_id)

    # Reader B reads again after propagation should have committed.
    sim.schedule_read(t=7, agent_id="B", artifact_id=artifact_id)

    result = sim.run()

    # Global memory eventually converges to the new version.
    assert result.global_memory.store[artifact_id].version_id == 2

    # First read misses and returns old version from global; second read is cache hit on B.
    read_events = [line for line in result.trace if line.event == "EV_READ_RESP" and line.metadata["agent"] == "B"]
    assert read_events[0].metadata["version_id"] == 1
    assert read_events[1].metadata["version_id"] == 1

    # The delayed commit should happen at t=6: t=0 write + (2*latency sync) + latency commit.
    commit_events = [line for line in result.trace if line.event == "EV_WRITE_COMMIT"]
    assert commit_events[0].t == 6


def test_eventual_protocol_marks_provisional_then_commits_with_contested_state() -> None:
    artifact_id = ("T1", "shared_plan")
    global_memory = GlobalMemory(
        latency=3,
        store={
            artifact_id: Artifact(
                artifact_id=artifact_id,
                version_id=4,
                size=100,
                scope=ArtifactScope.TASK,
                confidence=0.95,
            )
        },
    )

    sim = Simulator(
        agents=[Agent("A")],
        global_memory=global_memory,
        protocol=EventualProtocol(propagation_delay=1),
    )
    sim.schedule_write(t=2, agent_id="A", artifact_id=artifact_id, size=128)

    result = sim.run()

    write_req = [line for line in result.trace if line.event == "EV_WRITE_REQ"][0]
    assert write_req.metadata["coherence_state"] == CoherenceState.PROVISIONAL.value

    committed = result.global_memory.store[artifact_id]
    assert committed.version_id == 5
    assert committed.coherence_state == CoherenceState.CONTESTED

    # Eventual propagation adds one sync/check step before commit.
    conflict_events = [line for line in result.trace if line.event == "EV_CONFLICT_CHECK"]
    assert len(conflict_events) == 1
