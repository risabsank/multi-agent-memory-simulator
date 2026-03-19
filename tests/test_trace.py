from memory.model import (
    Agent,
    Artifact,
    ArtifactScope,
    Cache,
    CoherenceState,
    GlobalMemory,
    Memory,
)
from memory.protocols import WriteThroughStrongProtocol
from memory.simulator import Simulator
from memory.lib import human2bytes


def _build_scenario() -> tuple[Simulator, tuple[str, str]]:
    artifact_id = ("T1", "shared_plan")
    global_memory = GlobalMemory(
        read_latency=3,
        write_latency=3,
        swap_latency=100,
        total_size=1_000_000,
        block_size=1_000,
    )

    artifact = Artifact(
        artifact_id=artifact_id, version_id=1, size=100, scope=ArtifactScope.TASK
    )
    global_memory.store_artifact(artifact)

    sim = Simulator(
        agents=[
            Agent("A", Cache(0, 0, human2bytes("1 gb"), 4096)),
            Agent("B", Cache(0, 0, human2bytes("1 gb"), 4096)),
        ],
        global_memory=global_memory,
        protocol=WriteThroughStrongProtocol(),
    )
    sim.schedule_read(t=0, agent_id="A", artifact_id=artifact_id)
    sim.schedule_read(t=5, agent_id="A", artifact_id=artifact_id)
    sim.schedule_write(t=6, agent_id="A", artifact_id=artifact_id, size=120)
    sim.schedule_read(t=15, agent_id="B", artifact_id=artifact_id)
    return sim, artifact_id


def test_read_write_flow_and_versions() -> None:
    sim, artifact_id = _build_scenario()

    result = sim.run()

    assert result.agents["A"].stats.misses == 1
    assert result.agents["A"].stats.hits == 1
    assert result.global_memory._store[artifact_id].version_id == 2  # type: ignore[union-attr]
    assert result.agents["B"].cache.get_artifact(artifact_id).version_id == 2
    assert result.avg_latency("A") >= 0
    assert result.avg_write_latency("A") >= 0


def test_trace_behavior_matches_pre_refactor_expectations() -> None:
    sim, artifact_id = _build_scenario()

    result = sim.run()

    assert result.agents["A"].stats.hits == 1
    assert result.agents["A"].stats.misses == 1
    assert result.agents["B"].stats.hits == 0
    assert result.agents["B"].stats.misses == 1

    assert result.global_memory._store[artifact_id].version_id == 2  # type: ignore[union-attr]

    latencies = [
        int(line.detail.split("latency=")[1])
        for line in result.trace
        if line.event == "EV_READ_RESP"
    ]
    assert latencies == [3, 0, 3]


def test_phase2_schema_and_observability_report() -> None:
    sim, artifact_id = _build_scenario()

    result = sim.run()
    report = sim.build_report()

    assert result.global_memory is not None
    artifact = result.global_memory.get_artifact(artifact_id)
    assert artifact.provenance == "A"
    assert artifact.coherence_state in {
        CoherenceState.ACCEPTED,
        CoherenceState.CONTESTED,
    }
    assert artifact.claim_type.value == "fact"

    conflict_events = [
        line for line in result.trace if line.event == "EV_CONFLICT_CHECK"
    ]
    assert len(conflict_events) == 1
    assert conflict_events[0].metadata["artifact_id"] == artifact_id

    assert report.conflict_checks == 1
    assert report.cache_hits == 1
    assert report.cache_misses == 2
    assert report.avg_read_latency == 2.0
    assert report.avg_write_latency == 3.0
