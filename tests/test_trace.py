from memory.model import Agent, GlobalMemory, Artifact, ArtifactScope
from memory.simulator import Simulator
from memory.protocols import WriteThroughStrongProtocol


def _build_scenario() -> tuple[Simulator, tuple[str, str]]:
    artifact_id = ("T1", "shared_plan")
    global_memory = GlobalMemory(
        read_latency=5,
        write_latency=5,
        swap_latency=100,
        total_size=1_000_000,
        block_size=1_000,
    )

    artifact = Artifact(artifact_id=artifact_id, version_id=1, size=100, scope=ArtifactScope.TASK)
    global_memory.store_artifact(artifact)

    sim = Simulator(
        agents=[Agent("A"), Agent("B")],
        model=global_memory,
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
    assert result.model._store[artifact_id].version_id == 2 # type: ignore[union-attr]
    assert result.agents["B"].cache[artifact_id].version_id == 2
    assert result.avg_latency("A") >= 0

def test_trace_behavior_matches_pre_refactor_expectations() -> None:
    sim, artifact_id = _build_scenario()

    result = sim.run()

    assert result.agents["A"].stats.hits == 1
    assert result.agents["A"].stats.misses == 1
    assert result.agents["B"].stats.hits == 0
    assert result.agents["B"].stats.misses == 1

    assert result.model._store[artifact_id].version_id == 2 # type: ignore[union-attr]

    latencies = [
        int(line.detail.split("latency=")[1])
        for line in result.trace
        if line.event == "EV_READ_RESP"
    ]
    assert latencies == [3, 0, 3]
