from memory.model import (
    Agent,
    Artifact,
    ArtifactScope,
    Cache,
    GlobalMemory,
)
from memory.protocols.mesi import MesiProtocol, STATES
from memory.simulator import Simulator
from memory.lib import human2bytes


def test_mesi_decision_eviction1() -> None:
    artifact_id0 = ("T1", "shared")
    artifact_id1 = ("T2", "shared")
    artifact0 = Artifact(
        artifact_id=artifact_id0,
        version_id=1,
        size=20,
        scope=ArtifactScope.TASK,
        confidence=0.9,
    )
    artifact1 = Artifact(
        artifact_id=artifact_id1,
        version_id=1,
        size=20,
        scope=ArtifactScope.TASK,
        confidence=0.9,
    )

    global_memory = GlobalMemory(
        read_latency=2,
        write_latency=2,
        swap_latency=100,
        total_size=human2bytes("4 gb"),
        block_size=4096,
    )
    global_memory.store_artifact(artifact0)

    agents = [
        Agent("A", Cache(1, 1, human2bytes("1 gb"), 4096)),
        Agent("B", Cache(1, 1, human2bytes("1 gb"), 4096)),
    ]
    sim = Simulator(
        agents=agents,
        global_memory=global_memory,
        protocol=MesiProtocol(bus_latency=1, agents=agents),
    )
    # B is after A in event ordering but will clash for bus arbitration
    # Since A is first, A will change state/write/commit first, but write to B will evict A
    # Final state should be:
    # A does not have artifact_id0, B does have artifact_id0
    sim.schedule_write(t=0, agent_id="A", artifact_id=artifact_id0, size=20)
    sim.schedule_write(t=0, agent_id="B", artifact_id=artifact_id0, size=20)
    result = sim.run()
    assert isinstance(sim.protocol, MesiProtocol)
    final_states = sim.protocol.states
    assert final_states[artifact_id0]["A"] == STATES.I
    assert final_states[artifact_id0]["B"] == STATES.M
    assert not result.agents["A"].cache.artifact_exists(artifact_id0)
    assert result.agents["B"].cache.artifact_exists(artifact_id0)
