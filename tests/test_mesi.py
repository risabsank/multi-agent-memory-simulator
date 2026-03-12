from memory.model import (
    Agent,
    Artifact,
    ArtifactScope,
    Cache,
    CoherenceState,
    GlobalMemory,
    Memory,
)
from memory.protocols import (
    ConflictDecision,
    DeterministicConflictJudge,
    EventualProtocol,
    WriteThroughStrongProtocol,
)
from memory.protocols.mesi import MesiProtocol
from memory.simulator import Simulator
from memory.lib import human2bytes


def test_mesi_decisions() -> None:
    artifact_id = ("T1", "shared")
    artifact = Artifact(
        artifact_id=artifact_id,
        version_id=1,
        size=10,
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
    global_memory.store_artifact(artifact)

    agents = [Agent("A", Cache(1, 1, human2bytes("1 gb"), 4096))]
    sim = Simulator(
        agents=agents,
        global_memory=global_memory,
        protocol=MesiProtocol(bus_latency=1, agents=agents),
    )
    sim.schedule_write(t=0, agent_id="A", artifact_id=artifact_id, size=20)
    result = sim.run()

