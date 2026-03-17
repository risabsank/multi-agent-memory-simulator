from memory.model import Agent, Artifact, ArtifactScope, Cache, GlobalMemory, ArtifactId
from memory.protocols.mesi import MesiProtocol, STATES
from memory.simulator import Simulator
from memory.lib import human2bytes

"""
Some quirks about MESI behavior:
When scheduling reads/writes, the scheduled operation will always be prioritized lower than the internally spawned operations in the event queue
For example, if there is a read request at t=0 that will respond at t=4 and you specify a write request at t=4, then the read request will run first
This is thanks to (and can be controlled by) the new priority field in the Event class
"""


def init_scenario1() -> tuple[Simulator, ArtifactId]:
    artifact_id0 = ("T1", "shared")
    artifact0 = Artifact(
        artifact_id=artifact_id0,
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
        Agent("C", Cache(1, 1, human2bytes("1 gb"), 4096)),
    ]
    sim = Simulator(
        agents=agents,
        global_memory=global_memory,
        protocol=MesiProtocol(bus_latency=1, agents=agents),
    )
    return sim, artifact_id0


def test_mesi_decision_invalidate1() -> None:
    sim, artifact_id0 = init_scenario1()
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


def test_mesi_decision_invalidate2() -> None:
    sim, artifact_id0 = init_scenario1()
    sim.schedule_write(t=0, agent_id="A", artifact_id=artifact_id0, size=20)
    sim.schedule_write(t=1, agent_id="B", artifact_id=artifact_id0, size=20)
    sim.schedule_write(t=1, agent_id="C", artifact_id=artifact_id0, size=20)
    sim.schedule_write(t=4, agent_id="B", artifact_id=artifact_id0, size=20)
    result = sim.run()
    assert isinstance(sim.protocol, MesiProtocol)
    final_states = sim.protocol.states
    assert final_states[artifact_id0]["A"] == STATES.I
    assert final_states[artifact_id0]["B"] == STATES.M
    assert final_states[artifact_id0]["C"] == STATES.I
    assert not result.agents["A"].cache.artifact_exists(artifact_id0)
    assert not result.agents["C"].cache.artifact_exists(artifact_id0)
    assert result.agents["B"].cache.artifact_exists(artifact_id0)


def test_mesi_decision_exclusive() -> None:
    sim, artifact_id0 = init_scenario1()
    sim.schedule_read(t=0, agent_id="A", artifact_id=artifact_id0)
    result = sim.run()
    assert isinstance(sim.protocol, MesiProtocol)
    final_states = sim.protocol.states
    assert final_states[artifact_id0]["A"] == STATES.E
    assert result.agents["A"].cache.artifact_exists(artifact_id0)


def test_mesi_decision_exclusive_to_modified() -> None:
    sim, artifact_id0 = init_scenario1()
    sim.schedule_read(t=0, agent_id="A", artifact_id=artifact_id0)
    sim.schedule_write(t=4, agent_id="A", artifact_id=artifact_id0, size=20)
    result = sim.run()
    assert isinstance(sim.protocol, MesiProtocol)
    final_states = sim.protocol.states
    assert final_states[artifact_id0]["A"] == STATES.M
    assert result.agents["A"].cache.artifact_exists(artifact_id0)
    # global memory should have never been written too, only read out of
    assert result.global_memory is not None
    assert result.global_memory.get_artifact(artifact_id0).version_id == 1
    # First read will be 2 (global memory read latency) + 1 (cache latency for check) + 1 (bus snooping)
    # Write will be 1 (cache latency for check)
    assert result.agents["A"].stats.read_latency_total == 4
    assert result.agents["A"].stats.write_latency_total == 1


def test_mesi_decision_exclusive_to_shared() -> None:
    sim, artifact_id0 = init_scenario1()
    sim.schedule_read(t=0, agent_id="A", artifact_id=artifact_id0)
    sim.schedule_read(t=5, agent_id="B", artifact_id=artifact_id0)
    result = sim.run()
    assert isinstance(sim.protocol, MesiProtocol)
    final_states = sim.protocol.states
    assert final_states[artifact_id0]["A"] == STATES.S
    assert final_states[artifact_id0]["B"] == STATES.S

    sim, artifact_id0 = init_scenario1()
    sim.schedule_read(t=0, agent_id="A", artifact_id=artifact_id0)
    sim.schedule_read(t=0, agent_id="B", artifact_id=artifact_id0)
    result = sim.run()
    assert isinstance(sim.protocol, MesiProtocol)
    final_states = sim.protocol.states
    assert final_states[artifact_id0]["A"] == STATES.S
    assert final_states[artifact_id0]["B"] == STATES.S


def test_mesi_decision_shared_to_modified() -> None:
    sim, artifact_id0 = init_scenario1()
    sim.schedule_read(t=0, agent_id="A", artifact_id=artifact_id0)
    sim.schedule_read(t=0, agent_id="B", artifact_id=artifact_id0)
    sim.schedule_write(t=5, agent_id="A", artifact_id=artifact_id0, size=20)
    result = sim.run()
    assert isinstance(sim.protocol, MesiProtocol)
    final_states = sim.protocol.states
    assert final_states[artifact_id0]["A"] == STATES.M
    assert final_states[artifact_id0]["B"] == STATES.I


def test_mesi_decision_modified_to_shared() -> None:
    sim, artifact_id0 = init_scenario1()
    sim.schedule_write(t=0, agent_id="A", artifact_id=artifact_id0, size=20)
    sim.schedule_read(t=3, agent_id="B", artifact_id=artifact_id0)
    result = sim.run()
    assert isinstance(sim.protocol, MesiProtocol)
    final_states = sim.protocol.states
    assert final_states[artifact_id0]["A"] == STATES.S
    assert final_states[artifact_id0]["B"] == STATES.S


def test_mesi_decision_mass_invalidate() -> None:
    sim, artifact_id0 = init_scenario1()
    sim.schedule_read(t=0, agent_id="A", artifact_id=artifact_id0)
    sim.schedule_read(t=1, agent_id="B", artifact_id=artifact_id0)
    sim.schedule_read(t=2, agent_id="C", artifact_id=artifact_id0)
    sim.schedule_write(t=3, agent_id="A", artifact_id=artifact_id0, size=20)
    result = sim.run()
    assert isinstance(sim.protocol, MesiProtocol)
    final_states = sim.protocol.states
    assert final_states[artifact_id0]["A"] == STATES.M
    assert final_states[artifact_id0]["B"] == STATES.I
    assert final_states[artifact_id0]["C"] == STATES.I


def test_mesi_decision_multi_read() -> None:
    sim, artifact_id0 = init_scenario1()
    sim.schedule_write(t=0, agent_id="A", artifact_id=artifact_id0, size=20)
    sim.schedule_read(t=0, agent_id="B", artifact_id=artifact_id0)
    sim.schedule_read(t=0, agent_id="C", artifact_id=artifact_id0)
    result = sim.run()
    assert isinstance(sim.protocol, MesiProtocol)
    final_states = sim.protocol.states
    assert final_states[artifact_id0]["A"] == STATES.S
    assert final_states[artifact_id0]["B"] == STATES.S
    assert final_states[artifact_id0]["C"] == STATES.S


def test_mesi_decision_multi_write_same_time() -> None:
    sim, artifact_id0 = init_scenario1()
    sim.schedule_write(t=0, agent_id="A", artifact_id=artifact_id0, size=20)
    sim.schedule_write(t=0, agent_id="B", artifact_id=artifact_id0, size=20)
    sim.schedule_write(t=0, agent_id="C", artifact_id=artifact_id0, size=20)
    result = sim.run()
    assert isinstance(sim.protocol, MesiProtocol)
    final_states = sim.protocol.states
    assert final_states[artifact_id0]["A"] == STATES.I
    assert final_states[artifact_id0]["B"] == STATES.I
    assert final_states[artifact_id0]["C"] == STATES.M

def test_mesi_decision_exclusive_to_shared_to_modified() -> None:
    sim, artifact_id0 = init_scenario1()
    sim.schedule_read(t=0, agent_id="A", artifact_id=artifact_id0)
    sim.schedule_write(t=1, agent_id="A", artifact_id=artifact_id0, size=20)
    sim.schedule_read(t=1, agent_id="B", artifact_id=artifact_id0)
    sim.schedule_write(t=2, agent_id="C", artifact_id=artifact_id0, size=20)
    result = sim.run()
    assert isinstance(sim.protocol, MesiProtocol)
    final_states = sim.protocol.states
    assert final_states[artifact_id0]["A"] == STATES.I
    assert final_states[artifact_id0]["B"] == STATES.I
    assert final_states[artifact_id0]["C"] == STATES.M

