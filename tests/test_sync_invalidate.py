from memory.model import Agent, Artifact, ArtifactScope, GlobalMemory, Memory
from memory.protocols import EventualProtocol, WriteThroughStrongProtocol
from memory.simulator import Simulator
from memory.lib import human2bytes


def test_manual_sync_updates_stale_cache_entry() -> None:
    artifact_id = ("T1", "shared")
    artifact = Artifact(artifact_id=artifact_id, version_id=1, size=10, scope=ArtifactScope.TASK)
    sim = Simulator(
        agents=[Agent("A", Memory(1, 1, human2bytes("1 gb"), 4096)),
                Agent("B", Memory(1, 1, human2bytes("1 gb"), 4096))],
        global_memory=GlobalMemory(
            read_latency=1,
            write_latency=1,
            swap_latency=100,
            total_size=human2bytes("4 gb"),
            block_size=4096,
        ),
        protocol=WriteThroughStrongProtocol(),
    )
    sim.global_memory.store_artifact(artifact)

    # B gets v1 cached
    sim.schedule_read(0, "B", artifact_id)
    # A writes v2 and commits globally
    sim.schedule_write(2, "A", artifact_id, 20)
    # B explicitly syncs after commit to refresh cache
    sim.schedule_sync(4, "B", artifact_id)

    result = sim.run()

    assert result.agents["B"].cache.get_artifact(artifact_id).version_id == 2
    sync_events = [line for line in result.trace if line.event == "EV_SYNC_REQ"]
    assert len(sync_events) == 1
    assert sync_events[0].metadata["stale_before"] == 1

    report = sim.build_report()
    assert report.sync_requests == 1


def test_manual_invalidate_removes_cache_entry() -> None:
    artifact_id = ("T1", "shared")
    artifact = Artifact(artifact_id=artifact_id, version_id=1, size=10, scope=ArtifactScope.TASK)
    sim = Simulator(
        agents=[Agent("A", Memory(1, 1, human2bytes("1 gb"), 4096))],
        global_memory=GlobalMemory(
            read_latency=1,
            write_latency=1,
            swap_latency=100,
            total_size=human2bytes("4 gb"),
            block_size=4096,
        ),
        protocol=WriteThroughStrongProtocol(),
    )
    sim.global_memory.store_artifact(artifact)

    sim.schedule_read(0, "A", artifact_id)
    sim.schedule_invalidate(2, "A", artifact_id, reason="manual")

    result = sim.run()

    assert not result.agents["A"].cache.artifact_exists(artifact_id)
    invalidate_events = [line for line in result.trace if line.event == "EV_INVALIDATE"]
    assert len(invalidate_events) == 1
    assert invalidate_events[0].metadata["had_entry"] is True

    report = sim.build_report()
    assert report.invalidations == 1


def test_eventual_auto_invalidate_forces_remote_miss() -> None:
    artifact_id = ("T1", "shared")
    artifact = Artifact(artifact_id=artifact_id, version_id=1, size=10, scope=ArtifactScope.TASK)
    sim = Simulator(
        agents=[Agent("A", Memory(1, 1, human2bytes("1 gb"), 4096)),
                Agent("B", Memory(1, 1, human2bytes("1 gb"), 4096))],
        global_memory=GlobalMemory(
            read_latency=1,
            write_latency=1,
            swap_latency=100,
            total_size=human2bytes("4 gb"),
            block_size=4096,
        ),
        protocol=EventualProtocol(propagation_delay=1, auto_invalidate_on_commit=True),
    )
    sim.global_memory.store_artifact(artifact)

    # B caches v1, then A updates. Commit should invalidate B's cache.
    sim.schedule_read(0, "B", artifact_id)
    sim.schedule_write(2, "A", artifact_id, 22)
    sim.schedule_read(5, "B", artifact_id)

    result = sim.run()

    b_reads = [line for line in result.trace if line.event == "EV_READ_RESP" and line.metadata.get("agent") == "B"]
    assert b_reads[-1].metadata["read_source"] == "global"

    invalidations = [line for line in result.trace if line.event == "EV_INVALIDATE"]
    assert len(invalidations) == 1
    assert invalidations[0].metadata["agent"] == "B"
