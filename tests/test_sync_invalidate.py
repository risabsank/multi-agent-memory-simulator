from memory.model import Agent, Artifact, ArtifactScope, GlobalMemory
from memory.protocols import EventualProtocol, WriteThroughStrongProtocol
from memory.simulator import Simulator


def test_manual_sync_updates_stale_cache_entry() -> None:
    artifact_id = ("T1", "shared")
    sim = Simulator(
        agents=[Agent("A"), Agent("B")],
        global_memory=GlobalMemory(
            latency=1,
            store={artifact_id: Artifact(artifact_id=artifact_id, version_id=1, size=10, scope=ArtifactScope.TASK)},
        ),
        protocol=WriteThroughStrongProtocol(),
    )

    # B gets v1 cached
    sim.schedule_read(0, "B", artifact_id)
    # A writes v2 and commits globally
    sim.schedule_write(2, "A", artifact_id, 20)
    # B explicitly syncs after commit to refresh cache
    sim.schedule_sync(4, "B", artifact_id)

    result = sim.run()

    assert result.agents["B"].cache[artifact_id].version_id == 2
    sync_events = [line for line in result.trace if line.event == "EV_SYNC_REQ"]
    assert len(sync_events) == 1
    assert sync_events[0].metadata["stale_before"] == 1

    report = sim.build_report()
    assert report.sync_requests == 1


def test_manual_invalidate_removes_cache_entry() -> None:
    artifact_id = ("T1", "shared")
    sim = Simulator(
        agents=[Agent("A")],
        global_memory=GlobalMemory(
            latency=1,
            store={artifact_id: Artifact(artifact_id=artifact_id, version_id=1, size=10, scope=ArtifactScope.TASK)},
        ),
        protocol=WriteThroughStrongProtocol(),
    )

    sim.schedule_read(0, "A", artifact_id)
    sim.schedule_invalidate(2, "A", artifact_id, reason="manual")

    result = sim.run()

    assert artifact_id not in result.agents["A"].cache
    invalidate_events = [line for line in result.trace if line.event == "EV_INVALIDATE"]
    assert len(invalidate_events) == 1
    assert invalidate_events[0].metadata["had_entry"] is True

    report = sim.build_report()
    assert report.invalidations == 1


def test_eventual_auto_invalidate_forces_remote_miss() -> None:
    artifact_id = ("T1", "shared")
    sim = Simulator(
        agents=[Agent("A"), Agent("B")],
        global_memory=GlobalMemory(
            latency=1,
            store={artifact_id: Artifact(artifact_id=artifact_id, version_id=1, size=10, scope=ArtifactScope.TASK)},
        ),
        protocol=EventualProtocol(propagation_delay=1, auto_invalidate_on_commit=True),
    )

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
