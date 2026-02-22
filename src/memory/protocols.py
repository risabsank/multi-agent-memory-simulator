from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from .events import Event, EventType
from .model import Artifact, ArtifactScope, CacheEntry

if TYPE_CHECKING:
    from .simulator import Simulator


class ConsistencyProtocol(Protocol):
    def on_read_req(self, simulator: Simulator, event: Event) -> None:
        ...

    def on_read_resp(self, simulator: Simulator, event: Event) -> None:
        ...

    def on_write_req(self, simulator: Simulator, event: Event) -> None:
        ...

    def on_write_commit(self, simulator: Simulator, event: Event) -> None:
        ...

    def on_sync_req(self, simulator: Simulator, event: Event) -> None:
        ...


class WriteThroughStrongProtocol:
    """Current write-through, strong consistency behavior."""

    def on_read_req(self, simulator: Simulator, event: Event) -> None:
        agent = simulator.agents[event.src]
        artifact_id = tuple(event.payload["artifact_id"])
        requested_t = event.payload["requested_t"]

        if artifact_id in agent.cache:
            entry = agent.cache[artifact_id]
            agent.stats.hits += 1
            entry.last_access_t = simulator.now
            simulator.trace.append(
                simulator.trace_line_type(
                    simulator.now,
                    "EV_CACHE_HIT",
                    f"{agent.agent_id} {artifact_id} v{entry.version_id}",
                )
            )
            simulator.queue.push(
                t=simulator.now,
                event_type=EventType.EV_READ_RESP,
                src="cache",
                dst=agent.agent_id,
                payload={
                    "artifact_id": artifact_id,
                    "version_id": entry.version_id,
                    "requested_t": requested_t,
                    "hit": True,
                },
            )
            return

        agent.stats.misses += 1
        simulator.trace.append(
            simulator.trace_line_type(simulator.now, "EV_CACHE_MISS", f"{agent.agent_id} {artifact_id}")
        )
        simulator.queue.push(
            t=simulator.now + simulator.global_memory.latency,
            event_type=EventType.EV_READ_RESP,
            src="global",
            dst=agent.agent_id,
            payload={
                "artifact_id": artifact_id,
                "version_id": simulator.global_memory.store[artifact_id].version_id,
                "requested_t": requested_t,
                "hit": False,
            },
        )

    def on_read_resp(self, simulator: Simulator, event: Event) -> None:
        agent = simulator.agents[event.dst]
        artifact_id = tuple(event.payload["artifact_id"])
        version_id = event.payload["version_id"]
        requested_t = event.payload["requested_t"]

        artifact = simulator.global_memory.store[artifact_id]
        agent.cache[artifact_id] = CacheEntry(
            artifact_id=artifact_id,
            version_id=version_id,
            size=artifact.size,
            last_access_t=simulator.now,
        )
        latency = simulator.now - requested_t
        agent.stats.read_latency_total += latency
        agent.stats.read_count += 1
        simulator.trace.append(
            simulator.trace_line_type(
                simulator.now,
                EventType.EV_READ_RESP.value,
                f"{agent.agent_id} got {artifact_id} v{version_id} latency={latency}",
            )
        )

    def on_write_req(self, simulator: Simulator, event: Event) -> None:
        agent = simulator.agents[event.src]
        artifact_id = tuple(event.payload["artifact_id"])
        size = event.payload["size"]

        new_version = simulator.clock.next(artifact_id)
        old_artifact = simulator.global_memory.store.get(artifact_id)
        scope = old_artifact.scope if old_artifact else ArtifactScope.TASK

        agent.cache[artifact_id] = CacheEntry(
            artifact_id=artifact_id,
            version_id=new_version,
            size=size,
            last_access_t=simulator.now,
        )
        simulator.trace.append(
            simulator.trace_line_type(
                simulator.now,
                EventType.EV_WRITE_REQ.value,
                f"{agent.agent_id} wrote {artifact_id} v{new_version}",
            )
        )

        simulator.queue.push(
            t=simulator.now + simulator.global_memory.latency,
            event_type=EventType.EV_WRITE_COMMIT,
            src=agent.agent_id,
            dst="global",
            payload={
                "artifact_id": artifact_id,
                "version_id": new_version,
                "size": size,
                "scope": scope,
            },
        )

    def on_write_commit(self, simulator: Simulator, event: Event) -> None:
        artifact_id = tuple(event.payload["artifact_id"])
        artifact = Artifact(
            artifact_id=artifact_id,
            version_id=event.payload["version_id"],
            size=event.payload["size"],
            scope=event.payload["scope"],
        )
        simulator.global_memory.store[artifact_id] = artifact
        simulator.trace.append(
            simulator.trace_line_type(
                simulator.now,
                EventType.EV_WRITE_COMMIT.value,
                f"global committed {artifact_id} v{artifact.version_id}",
            )
        )

    def on_sync_req(self, simulator: Simulator, event: Event) -> None:
        return None


class EventualProtocol:
    """Scaffold for milestone-1 eventual consistency implementation."""

    def on_read_req(self, simulator: Simulator, event: Event) -> None:
        raise NotImplementedError("EventualProtocol.on_read_req is not implemented yet")

    def on_read_resp(self, simulator: Simulator, event: Event) -> None:
        raise NotImplementedError("EventualProtocol.on_read_resp is not implemented yet")

    def on_write_req(self, simulator: Simulator, event: Event) -> None:
        raise NotImplementedError("EventualProtocol.on_write_req is not implemented yet")

    def on_write_commit(self, simulator: Simulator, event: Event) -> None:
        raise NotImplementedError("EventualProtocol.on_write_commit is not implemented yet")

    def on_sync_req(self, simulator: Simulator, event: Event) -> None:
        raise NotImplementedError("EventualProtocol.on_sync_req is not implemented yet")
