from __future__ import annotations

from dataclasses import dataclass, field

from .events import Event, EventQueue, EventType
from .model import Agent, Artifact, ArtifactId, ArtifactScope, CacheEntry, GlobalMemory, VersionClock


@dataclass(slots=True)
class TraceLine:
    t: int
    event: str
    detail: str


@dataclass(slots=True)
class SimulationResult:
    trace: list[TraceLine] = field(default_factory=list)
    agents: dict[str, Agent] = field(default_factory=dict)
    global_memory: GlobalMemory | None = None

    def avg_latency(self, agent_id: str) -> float:
        stats = self.agents[agent_id].stats
        if stats.read_count == 0:
            return 0.0
        return stats.read_latency_total / stats.read_count


class Simulator:
    """Minimal discrete-event simulator vertical slice.

    Supports one cache level, write-through protocol, and 4 event types.
    TODO(m1): add SUMMARIZE/SYNC and protocol plug-ins.
    """

    def __init__(self, agents: list[Agent], global_memory: GlobalMemory) -> None:
        self.agents = {a.agent_id: a for a in agents} # agents indexed
        self.global_memory = global_memory # 
        self.queue = EventQueue()
        self.clock = VersionClock()
        for artifact_id, artifact in self.global_memory.store.items():
            self.clock._versions[artifact_id] = artifact.version_id

        self.now = 0
        self.trace: list[TraceLine] = []

    def schedule_read(self, t: int, agent_id: str, artifact_id: ArtifactId) -> None:
        self.queue.push(
            t=t,
            event_type=EventType.EV_READ_REQ,
            src=agent_id,
            dst=agent_id,
            payload={"artifact_id": artifact_id, "requested_t": t},
        )

    def schedule_write(self, t: int, agent_id: str, artifact_id: ArtifactId, size: int) -> None:
        self.queue.push(
            t=t,
            event_type=EventType.EV_WRITE_REQ,
            src=agent_id,
            dst=agent_id,
            payload={"artifact_id": artifact_id, "size": size, "requested_t": t},
        )

    def run(self) -> SimulationResult:
        while len(self.queue) > 0: # while queue not empry
            event = self.queue.pop() # pop next event
            self.now = event.t
            self._handle(event) # based on event type, perform an action

        return SimulationResult(trace=self.trace, agents=self.agents, global_memory=self.global_memory)

    def _handle(self, event: Event) -> None:
        if event.type == EventType.EV_READ_REQ:
            self._on_read_req(event)
        elif event.type == EventType.EV_READ_RESP:
            self._on_read_resp(event)
        elif event.type == EventType.EV_WRITE_REQ:
            self._on_write_req(event)
        elif event.type == EventType.EV_WRITE_COMMIT:
            self._on_write_commit(event)

    def _on_read_req(self, event: Event) -> None:
        agent = self.agents[event.src]
        artifact_id = tuple(event.payload["artifact_id"])
        requested_t = event.payload["requested_t"]

        if artifact_id in agent.cache: # if cache hit
            entry = agent.cache[artifact_id]
            agent.stats.hits += 1
            entry.last_access_t = self.now
            self.trace.append(TraceLine(self.now, "EV_CACHE_HIT", f"{agent.agent_id} {artifact_id} v{entry.version_id}"))
            self.queue.push(
                t=self.now,
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
        self.trace.append(TraceLine(self.now, "EV_CACHE_MISS", f"{agent.agent_id} {artifact_id}"))
        self.queue.push(
            t=self.now + self.global_memory.latency,
            event_type=EventType.EV_READ_RESP,
            src="global",
            dst=agent.agent_id,
            payload={
                "artifact_id": artifact_id,
                "version_id": self.global_memory.store[artifact_id].version_id,
                "requested_t": requested_t,
                "hit": False,
            },
        )

    def _on_read_resp(self, event: Event) -> None:
        agent = self.agents[event.dst]
        artifact_id = tuple(event.payload["artifact_id"])
        version_id = event.payload["version_id"]
        requested_t = event.payload["requested_t"]

        artifact = self.global_memory.store[artifact_id]
        agent.cache[artifact_id] = CacheEntry(
            artifact_id=artifact_id,
            version_id=version_id,
            size=artifact.size,
            last_access_t=self.now,
        )
        latency = self.now - requested_t
        agent.stats.read_latency_total += latency
        agent.stats.read_count += 1
        self.trace.append(
            TraceLine(
                self.now,
                EventType.EV_READ_RESP.value,
                f"{agent.agent_id} got {artifact_id} v{version_id} latency={latency}",
            )
        )

    def _on_write_req(self, event: Event) -> None:
        agent = self.agents[event.src]
        artifact_id = tuple(event.payload["artifact_id"])
        size = event.payload["size"]

        new_version = self.clock.next(artifact_id) # allocate bew version
        old_artifact = self.global_memory.store.get(artifact_id)
        scope = old_artifact.scope if old_artifact else ArtifactScope.TASK

        agent.cache[artifact_id] = CacheEntry(
            artifact_id=artifact_id,
            version_id=new_version,
            size=size,
            last_access_t=self.now,
        )
        self.trace.append(
            TraceLine(self.now, EventType.EV_WRITE_REQ.value, f"{agent.agent_id} wrote {artifact_id} v{new_version}")
        )

        self.queue.push(
            t=self.now + self.global_memory.latency,
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

    def _on_write_commit(self, event: Event) -> None:
        artifact_id = tuple(event.payload["artifact_id"])
        artifact = Artifact(
            artifact_id=artifact_id,
            version_id=event.payload["version_id"],
            size=event.payload["size"],
            scope=event.payload["scope"],
        )
        self.global_memory.store[artifact_id] = artifact
        self.trace.append(
            TraceLine(
                self.now,
                EventType.EV_WRITE_COMMIT.value,
                f"global committed {artifact_id} v{artifact.version_id}",
            )
        )
