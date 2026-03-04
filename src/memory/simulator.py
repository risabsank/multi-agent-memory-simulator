from __future__ import annotations

from dataclasses import dataclass, field

from .events import Event, EventQueue, EventType
from .model import Agent, Artifact, ArtifactId, CacheEntry, GlobalMemory, VersionClock
from .protocols import ConsistencyProtocol, WriteThroughStrongProtocol


@dataclass(slots=True)
class TraceLine:
    t: int
    event: str
    detail: str


@dataclass(slots=True)
class SimulationResult:
    trace: list[TraceLine] = field(default_factory=list)
    agents: dict[str, Agent] = field(default_factory=dict)
    model: GlobalMemory | None = None

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

    def __init__(
        self,
        agents: list[Agent],
        model: GlobalMemory,
        protocol: ConsistencyProtocol | None = None,
    ) -> None:
        self.agents = {a.agent_id: a for a in agents} # agents indexed
        self.model = model
        self.queue = EventQueue()
        self.protocol = protocol or WriteThroughStrongProtocol()
        self.clock = VersionClock()
        for artifact_id, artifact in self.model.get_all_artifacts():
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

        return SimulationResult(trace=self.trace, agents=self.agents, model=self.model)

    def _handle(self, event: Event) -> None:
        if event.type == EventType.EV_READ_REQ:
            self.protocol.on_read_req(self, event)
        elif event.type == EventType.EV_READ_RESP:
            self.protocol.on_read_resp(self, event)
        elif event.type == EventType.EV_WRITE_REQ:
            self.protocol.on_write_req(self, event)
        elif event.type == EventType.EV_WRITE_COMMIT:
            self.protocol.on_write_commit(self, event)

    @staticmethod
    def trace_line_type(t: int, event: str, detail: str) -> TraceLine:
        return TraceLine(t, event, detail)
