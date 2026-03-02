from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .events import Event, EventQueue, EventType
from .model import Agent, ArtifactId, GlobalMemory, VersionClock
from .protocols import ConsistencyProtocol, WriteThroughStrongProtocol

# structured execution log entry for each simulation step
# acts as an event journal
@dataclass
class TraceLine:
    t: int # simulation time
    event: str # event label
    detail: str # summary
    metadata: dict[str, object] = field(default_factory=dict) # context


@dataclass
class SimulationResult:
    trace: list[TraceLine] = field(default_factory=list)
    agents: dict[str, Agent] = field(default_factory=dict)
    global_memory: GlobalMemory | None = None

    def avg_latency(self, agent_id: str) -> float:
        stats = self.agents[agent_id].stats
        if stats.read_count == 0:
            return 0.0
        return stats.read_latency_total / stats.read_count
    
    def avg_write_latency(self, agent_id: str) -> float:
        stats = self.agents[agent_id].stats
        if stats.write_count == 0:
            return 0.0
        return stats.write_latency_total / stats.write_count


@dataclass
class RunReport:
    total_events: int
    cache_hits: int
    cache_misses: int
    conflict_checks: int
    contested_writes: int
    accepted_writes: int
    avg_read_latency: float
    avg_write_latency: float
    judge_provider_breakdown: dict[str, int] = field(default_factory=dict)
    fallback_count: int = 0
    llm_failure_categories: dict[str, int] = field(default_factory=dict)
    reason_code_counts: dict[str, int] = field(default_factory=dict)
    avg_judge_latency: float | None = None


class Simulator:
    """Minimal discrete-event simulator vertical slice.

    Supports one cache level, write-through protocol, and 4 event types.
    TODO(m1): add SUMMARIZE/SYNC and protocol plug-ins.
    """

    def __init__(
        self,
        agents: list[Agent],
        global_memory: GlobalMemory,
        protocol: ConsistencyProtocol | None = None,
    ) -> None:
        self.agents = {a.agent_id: a for a in agents} # agents indexed
        self.global_memory = global_memory # 
        self.queue = EventQueue()
        self.protocol = protocol or WriteThroughStrongProtocol()
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
    
    def build_report(self) -> RunReport: # creates benchmark output aligned with protocol comparison goals
        total_events = len(self.trace)
        cache_hits = sum(1 for t in self.trace if t.event == "EV_CACHE_HIT")
        cache_misses = sum(1 for t in self.trace if t.event == "EV_CACHE_MISS")
        conflict_checks = sum(1 for t in self.trace if t.event == EventType.EV_CONFLICT_CHECK.value)
        contested_writes = sum(
            1
            for t in self.trace
            if t.event == "EV_WRITE_REQ" and t.metadata.get("coherence_state") == "contested"
        )
        accepted_writes = sum(
            1
            for t in self.trace
            if t.event == "EV_WRITE_REQ" and t.metadata.get("coherence_state") == "accepted"
        )

        read_samples = [lat for a in self.agents.values() for lat in a.stats.read_latencies]
        write_samples = [lat for a in self.agents.values() for lat in a.stats.write_latencies]
        avg_read_latency = sum(read_samples) / len(read_samples) if read_samples else 0.0
        avg_write_latency = sum(write_samples) / len(write_samples) if write_samples else 0.0

        # filter for conflict check metadata
        conflict_metadata = [
            t.metadata for t in self.trace if t.event == EventType.EV_CONFLICT_CHECK.value
        ]

        # specifies whether we are using LLM or deterministic fallback judge
        provider_counter = Counter(
            str(m.get("provider") or m.get("judge_provider"))
            for m in conflict_metadata
            if (m.get("provider") or m.get("judge_provider")) is not None
        )
        fallback_count = sum(1 for m in conflict_metadata if bool(m.get("fallback_used") or m.get("judge_fallback_used")))

        # breaks down LLM failure distribution
        failure_counter = Counter()
        for m in conflict_metadata:
            warning = m.get("warning") or m.get("judge_warning")
            if isinstance(warning, str) and warning:
                failure_counter[warning] += 1

        # how often each decision reason was invoked
        reason_counter = Counter()
        for m in conflict_metadata:
            for reason in m.get("reason_codes", []):
                reason_counter[str(reason)] += 1

        latency_samples = [
            float(m["judge_latency_ms"])
            for m in conflict_metadata
            if isinstance(m.get("judge_latency_ms"), (int, float))
        ]

        # latency contributed by judge
        avg_judge_latency = sum(latency_samples) / len(latency_samples) if latency_samples else None

        return RunReport(
            total_events=total_events,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            conflict_checks=conflict_checks,
            contested_writes=contested_writes,
            accepted_writes=accepted_writes,
            avg_read_latency=avg_read_latency,
            avg_write_latency=avg_write_latency,
            judge_provider_breakdown=dict(provider_counter),
            fallback_count=fallback_count,
            llm_failure_categories=dict(failure_counter),
            reason_code_counts=dict(reason_counter),
            avg_judge_latency=avg_judge_latency,
        )

    def _handle(self, event: Event) -> None:
        handlers = {
            EventType.EV_READ_REQ: self.protocol.on_read_req,
            EventType.EV_READ_RESP: self.protocol.on_read_resp,
            EventType.EV_WRITE_REQ: self.protocol.on_write_req,
            EventType.EV_WRITE_COMMIT: self.protocol.on_write_commit,
            EventType.EV_CONFLICT_CHECK: self.protocol.on_sync_req,
        }
        handlers[event.type](self, event)

    @staticmethod
    def trace_line_type(t: int, event: str, detail: str, metadata: dict[str, object] | None = None) -> TraceLine:
        return TraceLine(t=t, event=event, detail=detail, metadata=metadata or {})