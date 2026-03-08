from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

from .events import Event, EventQueue, EventType
from .model import Agent, ArtifactId, GlobalMemory, VersionClock
from .protocols import ConsistencyProtocol, WriteThroughStrongProtocol


# structured execution log entry for each simulation step
# acts as an event journal
@dataclass
class TraceLine:
    t: int  # simulation time
    event: str  # event label
    detail: str  # summary
    metadata: dict[str, object | Iterable[Any]] = field(default_factory=dict)  # context


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
    sync_requests: int = 0
    invalidations: int = 0
    cache_hit_rate: float = 0.0
    contested_ratio: float = 0.0
    avg_visibility_lag: float = 0.0
    stale_reads: int = 0
    stale_read_rate: float = 0.0
    avg_convergence_time: float = 0.0
    read_latency_p50: float = 0.0
    read_latency_p95: float = 0.0
    read_latency_p99: float = 0.0
    write_latency_p50: float = 0.0
    write_latency_p95: float = 0.0
    write_latency_p99: float = 0.0
    read_your_writes_checks: int = 0
    read_your_writes_successes: int = 0
    read_your_writes_success_rate: float = 0.0
    monotonic_read_violations: int = 0
    avg_stale_version_gap: float = 0.0
    p95_stale_version_gap: float = 0.0
    max_stale_version_gap: int = 0

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
        self.agents = {a.agent_id: a for a in agents}  # agents indexed
        self.global_memory = global_memory  #
        self.queue = EventQueue()
        self.protocol = protocol or WriteThroughStrongProtocol()
        self.clock = VersionClock()

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

    def schedule_write(
        self, t: int, agent_id: str, artifact_id: ArtifactId, size: int
    ) -> None:
        self.queue.push(
            t=t,
            event_type=EventType.EV_WRITE_REQ,
            src=agent_id,
            dst=agent_id,
            payload={"artifact_id": artifact_id, "size": size, "requested_t": t},
        )

    def schedule_sync(self, t: int, agent_id: str, artifact_id: ArtifactId) -> None:
        self.queue.push(
            t=t,
            event_type=EventType.EV_SYNC_REQ,
            src=agent_id,
            dst="global",
            payload={"artifact_id": artifact_id, "requested_t": t},
        )

    def schedule_invalidate(
        self, t: int, agent_id: str, artifact_id: ArtifactId, reason: str = "manual"
    ) -> None:
        self.queue.push(
            t=t,
            event_type=EventType.EV_INVALIDATE,
            src="global",
            dst=agent_id,
            payload={"artifact_id": artifact_id, "reason": reason},
        )

    def pre_run(self) -> None:
        """
        Run some bookkeeping/initialization code prior to run() but after __init__()
        """
        for artifact_id, artifact in self.global_memory.get_all_artifacts():
            self.clock._versions[artifact_id] = artifact.version_id

    def run(self) -> SimulationResult:
        self.pre_run()
        while len(self.queue) > 0:  # while queue not empry
            event = self.queue.pop()  # pop next event
            self.now = event.t
            self._handle(event)  # based on event type, perform an action

        return SimulationResult(
            trace=self.trace, agents=self.agents, global_memory=self.global_memory
        )

    def build_report(
        self,
    ) -> RunReport:  # creates benchmark output aligned with protocol comparison goals
        ordered_trace = sorted(self.trace, key=lambda line: line.t)
        total_events = len(self.trace)
        cache_hits = sum(1 for t in ordered_trace if t.event == "EV_CACHE_HIT")
        cache_misses = sum(1 for t in ordered_trace if t.event == "EV_CACHE_MISS")
        conflict_checks = sum(
            1 for t in ordered_trace if t.event == EventType.EV_CONFLICT_CHECK.value
        )

        sync_requests = sum(
            1 for t in ordered_trace if t.event == EventType.EV_SYNC_REQ.value
        )
        invalidations = sum(
            1 for t in ordered_trace if t.event == EventType.EV_INVALIDATE.value
        )

        contested_writes = sum(
            1
            for t in ordered_trace
            if t.event == "EV_WRITE_REQ"
            and t.metadata.get("coherence_state") == "contested"
        )
        accepted_writes = sum(
            1
            for t in ordered_trace
            if t.event == "EV_WRITE_REQ"
            and t.metadata.get("coherence_state") == "accepted"
        )

        read_samples = [
            lat for a in self.agents.values() for lat in a.stats.read_latencies
        ]
        write_samples = [
            lat for a in self.agents.values() for lat in a.stats.write_latencies
        ]
        avg_read_latency = (
            sum(read_samples) / len(read_samples) if read_samples else 0.0
        )
        avg_write_latency = (
            sum(write_samples) / len(write_samples) if write_samples else 0.0
        )

        def _percentile(samples: list[int], pct: float) -> float:
            if not samples:
                return 0.0
            ordered = sorted(samples)
            idx = int((len(ordered) - 1) * pct)
            return float(ordered[idx])

        read_latency_p50 = _percentile(read_samples, 0.50)
        read_latency_p95 = _percentile(read_samples, 0.95)
        read_latency_p99 = _percentile(read_samples, 0.99)
        write_latency_p50 = _percentile(write_samples, 0.50)
        write_latency_p95 = _percentile(write_samples, 0.95)
        write_latency_p99 = _percentile(write_samples, 0.99)

        # filter for conflict check metadata
        conflict_metadata = [
            t.metadata
            for t in ordered_trace
            if t.event == EventType.EV_CONFLICT_CHECK.value
        ]


        cache_total = cache_hits + cache_misses
        cache_hit_rate = cache_hits / cache_total if cache_total else 0.0

        write_total = accepted_writes + contested_writes
        contested_ratio = contested_writes / write_total if write_total else 0.0

        write_commits = [
            t
            for t in ordered_trace
            if t.event == EventType.EV_WRITE_COMMIT.value
            and isinstance(t.metadata.get("latency"), (int, float))
        ]
        avg_visibility_lag = (
            sum(float(t.metadata["latency"]) for t in write_commits) / len(write_commits)
            if write_commits
            else 0.0
        )

        read_events = [
            t for t in ordered_trace if t.event == EventType.EV_READ_RESP.value
        ]
        stale_reads = sum(
            1
            for t in read_events
            if isinstance(t.metadata.get("version_id"), int)
            and isinstance(t.metadata.get("global_version_at_read"), int)
            and int(t.metadata["version_id"]) < int(t.metadata["global_version_at_read"])
        )
        stale_read_rate = stale_reads / len(read_events) if read_events else 0.0

        stale_version_gaps = [
            int(t.metadata["global_version_at_read"]) - int(t.metadata["version_id"])
            for t in read_events
            if isinstance(t.metadata.get("version_id"), int)
            and isinstance(t.metadata.get("global_version_at_read"), int)
            and int(t.metadata["version_id"]) < int(t.metadata["global_version_at_read"])
        ]
        avg_stale_version_gap = (
            sum(stale_version_gaps) / len(stale_version_gaps)
            if stale_version_gaps
            else 0.0
        )
        p95_stale_version_gap = (
            _percentile(stale_version_gaps, 0.95) if stale_version_gaps else 0.0
        )
        max_stale_version_gap = max(stale_version_gaps) if stale_version_gaps else 0

        # Approximate convergence: after global commit, how long until each other agent
        # first reads that version (or newer) for the same artifact.
        convergence_samples: list[int] = []
        for commit in write_commits:
            artifact_id = commit.metadata.get("artifact_id")
            version_id = commit.metadata.get("version_id")
            writer = commit.metadata.get("provenance")
            if not isinstance(artifact_id, tuple) or not isinstance(version_id, int):
                continue
            for reader_id in self.agents:
                if reader_id == writer:
                    continue
                catchup = next(
                    (
                        r
                        for r in read_events
                        if r.t >= commit.t
                        and r.metadata.get("agent") == reader_id
                        and r.metadata.get("artifact_id") == artifact_id
                        and isinstance(r.metadata.get("version_id"), int)
                        and int(r.metadata["version_id"]) >= version_id
                    ),
                    None,
                )
                if catchup is not None:
                    convergence_samples.append(catchup.t - commit.t)
        avg_convergence_time = (
            sum(convergence_samples) / len(convergence_samples)
            if convergence_samples
            else 0.0
        )

        # Read-your-writes approximation: writer's first post-write read should observe
        # at least the written version for the same artifact.
        write_requests = [
            t
            for t in ordered_trace
            if t.event == EventType.EV_WRITE_REQ.value
            and isinstance(t.metadata.get("agent"), str)
            and isinstance(t.metadata.get("artifact_id"), tuple)
            and isinstance(t.metadata.get("version_id"), int)
        ]
        read_your_writes_checks = 0
        read_your_writes_successes = 0
        for write_req in write_requests:
            writer = str(write_req.metadata["agent"])
            artifact_id = write_req.metadata["artifact_id"]
            written_version = int(write_req.metadata["version_id"])
            writer_followup = next(
                (
                    r
                    for r in read_events
                    if r.t >= write_req.t
                    and r.metadata.get("agent") == writer
                    and r.metadata.get("artifact_id") == artifact_id
                    and isinstance(r.metadata.get("version_id"), int)
                ),
                None,
            )
            if writer_followup is None:
                continue
            read_your_writes_checks += 1
            if int(writer_followup.metadata["version_id"]) >= written_version:
                read_your_writes_successes += 1
        read_your_writes_success_rate = (
            read_your_writes_successes / read_your_writes_checks
            if read_your_writes_checks
            else 0.0
        )

        monotonic_read_violations = 0
        per_reader_stream: dict[tuple[str, object], int] = {}
        for read_event in read_events:
            agent_id = read_event.metadata.get("agent")
            artifact_id = read_event.metadata.get("artifact_id")
            version_id = read_event.metadata.get("version_id")
            if (
                not isinstance(agent_id, str)
                or not isinstance(version_id, int)
                or artifact_id is None
            ):
                continue
            key = (agent_id, artifact_id)
            previous_version = per_reader_stream.get(key)
            if previous_version is not None and version_id < previous_version:
                monotonic_read_violations += 1
            per_reader_stream[key] = version_id

        # specifies whether we are using LLM or deterministic fallback judge
        provider_counter = Counter(
            str(m.get("provider") or m.get("judge_provider"))
            for m in conflict_metadata
            if (m.get("provider") or m.get("judge_provider")) is not None
        )
        fallback_count = sum(
            1
            for m in conflict_metadata
            if bool(m.get("fallback_used") or m.get("judge_fallback_used"))
        )

        # breaks down LLM failure distribution
        failure_counter = Counter()
        for m in conflict_metadata:
            warning = m.get("warning") or m.get("judge_warning")
            if isinstance(warning, str) and warning:
                failure_counter[warning] += 1

        # how often each decision reason was invoked
        reason_counter = Counter()
        for m in conflict_metadata:
            reason_codes = m.get("reason_codes", [])
            assert isinstance(reason_codes, Iterable)
            for reason in reason_codes:
                reason_counter[str(reason)] += 1

        latency_samples = [
            float(m["judge_latency_ms"])  # type: ignore[reportArgumentType]
            for m in conflict_metadata
            if isinstance(m.get("judge_latency_ms"), (int, float))
        ]

        # latency contributed by judge
        avg_judge_latency = (
            sum(latency_samples) / len(latency_samples) if latency_samples else None
        )

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
            sync_requests=sync_requests,
            invalidations=invalidations,
            cache_hit_rate=cache_hit_rate,
            contested_ratio=contested_ratio,
            avg_visibility_lag=avg_visibility_lag,
            stale_reads=stale_reads,
            stale_read_rate=stale_read_rate,
            avg_convergence_time=avg_convergence_time,
            read_latency_p50=read_latency_p50,
            read_latency_p95=read_latency_p95,
            read_latency_p99=read_latency_p99,
            write_latency_p50=write_latency_p50,
            write_latency_p95=write_latency_p95,
            write_latency_p99=write_latency_p99,
            read_your_writes_checks=read_your_writes_checks,
            read_your_writes_successes=read_your_writes_successes,
            read_your_writes_success_rate=read_your_writes_success_rate,
            monotonic_read_violations=monotonic_read_violations,
            avg_stale_version_gap=avg_stale_version_gap,
            p95_stale_version_gap=p95_stale_version_gap,
            max_stale_version_gap=max_stale_version_gap,
        )

    def _handle(self, event: Event) -> None:
        handlers = {
            EventType.EV_READ_REQ: self.protocol.on_read_req,
            EventType.EV_READ_RESP: self.protocol.on_read_resp,
            EventType.EV_WRITE_REQ: self.protocol.on_write_req,
            EventType.EV_WRITE_COMMIT: self.protocol.on_write_commit,
            EventType.EV_CONFLICT_CHECK: self.protocol.on_sync_req,
            EventType.EV_SYNC_REQ: self.protocol.on_sync_req,
            EventType.EV_INVALIDATE: self.protocol.on_invalidate_req,
        }
        handlers[event.type](self, event)

    @staticmethod
    def trace_line_type(
        t: int, event: str, detail: str, metadata: dict[str, object] | None = None
    ) -> TraceLine:
        return TraceLine(t=t, event=event, detail=detail, metadata=metadata or {})
