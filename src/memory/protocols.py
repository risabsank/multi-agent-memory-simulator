from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from .events import Event, EventType
from .model import Artifact, ArtifactScope, CacheEntry, ClaimType, CoherenceState

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
    
    @staticmethod
    def _resolve_coherence_state(old_artifact: Artifact | None, confidence: float) -> CoherenceState:
        if old_artifact and confidence < old_artifact.confidence:
            return CoherenceState.CONTESTED
        return CoherenceState.ACCEPTED

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
                    metadata={"agent": agent.agent_id, "artifact_id": artifact_id, "read_source": "cache"},
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
                    "read_source": "cache",
                },
            )
            return

        agent.stats.misses += 1
        simulator.trace.append(
            simulator.trace_line_type(
                simulator.now,
                "EV_CACHE_MISS",
                f"{agent.agent_id} {artifact_id}",
                metadata={"agent": agent.agent_id, "artifact_id": artifact_id, "read_source": "global"},
            )
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
                "read_source": "global",
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
        agent.stats.read_latencies.append(latency)
        simulator.trace.append(
            simulator.trace_line_type(
                simulator.now,
                EventType.EV_READ_RESP.value,
                f"{agent.agent_id} got {artifact_id} v{version_id} latency={latency}",
                metadata={
                    "agent": agent.agent_id,
                    "artifact_id": artifact_id,
                    "version_id": version_id,
                    "latency": latency,
                    "read_source": event.payload.get("read_source", event.src),
                    "coherence_state": artifact.coherence_state.value,
                },
                # observabiliy and semantic state become queryable per read event
            )
        )

    def on_write_req(self, simulator: Simulator, event: Event) -> None:
        agent = simulator.agents[event.src]
        artifact_id = tuple(event.payload["artifact_id"])
        size = event.payload["size"]
        requested_t = event.payload["requested_t"]

        new_version = simulator.clock.next(artifact_id)
        old_artifact = simulator.global_memory.store.get(artifact_id)

        scope = old_artifact.scope if old_artifact else ArtifactScope.TASK
        claim_type = old_artifact.claim_type if old_artifact else ClaimType.PLAN
        confidence = float(event.payload.get("confidence", 0.8)) # confidence defaults to 0.8 only if caller didn't prvide one
        coherence_state = self._resolve_coherence_state(old_artifact, confidence)

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
                metadata={
                    "agent": agent.agent_id,
                    "artifact_id": artifact_id,
                    "version_id": new_version,
                    "coherence_state": coherence_state.value,
                    "confidence": confidence,
                },
            )
        )

        simulator.queue.push(
            t=simulator.now,
            event_type=EventType.EV_CONFLICT_CHECK,
            src=agent.agent_id,
            dst="global",
            payload={
                "artifact_id": artifact_id,
                "new_version": new_version,
                "old_version": old_artifact.version_id if old_artifact else None,
                "coherence_state": coherence_state.value,
                "confidence": confidence,
            },
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
                "claim_type": claim_type,
                "provenance": agent.agent_id,
                "confidence": confidence,
                "coherence_state": coherence_state,
                "observed_at": simulator.now,
                "valid_at": None,
                "requested_t": requested_t,
            },
        )

    def on_write_commit(self, simulator: Simulator, event: Event) -> None:
        artifact_id = tuple(event.payload["artifact_id"])
        scope = event.payload["scope"]
        claim_type = event.payload["claim_type"]
        coherence_state = event.payload["coherence_state"]

        if isinstance(scope, str):
            scope = ArtifactScope(scope)
        if isinstance(claim_type, str):
            claim_type = ClaimType(claim_type)
        if isinstance(coherence_state, str):
            coherence_state = CoherenceState(coherence_state)

        artifact = Artifact( # constructs full Artifact with metadata
            artifact_id=artifact_id,
            version_id=event.payload["version_id"],
            size=event.payload["size"],
            scope=scope,
            claim_type=claim_type,
            provenance=event.payload["provenance"],
            confidence=event.payload["confidence"],
            coherence_state=coherence_state,
            observed_at=event.payload["observed_at"],
            valid_at=event.payload["valid_at"],
        )
        simulator.global_memory.store[artifact_id] = artifact

        latency = simulator.now - int(event.payload["requested_t"])
        writer = simulator.agents[event.src]
        writer.stats.write_latency_total += latency
        writer.stats.write_count += 1
        writer.stats.write_latencies.append(latency)

        simulator.trace.append(
            simulator.trace_line_type(
                simulator.now,
                EventType.EV_WRITE_COMMIT.value,
                f"global committed {artifact_id} v{artifact.version_id}",
                metadata={
                    "artifact_id": artifact_id,
                    "version_id": artifact.version_id,
                    "coherence_state": artifact.coherence_state.value,
                    "confidence": artifact.confidence,
                    "provenance": artifact.provenance,
                    "latency": latency,
                },
            )
        )

    def on_sync_req(self, simulator: Simulator, event: Event) -> None:
        artifact_id = tuple(event.payload["artifact_id"])
        simulator.trace.append( # appends conflict-check trace with comparison metadata
            simulator.trace_line_type(
                simulator.now,
                EventType.EV_CONFLICT_CHECK.value,
                f"checked {artifact_id} coherence={event.payload['coherence_state']}",
                metadata={
                    "artifact_id": artifact_id,
                    "coherence_state": event.payload["coherence_state"],
                    "new_version": event.payload["new_version"],
                    "old_version": event.payload["old_version"],
                    "confidence": event.payload["confidence"],
                },
            )
        )


class EventualProtocol:
    """Eventual consistency protocol with deferred global propagation.

    Reads always prefer the local cache. Writes become visible immediately to the
    writing agent's cache, then propagate to global memory later via
    ``EV_CONFLICT_CHECK`` and ``EV_WRITE_COMMIT``.
    """

    def __init__(self, propagation_delay: int = 2) -> None:
        # Number of global-memory latency windows to wait before syncing writes.
        # A value >1 creates an observable stale-read window.
        self.propagation_delay = max(1, propagation_delay)

    @staticmethod
    def _resolve_commit_state(old_artifact: Artifact | None, confidence: float) -> CoherenceState:
        # The final committed state is accepted unless confidence regresses.
        if old_artifact and confidence < old_artifact.confidence:
            return CoherenceState.CONTESTED
        return CoherenceState.ACCEPTED

    def on_read_req(self, simulator: Simulator, event: Event) -> None:
        agent = simulator.agents[event.src]
        artifact_id = tuple(event.payload["artifact_id"])
        requested_t = event.payload["requested_t"]

        if artifact_id in agent.cache:
            # Eventual consistency favors local responsiveness: cache hits are
            # served immediately even if global memory may have a newer version.
            entry = agent.cache[artifact_id]
            agent.stats.hits += 1
            entry.last_access_t = simulator.now
            simulator.trace.append(
                simulator.trace_line_type(
                    simulator.now,
                    "EV_CACHE_HIT",
                    f"{agent.agent_id} {artifact_id} v{entry.version_id}",
                    metadata={
                        "agent": agent.agent_id,
                        "artifact_id": artifact_id,
                        "read_source": "cache",
                        "stale_possible": True,
                    },
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
                    "read_source": "cache",
                    "stale_possible": True,
                },
            )
            return

        # On miss we still fetch from global memory. If propagation has not
        # occurred yet, this can return an older version.
        agent.stats.misses += 1
        simulator.trace.append(
            simulator.trace_line_type(
                simulator.now,
                "EV_CACHE_MISS",
                f"{agent.agent_id} {artifact_id}",
                metadata={"agent": agent.agent_id, "artifact_id": artifact_id, "read_source": "global"},
            )
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
                "read_source": "global",
                "stale_possible": False,
            },
        )

    def on_read_resp(self, simulator: Simulator, event: Event) -> None:
        agent = simulator.agents[event.dst]
        artifact_id = tuple(event.payload["artifact_id"])
        version_id = int(event.payload["version_id"])
        requested_t = int(event.payload["requested_t"])

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
        agent.stats.read_latencies.append(latency)

        simulator.trace.append(
            simulator.trace_line_type(
                simulator.now,
                EventType.EV_READ_RESP.value,
                f"{agent.agent_id} got {artifact_id} v{version_id} latency={latency}",
                metadata={
                    "agent": agent.agent_id,
                    "artifact_id": artifact_id,
                    "version_id": version_id,
                    "latency": latency,
                    "read_source": event.payload.get("read_source", event.src),
                    "stale_possible": bool(event.payload.get("stale_possible", False)),
                    "global_version_at_read": artifact.version_id,
                    "coherence_state": artifact.coherence_state.value,
                },
            )
        )

    def on_write_req(self, simulator: Simulator, event: Event) -> None:
        agent = simulator.agents[event.src]
        artifact_id = tuple(event.payload["artifact_id"])
        size = int(event.payload["size"])
        requested_t = int(event.payload["requested_t"])

    def on_write_commit(self, simulator: Simulator, event: Event) -> None:
        new_version = simulator.clock.next(artifact_id)
        old_artifact = simulator.global_memory.store.get(artifact_id)
        scope = old_artifact.scope if old_artifact else ArtifactScope.TASK
        claim_type = old_artifact.claim_type if old_artifact else ClaimType.PLAN
        confidence = float(event.payload.get("confidence", 0.8))

        # Write is immediately visible locally but not globally committed yet.
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
                f"{agent.agent_id} wrote {artifact_id} v{new_version} (pending sync)",
                metadata={
                    "agent": agent.agent_id,
                    "artifact_id": artifact_id,
                    "version_id": new_version,
                    "coherence_state": CoherenceState.PROVISIONAL.value,
                    "confidence": confidence,
                    "pending_global_commit": True,
                },
            )
        )

        # The sync/check event represents asynchronous dissemination.
        simulator.queue.push(
            t=simulator.now + simulator.global_memory.latency * self.propagation_delay,
            event_type=EventType.EV_CONFLICT_CHECK,
            src=agent.agent_id,
            dst="global",
            payload={
                "artifact_id": artifact_id,
                "version_id": new_version,
                "size": size,
                "scope": scope,
                "claim_type": claim_type,
                "provenance": agent.agent_id,
                "confidence": confidence,
                "requested_t": requested_t,
                "observed_at": simulator.now,
                "valid_at": None,
                "old_version": old_artifact.version_id if old_artifact else None,
            },
        )

    def on_sync_req(self, simulator: Simulator, event: Event) -> None:
        artifact_id = tuple(event.payload["artifact_id"])
        confidence = float(event.payload["confidence"])
        previous = simulator.global_memory.store.get(artifact_id)
        coherence_state = self._resolve_commit_state(previous, confidence)

        simulator.trace.append(
            simulator.trace_line_type(
                simulator.now,
                EventType.EV_CONFLICT_CHECK.value,
                f"eventual sync {artifact_id} v{event.payload['version_id']}",
                metadata={
                    "artifact_id": artifact_id,
                    "new_version": event.payload["version_id"],
                    "old_version": event.payload["old_version"],
                    "coherence_state": coherence_state.value,
                    "confidence": confidence,
                    "delayed_propagation": True,
                },
            )
        )

        # Commit is intentionally delayed until after sync to model eventuality.
        simulator.queue.push(
            t=simulator.now + simulator.global_memory.latency,
            event_type=EventType.EV_WRITE_COMMIT,
            src=event.src,
            dst="global",
            payload={
                "artifact_id": artifact_id,
                "version_id": event.payload["version_id"],
                "size": event.payload["size"],
                "scope": event.payload["scope"],
                "claim_type": event.payload["claim_type"],
                "provenance": event.payload["provenance"],
                "confidence": confidence,
                "coherence_state": coherence_state,
                "observed_at": event.payload["observed_at"],
                "valid_at": event.payload["valid_at"],
                "requested_t": event.payload["requested_t"],
            },
        )

    def on_write_commit(self, simulator: Simulator, event: Event) -> None:
        artifact_id = tuple(event.payload["artifact_id"])
        scope = event.payload["scope"]
        claim_type = event.payload["claim_type"]
        coherence_state = event.payload["coherence_state"]

        if isinstance(scope, str):
            scope = ArtifactScope(scope)
        if isinstance(claim_type, str):
            claim_type = ClaimType(claim_type)
        if isinstance(coherence_state, str):
            coherence_state = CoherenceState(coherence_state)

        artifact = Artifact(
            artifact_id=artifact_id,
            version_id=int(event.payload["version_id"]),
            size=int(event.payload["size"]),
            scope=scope,
            claim_type=claim_type,
            provenance=event.payload["provenance"],
            confidence=float(event.payload["confidence"]),
            coherence_state=coherence_state,
            observed_at=int(event.payload["observed_at"]),
            valid_at=event.payload["valid_at"],
        )
        simulator.global_memory.store[artifact_id] = artifact

        writer = simulator.agents[event.src]
        latency = simulator.now - int(event.payload["requested_t"])
        writer.stats.write_latency_total += latency
        writer.stats.write_count += 1
        writer.stats.write_latencies.append(latency)

        simulator.trace.append(
            simulator.trace_line_type(
                simulator.now,
                EventType.EV_WRITE_COMMIT.value,
                f"eventual commit {artifact_id} v{artifact.version_id}",
                metadata={
                    "artifact_id": artifact_id,
                    "version_id": artifact.version_id,
                    "coherence_state": artifact.coherence_state.value,
                    "confidence": artifact.confidence,
                    "provenance": artifact.provenance,
                    "latency": latency,
                    "delayed_propagation": True,
                },
            )
        )
