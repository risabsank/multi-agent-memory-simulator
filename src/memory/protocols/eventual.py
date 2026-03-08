from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from memory.protocols.base import ConsistencyProtocol

from ..events import Event, EventType
from ..model import Artifact, ArtifactScope, CacheEntry, ClaimType, CoherenceState
from .judges import ConflictJudge, DeterministicConflictJudge, build_conflict_judge

if TYPE_CHECKING:
    from ..simulator import Simulator


class EventualProtocol(ConsistencyProtocol):
    """Eventual consistency protocol with deferred global propagation.

    Reads always prefer the local cache. Writes become visible immediately to the
    writing agent's cache, then propagate to global memory later via
    ``EV_CONFLICT_CHECK`` and ``EV_WRITE_COMMIT``.
    """

    def __init__(
        self,
        propagation_delay: int = 2,
        conflict_judge: ConflictJudge | None = None,
        auto_invalidate_on_commit: bool = False,
        *,
        judge_mode: str = "deterministic",
        llm_inference_fn: Callable[[str], str] | None = None,
        llm_provider: str = "llm",
        llm_model: str = "unknown",
        llm_timeout_s: float = 0.25,
        deterministic_profile: str = "balanced",
    ) -> None:
        self.propagation_delay = max(1, propagation_delay)
        self.auto_invalidate_on_commit = auto_invalidate_on_commit
        self.deterministic_profile = deterministic_profile
        self.conflict_judge = conflict_judge or build_conflict_judge(
            judge_mode=judge_mode,
            llm_inference_fn=llm_inference_fn,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_timeout_s=llm_timeout_s,
        )

    def _resolve_commit_state(
        self, old_artifact: Artifact | None, confidence: float
    ) -> CoherenceState:
        if old_artifact is None:
            return CoherenceState.ACCEPTED
        min_delta = DeterministicConflictJudge.PROFILE_THRESHOLDS[
            self.deterministic_profile
        ]
        delta = confidence - old_artifact.confidence
        if delta < min_delta:
            return CoherenceState.CONTESTED
        return CoherenceState.ACCEPTED

    def on_read_req(self, simulator: Simulator, event: Event) -> None:
        agent = simulator.agents[event.src]
        artifact_id = tuple(event.payload["artifact_id"])
        requested_t = event.payload["requested_t"]
        t = simulator.now

        # prior todo: technically, the latency for a cache hit and cache miss are probably different
        # edit: I think I resolved this in the Cache class
        t += agent.cache.read_artifact_latency(artifact_id)
        if agent.cache.artifact_exists(artifact_id):
            agent.cache.read_artifact(artifact_id)
            entry = agent.cache.get_artifact(artifact_id)
            agent.stats.hits += 1
            simulator.trace.append(
                simulator.trace_line_type(
                    t,
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
                t=t,
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

        agent.stats.misses += 1
        simulator.trace.append(
            simulator.trace_line_type(
                t,
                "EV_CACHE_MISS",
                f"{agent.agent_id} {artifact_id}",
                metadata={
                    "agent": agent.agent_id,
                    "artifact_id": artifact_id,
                    "read_source": "global",
                },
            )
        )

        t += simulator.global_memory.read_artifact_latency(artifact_id)
        simulator.queue.push(
            t=t,
            event_type=EventType.EV_READ_RESP,
            src="global",
            dst=agent.agent_id,
            payload={
                "artifact_id": artifact_id,
                "version_id": simulator.global_memory.get_artifact(
                    artifact_id
                ).version_id,
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

        artifact = simulator.global_memory.get_artifact(artifact_id)
        agent.cache.store_artifact(artifact)

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
        new_version = simulator.clock.next(artifact_id)
        old_artifact = (
            simulator.global_memory.get_artifact(artifact_id)
            if simulator.global_memory.artifact_exists(artifact_id)
            else None
        )
        scope = old_artifact.scope if old_artifact else ArtifactScope.TASK
        claim_type = old_artifact.claim_type if old_artifact else ClaimType.PLAN
        confidence = float(event.payload.get("confidence", 0.8))

        # a write request will trigger two EV_WRITE_COMMIT (eventually), one for cache and one for global
        # cache will be immediate and global will be eventual

        # TODO: this function requires a bunch of prefilled fields in event, but I don't know if they are filled in at this time
        pending_artifact = Artifact(  # changed this to build an explicit pending artifact so latency estimation does not depend on pre-existing store state
            artifact_id=artifact_id,
            version_id=new_version,
            size=size,
            scope=scope,
            claim_type=claim_type,
            provenance=agent.agent_id,
            confidence=confidence,
            coherence_state=CoherenceState.ACCEPTED,  # changed this to keep pending cache-commit artifact construction explicit without relying on an undefined local state
            observed_at=simulator.now,
            valid_at=None,
        )

        cache_commit_time = simulator.now + agent.cache.store_artifact_latency(
            pending_artifact
        )
        simulator.queue.push(
            t=cache_commit_time,
            event_type=EventType.EV_WRITE_COMMIT,
            src=agent.agent_id,
            dst="cache",
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
                "coherence_state": CoherenceState.ACCEPTED.value,  # TODO: verify that this payload is correct for generated an artifact
            },
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

        global_memory_commit_time = (
            simulator.now
            + simulator.global_memory.store_artifact_latency(pending_artifact)
            * self.propagation_delay
        )
        # To be extra careful, ensure global memory is slower than cache, else the state machine behavior will be weird/undefined
        # even being equal might result in the wrong order of operations, dependent on Python's heap sorting algorithms
        # from my testing, this will ensure that global memory latency > cache latency
        if not global_memory_commit_time > cache_commit_time:
            raise Exception(
                f"Global memory commit time {global_memory_commit_time} is not greater than cache commit time {cache_commit_time}! Ensure that global memory latency is configured to have higher latency that cache latency."
            )
        simulator.queue.push(
            t=global_memory_commit_time,  # changed this to use the explicit pending artifact for dynamic global write latency calculation
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
                "coherence_state": "contested",  # not a real coherence state, required for building a fake artifact in EV_SYNC_REQ to calculate artifact write latency
            },
        )

    def on_sync_req(self, simulator: Simulator, event: Event) -> None:
        if event.type == EventType.EV_SYNC_REQ:
            agent = simulator.agents[event.src]
            artifact_id = tuple(event.payload["artifact_id"])
            if not simulator.global_memory.artifact_exists(artifact_id):
                return

            artifact = simulator.global_memory.get_artifact(artifact_id)
            local_entry = (
                agent.cache.get_artifact(artifact_id)
                if agent.cache.artifact_exists(artifact_id)
                else None
            )
            stale_before = local_entry.version_id if local_entry else None
            agent.cache.store_artifact(artifact)
            simulator.trace.append(
                simulator.trace_line_type(
                    simulator.now,
                    EventType.EV_SYNC_REQ.value,
                    f"{agent.agent_id} synced {artifact_id} to v{artifact.version_id}",
                    metadata={
                        "agent": agent.agent_id,
                        "artifact_id": artifact_id,
                        "version_id": artifact.version_id,
                        "stale_before": stale_before,
                    },
                )
            )
            return

        artifact_id = tuple(event.payload["artifact_id"])
        confidence = float(event.payload["confidence"])
        previous = (
            simulator.global_memory.get_artifact(artifact_id)
            if simulator.global_memory.artifact_exists(artifact_id)
            else None
        )
        decision = self.conflict_judge.judge(
            previous=previous,
            candidate_confidence=confidence,
            candidate_payload=event.payload,
        )

        simulator.trace.append(
            simulator.trace_line_type(
                simulator.now,
                EventType.EV_CONFLICT_CHECK.value,
                f"eventual sync {artifact_id} v{event.payload['version_id']}",
                metadata={
                    "artifact_id": artifact_id,
                    "new_version": event.payload["version_id"],
                    "old_version": event.payload["old_version"],
                    **decision.to_metadata(),
                    "confidence": confidence,
                    "delayed_propagation": True,
                },
            )
        )

        simulator.queue.push(
            t=simulator.now
            + simulator.global_memory.store_artifact_latency(
                ConsistencyProtocol.create_artifact(simulator, event)
            ),
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
                "coherence_state": decision.coherence_state,
                "observed_at": event.payload["observed_at"],
                "valid_at": event.payload["valid_at"],
                "requested_t": event.payload["requested_t"],
            },
        )

    def on_invalidate_req(self, simulator: Simulator, event: Event) -> None:
        agent = simulator.agents[event.dst]
        artifact_id = tuple(event.payload["artifact_id"])
        entry = (
            agent.cache.remove_artifact(artifact_id)
            if agent.cache.artifact_exists(artifact_id)
            else None
        )
        simulator.trace.append(
            simulator.trace_line_type(
                simulator.now,
                EventType.EV_INVALIDATE.value,
                f"{agent.agent_id} invalidated {artifact_id}",
                metadata={
                    "agent": agent.agent_id,
                    "artifact_id": artifact_id,
                    "had_entry": entry is not None,
                    "invalidated_version": entry.version_id if entry else None,
                    "reason": event.payload.get("reason", "unknown"),
                },
            )
        )

    def on_write_commit(self, simulator: Simulator, event: Event) -> None:
        agent = simulator.agents[event.src]
        artifact_id = tuple(event.payload["artifact_id"])

        artifact = ConsistencyProtocol.create_artifact(simulator, event)
        if event.dst == "cache":
            agent.cache.store_artifact(artifact)
            # TODO: some trace here for cache commit??
            # should there be any reason to invalidate a cache write?
            return

        # Since we write to global memory the correct artifact after an EV_SYNC_REQ, we also need to update the cache
        agent.cache.store_artifact(artifact)
        simulator.global_memory.store_artifact(artifact)

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

        if self.auto_invalidate_on_commit:
            for agent_id in simulator.agents:
                if agent_id == event.src:
                    continue
                simulator.queue.push(
                    t=simulator.now,
                    event_type=EventType.EV_INVALIDATE,
                    src="global",
                    dst=agent_id,
                    payload={"artifact_id": artifact_id, "reason": "write_commit"},
                )
