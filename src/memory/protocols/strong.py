from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from memory.protocols.base import ConsistencyProtocol

from ..events import Event, EventType
from ..model import Artifact, ArtifactScope, CacheEntry, ClaimType, CoherenceState
from .judges import ConflictJudge, build_conflict_judge

if TYPE_CHECKING:
    from ..simulator import Simulator


class WriteThroughStrongProtocol(ConsistencyProtocol):
    """Current write-through, strong consistency behavior."""

    def __init__(
        self,
        conflict_judge: ConflictJudge | None = None,
        auto_invalidate_on_commit: bool = False,
        *,
        judge_mode: str = "deterministic",
        llm_inference_fn: Callable[[str], str] | None = None,
        llm_provider: str = "llm",
        llm_model: str = "unknown",
        llm_timeout_s: float = 0.25,
    ) -> None:
        self.conflict_judge = conflict_judge or build_conflict_judge(
            judge_mode=judge_mode,
            llm_inference_fn=llm_inference_fn,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_timeout_s=llm_timeout_s,
        )
        self.auto_invalidate_on_commit = auto_invalidate_on_commit

    @staticmethod
    def _resolve_coherence_state(old_artifact: Artifact | None, confidence: float) -> CoherenceState:
        if old_artifact and confidence < old_artifact.confidence:
            return CoherenceState.CONTESTED
        return CoherenceState.ACCEPTED

    def on_read_req(self, simulator: Simulator, event: Event) -> None:
        agent = simulator.agents[event.src]
        artifact_id = tuple(event.payload["artifact_id"])
        requested_t = event.payload["requested_t"]

        if agent.cache.artifact_exists(artifact_id):
            entry = agent.cache.get_artifact(artifact_id)
            agent.stats.hits += 1
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
            t=simulator.now + simulator.global_memory.read_artifact_latency(artifact_id),
            event_type=EventType.EV_READ_RESP,
            src="global",
            dst=agent.agent_id,
            payload={
                "artifact_id": artifact_id,
                "version_id": simulator.global_memory.get_artifact(artifact_id).version_id,
                "requested_t": requested_t,
                "hit": False,
                "read_source": "global",
            },
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

    def on_read_resp(self, simulator: Simulator, event: Event) -> None:
        agent = simulator.agents[event.dst]
        artifact_id = tuple(event.payload["artifact_id"])
        version_id = event.payload["version_id"]
        requested_t = event.payload["requested_t"]

        artifact = simulator.global_memory.get_artifact(artifact_id)
        if agent.cache.artifact_exists(artifact_id):
            agent.cache.overwrite_artifact(artifact)
        else:
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
        old_artifact = simulator.global_memory.get_artifact(artifact_id) if simulator.global_memory.artifact_exists(artifact_id) else None

        scope = old_artifact.scope if old_artifact else ArtifactScope.TASK
        claim_type = old_artifact.claim_type if old_artifact else ClaimType.PLAN
        confidence = float(event.payload.get("confidence", 0.8))
        coherence_state = self._resolve_coherence_state(old_artifact, confidence)

        artifact = ConsistencyProtocol.create_artifact(simulator, event)
        simulator.queue.push(
            t=simulator.now + agent.cache.store_artifact_latency(artifact),
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
                "coherence_state": coherence_state,
                "observed_at": simulator.now,
                "valid_at": None,
                "requested_t": requested_t,
            },
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
            t=simulator.now + simulator.global_memory.store_artifact_latency(artifact),
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
        agent = simulator.agents[event.src]
        if event.src == "cache":
            artifact = ConsistencyProtocol.create_artifact(simulator, event)
            if agent.cache.artifact_exists(artifact_id):
                agent.cache.overwrite_artifact(artifact)
            else:
                agent.cache.store_artifact(artifact)
            # TODO: add trace log statement
            return

        artifact = ConsistencyProtocol.create_artifact(simulator, event)
        simulator.global_memory.store_artifact(artifact)

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
        if event.type == EventType.EV_SYNC_REQ:
            agent = simulator.agents[event.src]
            artifact_id = tuple(event.payload["artifact_id"])
            if simulator.global_memory.artifact_exists(artifact_id):
                return

            artifact = simulator.global_memory.get_artifact(artifact_id)
            local_entry = agent.cache.get_artifact(artifact_id) if agent.cache.artifact_exists(artifact_id) else None
            stale_before = local_entry.version_id if local_entry else None
            if agent.cache.artifact_exists(artifact_id):
                agent.cache.overwrite_artifact(artifact)
            else:
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

        # Retrieve the currently accepted artifact
        previous = simulator.global_memory.get_artifact(artifact_id) if simulator.global_memory.artifact_exists(artifact_id) else None
        confidence = float(event.payload["confidence"])  # Extract the candidate write's confidence score from the event.
        # conflict resolution policy
        decision = self.conflict_judge.judge(
            previous=previous,
            candidate_confidence=confidence,
            candidate_payload=event.payload,
        )

        simulator.trace.append(
            simulator.trace_line_type(
                simulator.now,
                EventType.EV_CONFLICT_CHECK.value,
                f"checked {artifact_id} coherence={decision.coherence_state.value}",
                metadata={
                    "artifact_id": artifact_id,
                    **decision.to_metadata(),
                    "new_version": event.payload["new_version"],
                    "old_version": event.payload["old_version"],
                    "confidence": confidence,
                },
            )
        )

    def on_invalidate_req(self, simulator: Simulator, event: Event) -> None:
        agent = simulator.agents[event.dst]
        artifact_id = tuple(event.payload["artifact_id"])
        entry = agent.cache.remove_artifact(artifact_id) if agent.cache.artifact_exists(artifact_id) else None
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
