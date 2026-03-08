from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterable

from memory.protocols.base import ConsistencyProtocol

from ..events import Event, EventType
from ..model import Artifact, ArtifactScope, ClaimType, CoherenceState
from .eventual import EventualProtocol
from .judges import ConflictJudge, build_conflict_judge

if TYPE_CHECKING:
    from ..simulator import Simulator


class HybridProtocol(ConsistencyProtocol):
    """Hybrid consistency protocol.

    Defaults to eventual propagation, but escalates specific writes to strong/sync
    handling when:
      - the claim type is high-risk, or
      - confidence gap vs. current global value is small.
    """

    def __init__(
        self,
        propagation_delay: int = 2,
        conflict_judge: ConflictJudge | None = None,
        auto_invalidate_on_commit: bool = False,
        *,
        high_risk_claim_types: Iterable[ClaimType | str] | None = None,
        confidence_gap_threshold: float = 0.05,
        judge_mode: str = "deterministic",
        llm_inference_fn: Callable[[str], str] | None = None,
        llm_provider: str = "llm",
        llm_model: str = "unknown",
        llm_timeout_s: float = 0.25,
    ) -> None:
        self.propagation_delay = max(1, propagation_delay)
        self.auto_invalidate_on_commit = auto_invalidate_on_commit
        self.confidence_gap_threshold = max(0.0, confidence_gap_threshold)
        self.conflict_judge = conflict_judge or build_conflict_judge(
            judge_mode=judge_mode,
            llm_inference_fn=llm_inference_fn,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_timeout_s=llm_timeout_s,
        )
        default_high_risk = {ClaimType.FACT, ClaimType.PLAN}
        configured = high_risk_claim_types or default_high_risk
        self.high_risk_claim_types = {
            ClaimType(v) if isinstance(v, str) else v for v in configured
        }

        # Reuse eventual read/response/invalidation semantics.
        self._eventual_reader = EventualProtocol(
            propagation_delay=propagation_delay,
            conflict_judge=self.conflict_judge,
            auto_invalidate_on_commit=auto_invalidate_on_commit,
        )

    def on_read_req(self, simulator: Simulator, event: Event) -> None:
        self._eventual_reader.on_read_req(simulator, event)

    def on_read_resp(self, simulator: Simulator, event: Event) -> None:
        self._eventual_reader.on_read_resp(simulator, event)

    def _resolve_write_mode(
        self,
        *,
        previous: Artifact | None,
        claim_type: ClaimType,
        candidate_confidence: float,
    ) -> tuple[str, str]:
        if claim_type in self.high_risk_claim_types:
            return ("strong", "high_risk_claim")

        if previous is not None:
            gap = abs(candidate_confidence - previous.confidence)
            if gap <= self.confidence_gap_threshold:
                return ("strong", "small_confidence_gap")

        return ("eventual", "default_eventual")

    def on_write_req(self, simulator: Simulator, event: Event) -> None:
        agent = simulator.agents[event.src]
        artifact_id = tuple(event.payload["artifact_id"])
        size = int(event.payload["size"])
        requested_t = int(event.payload["requested_t"])
        new_version = simulator.clock.next(artifact_id)

        previous = (
            simulator.global_memory.get_artifact(artifact_id)
            if simulator.global_memory.artifact_exists(artifact_id)
            else None
        )
        scope = event.payload.get(
            "scope", previous.scope if previous else ArtifactScope.TASK
        )
        claim_type = event.payload.get(
            "claim_type", previous.claim_type if previous else ClaimType.PLAN
        )
        if isinstance(scope, str):
            scope = ArtifactScope(scope)
        if isinstance(claim_type, str):
            claim_type = ClaimType(claim_type)

        confidence = float(event.payload.get("confidence", 0.8))
        write_mode, force_reason = self._resolve_write_mode(
            previous=previous,
            claim_type=claim_type,
            candidate_confidence=confidence,
        )

        coherence_state = CoherenceState.ACCEPTED
        if previous and confidence < previous.confidence:
            coherence_state = CoherenceState.CONTESTED

        pending_artifact = Artifact(
            artifact_id=artifact_id,
            version_id=new_version,
            size=size,
            scope=scope,
            claim_type=claim_type,
            provenance=agent.agent_id,
            confidence=confidence,
            coherence_state=coherence_state,
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
                "old_version": previous.version_id if previous else None,
                "coherence_state": coherence_state,
                "hybrid_mode": write_mode,
            },
        )

        simulator.trace.append(
            simulator.trace_line_type(
                simulator.now,
                EventType.EV_WRITE_REQ.value,
                f"{agent.agent_id} hybrid wrote {artifact_id} v{new_version} mode={write_mode}",
                metadata={
                    "agent": agent.agent_id,
                    "artifact_id": artifact_id,
                    "version_id": new_version,
                    "coherence_state": CoherenceState.PROVISIONAL.value,
                    "confidence": confidence,
                    "hybrid_mode": write_mode,
                    "hybrid_force_reason": force_reason,
                },
            )
        )

        if write_mode == "strong":
            simulator.queue.push(
                t=simulator.now,
                event_type=EventType.EV_CONFLICT_CHECK,
                src=agent.agent_id,
                dst="global",
                payload={
                    "artifact_id": artifact_id,
                    "new_version": new_version,
                    "old_version": previous.version_id if previous else None,
                    "coherence_state": coherence_state.value,
                    "confidence": confidence,
                    "hybrid_mode": "strong",
                    "hybrid_force_reason": force_reason,
                },
            )
            global_commit_time = (
                simulator.now
                + simulator.global_memory.store_artifact_latency(pending_artifact)
            )
            assert global_commit_time > cache_commit_time
            simulator.queue.push(
                t=global_commit_time,
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
                    "hybrid_mode": "strong",
                },
            )
            return

        global_conflict_time = (
            simulator.now
            + simulator.global_memory.store_artifact_latency(pending_artifact)
            * self.propagation_delay
        )
        if not global_conflict_time > cache_commit_time:
            raise Exception(
                "Hybrid eventual conflict check must be scheduled after cache commit"
            )
        simulator.queue.push(
            t=global_conflict_time,
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
                "old_version": previous.version_id if previous else None,
                "coherence_state": "contested",
                "hybrid_mode": "eventual",
            },
        )

    def on_sync_req(self, simulator: Simulator, event: Event) -> None:
        if event.type == EventType.EV_SYNC_REQ:
            self._eventual_reader.on_sync_req(simulator, event)
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

        mode = str(event.payload.get("hybrid_mode", "eventual"))
        simulator.trace.append(
            simulator.trace_line_type(
                simulator.now,
                EventType.EV_CONFLICT_CHECK.value,
                f"hybrid checked {artifact_id} mode={mode}",
                metadata={
                    "artifact_id": artifact_id,
                    **decision.to_metadata(),
                    "new_version": event.payload.get(
                        "new_version", event.payload.get("version_id")
                    ),
                    "old_version": event.payload.get("old_version"),
                    "confidence": confidence,
                    "hybrid_mode": mode,
                    "hybrid_force_reason": event.payload.get("hybrid_force_reason"),
                },
            )
        )

        if mode != "eventual":
            return

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
                "hybrid_mode": "eventual",
            },
        )

    def on_invalidate_req(self, simulator: Simulator, event: Event) -> None:
        self._eventual_reader.on_invalidate_req(simulator, event)

    def on_write_commit(self, simulator: Simulator, event: Event) -> None:
        artifact = ConsistencyProtocol.create_artifact(simulator, event)
        agent = simulator.agents[event.src]

        if event.dst == "cache":
            agent.cache.store_artifact(artifact)
            return

        mode = str(event.payload.get("hybrid_mode", "eventual"))
        if mode == "eventual":
            agent.cache.store_artifact(artifact)
        simulator.global_memory.store_artifact(artifact)

        latency = simulator.now - int(event.payload["requested_t"])
        agent.stats.write_latency_total += latency
        agent.stats.write_count += 1
        agent.stats.write_latencies.append(latency)

        simulator.trace.append(
            simulator.trace_line_type(
                simulator.now,
                EventType.EV_WRITE_COMMIT.value,
                f"hybrid commit {tuple(event.payload['artifact_id'])} v{artifact.version_id} mode={mode}",
                metadata={
                    "artifact_id": tuple(event.payload["artifact_id"]),
                    "version_id": artifact.version_id,
                    "coherence_state": artifact.coherence_state.value,
                    "confidence": artifact.confidence,
                    "provenance": artifact.provenance,
                    "latency": latency,
                    "hybrid_mode": mode,
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
                    payload={"artifact_id": tuple(event.payload["artifact_id"]), "reason": "write_commit"},
                )
