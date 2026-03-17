from __future__ import annotations

from heapq import heappush, heappop
from typing import TYPE_CHECKING, Any, Optional

from memory.protocols.base import ConsistencyProtocol
from memory.protocols.strong import WriteThroughStrongProtocol
from uuid import uuid4

from ..events import Event, EventType
from ..model import (
    Artifact,
    ArtifactScope,
    ClaimType,
    CoherenceState,
    Agent,
    ArtifactId,
)

from enum import Enum

if TYPE_CHECKING:
    from ..simulator import Simulator

STATES = Enum("MesiStates", [("M", 1), ("E", 2), ("S", 3), ("I", 4)])


class MesiProtocol(WriteThroughStrongProtocol):
    """
    Implementation details:
    Since the goal is MESI simulation, the implementation for MESI basically uses a write/store buffer
    When the protocol needs to snoop on other processes, it will check other agent states in two steps:
    1. Check the latest inflight state
    2. Check the currently written state

    The inflight field basically holds a timeline of all state changes
    The idea is that every request will consult the timeline if necessary, then heappush a state change onto the timeline
    Every response will heappop the timeline for the latest (technically earliest) change it needs to make

    Since the simulator runs in order, it should always pop the correct state changes from the timeline
    """

    # TODO: Tech debt here where agents must be passed into protocol as well as simulator
    # Another drawback is this MESI implementation is tied to a per-artifact granularity, rather than per block
    # I believe this granularity is correct/ok for inter-agent MESI, but may cause issues if per-block VM simulation is implemented down the line
    def __init__(self, bus_latency: int, agents: list[Agent]) -> None:
        self.bus_latency = bus_latency
        self.states: dict[ArtifactId, dict[str, STATES]] = {}
        self.inflight: dict[
            ArtifactId, list[tuple[int, int, dict[str, STATES], dict[str, Any]]]
        ] = {}
        self.verification: dict[ArtifactId, list[tuple[int, int, Any]]] = {}
        self.tiebreaker = 0
        # TODO: probably won't have time, but this is massive tech debt to get around the issue of all write commit calls requiring all artifact details at time of request instead of time of write
        # This works for most situations, but for inflight requests in MESI, a request for one artifact can trigger a write back for a different artifact, and inflight requests have not been written back to the caches yet
        self.artifact_inflight: dict[ArtifactId, list[tuple[int, int, Artifact]]] = {}

    def _add_to_heap(self, artifact_id, t, states, metadata):
        count = self.tiebreaker  # deal with dict to dict comparisons with heapq
        heappush(self.inflight[artifact_id], (t, count, states, metadata))
        heappush(self.verification.setdefault(artifact_id, []), (t, count, metadata))
        self.tiebreaker += 1

    def _process_states(self, host_agent_id: str, states: dict[str, STATES]):
        invalid, shared, exclusive, modified = [], [], [], []
        for other_agent_id, other_agent_state in states.items():
            if other_agent_id == host_agent_id:
                continue
            elif other_agent_state == STATES.I:
                invalid.append(other_agent_id)
            elif other_agent_state == STATES.S:
                shared.append(other_agent_id)
            elif other_agent_state == STATES.E:
                exclusive.append(other_agent_id)
            elif other_agent_state == STATES.M:
                modified.append(other_agent_id)
        return invalid, shared, exclusive, modified

    def _process_mesi_load_miss(
        self, host_agent_id: str, states: dict[str, STATES]
    ) -> tuple[dict[str, STATES], dict[str, Any]]:
        invalid, shared, exclusive, modified = self._process_states(
            host_agent_id, states
        )
        new_states = states.copy()
        metadata = {}
        if len(shared) > 0:
            new_states[host_agent_id] = STATES.S
        elif len(exclusive) > 0:
            for other_agent_id in exclusive:
                new_states[other_agent_id] = STATES.S
            new_states[host_agent_id] = STATES.S
        elif len(modified) > 0:
            for other_agent_id in modified:
                new_states[other_agent_id] = STATES.S
            new_states[host_agent_id] = STATES.S
            metadata["write_back"] = modified
        else:
            new_states[host_agent_id] = STATES.E
        return new_states, metadata

    def _process_mesi_write_miss(self, host_agent_id: str, states: dict[str, STATES]):
        invalid, shared, exclusive, modified = self._process_states(
            host_agent_id, states
        )
        new_states = states.copy()
        metadata = {}
        new_states[host_agent_id] = STATES.M
        if len(shared) > 0:
            for other_agent_id in shared:
                new_states[other_agent_id] = STATES.I
        elif len(exclusive) > 0:
            for other_agent_id in exclusive:
                new_states[other_agent_id] = STATES.I
        elif len(modified) > 0:
            for other_agent_id in modified:
                new_states[other_agent_id] = STATES.I
            metadata["write_back"] = modified
        return new_states, metadata

    def _get_latest_in_inflight(self, artifact_id: ArtifactId, t=float("inf")):
        queue = self.inflight.setdefault(artifact_id, [])
        latest = None
        for item in queue:
            if latest is None or (latest[0] <= item[0] and item[0] < t):
                latest = item
        assert latest is not None
        return latest

    def _get_latest_artifact_inflight(self, artifact_id: ArtifactId, t=float("inf")):
        queue = self.artifact_inflight.setdefault(artifact_id, [])
        latest = None
        for item in queue:
            if latest is None or (latest[0] <= item[0] and item[0] < t):
                latest = item
        assert latest is not None
        return latest

    def snoop_for_load_miss(
        self, simulator, event
    ) -> tuple[dict[str, STATES], dict[str, Any]]:
        agent = simulator.agents[event.src]
        artifact_id = tuple(event.payload["artifact_id"])
        agent_id = agent.agent_id
        # Must check inflight requests according to the MESI protocol
        # Ex: A write after another write must be aware of the first write to avoid desynchronizing states
        # TODO: this isn't really important but performance wise, a deque implementation would be more efficient (will turn log and linear time into (mostly) constant)
        inflight_states = (
            self._get_latest_in_inflight(artifact_id)[2]
            if len(self.inflight.setdefault(artifact_id, [])) > 0
            else {}
        )
        if len(inflight_states) > 0:
            new_state, metadata = self._process_mesi_load_miss(
                agent_id, inflight_states
            )
            metadata["inflight"] = True
        else:
            new_state, metadata = self._process_mesi_load_miss(
                agent_id, self.states[artifact_id]
            )
        return new_state, metadata

    def snoop_for_write_miss(self, simulator, event):
        agent = simulator.agents[event.src]
        artifact_id = tuple(event.payload["artifact_id"])
        agent_id = agent.agent_id
        inflight_states = (
            self._get_latest_in_inflight(artifact_id)[2]
            if len(self.inflight.setdefault(artifact_id, [])) > 0
            else {}
        )
        if len(inflight_states) > 0:
            new_state, metadata = self._process_mesi_write_miss(
                agent_id, inflight_states
            )
            metadata["inflight"] = True
        else:
            new_state, metadata = self._process_mesi_write_miss(
                agent_id, self.states[artifact_id]
            )
        return new_state, metadata

    def process_mesi_update(
        self,
        simulator: Simulator,
        event: Event,
        artifact_id: ArtifactId,
        target_uuid: Optional[Any] = None,
    ):
        # Commit the MESI update corresponding to the caller
        t0, _, states, metadata = heappop(self.inflight[artifact_id])
        t1, _, metadata_v = heappop(self.verification[artifact_id])
        if target_uuid:
            assert t0 == t1
            assert metadata_v["gen_id"] == target_uuid
        for agent_id, state in states.items():
            if state == STATES.I:
                # invalidate
                # I'm worried about any desync bugs
                if simulator.agents[agent_id].cache.artifact_exists(artifact_id):
                    simulator.agents[agent_id].cache.remove_artifact(artifact_id)
            self.states[artifact_id][agent_id] = state

    def get_mesi_state(self, artifact_id: ArtifactId, agent_id: str):
        # Also does initialization
        return self.states.setdefault(artifact_id, {}).setdefault(agent_id, STATES.I)

    def commit_writes_for_modified_states(
        self, simulator: Simulator, event, metadata: dict, t: int
    ):
        # Do a write for every M->S or M->I
        artifact_id = tuple(event.payload["artifact_id"])
        agent = simulator.agents[event.src]
        requested_t = event.payload["requested_t"]
        modified = metadata.get("write_back", [])
        inflight = metadata.get("inflight", False)
        # Either commit a write for the cache or the in flight request
        for other_agent_id in modified:
            if inflight:
                # An inflight request needs special handling; we need to commit a write (also t after the commit to cache) *before* the commit to cache
                artifact = self._get_latest_artifact_inflight(artifact_id, t)[2]
            else:
                artifact = simulator.agents[other_agent_id].cache.get_artifact(
                    artifact_id
                )
            cache_write_time = simulator.now + agent.cache.store_artifact_latency(
                artifact
            )
            simulator.queue.push(
                t=cache_write_time,
                event_type=EventType.EV_WRITE_COMMIT,
                src=agent.agent_id,
                dst="global",
                payload={
                    "artifact_id": artifact_id,
                    "version_id": artifact.version_id,
                    "size": artifact.size,
                    "scope": artifact.scope,
                    "claim_type": artifact.claim_type,
                    "provenance": agent.agent_id,
                    "confidence": artifact.confidence,
                    "coherence_state": artifact.coherence_state,
                    "observed_at": simulator.now,
                    "valid_at": None,
                    "requested_t": requested_t,
                },
            )

    def on_read_req(self, simulator: Simulator, event: Event) -> None:
        agent = simulator.agents[event.src]
        artifact_id = tuple(event.payload["artifact_id"])
        requested_t = event.payload["requested_t"]
        agent_id = agent.agent_id
        t = simulator.now + agent.cache.read_artifact_latency(artifact_id)

        state = self.get_mesi_state(artifact_id, agent_id)
        # Note: There is still a race condition caused by the delay it takes to invalidate things on the bus
        # However, I believe this is a flaw inherent in MESI
        # Though I'm not sure if the state change/invalidation should be done after receiving the invalidation request or after updating its cache line, which are separate latencies
        if state == STATES.E or state == STATES.S or state == STATES.M:
            # Note: Same code as WriteThroughStrongProtocol, could dedup later
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
                    },
                )
            )
            gen_id = uuid4()  # For testing purposes
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
                    "snoop_id": gen_id,
                },
            )
            # push something onto the heap for behavioral consistency
            metadata = {"id": gen_id}
            self._add_to_heap(artifact_id, t, self.states[artifact_id].copy(), metadata)
            return
        # Else, state is I
        # Also same as WriteThroughStrongProtocol
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
        # Try to snoop first
        new_states, metadata = self.snoop_for_load_miss(simulator, event)
        # Since we used the bus, incur latency
        t += self.bus_latency
        snooped_successful = (
            new_states[agent_id] != STATES.E
        )  # E means no other state cached it, so need a global memory read
        # Need to remember to push state change up
        gen_id = uuid4()  # For testing purposes
        metadata["gen_id"] = gen_id
        self._add_to_heap(artifact_id, t, new_states, metadata)

        modified = metadata.get(
            "write_back", []
        )  # The existence of it, added in _process_mesi_load_miss, implies a write back
        self.commit_writes_for_modified_states(simulator, event, modified, t)

        if snooped_successful is True:
            # Successfully, snooped, so no need to read global memory

            # TODO: verify if this behavior is correct
            # Since the snooped agents accessed the object, push it up in LRU priority
            snooped_artifact = None
            for snooped_agent_id, state in new_states.items():
                if state != STATES.I and snooped_agent_id != agent_id:
                    simulator.agents[snooped_agent_id].cache.read_artifact(artifact_id)
                    if snooped_artifact is None:
                        snooped_artifact = simulator.agents[
                            snooped_agent_id
                        ].cache.get_artifact(artifact_id)
            assert snooped_artifact is not None
            simulator.queue.push(
                t=t,
                event_type=EventType.EV_READ_RESP,
                src="snoop",  # TODO: ensure the src is correct
                dst=agent.agent_id,
                payload={
                    "artifact_id": artifact_id,
                    "version_id": snooped_artifact.version_id,  # grab the latest version id from the snooped cache
                    "requested_t": requested_t,
                    "hit": False,
                    "read_source": "snoop",  # TODO: ensure the read_source is correct
                    "snoop_id": gen_id,
                    "snooped_artifact": snooped_artifact,
                },
            )
        else:
            # Same as WriteThroughStrongProtocol
            simulator.queue.push(
                t=t + simulator.global_memory.read_artifact_latency(artifact_id),
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
                },
            )

    def on_read_resp(self, simulator: Simulator, event: Event) -> None:
        agent = simulator.agents[event.dst]
        artifact_id = tuple(event.payload["artifact_id"])
        version_id = event.payload["version_id"]
        requested_t = event.payload["requested_t"]

        self.process_mesi_update(
            simulator, event, artifact_id, event.payload.get("snoop_id")
        )
        if event.src == "global" or event.src == "cache":
            # Only do WriteThroughStrongProtocol on_read_resp when the artifact was not able to be snooped, thus requiring a global memory read
            super().on_read_resp(simulator, event)
        elif event.src == "snoop":
            latency = simulator.now - requested_t
            agent.stats.read_latency_total += latency
            agent.stats.read_count += 1
            agent.stats.read_latencies.append(latency)
            # There is a minor risk where get_artifact on the snooped agent can fail if enough events between the read_req and read_resp times evict the artifact out of the cache through excessive stores
            # It is probably unlikely, but still a bug. Assuming large enough cache size (ex: cache can fit some n artifacts), this is unlikely to happen
            # Therefore, carry the artifact through the payload
            snooped_artifact = event.payload["snooped_artifact"]
            # Commit the MESI update here (I think doing the MESI state update in req would cause state divergence/drift)
            simulator.agents[agent.agent_id].cache.store_artifact(snooped_artifact)

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
                        "global_version_at_read": snooped_artifact.version_id,
                        "coherence_state": snooped_artifact.coherence_state.value,
                    },
                    # observabiliy and semantic state become queryable per read event
                )
            )
        else:
            # Should just be 3 cases
            raise NotImplementedError()

    def on_write_req(self, simulator: Simulator, event: Event) -> None:
        agent = simulator.agents[event.src]
        artifact_id = tuple(event.payload["artifact_id"])
        size = event.payload["size"]
        requested_t = event.payload["requested_t"]
        new_version = simulator.clock.next(artifact_id)
        old_artifact = (
            simulator.global_memory.get_artifact(artifact_id)
            if simulator.global_memory.artifact_exists(artifact_id)
            else None
        )

        scope = event.payload.get(
            "scope", old_artifact.scope if old_artifact else ArtifactScope.TASK
        )
        claim_type = event.payload.get(
            "claim_type", old_artifact.claim_type if old_artifact else ClaimType.PLAN
        )
        if isinstance(scope, str):
            scope = ArtifactScope(scope)
        if isinstance(claim_type, str):
            claim_type = ClaimType(claim_type)
        confidence = float(event.payload.get("confidence", 0.8))
        # coherence_state = self._resolve_coherence_state(old_artifact, confidence)
        coherence_state = (
            CoherenceState.ACCEPTED
        )  # Isn't really important for MESI I think

        # Same as WriteThroughStrongProtocol
        pending_artifact = Artifact(  # changed this to build an explicit pending artifact so latency estimation does not depend on pre-existing store state
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

        agent_id = agent.agent_id
        state = self.get_mesi_state(artifact_id, agent_id)
        t = simulator.now + agent.cache.store_artifact_latency(pending_artifact)
        gen_id = uuid4()
        if state == STATES.M or state == STATES.E:
            # Either this is a hit, and I don't need to update states, or the agent is the sole owner
            # In both, just update local cache
            simulator.queue.push(
                t=t,  # no bus latency needed here as only my agent's state was checked
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
            metadata = {"id": gen_id}
            self._add_to_heap(
                artifact_id, t, self.states[artifact_id].copy(), metadata
            )  # push dud info for behavioral consistency and simplicity
        elif state == STATES.I or state == STATES.S:
            new_states, metadata = self.snoop_for_write_miss(simulator, event)
            self.commit_writes_for_modified_states(simulator, event, metadata, t)
            # Fortunately, for writes, we don't need to consult global memory; all writes are done to local caches
            t += self.bus_latency
            gen_id = uuid4()  # For testing purposes
            new_metadata = {"gen_id": gen_id}

            heappush(
                self.artifact_inflight.setdefault(artifact_id, []),
                (t, self.tiebreaker, pending_artifact),
            )
            self._add_to_heap(artifact_id, t, new_states, new_metadata)
            # Similar to read_req, update LRU
            for snooped_agent_id, state in new_states.items():
                if state != STATES.I and snooped_agent_id != agent_id:
                    simulator.agents[snooped_agent_id].cache.read_artifact(artifact_id)

            simulator.queue.push(
                t=t,
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

    def on_write_commit(self, simulator: Simulator, event: Event) -> None:
        artifact_id = tuple(event.payload["artifact_id"])
        agent = simulator.agents[event.src]
        if event.dst == "cache":
            test = heappop(self.artifact_inflight[artifact_id])
            self.process_mesi_update(simulator, event, artifact_id)
            artifact = ConsistencyProtocol.create_artifact(simulator, event)
            agent.cache.store_artifact(artifact)
            # TODO: add trace log statement
            return

        # Global memory commits are all writebacks, don't need to deal with inflight timeline
        artifact = ConsistencyProtocol.create_artifact(simulator, event)
        simulator.global_memory.store_artifact(artifact)
        # TODO: what traces to push?

    def on_sync_req(self, simulator: Simulator, event: Event) -> None: ...

    def on_invalidate_req(self, simulator: Simulator, event: Event) -> None: ...
