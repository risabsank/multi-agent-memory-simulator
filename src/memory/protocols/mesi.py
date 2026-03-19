from __future__ import annotations

from heapq import heapify, heappush, heappop
from typing import TYPE_CHECKING, Any, Optional

from memory.protocols.base import ConsistencyProtocol
from memory.protocols.strong import WriteThroughStrongProtocol
from uuid import uuid4, UUID

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
    The idea is that every request will consult the timeline if necessary, then ~~heappush~~ push a state change onto the timeline
    Every response will ~~heappop the timeline~~ pop the corresponding element for the latest (technically earliest) change it needs to make

    ~~Since the simulator runs in order, it should always pop the correct state changes from the timeline~~
    The ordering correctness is enforced by the push/pop logic
    """

    # TODO:
    # Another drawback is this MESI implementation is tied to a per-artifact granularity, rather than per block
    # I believe this granularity is correct/ok for inter-agent MESI, but may cause issues if per-block VM simulation is implemented down the line
    def __init__(self, bus_latency: int) -> None:
        self.bus_latency = bus_latency
        self.states: dict[ArtifactId, dict[str, STATES]] = {}
        self.inflight: dict[
            ArtifactId, dict[UUID, tuple[int, dict[str, STATES], dict[str, Any]]]
        ] = {}
        # TODO: probably won't have time, but this is massive tech debt to get around the issue of all write commit calls requiring all artifact details at time of request instead of time of write
        # This works for most situations, but for inflight requests in MESI, a request for one artifact can trigger a write back for a different artifact, and inflight requests have not been written back to the caches yet
        self.artifact_inflight: dict[ArtifactId, dict[UUID, tuple[int, Artifact]]] = {}

    def _add_to_inflight(self, artifact_id, gen_id, t, states, metadata):
        self.inflight.setdefault(artifact_id, {})[gen_id] = (t, states, metadata)

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
            metadata["sharer"] = shared[
                0
            ]  # TODO: I dont *think* which sharer matters as they will all get invalidated at the same time, ex when another agent does a store miss, but I should double check
        elif len(exclusive) > 0:
            assert len(exclusive) == 1
            for other_agent_id in exclusive:
                new_states[other_agent_id] = STATES.S
            new_states[host_agent_id] = STATES.S
            metadata["was_exclusive"] = exclusive[0]  # this should just be len 1
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

    def _get_latest_in_inflight(self, artifact_id: ArtifactId, current_t=float("inf")):
        inflight = self.inflight.get(artifact_id, {})
        if not inflight:
            return None

        latest = None
        for t, states, metadata in inflight.values():
            if t <= current_t and (latest is None or t >= latest[0]):
                latest = (t, states, metadata)

        return latest

    def _get_latest_artifact_inflight(
        self, artifact_id: ArtifactId, current_t=float("inf")
    ):
        inflight_artifacts = self.artifact_inflight.setdefault(artifact_id, {})
        if not inflight_artifacts:
            return None
        latest = None
        for t, artifact in inflight_artifacts.values():
            if t <= current_t and (latest is None or t >= latest[0]):
                latest = (t, artifact)
        return latest

    def snoop_for_load_miss(
        self, simulator, event
    ) -> tuple[int | None, dict[str, STATES], dict[str, Any]]:
        agent = simulator.agents[event.src]
        artifact_id = tuple(event.payload["artifact_id"])
        agent_id = agent.agent_id
        # Must check inflight requests according to the MESI protocol
        # Ex: A write after another write must be aware of the first write to avoid desynchronizing states
        # TODO: this isn't really important but performance wise, a deque implementation would be more efficient (will turn log and linear time into (mostly) constant)
        inflight = (
            self._get_latest_in_inflight(artifact_id)
            if len(self.inflight.setdefault(artifact_id, {})) > 0
            else None
        )
        if inflight is not None and len(inflight[2]) > 0:
            t, inflight_states, _ = inflight
            new_state, metadata = self._process_mesi_load_miss(
                agent_id, inflight_states
            )
            metadata["inflight"] = True
        else:
            # TODO: Not the best semantically to return dud information here, maybe use metadata?
            t = None
            new_state, metadata = self._process_mesi_load_miss(
                agent_id, self.states[artifact_id]
            )
        return t, new_state, metadata

    def snoop_for_write_miss(self, simulator, event):
        agent = simulator.agents[event.src]
        artifact_id = tuple(event.payload["artifact_id"])
        agent_id = agent.agent_id
        inflight = (
            self._get_latest_in_inflight(artifact_id)
            if len(self.inflight.setdefault(artifact_id, {})) > 0
            else None
        )
        if inflight is not None and len(inflight[2]) > 0:
            t, inflight_states, _ = inflight
            new_state, metadata = self._process_mesi_write_miss(
                agent_id, inflight_states
            )
            metadata["inflight"] = True
        else:
            # TODO: Not the best semantically to return dud information here, maybe use metadata?
            t = None
            new_state, metadata = self._process_mesi_write_miss(
                agent_id, self.states[artifact_id]
            )
        return t, new_state, metadata

    def process_mesi_update(
        self,
        simulator: Simulator,
        event: Event,
        artifact_id: ArtifactId,
        target_uuid: Optional[Any] = None,
    ):
        # Note: I was stubbornly trying to force efficient O(logn) popping operations for processing, but I don't think this is actually possible
        # When a bunch of events fall onto the same timestamp t (very common when internal events are spawned), it is impossible to match the event queue with the inflight queue due to the different tiebreakers (or at least not possible without some significant effort)

        # Commit the MESI update corresponding to the caller
        if target_uuid is None:
            target_uuid = event.payload.get("snoop_id")
        assert target_uuid is not None
        t0, states, metadata = self.inflight[artifact_id].pop(target_uuid)

        for agent_id, state in states.items():
            if state == STATES.I:
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
            # a cache hit can result in another agent snooping on inflight requests even though no new artifact needs to be written (aka not in inflight_artifacts)
            # so fall back if not found as that means the correct data is in the cache
            artifact = None
            if inflight:
                # An inflight request needs special handling; we need to commit a write (also t after the commit to cache) *before* the commit to cache
                latest_inflight = self._get_latest_artifact_inflight(
                    artifact_id
                )  # TODO: do I pass t here
                if latest_inflight is not None:
                    artifact = latest_inflight[1]
            if artifact is None:
                artifact = simulator.agents[other_agent_id].cache.get_artifact(
                    artifact_id
                )
            cache_write_time = simulator.now + agent.cache.store_artifact_latency(
                artifact
            )
            simulator.queue.push(
                t=cache_write_time,
                event_type=EventType.EV_WRITE_COMMIT,
                src=other_agent_id,
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
                priority=0,
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
        if (
            (state == STATES.E or state == STATES.S or state == STATES.M)
            and agent.cache.artifact_exists(artifact_id)
        ):  # ig there is a chance the artifact actually could have been silently evicted *before* the state change was able to propagate
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
                        "state": state,
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
                priority=0,
            )
            # push something onto the heap for behavioral consistency
            metadata = {"gen_id": gen_id, "version_id": entry.version_id}
            self._add_to_inflight(
                artifact_id, gen_id, t, self.states[artifact_id].copy(), metadata
            )
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
                    "state": state,
                },
            )
        )
        # Try to snoop first
        snoop_t, new_states, metadata = self.snoop_for_load_miss(simulator, event)
        # Since we used the bus, incur latency
        t += self.bus_latency
        snooped_successful = (
            new_states[agent_id] != STATES.E
        )  # E means no other state cached it, so need a global memory read
        # Need to remember to push state change up
        gen_id = uuid4()  # For testing purposes

        global_version_id = simulator.global_memory.get_artifact(artifact_id).version_id
        version_id = metadata.get(
            "version_id", global_version_id
        )  # the version_id for read_resp is either the global version_id, implying we need a global memory read, or an inflight version_id, which is achieved through snooping
        new_metadata = {"gen_id": gen_id, "version_id": version_id}

        # The existence of write_back in metadata, added in _process_mesi_load_miss, implies a write back
        self.commit_writes_for_modified_states(simulator, event, metadata, t)

        if snooped_successful is True:
            # Successfully snooped, so no need to read global memory

            # TODO: verify if this behavior is correct
            # Since the snooped agents accessed the object, push it up in LRU priority
            for snooped_agent_id, state in new_states.items():
                if (
                    state != STATES.I
                    and snooped_agent_id != agent_id
                    and simulator.agents[snooped_agent_id].cache.artifact_exists(
                        artifact_id
                    )
                ):  # artifact might not exist yet if its inflight and we snooped off of the bus (bus says write in progress)
                    simulator.agents[snooped_agent_id].cache.read_artifact(artifact_id)
            # TODO: I commented something similar up in snoop_for_load_miss, but this is a synctatically very poor way of writing this, but I will try to implement something that works first (I kind of doubt I'll have enough time to clean this up, sadly)
            # snoop_t existing means an inflight request was picked up
            # snoop_from indicates the agent that the current agent will snoop on for the artifact information
            # annoyingly, MESI has 2 cases: M and E
            if snoop_t is not None:
                t = max(
                    t, snoop_t
                )  # We actually have to wait until the artifact is loaded into the snooped agent, if we detected an inflight update
            if metadata.get("was_exclusive") is not None:
                snoop_from = metadata["was_exclusive"]
            elif metadata.get("write_back") is not None:
                snoop_from = (
                    metadata["write_back"][0]
                    if isinstance(metadata["write_back"], list)
                    else metadata["write_back"]
                )
            else:
                snoop_from = metadata["sharer"]
            simulator.queue.push(
                t=t,
                event_type=EventType.EV_READ_RESP,
                src="snoop",  # Note: be careful of the src here
                dst=agent.agent_id,
                payload={
                    "artifact_id": artifact_id,
                    "version_id": version_id,  # grab the latest version id from the snooped cache
                    "requested_t": requested_t,
                    "hit": False,
                    "read_source": "snoop",  # Note: I don't actually know what read_source is for
                    "snoop_id": gen_id,
                    "snoop_from": snoop_from,
                },
                priority=0,
            )
            self._add_to_inflight(artifact_id, gen_id, t, new_states, new_metadata)
        else:
            t += simulator.global_memory.read_artifact_latency(artifact_id)
            # Same as WriteThroughStrongProtocol
            simulator.queue.push(
                t=t,
                event_type=EventType.EV_READ_RESP,
                src="global",
                dst=agent.agent_id,
                payload={
                    "artifact_id": artifact_id,
                    "version_id": global_version_id,
                    "requested_t": requested_t,
                    "hit": False,
                    "read_source": "global",
                    "snoop_id": gen_id,
                },
                priority=0,
            )
            self._add_to_inflight(artifact_id, gen_id, t, new_states, new_metadata)

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
            snooped_agent_id = event.payload["snoop_from"]
            # Note: I'm not too certain why, but here, it is possible for a snoop read_resp to come before the write_commit for the snooped agent, meaning it won't be there in the cache to snoop out
            # I *think* this is because I got rid of some of my priority constructs; putting snoop priorities to 1 level lower in priority in theory should also resolev this, but I'm concerned about the race conditions/inconsistent ordering that may occur then (or still exist)
            snooped_artifact = None
            if simulator.agents[snooped_agent_id].cache.artifact_exists(artifact_id):
                snooped_artifact = simulator.agents[
                    snooped_agent_id
                ].cache.get_artifact(artifact_id)
            if snooped_artifact is None:
                latest_inflight = self._get_latest_artifact_inflight(artifact_id)
                if latest_inflight is not None:
                    snooped_artifact = latest_inflight[1]
            if snooped_artifact is None:
                # I dont think this should ever really happen, but just in case ig
                snooped_artifact = simulator.global_memory.get_artifact(artifact_id)
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
                        "state": self.get_mesi_state(artifact_id, agent.agent_id),
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
                    "snoop_id": gen_id,
                },
                priority=0,
            )
            simulator.trace.append(
                simulator.trace_line_type(
                    simulator.now,
                    EventType.EV_WRITE_REQ.value,
                    f"{agent.agent_id} wrote {artifact_id} v{new_version} with no snooping",
                    metadata={
                        "agent": agent.agent_id,
                        "artifact_id": artifact_id,
                        "version_id": new_version,
                        "coherence_state": coherence_state.value,
                        "confidence": confidence,
                        "state": state,
                    },
                )
            )
            metadata = {"gen_id": gen_id, "version_id": new_version}
            new_states = self.states[artifact_id].copy()
            # In MESI, this is the only case when the state changes due to a hit
            new_states[agent_id] = STATES.M
            self.artifact_inflight.setdefault(artifact_id, {})[gen_id] = (
                t,
                pending_artifact,
            )
            self._add_to_inflight(
                artifact_id, gen_id, t, new_states, metadata
            )  # push dud info for behavioral consistency and simplicity
        elif state == STATES.I or state == STATES.S:
            snoop_t, new_states, metadata = self.snoop_for_write_miss(simulator, event)
            self.commit_writes_for_modified_states(simulator, event, metadata, t)
            # Fortunately, for writes, we don't need to consult global memory; all writes are done to local caches
            t += self.bus_latency
            # similar to read_req
            if snoop_t:
                t = max(t, snoop_t)
            gen_id = uuid4()  # For testing purposes
            metadata["gen_id"] = (
                gen_id  # similar to above, it should be ok that the rest of the metadata is the same, but need to be careful if this code changes
            )

            self.artifact_inflight.setdefault(artifact_id, {})[gen_id] = (
                t,
                pending_artifact,
            )
            self._add_to_inflight(artifact_id, gen_id, t, new_states, metadata)
            # Similar to read_req, update LRU
            for snooped_agent_id, state in new_states.items():
                if (
                    state != STATES.I
                    and snooped_agent_id != agent_id
                    and simulator.agents[snooped_agent_id].cache.artifact_exists(
                        artifact_id
                    )
                ):  # artifact might not exist yet if its inflight and we snooped off of the bus (bus says write in progress)
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
                    "snoop_id": gen_id,
                },
                priority=0,
            )
            simulator.trace.append(
                simulator.trace_line_type(
                    simulator.now,
                    EventType.EV_WRITE_REQ.value,
                    f"{agent.agent_id} wrote {artifact_id} v{new_version} (Write Miss/Upgrade)",
                    metadata={
                        "agent": agent.agent_id,
                        "artifact_id": artifact_id,
                        "version_id": new_version,
                        "coherence_state": coherence_state.value,
                        "confidence": confidence,
                        "state": state,
                    },
                )
            )

    def on_write_commit(self, simulator: Simulator, event: Event) -> None:
        artifact_id = tuple(event.payload["artifact_id"])
        agent = simulator.agents[event.src]
        latency = simulator.now - int(event.payload["requested_t"])
        writer = simulator.agents[event.src]
        writer.stats.write_latency_total += latency
        writer.stats.write_count += 1
        writer.stats.write_latencies.append(latency)
        snoop_id = event.payload.get("snoop_id")
        if snoop_id:
            self.artifact_inflight.get(artifact_id, {}).pop(snoop_id, None)

        if event.dst == "cache":
            self.process_mesi_update(
                simulator, event, artifact_id, target_uuid=snoop_id
            )
            artifact = ConsistencyProtocol.create_artifact(simulator, event)
            agent.cache.store_artifact(artifact)
            simulator.trace.append(
                simulator.trace_line_type(
                    simulator.now,
                    EventType.EV_WRITE_COMMIT.value,
                    f"cache committed {artifact_id} v{artifact.version_id} for agent {agent.agent_id}",
                    metadata={
                        "artifact_id": artifact_id,
                        "version_id": artifact.version_id,
                        "coherence_state": artifact.coherence_state.value,
                        "confidence": artifact.confidence,
                        "provenance": artifact.provenance,
                        "latency": latency,
                        "state": self.get_mesi_state(artifact_id, agent.agent_id),
                    },
                )
            )
            return
        else:
            # Global memory commits are all writebacks, don't need to deal with inflight timeline
            artifact = ConsistencyProtocol.create_artifact(simulator, event)
            simulator.global_memory.store_artifact(artifact)

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
                        "state": self.get_mesi_state(artifact_id, agent.agent_id),
                    },
                )
            )

    def on_sync_req(self, simulator: Simulator, event: Event) -> None: ...

    def on_invalidate_req(self, simulator: Simulator, event: Event) -> None: ...
