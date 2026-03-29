from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from ..events import Event
from ..model import Artifact, ArtifactScope, ClaimType, CoherenceState

if TYPE_CHECKING:
    from ..simulator import Simulator


class ConsistencyProtocol(Protocol):
    @staticmethod
    def create_artifact(simulator: Simulator, event: Event) -> Artifact:
        # I cannot actually check for existence here as the artifact in the agent cache may defer from the specifications for the artifact to be written defined in event
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
        return artifact

    def on_read_req(self, simulator: Simulator, event: Event) -> None:
        ...

    def on_read_resp(self, simulator: Simulator, event: Event) -> None:
        ...

    def on_write_req(self, simulator: Simulator, event: Event) -> None:
        ...

    def on_write_commit(self, simulator: Simulator, event: Event) -> None:
        ...
    
    def on_summarize_req(self, simulator: Simulator, event: Event) -> None:
        ...

    def on_summarize_commit(self, simulator: Simulator, event: Event) -> None:
        ...

    def on_sync_req(self, simulator: Simulator, event: Event) -> None:
        ...  

    def on_invalidate_req(self, simulator: Simulator, event: Event) -> None:
        ...
