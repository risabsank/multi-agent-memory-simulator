from __future__ import annotations

from typing import Protocol

from runtime.schemas import CoherenceState


class MemoryService(Protocol):
    """Tenant-scoped memory service used by the runtime orchestrator.

    Implementations should maintain audit trails and support deterministic
    replay for write paths.
    """

    def read(self, *, tenant_id: str, key: str) -> dict | None:
        """Read the latest value for a tenant-scoped key."""

    def write(
        self,
        *,
        tenant_id: str,
        key: str,
        value: dict,
        expected_version: int | None = None,
        coherence_state: CoherenceState = CoherenceState.ACCEPTED,
    ) -> int:
        """Write a value and return the resulting version."""
