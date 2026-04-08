from __future__ import annotations

from typing import Protocol

from runtime.schemas import JournalEvent


class EventJournal(Protocol):
    """Append-only event journal supporting audit and replay semantics."""

    def append(self, event: JournalEvent) -> None:
        """Append one immutable event to the journal."""

    def replay(self, *, tenant_id: str) -> list[JournalEvent]:
        """Return ordered events for the given tenant for deterministic replay."""
