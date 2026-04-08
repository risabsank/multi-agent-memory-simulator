from __future__ import annotations

from typing import Protocol

from runtime.schemas import JournalEvent


class Orchestrator(Protocol):
    """Coordinates deterministic runtime progression from inbound events."""

    def handle(self, event: JournalEvent) -> list[JournalEvent]:
        """Process a single event and return follow-up events to emit."""
