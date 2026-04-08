from __future__ import annotations

from typing import Protocol

from runtime.schemas import JournalEvent


class ControlAPI(Protocol):
    """Public control-plane interface for submitting runtime requests.

    Implementations should be idempotent per request key and tenant-scoped.
    """

    def submit(self, event: JournalEvent) -> str:
        """Submit a control request and return a correlation identifier."""
