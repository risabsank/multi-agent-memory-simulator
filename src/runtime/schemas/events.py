from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .enums import RuntimeEventType


@dataclass(frozen=True)
class JournalEvent:
    """Immutable event envelope for runtime journaling.

    The event envelope is designed for auditability and replayability. `payload`
    is intentionally generic in this scaffold; schema-specific payload contracts
    should be introduced incrementally per event type.
    """

    tenant_id: str
    event_type: RuntimeEventType
    event_id: str
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    payload: dict[str, Any] = field(default_factory=dict)
