from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from ..events import Event

if TYPE_CHECKING:
    from ..simulator import Simulator


class ConsistencyProtocol(Protocol):
    def on_read_req(self, simulator: Simulator, event: Event) -> None:
        ...

    def on_read_resp(self, simulator: Simulator, event: Event) -> None:
        ...

    def on_write_req(self, simulator: Simulator, event: Event) -> None:
        ...

    def on_write_commit(self, simulator: Simulator, event: Event) -> None:
        ...

    def on_sync_req(self, simulator: Simulator, event: Event) -> None:
        ...
