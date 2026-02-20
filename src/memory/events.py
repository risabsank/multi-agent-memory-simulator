from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from heapq import heappop, heappush


class EventType(str, Enum):
    EV_READ_REQ = "EV_READ_REQ"
    EV_READ_RESP = "EV_READ_RESP"
    EV_WRITE_REQ = "EV_WRITE_REQ"
    EV_WRITE_COMMIT = "EV_WRITE_COMMIT"


@dataclass(order=True, slots=True)
class Event:
    t: int # scheduled simulation time
    eid: int # unique event id
    type: EventType = field(compare=False)
    src: str = field(compare=False)
    dst: str = field(compare=False)
    payload: dict = field(default_factory=dict, compare=False)


class EventQueue:
    def __init__(self) -> None:
        self._heap: list[Event] = []
        self._next_eid = 0

    def push(self, t: int, event_type: EventType, src: str, dst: str, payload: dict | None = None) -> Event:
        ev = Event(
            t=t,
            eid=self._next_eid,
            type=event_type,
            src=src,
            dst=dst,
            payload=payload or {},
        )
        self._next_eid += 1
        heappush(self._heap, ev) # push event to the heapq
        return ev

    def pop(self) -> Event:
        return heappop(self._heap) # returns the next event via heappop()

    def __len__(self) -> int:
        return len(self._heap) # length of heap
