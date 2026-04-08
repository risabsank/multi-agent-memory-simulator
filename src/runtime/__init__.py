"""Production runtime scaffold for the Multi-Agent Memory system.

This package intentionally remains isolated from simulator internals (`memory.*`).
It defines runtime-facing contracts, service interfaces, and shared schemas that
can evolve toward a production orchestration runtime while preserving simulator
code as an evaluation environment.
"""

from .config import RuntimeConfig

__all__ = ["RuntimeConfig"]
