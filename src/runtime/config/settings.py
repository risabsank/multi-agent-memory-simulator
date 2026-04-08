from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeConfig:
    """Top-level runtime configuration.

    Fields are intentionally minimal for the initial scaffold and are expected
    to grow as concrete adapters and transports are added.
    """

    tenant_id: str
    environment: str = "dev"
    deterministic_only: bool = True
    advisory_llm_enabled: bool = False
