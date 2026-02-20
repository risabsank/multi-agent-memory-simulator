from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# defines the semantic visibility of an artifact
# relevance of the artifact to task, local subsystem, or full scope
class ArtifactScope(str, Enum):
    LOCAL = "local"
    TASK = "task"
    GLOBAL = "global"


ArtifactId = tuple[str, str] # (task_id, local_name)

# logicl memory unit being read/written
@dataclass(slots=True)
class Artifact:
    artifact_id: ArtifactId
    version_id: int # tracks semantic evolution over time
    size: int
    scope: ArtifactScope

# artifact copy stored in an agent's local cache
# working memory view for an agent
@dataclass(slots=True)
class CacheEntry:
    artifact_id: ArtifactId
    version_id: int
    size: int
    last_access_t: int


@dataclass(slots=True)
class AgentStats:
    hits: int = 0
    misses: int = 0
    read_latency_total: int = 0
    read_count: int = 0

# one simulated worker/actor
@dataclass(slots=True)
class Agent:
    agent_id: str
    cache: dict[ArtifactId, CacheEntry] = field(default_factory=dict)
    stats: AgentStats = field(default_factory=AgentStats)

# coordination name space
@dataclass(slots=True)
class Task:
    task_id: str
    agent_ids: list[str] # list of involved agents
    artifact_ids: list[ArtifactId] # list of involved artifacts

# shared store
# souece of truth for artifact versions
@dataclass(slots=True)
class GlobalMemory:
    latency: int # fixed delay model used in read misses and write commits
    store: dict[ArtifactId, Artifact] = field(default_factory=dict)


class VersionClock:
    """Monotonically increasing version counter per artifact."""

    def __init__(self) -> None:
        self._versions: dict[ArtifactId, int] = {}

    def next(self, artifact_id: ArtifactId) -> int:
        current = self._versions.get(artifact_id, 0) + 1 # increment from current version number
        self._versions[artifact_id] = current
        return current

    def current(self, artifact_id: ArtifactId) -> int:
        return self._versions.get(artifact_id, 0) # get the current version id
