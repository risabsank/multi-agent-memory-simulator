from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# defines the semantic visibility of an artifact
# relevance of the artifact to task, local subsystem, or full scope
class ArtifactScope(str, Enum):
    """Semantic visibility of an artifact."""

    LOCAL = "local"
    TASK = "task"
    GLOBAL = "global"

# semantic enums
class ClaimType(str, Enum):
    """Type of statement represented by an artifact."""

    FACT = "fact"
    HYPOTHESIS = "hypothesis"
    PLAN = "plan"
    OBSERVATION = "observation"


class CoherenceState(str, Enum):
    """Quality/conflict state for semantic coherence."""

    ACCEPTED = "accepted" # usable as stable context
    CONTESTED = "contested" # conflicting evidence exists
    PROVISIONAL = "provisional" # low confidence or weak provenance
    DEPRECATED = "deprecated" # deprecated

ArtifactId = tuple[str, str] # (task_id, local_name)

@dataclass(slots=True)
class Artifact:
    """Logical memory unit read/written by agents."""

    artifact_id: ArtifactId
    version_id: int # tracks semantic evolution over time
    size: int
    scope: ArtifactScope
    claim_type: ClaimType = ClaimType.FACT # type of claim
    provenance: str = "system" # who/what created the artifact
    confidence: float = 1.0
    coherence_state: CoherenceState = CoherenceState.ACCEPTED
    observed_at: int = 0 # point at which artifact was observed
    valid_at: int | None = None # timestamp at which the artifact is valid

@dataclass(slots=True)
class CacheEntry:
    """Artifact copy stored in an agent's local cache."""
    
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
    read_latencies: list[int] = field(default_factory=list)
    write_latency_total: int = 0
    write_count: int = 0
    write_latencies: list[int] = field(default_factory=list)

@dataclass(slots=True)
class Agent:
    """One simulated worker/actor."""

    agent_id: str
    cache: dict[ArtifactId, CacheEntry] = field(default_factory=dict)
    stats: AgentStats = field(default_factory=AgentStats)

@dataclass(slots=True)
class Task:
    """Coordination namespace over agents and artifacts."""

    task_id: str
    agent_ids: list[str] # list of involved agents
    artifact_ids: list[ArtifactId] # list of involved artifacts

@dataclass(slots=True)
class GlobalMemory:
    """Shared source of truth for committed artifact versions."""

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
