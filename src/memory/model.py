from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, OrderedDict


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

    ACCEPTED = "accepted"  # usable as stable context
    CONTESTED = "contested"  # conflicting evidence exists
    PROVISIONAL = "provisional"  # low confidence or weak provenance
    DEPRECATED = "deprecated"  # deprecated


ArtifactId = tuple[str, str]  # (task_id, local_name)


@dataclass
class Artifact:
    """Logical memory unit read/written by agents."""

    artifact_id: ArtifactId
    version_id: int  # tracks semantic evolution over time
    size: int
    scope: ArtifactScope
    claim_type: ClaimType = ClaimType.FACT  # type of claim
    provenance: str = "system"  # who/what created the artifact
    confidence: float = 1.0
    coherence_state: CoherenceState = CoherenceState.ACCEPTED
    observed_at: int = 0  # point at which artifact was observed
    valid_at: int | None = None  # timestamp at which the artifact is valid


# artifact copy stored in an agent's local cache
# working memory view for an agent
@dataclass
class CacheEntry:
    """Artifact copy stored in an agent's local cache."""

    artifact_id: ArtifactId
    version_id: int
    size: int
    last_access_t: int


@dataclass
class AgentStats:
    hits: int = 0
    misses: int = 0
    read_latency_total: int = 0
    read_count: int = 0
    read_latencies: list[int] = field(default_factory=list)
    write_latency_total: int = 0
    write_count: int = 0
    write_latencies: list[int] = field(default_factory=list)


# one simulated worker/actor
@dataclass
class Agent:
    """One simulated worker/actor."""

    agent_id: str
    cache: Cache
    stats: AgentStats = field(default_factory=AgentStats)


# coordination name space
@dataclass
class Task:
    """Coordination namespace over agents and artifacts."""

    task_id: str
    agent_ids: list[str]  # list of involved agents
    artifact_ids: list[ArtifactId]  # list of involved artifacts


@dataclass
class Memory:
    read_latency: int  # read latency PER BLOCK
    write_latency: int  # write latency PER BLOCK
    total_size: int
    block_size: int
    _store: dict[ArtifactId, Artifact] = field(default_factory=dict)
    _occupancy: dict[ArtifactId, int] = field(
        default_factory=lambda: defaultdict(int)
    )  # store number of used blocks per artifact
    _used_size: int = field(init=False)
    _lru: LRUEviction = field(init=False)

    # TODO: can refactor this into a subclass of an eviction policy
    class LRUEviction:
        """
        Some liberties are taken in this LRU implementation:

        - LRU is tracked per object rather than per block.
            - This assumes a client accessing a memory object always results in an access to the entire object
        - Eviction is done on a per object rather than a per block basis
            - This means more blocks than necessary could be evicted
            - TODO: implement per block eviction, unless the differences are marginal
        """

        def __init__(self):
            self.lru: OrderedDict[ArtifactId, None] = OrderedDict()

        def get_evict_candidate(self) -> ArtifactId:
            if len(self.lru) == 0:
                raise Exception(
                    "tried to find eviction candidate LRU for global memory is empty. eviction was called before any memory was committed. is total_memory size 0?"
                )
            evict_id, _ = self.lru.popitem(last=False)
            return evict_id

        def add_artifact_id(self, id: ArtifactId) -> None:
            self.lru[id] = None

        def remove_artifact_id(self, id: ArtifactId) -> None:
            del self.lru[id]

        def update_artifact_id(self, id: ArtifactId) -> None:
            """
            Add/Move ArtifactId to end of LRU order
            """
            if id not in self.lru:
                self.add_artifact_id(id)
                return
            self.lru.move_to_end(id, last=True)

    def __post_init__(self):
        self._used_size = 0
        self._lru = Memory.LRUEviction()

    def store_artifact_latency(
        self, artifact: Artifact
    ) -> int:  # TODO: read vs store latency functions take artifact id and artifact respectively, maybe they should be the same? issue is artifact is better for store since id won't exist within memory at that point in time
        return self._calc_total_latency_for_size(self.write_latency, artifact.size)

    def store_artifact_unique(self, artifact: Artifact) -> None:
        """
        Artifact must not already be in store
        """
        artifact_id, artifact_size = artifact.artifact_id, artifact.size
        assert artifact_id not in self._store
        num_blocks = Memory.num_blocks(artifact_size, self.block_size)
        req_size = num_blocks * self.block_size
        if req_size > self.total_size:
            raise Exception("Requested global memory store larger than total memory")
        if req_size > self.total_size - self._used_size:
            self._evict(req_size)
        self._used_size += req_size
        self._occupancy[artifact_id] += num_blocks
        self._store[artifact_id] = artifact

        self._lru.update_artifact_id(artifact_id)

    def store_artifact(self, artifact: Artifact) -> None:
        """Overwrite if necessary"""  # I believe we only want to write/overwrite when necessary
        if self.artifact_exists(artifact.artifact_id):
            self.overwrite_artifact(artifact)
        else:
            self.store_artifact_unique(artifact)

    def overwrite_artifact(self, artifact: Artifact) -> None:
        # can this happen? should the parameter be artifact or artifact_id + size?
        self.remove_artifact(artifact.artifact_id)
        self.store_artifact(artifact)

    def read_artifact_latency(self, artifact_id: ArtifactId) -> int:
        if artifact_id in self._store:
            return self._calc_artifact_latency(self.read_latency, artifact_id)
        else:
            raise Exception(
                "Attempted to calculate read artifact latency, but artifact id not found in both memory and swap"
            )

    def read_artifact(self, artifact_id: ArtifactId) -> None:
        """
        Get the artifact out of swap if necessary, returning the latency to read
        """
        if artifact_id in self._store:
            self._lru.update_artifact_id(artifact_id)
        else:
            raise Exception(
                "Attempted get artifact_id, but not found in both memory and swap"
            )

    def get_artifact(self, artifact_id: ArtifactId) -> Artifact:
        """
        Return the artifact given the id out of the memory store (either main or swap).

        Note: This is a simple get(). This does not perform eviction. Therefore, for protocols, get_artifact should be used instead
        """
        return self._store[artifact_id]

    def remove_artifact(self, artifact_id: ArtifactId) -> Artifact:
        if artifact_id not in self._store:
            raise Exception("artifact_id not in global memory store")
        artifact = self._store[artifact_id]
        artifact_size = artifact.size
        num_blocks = Memory.num_blocks(artifact_size, self.block_size)
        req_size = num_blocks * self.block_size
        self._used_size -= req_size
        self._occupancy[artifact_id] -= num_blocks
        del self._store[artifact_id]

        self._lru.remove_artifact_id(artifact_id)
        return artifact

    def _evict(self, size: int) -> None:
        # TODO: dedup
        evicted_size = 0
        while evicted_size < size:
            artifact_id = self._lru.get_evict_candidate()
            artifact = self.remove_artifact(artifact_id)
            req_size = (
                Memory.num_blocks(artifact.size, self.block_size) * self.block_size
            )
            assert (
                artifact.size != 0
            )  # ensure we're making progress since while loops can be undeterministic, this likely means an invalid object was stored to global memory
            evicted_size += req_size

    def artifact_exists(self, artifact_id: ArtifactId) -> bool:
        return self._store.get(artifact_id) is not None

    def _calc_total_latency_for_size(self, base_latency: int, size: int) -> int:
        return base_latency * Memory.num_blocks(size, self.block_size)

    def _calc_artifact_latency(self, base_latency: int, artifact_id: ArtifactId):
        return self._calc_total_latency_for_size(
            base_latency, self.get_artifact(artifact_id).size
        )

    @staticmethod
    def num_blocks(size: int, block_size: int) -> int:
        return (size + block_size - 1) // block_size

    def get_all_artifacts(self) -> list[tuple[ArtifactId, Artifact]]:
        return list(self._store.items())


# run memory model simulation, keeping track of processing cost
# shared store
# souece of truth for artifact versions
@dataclass(kw_only=True)  # deal with annoying inheritance, fields need kw specifier now
class GlobalMemory(Memory):
    swap_latency: int  # latency to move from swap
    _swap: Swap = field(init=False)

    class Swap:
        # The code here is very similar and duplicative of the overall GlobalMemory; a good refactor would be to define the memory class separately and instantiate two nested memory classes, one with infinite size
        # Assuming infinite swap space
        def __init__(self, block_size: int):
            self.swap_store: dict[ArtifactId, Artifact] = dict()
            self.swap_occupancy: dict[ArtifactId, int] = defaultdict(int)
            self.used_size = 0  # for bookkeeping
            self.block_size = block_size

        def add_to_swap(self, artifact: Artifact):
            artifact_id, artifact_size = artifact.artifact_id, artifact.size
            num_blocks = Memory.num_blocks(artifact_size, self.block_size)
            req_size = num_blocks * self.block_size
            self.swap_occupancy[artifact_id] += num_blocks
            self.swap_store[artifact_id] = artifact
            self.used_size += req_size

        def remove_from_swap(self, artifact_id: ArtifactId) -> Artifact:
            artifact = self.swap_store[artifact_id]
            artifact_size = artifact.size
            num_blocks = Memory.num_blocks(artifact_size, self.block_size)
            req_size = num_blocks * self.block_size
            self.used_size -= req_size
            self.swap_occupancy[artifact_id] -= num_blocks
            del self.swap_store[artifact_id]
            return artifact

        def get_all_artifacts(self) -> list[tuple[ArtifactId, Artifact]]:
            return list(self.swap_store.items())

    def __post_init__(self):
        super().__post_init__()
        self._swap = GlobalMemory.Swap(self.block_size)

    def store_artifact_latency(self, artifact: Artifact) -> int:
        req_size = Memory.num_blocks(artifact.size, self.block_size) * self.block_size
        base_latency = self._calc_total_latency_for_size(
            self.write_latency, artifact.size
        )  # changed this to compute write latency from candidate artifact size instead of requiring an existing stored artifact

        # TODO: dedup with _evict
        evicted_size = 0
        i = 0
        while (
            evicted_size < req_size
        ):  # changed this to use block-aligned required size so swap penalty estimate matches actual eviction need
            to_evict = self.get_artifact(list(self._lru.lru.keys())[i])
            i += 1
            to_evict_size = (
                Memory.num_blocks(to_evict.size, self.block_size) * self.block_size
            )
            assert to_evict_size != 0
            evicted_size += to_evict_size

        if req_size > self.total_size - self._used_size:
            return base_latency + self._calc_total_latency_for_size(
                self.swap_latency, evicted_size
            )
        return base_latency

    def store_artifact_unique(self, artifact: Artifact) -> None:
        """
        Artifact must not already be in store
        """
        artifact_id, artifact_size = artifact.artifact_id, artifact.size
        assert artifact_id not in self._store
        num_blocks = Memory.num_blocks(artifact_size, self.block_size)
        req_size = num_blocks * self.block_size
        if req_size > self.total_size:
            raise Exception("Requested global memory store larger than total memory")
        if req_size > self.total_size - self._used_size:
            self._evict(req_size)
        self._used_size += req_size
        self._occupancy[artifact_id] += num_blocks
        self._store[artifact_id] = artifact

        self._lru.update_artifact_id(artifact_id)

    def overwrite_artifact(self, artifact: Artifact) -> None:
        # can this happen? should the parameter be artifact or artifact_id + size?
        self.remove_artifact(artifact.artifact_id)
        self.store_artifact(artifact)

    def read_artifact_latency(self, artifact_id: ArtifactId) -> int:
        if artifact_id in self._store:
            return self._calc_artifact_latency(self.read_latency, artifact_id)
        elif artifact_id in self._swap.swap_store:
            return self._calc_artifact_latency(
                self.swap_latency, artifact_id
            ) + self._calc_artifact_latency(self.read_latency, artifact_id)
        else:
            raise Exception(
                "Attempted to calculate read artifact latency, but artifact id not found in both memory and swap"
            )

    def read_artifact(self, artifact_id: ArtifactId) -> None:
        """
        Get the artifact out of swap if necessary, returning the latency to read
        """
        if artifact_id in self._store:
            self._lru.update_artifact_id(artifact_id)
        elif artifact_id in self._swap.swap_store:
            self.store_artifact(self._swap.remove_from_swap(artifact_id))
            return self.read_artifact(
                artifact_id
            )  # recursive call but should resolve as long as store_artifact guarantees to add it to store
        else:
            raise Exception(
                "Attempted get artifact_id, but not found in both memory and swap"
            )

    def get_artifact(self, artifact_id: ArtifactId) -> Artifact:
        """
        Return the artifact given the id out of the memory store (either main or swap).

        Note: This is a simple get(). This does not perform eviction. Therefore, for protocols, get_artifact should be used instead
        """
        return self._store.get(artifact_id) or self._swap.swap_store[artifact_id]

    def artifact_exists(self, artifact_id: ArtifactId) -> bool:
        return (
            self._store.get(artifact_id) or self._swap.swap_store.get(artifact_id)
        ) is not None

    def _evict(self, size: int) -> None:
        # TODO: dedup
        evicted_size = 0
        while evicted_size < size:
            artifact_id = self._lru.get_evict_candidate()
            artifact = self.remove_artifact(artifact_id)
            req_size = (
                Memory.num_blocks(artifact.size, self.block_size) * self.block_size
            )
            self._swap.add_to_swap(artifact)
            assert (
                artifact.size != 0
            )  # ensure we're making progress since while loops can be undeterministic, this likely means an invalid object was stored to global memory
            evicted_size += req_size

    def get_all_artifacts(self) -> list[tuple[ArtifactId, Artifact]]:
        swap_artifacts = self._swap.get_all_artifacts()
        memory_artifacts = list(self._store.items())
        memory_artifacts.extend(swap_artifacts)
        return memory_artifacts


@dataclass(kw_only=True)
class Cache(Memory):
    def read_artifact_latency(self, artifact_id: ArtifactId) -> int:
        """
        Since memory in this model is represented per-object, the cache hit/miss latencies are different:
            - A cache hit will be the latency per block * number of blocks in the artifact
            - A cache miss will be the latency for one block
        The idea here is that a cache hit needs to read the entire artifact out of cache, which takes time, while a cache miss can test for one block in the cache and see if that is missing.
        This may not be exactly correct depending on the implementation of the cache (maybe they can compare and read in parallel, similar to how tag comparators are implemented in cpu caches), and is not correct for a per-block VM system

        Fortunately, the store can be the same
        """
        if self.artifact_exists(artifact_id):
            return super().read_artifact_latency(artifact_id)
        else:
            return self.read_latency


class VersionClock:
    """Monotonically increasing version counter per artifact."""

    def __init__(self) -> None:
        self._versions: dict[ArtifactId, int] = {}

    def next(self, artifact_id: ArtifactId) -> int:
        current = (
            self._versions.get(artifact_id, 0) + 1
        )  # increment from current version number
        self._versions[artifact_id] = current
        return current

    def current(self, artifact_id: ArtifactId) -> int:
        return self._versions.get(artifact_id, 0)  # get the current version id
