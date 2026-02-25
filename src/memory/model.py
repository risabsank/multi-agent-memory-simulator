from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from llist import dllist

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
    total_size: int
    store: dict[ArtifactId, Artifact] = field(default_factory=dict)
    occupancy: dict[ArtifactId, int] = field(default_factory=lambda: defaultdict(int)) # store number of used blocks per artifact
    block_size: int = 4096
    used_size: int = field(init=False)
    lru: LRUEviction = field(init=False)
    swap: Swap = field(init=False)

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
            self.lru = dllist()
            self.node_map = dict()

        def get_evict_candidate(self) -> ArtifactId:
            if self.lru.size == 0:
                raise Exception("tried to find eviction candidate LRU for global memory is empty. eviction was called before any memory was committed. is total_memory size 0?")
            id = self.lru.popleft()
            del self.node_map[id]
            return id
        
        def add_artifact_id(self, id: ArtifactId) -> None:
            self.node_map[id] = self.lru.appendright(id)

        def remove_artifact_id(self, id: ArtifactId) -> None:
            self.lru.remove(self.node_map[id])
            del self.node_map[id]

        def update_artifact_id(self, id: ArtifactId) -> None:
            """
            Add/Move ArtifactId to end of LRU order
            """
            if id not in self.node_map:
                self.add_artifact_id(id)
                return
            node = self.node_map[id]
            self.lru.remove(node)
            self.node_map[id] = self.lru.appendright(id) # this works because node.value and id are the same

    class Swap:
        # The code here is very similar and duplicative of the overall GlobalMemory; a good refactor would be to define the memory class separately and instantiate two nested memory classes, one with infinite size
        # Assuming infinite swap space
        def __init__(self, block_size: int):
            self.swap_store: dict[ArtifactId, Artifact] = dict()
            self.swap_occupancy: dict[ArtifactId, int] = defaultdict(int)
            self.used_size = 0 # for bookkeeping
            self.block_size = block_size

        def add_to_swap(self, artifact: Artifact):
            artifact_id, artifact_size = artifact.artifact_id, artifact.size
            num_blocks = GlobalMemory.num_blocks(artifact_size, self.block_size)
            req_size = num_blocks * self.block_size
            self.swap_occupancy[artifact_id] += num_blocks
            self.swap_store[artifact_id] = artifact
            self.used_size += req_size
            
        def remove_from_swap(self, artifact_id: ArtifactId) -> Artifact:
            artifact = self.swap_store[artifact_id]
            artifact_size = artifact.size
            num_blocks = GlobalMemory.num_blocks(artifact_size, self.block_size)
            req_size = num_blocks * self.block_size
            self.used_size -= req_size
            self.swap_occupancy[artifact_id] -= num_blocks
            del self.swap_store[artifact_id]
            return artifact
            

    def __post_init__(self):
        self.used_size = 0
        self.lru = GlobalMemory.LRUEviction()
        self.swap = GlobalMemory.Swap(self.block_size)
    
    def store_artifact(self, artifact: Artifact) -> None:
        """
        Artifact must not already be in store
        """
        artifact_id, artifact_size = artifact.artifact_id, artifact.size
        assert artifact_id not in self.store
        num_blocks = GlobalMemory.num_blocks(artifact_size, self.block_size)
        req_size = num_blocks * self.block_size
        if req_size > self.total_size:
            raise Exception("Requested global memory store larger than total memory")
        if req_size > self.total_size - self.used_size:
            self.evict(req_size)
        self.used_size += req_size
        self.occupancy[artifact_id] += num_blocks
        self.store[artifact_id] = artifact

        self.lru.update_artifact_id(artifact_id)

    def overwrite_artifact(self, artifact: Artifact) -> bool:
        # can this happen? should the parameter be artifact or artifact_id + size?
        ...

    def get_artifact(self, artifact_id: ArtifactId) -> Artifact:
        """
        Get the artifact out of memory or swap. If in swap, move it back to memory
        """
        if artifact_id in self.store:
            self.lru.update_artifact_id(artifact_id)
            return self.store[artifact_id]
        elif artifact_id in self.swap.swap_store:
            self.store_artifact(self.swap.remove_from_swap(artifact_id))
            return self.get_artifact(artifact_id) # recursive call but should resolve as long as store_artifact guarantees to add it to store
        else:
            raise Exception("Attempted get artifact_id, but not found in both memory and swap")
    
    def remove_artifact(self, artifact_id: ArtifactId) -> Artifact:
        if artifact_id not in self.store:
            raise Exception("artifact_id not in global memory store")
        artifact = self.store[artifact_id]
        artifact_size = artifact.size
        num_blocks = GlobalMemory.num_blocks(artifact_size, self.block_size)
        req_size = num_blocks * self.block_size
        self.used_size -= req_size
        self.occupancy[artifact_id] -= num_blocks
        del self.store[artifact_id]

        self.lru.remove_artifact_id(artifact_id)
        return artifact
    
    def evict(self, size):
        evicted_size = 0
        while evicted_size < size:
            artifact_id = self.lru.get_evict_candidate()
            artifact = self.remove_artifact(artifact_id)
            req_size = GlobalMemory.num_blocks(artifact.size, self.block_size) * self.block_size
            self.swap.add_to_swap(artifact)
            assert artifact.size != 0 # ensure we're making progress since while loops can be undeterministic, this likely means an invalid object was stored to global memory
            evicted_size += req_size


    @staticmethod
    def num_blocks(size: int, block_size: int) -> int:
        return (size + block_size - 1) // block_size


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
