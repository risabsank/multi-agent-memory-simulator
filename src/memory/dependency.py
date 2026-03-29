from __future__ import annotations

from collections import defaultdict, deque

from .model import ArtifactId


class DependencyGraph:
    """Directed dependency graph for artifact invalidation and sync planning.

    Edge A -> B means B depends on A and should be considered stale when A changes.
    """

    def __init__(self) -> None:
        self._dependents: dict[ArtifactId, set[ArtifactId]] = defaultdict(set)

    def add_dependency(self, source: ArtifactId, dependent: ArtifactId) -> None:
        if source == dependent:
            return
        self._dependents[source].add(dependent)

    def remove_dependency(self, source: ArtifactId, dependent: ArtifactId) -> None:
        self._dependents[source].discard(dependent)

    def direct_dependents(self, source: ArtifactId) -> set[ArtifactId]:
        return set(self._dependents.get(source, set()))

    def dependents_closure(self, source: ArtifactId) -> set[ArtifactId]:
        visited: set[ArtifactId] = set()
        queue: deque[ArtifactId] = deque([source])
        while queue:
            current = queue.popleft()
            for dep in self._dependents.get(current, set()):
                if dep in visited:
                    continue
                visited.add(dep)
                queue.append(dep)
        return visited
