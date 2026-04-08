from enum import Enum


class CoherenceState(str, Enum):
    """Authoritative coherence states for runtime write/read decisions."""

    ACCEPTED = "accepted"
    CONTESTED = "contested"
    PROVISIONAL = "provisional"
    DEPRECATED = "deprecated"


class RuntimeEventType(str, Enum):
    """Stable runtime event type names for orchestration and audit logging."""

    CONTROL_REQUEST = "control.request"
    ORCHESTRATION_STEP = "orchestration.step"
    MEMORY_READ = "memory.read"
    MEMORY_WRITE = "memory.write"
    JUDGE_EVALUATION = "judge.evaluation"
    JOURNAL_APPEND = "journal.append"
from enum import Enum


class CoherenceState(str, Enum):
    """Authoritative coherence states for runtime write/read decisions."""

    ACCEPTED = "accepted"
    CONTESTED = "contested"
    PROVISIONAL = "provisional"
    DEPRECATED = "deprecated"


class RuntimeEventType(str, Enum):
    """Stable runtime event type names for orchestration and audit logging."""

    CONTROL_REQUEST = "control.request"
    ORCHESTRATION_STEP = "orchestration.step"
    MEMORY_READ = "memory.read"
    MEMORY_WRITE = "memory.write"
    JUDGE_EVALUATION = "judge.evaluation"
    JOURNAL_APPEND = "journal.append"
