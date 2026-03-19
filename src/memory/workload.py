from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Literal

from .model import ArtifactId

OperationType = Literal["read", "write"]


@dataclass(frozen=True)
class WorkloadOp:
    """Single operation generated for a simulator run."""

    t: int
    agent_id: str
    artifact_id: ArtifactId
    op: OperationType
    size: int | None = None


@dataclass(frozen=True)
class BurstyWorkloadConfig:
    """Configuration for a bursty synthetic LLM-style workload.

    Uses a simple two-state Markov-modulated Poisson process (MMPP):
    - normal state for baseline traffic
    - burst state for traffic spikes
    """

    duration: int
    seed: int = 0
    p_enter_burst: float = 0.03
    p_exit_burst: float = 0.25
    base_rate_per_tick: float = 0.6
    burst_multiplier: float = 7.0
    read_probability: float = 0.75
    write_size_bytes: int = 1024
    zipf_alpha: float = 1.15


def _sample_poisson(lam: float, rng: random.Random) -> int:
    """Knuth Poisson sampler, good enough for small-medium lambda in this simulator."""
    if lam <= 0:
        return 0
    threshold = math.exp(-lam)
    k = 0
    p = 1.0
    while p > threshold:
        k += 1
        p *= rng.random()
    return k - 1


def _weighted_choice(items: list[ArtifactId], weights: list[float], rng: random.Random) -> ArtifactId:
    total = sum(weights)
    draw = rng.random() * total
    upto = 0.0
    for item, weight in zip(items, weights):
        upto += weight
        if upto >= draw:
            return item
    return items[-1]


def generate_bursty_workload(
    *,
    config: BurstyWorkloadConfig,
    agent_ids: list[str],
    artifact_ids: list[ArtifactId],
) -> list[WorkloadOp]:
    """Generate a synthetic but realistic-ish bursty workload.

    Design goals:
    - deterministic via seed
    - burst periods mixed with background traffic
    - read-heavy mix with occasional writes
    - hot artifacts more likely than long-tail artifacts (Zipf-like popularity)
    """
    if not agent_ids:
        raise ValueError("agent_ids must not be empty")
    if not artifact_ids:
        raise ValueError("artifact_ids must not be empty")
    if config.duration <= 0:
        raise ValueError("duration must be > 0")

    rng = random.Random(config.seed)
    weights = [1.0 / ((idx + 1) ** config.zipf_alpha) for idx in range(len(artifact_ids))]

    burst_mode = False
    ops: list[WorkloadOp] = []

    for t in range(config.duration):
        if burst_mode:
            if rng.random() < config.p_exit_burst:
                burst_mode = False
        else:
            if rng.random() < config.p_enter_burst:
                burst_mode = True

        lam = config.base_rate_per_tick * (config.burst_multiplier if burst_mode else 1.0)
        event_count = _sample_poisson(lam, rng)

        for _ in range(event_count):
            op_type: OperationType = "read" if rng.random() < config.read_probability else "write"
            artifact_id = _weighted_choice(artifact_ids, weights, rng)
            agent_id = agent_ids[rng.randrange(len(agent_ids))]

            ops.append(
                WorkloadOp(
                    t=t,
                    agent_id=agent_id,
                    artifact_id=artifact_id,
                    op=op_type,
                    size=config.write_size_bytes if op_type == "write" else None,
                )
            )

    ops.sort(key=lambda op: (op.t, op.agent_id, op.artifact_id, op.op))
    return ops