from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable

from memory.lib import human2bytes
from memory.model import Agent, Artifact, ArtifactScope, Cache, GlobalMemory
from memory.protocols import (
    EventualProtocol,
    HybridProtocol,
    MesiProtocol,
    WriteThroughStrongProtocol,
)
from memory.protocols.base import ConsistencyProtocol
from memory.protocols.judges.llm import build_openai_inference_fn
from memory.simulator import RunReport, Simulator
from memory.workload import BurstyWorkloadConfig, WorkloadOp, generate_bursty_workload


OUTPUT_ROOT = Path(__file__).parent / "generated"
WORKLOAD_DIR = OUTPUT_ROOT / "workloads"
RAW_METRICS_CSV = OUTPUT_ROOT / "evaluation_metrics_raw.csv"
PLOTTABLE_METRICS_CSV = OUTPUT_ROOT / "evaluation_metrics_plot_ready.csv"
SUMMARY_METRICS_CSV = OUTPUT_ROOT / "evaluation_metrics_summary.csv"
RAW_JSON = OUTPUT_ROOT / "evaluation_metrics_raw.json"

LLM_MODEL = "gpt-4.1-mini"
LLM_TIMEOUT_S = 1.5


@dataclass(frozen=True)
class Regime:
    name: str
    base_rate_per_tick: float
    read_probability: float
    p_enter_burst: float
    p_exit_burst: float
    burst_multiplier: float
    zipf_alpha: float


@dataclass(frozen=True)
class ScalePoint:
    name: str
    num_agents: int
    num_artifacts: int
    duration: int


@dataclass(frozen=True)
class JudgeConfig:
    name: str
    judge_mode: str
    deterministic_profile: str


def _build_protocol(
    protocol_name: str, judge: JudgeConfig, agents: list[Agent]
) -> ConsistencyProtocol:
    protocol_map: dict[str, Callable[..., object]] = {
        "strong": WriteThroughStrongProtocol,
        "eventual": EventualProtocol,
        "mesi": MesiProtocol,
        "hybrid": HybridProtocol,
    }
    if protocol_name not in protocol_map:
        raise ValueError(f"unsupported protocol: {protocol_name}")

    if protocol_name == "mesi":
        return MesiProtocol(bus_latency=1)

    kwargs = {
        "judge_mode": judge.judge_mode,
        "deterministic_profile": judge.deterministic_profile,
    }

    if judge.judge_mode == "llm":
        llm_inference_fn = build_openai_inference_fn(model=LLM_MODEL)
        kwargs.update(
            {  # type: ignore[reportCallIssue]
                "llm_inference_fn": llm_inference_fn,
                "llm_provider": "openai",
                "llm_model": LLM_MODEL,
                "llm_timeout_s": LLM_TIMEOUT_S,
            }
        )

    return protocol_map[protocol_name](**kwargs)  # type: ignore[reportReturnType]  pyright is pretty stupid when it comes to funky function/class return handling


def _artifact_ids(num_artifacts: int) -> list[tuple[str, str]]:
    return [("T1", f"artifact_{idx + 1}") for idx in range(num_artifacts)]


def _agent_ids(num_agents: int) -> list[str]:
    return [f"A{idx + 1}" for idx in range(num_agents)]


def _write_workload_file(path: Path, ops: list[WorkloadOp]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        for op in ops:
            f.write(
                json.dumps(
                    {
                        "t": op.t,
                        "agent_id": op.agent_id,
                        "artifact_id": list(op.artifact_id),
                        "op": op.op,
                        "size": op.size,
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def _setup_simulator(
    agent_ids: list[str],
    artifact_ids: list[tuple[str, str]],
    protocol_name: str,
    judge: JudgeConfig,
) -> Simulator:
    global_memory = GlobalMemory(
        read_latency=3,
        write_latency=4,
        swap_latency=100,
        total_size=human2bytes("8 gb"),
        block_size=4096,
    )
    for artifact_id in artifact_ids:
        global_memory.store_artifact(
            Artifact(
                artifact_id=artifact_id,
                version_id=1,
                size=1024,
                scope=ArtifactScope.TASK,
                confidence=1.0,
            )
        )

    agents = [
        Agent(agent_id, Cache(1, 1, human2bytes("1 gb"), 4096))
        for agent_id in agent_ids
    ]

    return Simulator(
        agents=agents,
        global_memory=global_memory,
        protocol=_build_protocol(protocol_name, judge, agents),
    )


def _zero_report() -> RunReport:
    return RunReport(
        total_events=0,
        cache_hits=0,
        cache_misses=0,
        conflict_checks=0,
        contested_writes=0,
        accepted_writes=0,
        avg_read_latency=0.0,
        avg_write_latency=0.0,
    )


def _run_once(
    ops: list[WorkloadOp],
    protocol_name: str,
    judge: JudgeConfig,
    agent_ids: list[str],
    artifact_ids: list[tuple[str, str]],
) -> tuple[RunReport, str | None]:
    sim = _setup_simulator(agent_ids, artifact_ids, protocol_name, judge)
    for op in ops:
        if op.op == "read":
            sim.schedule_read(t=op.t, agent_id=op.agent_id, artifact_id=op.artifact_id)
        else:
            sim.schedule_write(
                t=op.t,
                agent_id=op.agent_id,
                artifact_id=op.artifact_id,
                size=op.size or 2048,
            )
    try:
        sim.run()
        return sim.build_report(), None
    except Exception as exc:
        return _zero_report(), f"{type(exc).__name__}: {exc}"


def _flatten_for_csv(base: dict[str, object], report: RunReport) -> dict[str, object]:
    report_data = asdict(report)
    flat: dict[str, object] = dict(base)
    for key, value in report_data.items():
        if isinstance(value, dict):
            flat[key] = json.dumps(value, sort_keys=True)
        else:
            flat[key] = value
    return flat


def _aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_group: dict[tuple[str, str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            str(row["protocol"]),
            str(row["judge"]),
            str(row["regime"]),
            str(row["scale"]),
        )
        by_group.setdefault(key, []).append(row)

    numeric_keys = [
        "cache_hit_rate",
        "contested_ratio",
        "avg_read_latency",
        "avg_write_latency",
        "read_latency_p95",
        "write_latency_p95",
        "stale_read_rate",
        "avg_visibility_lag",
        "avg_convergence_time",
        "read_your_writes_success_rate",
        "monotonic_read_violations",
        "avg_stale_version_gap",
        "p95_stale_version_gap",
        "max_stale_version_gap",
        "avg_judge_latency",
        "fallback_count",
        "conflict_checks",
        "total_events",
    ]

    summary: list[dict[str, object]] = []
    for (protocol, judge, regime, scale), group in sorted(by_group.items()):
        row: dict[str, object] = {
            "protocol": protocol,
            "judge": judge,
            "regime": regime,
            "scale": scale,
            "replicates": len(group),
        }
        for key in numeric_keys:
            values = [float(g[key]) for g in group if g.get(key) is not None]  # type: ignore[reportArgumentType]
            row[f"{key}_mean"] = mean(values) if values else 0.0
            row[f"{key}_std"] = pstdev(values) if len(values) > 1 else 0.0
        summary.append(row)
    return summary


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    WORKLOAD_DIR.mkdir(parents=True, exist_ok=True)

    regimes = [
        Regime(
            "low",
            base_rate_per_tick=0.35,
            read_probability=0.85,
            p_enter_burst=0.02,
            p_exit_burst=0.30,
            burst_multiplier=4.0,
            zipf_alpha=1.0,
        ),
        Regime(
            "medium",
            base_rate_per_tick=0.8,
            read_probability=0.75,
            p_enter_burst=0.04,
            p_exit_burst=0.25,
            burst_multiplier=6.0,
            zipf_alpha=1.15,
        ),
        Regime(
            "high",
            base_rate_per_tick=1.4,
            read_probability=0.65,
            p_enter_burst=0.08,
            p_exit_burst=0.20,
            burst_multiplier=8.5,
            zipf_alpha=1.4,
        ),
    ]
    scales = [
        ScalePoint("small", num_agents=3, num_artifacts=4, duration=40),
        ScalePoint("medium", num_agents=5, num_artifacts=7, duration=65),
        ScalePoint("large", num_agents=8, num_artifacts=12, duration=90),
    ]
    seeds = [7, 17, 29]

    judges = [
        JudgeConfig(
            name="deterministic_strict",
            judge_mode="deterministic",
            deterministic_profile="strict",
        ),
        JudgeConfig(
            name="deterministic_balanced",
            judge_mode="deterministic",
            deterministic_profile="balanced",
        ),
        JudgeConfig(
            name="deterministic_lenient",
            judge_mode="deterministic",
            deterministic_profile="permissive",
        ),
        JudgeConfig(
            name="llm_openai", judge_mode="llm", deterministic_profile="balanced"
        ),
    ]
    protocols = ["strong", "eventual", "mesi", "hybrid"]

    workload_cache: dict[
        tuple[str, str, int], tuple[list[WorkloadOp], list[str], list[tuple[str, str]]]
    ] = {}

    for regime in regimes:
        for scale in scales:
            for seed in seeds:
                agent_ids = _agent_ids(scale.num_agents)
                artifact_ids = _artifact_ids(scale.num_artifacts)
                config = BurstyWorkloadConfig(
                    duration=scale.duration,
                    seed=seed,
                    p_enter_burst=regime.p_enter_burst,
                    p_exit_burst=regime.p_exit_burst,
                    base_rate_per_tick=regime.base_rate_per_tick,
                    burst_multiplier=regime.burst_multiplier,
                    read_probability=regime.read_probability,
                    write_size_bytes=2048,
                    zipf_alpha=regime.zipf_alpha,
                )
                ops = generate_bursty_workload(
                    config=config,
                    agent_ids=agent_ids,
                    artifact_ids=artifact_ids,
                )
                workload_cache[(regime.name, scale.name, seed)] = (
                    ops,
                    agent_ids,
                    artifact_ids,
                )
                workload_path = (
                    WORKLOAD_DIR
                    / f"workload_{regime.name}_{scale.name}_seed{seed}.jsonl"
                )
                _write_workload_file(workload_path, ops)

    raw_rows: list[dict[str, object]] = []
    for protocol_name in protocols:
        for judge in judges:
            for regime in regimes:
                for scale in scales:
                    for seed in seeds:
                        ops, agent_ids, artifact_ids = workload_cache[
                            (regime.name, scale.name, seed)
                        ]

                        print(
                            f"Running test: "
                            f"protocol={protocol_name}, "
                            f"judge={judge.name}, "
                            f"regime={regime.name}, "
                            f"scale={scale.name}, "
                            f"seed={seed}"
                        )

                        report, run_error = _run_once(
                            ops=ops,
                            protocol_name=protocol_name,
                            judge=judge,
                            agent_ids=agent_ids,
                            artifact_ids=artifact_ids,
                        )
                        row_base = {
                            "protocol": protocol_name,
                            "judge": judge.name,
                            "judge_mode": judge.judge_mode,
                            "deterministic_profile": judge.deterministic_profile,
                            "regime": regime.name,
                            "scale": scale.name,
                            "seed": seed,
                            "num_agents": scale.num_agents,
                            "num_artifacts": scale.num_artifacts,
                            "duration": scale.duration,
                            "operations": len(ops),
                            "run_status": "error" if run_error else "ok",
                            "run_error": run_error or "",
                        }
                        raw_rows.append(_flatten_for_csv(row_base, report))

    _write_csv(RAW_METRICS_CSV, raw_rows)

    plot_columns = [
        "protocol",
        "judge",
        "regime",
        "scale",
        "seed",
        "operations",
        "cache_hit_rate",
        "avg_read_latency",
        "avg_write_latency",
        "read_latency_p95",
        "write_latency_p95",
        "stale_read_rate",
        "avg_visibility_lag",
        "avg_convergence_time",
        "contested_ratio",
        "conflict_checks",
        "avg_judge_latency",
        "fallback_count",
        "read_your_writes_success_rate",
        "monotonic_read_violations",
        "avg_stale_version_gap",
        "p95_stale_version_gap",
        "max_stale_version_gap",
    ]
    plot_rows = [{k: row.get(k) for k in plot_columns} for row in raw_rows]
    _write_csv(PLOTTABLE_METRICS_CSV, plot_rows)

    summary_rows = _aggregate(raw_rows)
    _write_csv(SUMMARY_METRICS_CSV, summary_rows)

    with RAW_JSON.open("w", encoding="utf-8") as f:
        json.dump(raw_rows, f, indent=2, sort_keys=True)

    print(f"Wrote workloads to: {WORKLOAD_DIR}")
    print(f"Raw metrics: {RAW_METRICS_CSV}")
    print(f"Plot-ready metrics: {PLOTTABLE_METRICS_CSV}")
    print(f"Summary metrics: {SUMMARY_METRICS_CSV}")


if __name__ == "__main__":
    main()
