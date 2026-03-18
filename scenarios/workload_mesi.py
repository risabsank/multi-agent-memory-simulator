from dataclasses import asdict

from memory.lib import human2bytes
from memory.model import Agent, Artifact, ArtifactScope, Cache, GlobalMemory
from memory.protocols import MesiProtocol
from memory.simulator import Simulator
from memory.workload import BurstyWorkloadConfig, generate_bursty_workload


def main() -> None:
    agent_ids = ["A", "B", "C"]
    artifact_ids = [("T1", "spec"), ("T1", "plan"), ("T1", "tests")]

    config = BurstyWorkloadConfig(
        duration=30,
        seed=7,
        base_rate_per_tick=1.5,
        read_probability=0.7,
        write_size_bytes=2048,
    )
    ops = generate_bursty_workload(
        config=config,
        agent_ids=agent_ids,
        artifact_ids=artifact_ids,
    )

    global_memory = GlobalMemory(
        read_latency=3,
        write_latency=3,
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

    agents = [Agent(agent_id, Cache(1, 1, human2bytes("1 gb"), 4096)) for agent_id in agent_ids]
    sim = Simulator(
        agents=agents,
        global_memory=global_memory,
        protocol=MesiProtocol(
            bus_latency=1,
        ),
    )

    for op in ops:
        print(op.t, op.op, op.artifact_id, op.agent_id)
        if op.op == "read":
            sim.schedule_read(t=op.t, agent_id=op.agent_id, artifact_id=op.artifact_id)
        else:
            sim.schedule_write(
                t=op.t,
                agent_id=op.agent_id,
                artifact_id=op.artifact_id,
                size=op.size or config.write_size_bytes,
            )

    result = sim.run()
    report = sim.build_report()
    report_metrics = asdict(report)

    print(f"Generated ops: {len(ops)}")
    for metric, value in sorted(report_metrics.items()):
        print(f"{metric}={value}")

    print("\nSample conflict checks:")
    shown = 0
    for line in result.trace:
        if line.event != "EV_CONFLICT_CHECK":
            continue
        print(
            f"t={line.t} artifact={line.metadata.get('artifact_id')} "
            f"state={line.metadata.get('coherence_state')} "
            f"reason={line.metadata.get('reason_codes')}"
        )
        shown += 1
        if shown >= 5:
            break


if __name__ == "__main__":
    main()
