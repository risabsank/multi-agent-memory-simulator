from memory.lib import human2bytes
from memory.model import Agent, GlobalMemory, Memory, Task, Artifact, ArtifactScope
from memory.protocols import WriteThroughStrongProtocol
from memory.simulator import Simulator


def run() -> None:
    task = Task(
        task_id="T1", agent_ids=["A", "B"], artifact_ids=[("T1", "shared_plan")]
    )
    global_memory = GlobalMemory(
        read_latency=5,
        write_latency=5,
        swap_latency=100,
        total_size=human2bytes("1gb"),
        block_size=4096,
    )
    artifact = Artifact(
        artifact_id=("T1", "shared_plan"),
        version_id=1,
        size=1024,
        scope=ArtifactScope.TASK,
    )
    global_memory.store_artifact(artifact)

    sim = Simulator(
        agents=[
            Agent("A", Memory(1, 1, human2bytes("1 gb"), 4096)),
            Agent("B", Memory(1, 1, human2bytes("1 gb"), 4096)),
        ],
        global_memory=global_memory,
        protocol=WriteThroughStrongProtocol(),
    )
    sim.schedule_read(t=0, agent_id="A", artifact_id=("T1", "shared_plan"))
    sim.schedule_read(t=10, agent_id="A", artifact_id=("T1", "shared_plan"))
    sim.schedule_write(t=12, agent_id="A", artifact_id=("T1", "shared_plan"), size=1200)
    sim.schedule_read(t=30, agent_id="B", artifact_id=("T1", "shared_plan"))

    result = sim.run()
    report = sim.build_report()

    print(f"Task {task.task_id}")
    print("Event trace:")
    for line in result.trace:
        print(f"t={line.t:>3} {line.event:>18} | {line.detail} | meta={line.metadata}")

    print("\nMetrics:")
    for agent_id, agent in result.agents.items():
        avg_read = result.avg_latency(agent_id)
        avg_write = result.avg_write_latency(agent_id)
        print(
            f"agent={agent_id} hits={agent.stats.hits} misses={agent.stats.misses} "
            f"avg_read_latency={avg_read:.2f} avg_write_latency={avg_write:.2f} "
            f"reads={agent.stats.read_count} writes={agent.stats.write_count}"
        )

    print("\nRun report:")
    print(
        f"events={report.total_events} hits={report.cache_hits} misses={report.cache_misses} "
        f"conflict_checks={report.conflict_checks} accepted_writes={report.accepted_writes} "
        f"contested_writes={report.contested_writes} avg_read_latency={report.avg_read_latency:.2f} "
        f"avg_write_latency={report.avg_write_latency:.2f}"
    )

    assert result.global_memory is not None
    final_artifact = result.global_memory.get_artifact(("T1", "shared_plan"))
    print(
        f"final_version(shared_plan)={final_artifact.version_id} "
        f"coherence={final_artifact.coherence_state.value} "
        f"confidence={final_artifact.confidence:.2f}"
    )


if __name__ == "__main__":
    run()
