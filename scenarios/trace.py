from memory.model import Agent, Artifact, ArtifactScope, GlobalMemory, Task
from memory.simulator import Simulator


def run() -> None:
    task = Task(task_id="T1", agent_ids=["A", "B"], artifact_ids=[("T1", "shared_plan")])
    global_memory = GlobalMemory(
        latency=5,
        store={
            ("T1", "shared_plan"): Artifact(
                artifact_id=("T1", "shared_plan"),
                version_id=1,
                size=1024,
                scope=ArtifactScope.TASK,
            )
        },
    )

    sim = Simulator(agents=[Agent("A"), Agent("B")], global_memory=global_memory)
    sim.schedule_read(t=0, agent_id="A", artifact_id=("T1", "shared_plan"))
    sim.schedule_read(t=10, agent_id="A", artifact_id=("T1", "shared_plan"))
    sim.schedule_write(t=12, agent_id="A", artifact_id=("T1", "shared_plan"), size=1200)
    sim.schedule_read(t=30, agent_id="B", artifact_id=("T1", "shared_plan"))

    result = sim.run()
    print(f"Task {task.task_id}")
    print("Event trace:")
    for line in result.trace:
        print(f"t={line.t:>3} {line.event:>16} | {line.detail}")

    print("\nMetrics:")
    for agent_id, agent in result.agents.items():
        avg = result.avg_latency(agent_id)
        print(
            f"agent={agent_id} hits={agent.stats.hits} misses={agent.stats.misses} "
            f"avg_latency={avg:.2f} reads={agent.stats.read_count}"
        )

    final_version = result.global_memory.store[("T1", "shared_plan")].version_id
    print(f"final_version(shared_plan)={final_version}")


if __name__ == "__main__":
    run()
