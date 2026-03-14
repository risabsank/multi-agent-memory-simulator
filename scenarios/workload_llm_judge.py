import json
import re
from dataclasses import asdict

from memory.lib import human2bytes
from memory.model import Agent, Artifact, ArtifactScope, Cache, GlobalMemory
from memory.protocols import WriteThroughStrongProtocol
from memory.simulator import Simulator
from memory.workload import BurstyWorkloadConfig, generate_bursty_workload


NUM_RE = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def mock_llm_judge(prompt: str) -> str:
    """Local inference stub that mimics an LLM JSON response.

    This keeps the scenario runnable without external API keys while still exercising
    the LLM judge path (`judge_mode=\"llm\"`).
    """

    previous_match = re.search(rf"previous_confidence=({NUM_RE}|None)", prompt)
    candidate_match = re.search(rf"candidate_confidence=({NUM_RE})", prompt)

    previous = None
    if previous_match and previous_match.group(1) != "None":
        previous = float(previous_match.group(1))
    candidate = float(candidate_match.group(1)) if candidate_match else 0.8

    if previous is None:
        state = "accepted"
        reasons = ["llm_new_artifact"]
    elif candidate - previous < 0.1:
        state = "contested"
        reasons = ["llm_low_confidence_delta"]
    else:
        state = "accepted"
        reasons = ["llm_confidence_upgrade"]

    payload = {
        "coherence_state": state,
        "reason_codes": reasons,
        "confidence_delta": None if previous is None else round(candidate - previous, 3),
    }
    return json.dumps(payload)


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

    sim = Simulator(
        agents=[Agent(agent_id, Cache(1, 1, human2bytes("1 gb"), 4096)) for agent_id in agent_ids],
        global_memory=global_memory,
        protocol=WriteThroughStrongProtocol(
            judge_mode="llm",
            llm_inference_fn=mock_llm_judge,
            llm_provider="local-mock",
            llm_model="rule-based-judge-v1",
            llm_timeout_s=0.2,
        ),
    )

    for op in ops:
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
            f"reason={line.metadata.get('reason_codes')} "
            f"provider={line.metadata.get('judge_provider')} "
            f"model={line.metadata.get('judge_model')}"
        )
        shown += 1
        if shown >= 5:
            break


if __name__ == "__main__":
    main()
