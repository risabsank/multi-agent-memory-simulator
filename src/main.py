import logging

import sys
import argparse
import time
from memory.model import Artifact, Cache, GlobalMemory, Agent, ArtifactScope
from memory.simulator import Simulator
from memory.protocols.eventual import EventualProtocol
from memory.lib import human2bytes

logger = logging.getLogger(__name__)


def test1():
    def slow_bad_llm(_prompt: str) -> str:
        time.sleep(0.05)
        return "{bad json"

    artifact_id = ("T1", "shared")
    artifact = Artifact(
        artifact_id=artifact_id,
        version_id=1,
        size=1,
        scope=ArtifactScope.TASK,
        confidence=0.95,
    )

    sim = Simulator(
        agents=[Agent("A", Cache(1, 1, human2bytes("1 gb"), 4096))],
        global_memory=GlobalMemory(
            read_latency=2,
            write_latency=2,
            swap_latency=100,
            total_size=human2bytes("4 gb"),
            block_size=4096,
        ),
        protocol=EventualProtocol(
            propagation_delay=1,
            judge_mode="llm",
            llm_inference_fn=slow_bad_llm,
            llm_provider="openai",
            llm_model="gpt-sim",
            llm_timeout_s=0.001,
        ),
    )
    sim.global_memory.store_artifact(artifact)
    sim.schedule_write(0, "A", artifact_id, 5)
    result = sim.run()
    # conflict_event = next(
    #     line for line in result.trace if line.event == "EV_CONFLICT_CHECK"
    # )
    for line in result.trace:
        print(line.t, line.detail, line.event)
        print(line.metadata)


def main(args=None):
    args = sys.argv[1:]
    parser = argparse.ArgumentParser()
    test1()


if __name__ == "__main__":
    main()
