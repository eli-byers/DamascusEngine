from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from damascusengine.agents import build_agent
from damascusengine.benchmarks.sequence_memory import SequenceMemoryEnv


def run_benchmark(config: dict) -> dict:
    benchmark = config["benchmark"]
    env = SequenceMemoryEnv(
        sequence_length=int(benchmark["sequence_length"]),
        delay=int(benchmark["delay"]),
        seed=int(benchmark.get("seed", 0)),
    )
    episodes = int(benchmark["episodes"])

    summaries = []
    for agent_name in config["agents"]:
        agent = build_agent(agent_name)
        correctness = []
        traces = []
        for _ in range(episodes):
            episode = env.sample_episode()
            result = env.rollout(agent, episode)
            correctness.append(result["correct"])
            traces.append(result)

        summaries.append(
            {
                "agent": agent_name,
                "accuracy": mean(correctness),
                "episodes": episodes,
                "sequence_length": benchmark["sequence_length"],
                "delay": benchmark["delay"],
                "sample_trace": traces[0],
            }
        )

    return {
        "benchmark": benchmark,
        "results": summaries,
    }


def write_output(payload: dict, output_path: str | None) -> None:
    text = json.dumps(payload, indent=2)
    if output_path is None:
        print(text)
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\nWrote results to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a DamascusEngine benchmark.")
    parser.add_argument("--config", required=True, help="Path to a JSON config file.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    payload = run_benchmark(config)
    write_output(payload, config.get("output_path"))


if __name__ == "__main__":
    main()
