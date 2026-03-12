from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from damascusengine.agents import build_agent
from damascusengine.benchmarks import build_benchmark


def normalize_benchmarks(config: dict) -> list[dict]:
    if "benchmarks" in config:
        return list(config["benchmarks"])
    if "benchmark" in config:
        return [config["benchmark"]]
    raise ValueError("config must define 'benchmark' or 'benchmarks'")


def run_single_benchmark(benchmark: dict, agent_names: list[str]) -> dict:
    env = build_benchmark(benchmark)
    episodes = int(benchmark["episodes"])

    summaries = []
    for agent_name in agent_names:
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
                "benchmark_type": benchmark["type"],
                "sample_trace": traces[0],
            }
        )

    return {
        "benchmark": benchmark,
        "results": summaries,
    }


def summarize_suite(benchmark_runs: list[dict]) -> dict:
    per_agent: dict[str, list[float]] = {}
    for run in benchmark_runs:
        for result in run["results"]:
            per_agent.setdefault(result["agent"], []).append(result["accuracy"])

    aggregate = [
        {
            "agent": agent,
            "mean_accuracy": mean(scores),
            "benchmarks": len(scores),
        }
        for agent, scores in sorted(per_agent.items())
    ]

    return {
        "benchmarks": benchmark_runs,
        "aggregate": aggregate,
    }


def run_benchmark(config: dict) -> dict:
    benchmark_specs = normalize_benchmarks(config)
    benchmark_runs = [
        run_single_benchmark(benchmark_spec, config["agents"])
        for benchmark_spec in benchmark_specs
    ]
    return summarize_suite(benchmark_runs)


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
