from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from statistics import mean

from damascusengine.runner import run_benchmark, write_output


def search_configs(config: dict) -> dict:
    search = config["search"]
    benchmark_templates = config["benchmarks"]

    trials = []
    for template in benchmark_templates:
        sweep_keys = template["sweep"]
        sweep_values = [search[key] for key in sweep_keys]
        for values in product(*sweep_values):
            benchmark = {key: value for key, value in template.items() if key != "sweep"}
            benchmark.update(dict(zip(sweep_keys, values)))
            trial_config = {
                "benchmarks": [benchmark],
                "agents": config["agents"],
            }
            result = run_benchmark(trial_config)
            aggregate_scores = {
                row["agent"]: row["mean_accuracy"] for row in result["aggregate"]
            }
            gap = aggregate_scores["register"] - aggregate_scores["reactive"]
            scratchpad_gap = (
                aggregate_scores["scratchpad"] - aggregate_scores["reactive"]
            )
            trials.append(
                {
                    "benchmark_type": benchmark["type"],
                    "config": benchmark,
                    "scores": aggregate_scores,
                    "register_over_reactive": gap,
                    "scratchpad_over_reactive": scratchpad_gap,
                }
            )

    trials.sort(
        key=lambda row: (
            row["register_over_reactive"],
            row["scratchpad_over_reactive"],
        ),
        reverse=True,
    )
    benchmark_types = sorted({trial["benchmark_type"] for trial in trials})
    per_type_gap = {
        benchmark_type: mean(
            trial["register_over_reactive"]
            for trial in trials
            if trial["benchmark_type"] == benchmark_type
        )
        for benchmark_type in benchmark_types
    }
    return {
        "search_space": search,
        "benchmark_types": benchmark_types,
        "mean_register_advantage_by_type": per_type_gap,
        "trials": trials,
        "best_trial": trials[0] if trials else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a DamascusEngine local research loop.")
    parser.add_argument("--config", required=True, help="Path to a JSON config file.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    payload = search_configs(config)
    write_output(payload, config.get("output_path"))


if __name__ == "__main__":
    main()
