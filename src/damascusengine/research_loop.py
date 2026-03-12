from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

from damascusengine.runner import run_benchmark, write_output


def search_configs(config: dict) -> dict:
    base_benchmark = dict(config["benchmark"])
    search = config["search"]
    sequence_lengths = search["sequence_lengths"]
    delays = search["delays"]

    trials = []
    for sequence_length, delay in product(sequence_lengths, delays):
        trial_config = {
            "benchmark": {
                **base_benchmark,
                "sequence_length": sequence_length,
                "delay": delay,
            },
            "agents": config["agents"],
        }
        result = run_benchmark(trial_config)
        scores = {row["agent"]: row["accuracy"] for row in result["results"]}
        gap = scores["register"] - scores["reactive"]
        trials.append(
            {
                "sequence_length": sequence_length,
                "delay": delay,
                "scores": scores,
                "register_over_reactive": gap,
            }
        )

    trials.sort(key=lambda row: row["register_over_reactive"], reverse=True)
    return {
        "base_benchmark": base_benchmark,
        "search_space": search,
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
