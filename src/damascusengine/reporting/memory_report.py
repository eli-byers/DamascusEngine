from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["MPLCONFIGDIR"] = str(Path(".matplotlib").resolve())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_runs(root: Path) -> list[dict]:
    runs = []
    for summary_path in sorted(root.glob("*/*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics_path = summary_path.parent / "metrics.csv"
        if not metrics_path.exists():
            continue
        runs.append(
            {
                "agent": summary["config"]["agent"],
                "summary": summary,
                "metrics": pd.read_csv(metrics_path),
                "run_dir": summary_path.parent,
            }
        )
    return runs


def write_markdown(runs: list[dict], output_path: Path) -> None:
    lines = [
        "# Memory Run Report",
        "",
        "| agent | backend | cue delay | steps | best success50 | final success50 | output |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for run in runs:
        config = run["summary"]["config"]
        lines.append(
            "| {agent} | {backend} | {delay} | {steps} | {best:.2f} | {final:.2f} | {output} |".format(
                agent=config["agent"],
                backend=config["backend"],
                delay=config["cue_delay"],
                steps=config["total_timesteps"],
                best=run["summary"]["best_success_rate_50"],
                final=run["summary"]["final_success_rate_50"],
                output=run["run_dir"],
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_plot(runs: list[dict], output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    for run in runs:
        metrics = run["metrics"]
        plt.plot(metrics["global_step"], metrics["success_rate_50"], label=run["agent"])
    plt.title("Cue Recall Success Rate")
    plt.xlabel("Environment Steps")
    plt.ylabel("Success Rate (50-episode window)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize DamascusEngine memory runs.")
    parser.add_argument("--root", default="results/memory-proof", help="Root memory results directory.")
    parser.add_argument("--output-dir", default="results/reports", help="Output directory for report artifacts.")
    args = parser.parse_args()

    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(root)
    if not runs:
        raise SystemExit(f"No memory runs found under {root}")
    write_markdown(runs, output_dir / "memory_runs.md")
    save_plot(runs, output_dir / "memory_runs.png")
    print(f"Wrote memory report to {output_dir}")


if __name__ == "__main__":
    main()
