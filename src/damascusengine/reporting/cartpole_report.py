from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_runs(root: Path) -> list[dict]:
    runs = []
    for summary_path in sorted(root.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        run_dir = summary_path.parent
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists():
            continue
        metrics = pd.read_csv(metrics_path)
        runs.append(
            {
                "name": run_dir.parent.name,
                "run_dir": run_dir,
                "summary": summary,
                "metrics": metrics,
            }
        )
    return runs


def write_markdown(runs: list[dict], output_path: Path) -> None:
    lines = [
        "# CartPole Run Report",
        "",
        "| suite | run | backend | steps | best return20 | final return20 | episodes | output |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for run in runs:
        summary = run["summary"]
        config = summary["config"]
        lines.append(
            "| {suite} | {run_name} | {backend} | {steps} | {best:.2f} | {final:.2f} | {episodes} | {output} |".format(
                suite=run["name"],
                run_name=run["run_dir"].name,
                backend=config["backend"],
                steps=config["total_timesteps"],
                best=summary["best_mean_return_20"],
                final=summary["final_mean_return_20"],
                episodes=summary["episodes_completed"],
                output=run["run_dir"],
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_comparison_plot(runs: list[dict], output_path: Path) -> None:
    if not runs:
        return

    plt.figure(figsize=(10, 6))
    for run in runs:
        metrics = run["metrics"]
        label = f"{run['name']}:{run['summary']['config']['backend']}"
        plt.plot(metrics["global_step"], metrics["mean_return_20"], label=label)

    plt.title("DamascusEngine CartPole Runs")
    plt.xlabel("Environment Steps")
    plt.ylabel("Mean Return (20-episode window)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize DamascusEngine CartPole runs.")
    parser.add_argument("--root", default="results", help="Root results directory.")
    parser.add_argument("--output-dir", default="results/reports", help="Output directory for report artifacts.")
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for suite_root in sorted(root.glob("cartpole-*")):
        runs.extend(load_runs(suite_root))

    write_markdown(runs, output_dir / "cartpole_runs.md")
    save_comparison_plot(runs, output_dir / "cartpole_runs.png")
    print(f"Wrote report to {output_dir}")


if __name__ == "__main__":
    main()
