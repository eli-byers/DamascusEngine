"""Benchmark suites for DamascusEngine."""

from __future__ import annotations

from damascusengine.benchmarks.inventory_flow import InventoryFlowEnv
from damascusengine.benchmarks.sequence_memory import SequenceMemoryEnv


def build_benchmark(spec: dict):
    benchmark_type = spec["type"]
    seed = int(spec.get("seed", 0))

    if benchmark_type == "sequence_memory":
        return SequenceMemoryEnv(
            sequence_length=int(spec["sequence_length"]),
            delay=int(spec["delay"]),
            seed=seed,
        )
    if benchmark_type == "inventory_flow":
        return InventoryFlowEnv(
            num_items=int(spec["num_items"]),
            num_operations=int(spec["num_operations"]),
            delay=int(spec["delay"]),
            seed=seed,
        )

    raise ValueError(f"unknown benchmark type: {benchmark_type}")
