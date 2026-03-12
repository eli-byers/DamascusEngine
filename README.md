# DamascusEngine

DamascusEngine is a local research workbench for self-improving agents.

It is built around a simple thesis:

- Percepta suggests models become more useful when they can execute structured computation internally.
- PufferLib shows agent research gets traction when environments are fast, stateful, and measurable.
- `autoresearch` shows progress compounds when an agent can edit code, run experiments, and keep improvements locally.

This repo joins those ideas into one small system you can actually run and modify.

## What This Repo Is

DamascusEngine is not a new model architecture and not a full RL framework.

It is a local playground for testing a narrow but important question:

**What improves when an agent has stronger exact state manipulation on long-horizon tasks?**

The initial repo focuses on:

- algorithmic, long-horizon benchmark tasks,
- contrasting agent styles,
- a repeatable local experiment loop,
- results saved in a machine-readable format.

The name is meant to suggest a system forged under pressure: repeated local trials, hard benchmarks, and exact feedback loops.

## Working Name

`DamascusEngine` is a working project name. It has only been lightly screened for obvious collisions and is not legal clearance.

## Initial Design

The current version contains two benchmark families and three agent styles:

- `ReactiveAgent`
  - Stateless baseline. Responds only to the current observation.
- `ScratchpadAgent`
  - Uses an explicit external state object. This stands in for tool use or an outer orchestration loop.
- `RegisterAgent`
  - Uses compact internal register updates. This stands in for a more computation-native agent design.

The benchmark families are:

- `SequenceMemoryEnv`
  - Observe a hidden binary sequence, survive a delay, answer an exact indexed query.
- `InventoryFlowEnv`
  - Observe a stream of inventory updates across item IDs, survive a delay, report the exact final count of one queried item.

This does not reproduce Percepta directly. Instead, it creates a place to test the *behavioral consequences* of stronger internal computation against tool-like external bookkeeping.

## Repo Layout

```text
configs/                  Experiment configs
results/                  JSON summaries written by runs
src/damascusengine/
  agents.py               Baseline agent implementations
  runner.py               Benchmark runner and reporting
  research_loop.py        Local search loop over configs
  benchmarks/
    inventory_flow.py     Deterministic bookkeeping benchmark
    sequence_memory.py    Deterministic long-horizon benchmark
```

## Quick Start

Requires Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
damascusengine-benchmark --config configs/baseline.json
damascusengine-research --config configs/search.json
```

## Why This Is Interesting

This repo gives you a controlled place to explore three layers at once:

- inner loop: how exact state updates are represented,
- training or eval world: tasks where long-horizon reliability matters,
- outer loop: how a local research agent searches over designs.

The natural next steps are:

- add PufferLib-backed environments,
- swap heuristic agents for learned recurrent or transformer policies,
- add code-editing automation in the style of `autoresearch`,
- introduce compiled subroutines or constrained execution channels.

The first useful result is already visible: the repo can now compare stateful and stateless agents across both memory and bookkeeping workloads, not just one memory toy.

## Near-Term Roadmap

1. Add more tasks: planning, inventory bookkeeping, graph search.
2. Add learned policies and train/eval adapters.
3. Add result comparison dashboards.
4. Add autonomous proposal generation for new configs and benchmarks.

## References

- Percepta, "Can LLMs Be Computers?"
  - https://www.percepta.ai/blog/can-llms-be-computers
- PufferAI, `PufferLib`
  - https://github.com/PufferAI/PufferLib
- Andrej Karpathy, `autoresearch`
  - https://github.com/karpathy/autoresearch

## Current Status

This repository starts intentionally small. The current scaffold is useful if it helps you ask better questions and run tighter local experiments, not because it is already complex.
