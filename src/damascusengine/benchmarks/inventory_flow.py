from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass
class InventoryEpisode:
    num_items: int
    operations: list[tuple[int, int]]
    delay: int
    query_item: int

    @property
    def answer(self) -> int:
        totals = [0] * self.num_items
        for item_id, delta in self.operations:
            totals[item_id] += delta
        return totals[self.query_item]


class InventoryFlowEnv:
    """Exact bookkeeping benchmark.

    The agent observes a stream of inventory updates for multiple item IDs, then
    must report the final count of one queried item after a delay.
    """

    def __init__(
        self,
        num_items: int,
        num_operations: int,
        delay: int,
        seed: int = 0,
    ) -> None:
        self.num_items = num_items
        self.num_operations = num_operations
        self.delay = delay
        self._rng = random.Random(seed)

    def sample_episode(self) -> InventoryEpisode:
        operations = []
        for _ in range(self.num_operations):
            item_id = self._rng.randint(0, self.num_items - 1)
            delta = self._rng.choice([-2, -1, 1, 2])
            operations.append((item_id, delta))
        query_item = self._rng.randint(0, self.num_items - 1)
        return InventoryEpisode(
            num_items=self.num_items,
            operations=operations,
            delay=self.delay,
            query_item=query_item,
        )

    def rollout(self, agent, episode: InventoryEpisode) -> dict:
        agent.reset()

        for item_id, delta in episode.operations:
            agent.act(
                {
                    "task": "inventory_flow",
                    "phase": "observe",
                    "item_id": item_id,
                    "delta": delta,
                }
            )

        for _ in range(episode.delay):
            agent.act({"task": "inventory_flow", "phase": "delay"})

        prediction = agent.act(
            {
                "task": "inventory_flow",
                "phase": "query",
                "query_item": episode.query_item,
            }
        )
        correct = int(prediction == episode.answer)
        return {
            "prediction": int(prediction),
            "target": episode.answer,
            "correct": correct,
            "query_item": episode.query_item,
            "operations": episode.operations,
        }
