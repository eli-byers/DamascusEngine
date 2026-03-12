from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass
class Episode:
    sequence: list[int]
    delay: int
    query_index: int

    @property
    def answer(self) -> int:
        return self.sequence[self.query_index]


class SequenceMemoryEnv:
    """Deterministic exact-memory benchmark.

    The agent observes a binary sequence token by token, waits through a delay,
    and then must answer a query asking for one earlier token by index.
    """

    def __init__(self, sequence_length: int, delay: int, seed: int = 0) -> None:
        self.sequence_length = sequence_length
        self.delay = delay
        self._rng = random.Random(seed)

    def sample_episode(self) -> Episode:
        sequence = [self._rng.randint(0, 1) for _ in range(self.sequence_length)]
        query_index = self._rng.randint(0, self.sequence_length - 1)
        return Episode(sequence=sequence, delay=self.delay, query_index=query_index)

    def rollout(self, agent, episode: Episode) -> dict:
        agent.reset()

        for token in episode.sequence:
            agent.act({"phase": "observe", "token": token})

        for _ in range(episode.delay):
            agent.act({"phase": "delay"})

        prediction = agent.act(
            {
                "phase": "query",
                "query_index": episode.query_index,
            }
        )

        correct = int(prediction == episode.answer)
        return {
            "prediction": int(prediction),
            "target": episode.answer,
            "correct": correct,
            "query_index": episode.query_index,
            "sequence": episode.sequence,
        }
