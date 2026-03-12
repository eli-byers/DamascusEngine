from __future__ import annotations

from dataclasses import dataclass, field


class BaseAgent:
    name = "base"

    def reset(self) -> None:
        """Reset agent state before a new episode."""

    def act(self, observation: dict) -> int:
        raise NotImplementedError


@dataclass
class ReactiveAgent(BaseAgent):
    name: str = "reactive"

    def act(self, observation: dict) -> int:
        # Stateless baseline: during query, guess a constant answer.
        return int(observation.get("token", 0) or 0)


@dataclass
class ScratchpadAgent(BaseAgent):
    name: str = "scratchpad"
    history: list[int] = field(default_factory=list)

    def reset(self) -> None:
        self.history.clear()

    def act(self, observation: dict) -> int:
        phase = observation["phase"]
        if phase == "observe":
            self.history.append(int(observation["token"]))
            return 0
        if phase == "delay":
            return 0

        query_index = int(observation["query_index"])
        return self.history[query_index]


@dataclass
class RegisterAgent(BaseAgent):
    name: str = "register"
    capacity: int = 16
    registers: list[int] = field(default_factory=list)
    count: int = 0

    def reset(self) -> None:
        self.registers = [0] * self.capacity
        self.count = 0

    def act(self, observation: dict) -> int:
        phase = observation["phase"]
        if phase == "observe":
            if self.count >= self.capacity:
                raise ValueError("register capacity exceeded")
            self.registers[self.count] = int(observation["token"])
            self.count += 1
            return 0
        if phase == "delay":
            return 0

        query_index = int(observation["query_index"])
        return self.registers[query_index]


AGENT_FACTORIES = {
    "reactive": ReactiveAgent,
    "scratchpad": ScratchpadAgent,
    "register": RegisterAgent,
}


def build_agent(name: str) -> BaseAgent:
    try:
        agent_type = AGENT_FACTORIES[name]
    except KeyError as exc:
        raise ValueError(f"unknown agent: {name}") from exc

    agent = agent_type()
    agent.reset()
    return agent
