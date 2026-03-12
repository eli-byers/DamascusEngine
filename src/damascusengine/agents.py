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
        task = observation.get("task")
        if task == "inventory_flow":
            return int(observation.get("delta", 0) or 0)
        # Stateless baseline: during query, guess a constant answer.
        return int(observation.get("token", 0) or 0)


@dataclass
class ScratchpadAgent(BaseAgent):
    name: str = "scratchpad"
    history: list[int] = field(default_factory=list)
    inventory: dict[int, int] = field(default_factory=dict)

    def reset(self) -> None:
        self.history.clear()
        self.inventory.clear()

    def act(self, observation: dict) -> int:
        task = observation.get("task")
        if task == "inventory_flow":
            return self._act_inventory(observation)
        phase = observation["phase"]
        if phase == "observe":
            self.history.append(int(observation["token"]))
            return 0
        if phase == "delay":
            return 0

        query_index = int(observation["query_index"])
        return self.history[query_index]

    def _act_inventory(self, observation: dict) -> int:
        phase = observation["phase"]
        if phase == "observe":
            item_id = int(observation["item_id"])
            delta = int(observation["delta"])
            self.inventory[item_id] = self.inventory.get(item_id, 0) + delta
            return 0
        if phase == "delay":
            return 0

        item_id = int(observation["query_item"])
        return self.inventory.get(item_id, 0)


@dataclass
class RegisterAgent(BaseAgent):
    name: str = "register"
    capacity: int = 16
    registers: list[int] = field(default_factory=list)
    count: int = 0
    inventory_registers: list[int] = field(default_factory=list)

    def reset(self) -> None:
        self.registers = [0] * self.capacity
        self.inventory_registers = [0] * self.capacity
        self.count = 0

    def act(self, observation: dict) -> int:
        task = observation.get("task")
        if task == "inventory_flow":
            return self._act_inventory(observation)
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

    def _act_inventory(self, observation: dict) -> int:
        phase = observation["phase"]
        if phase == "observe":
            item_id = int(observation["item_id"])
            delta = int(observation["delta"])
            if item_id >= self.capacity:
                raise ValueError("inventory register capacity exceeded")
            self.inventory_registers[item_id] += delta
            return 0
        if phase == "delay":
            return 0

        item_id = int(observation["query_item"])
        return self.inventory_registers[item_id]


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
