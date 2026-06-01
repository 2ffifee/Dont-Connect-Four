"""Agent construction helpers shared by UI and experiments."""

from __future__ import annotations

from connect4_mcts.players.base import Agent
from connect4_mcts.players.minimax import MinimaxPlayer
from connect4_mcts.players.random import RandomPlayer


AgentName = str
AGENT_CHOICES = ("random", "minimax", "uct", "fpu", "lgr", "pmbp")

_AGENT_LABELS = {
    "random": "Random",
    "minimax": "Minimax",
    "uct": "UCT",
    "fpu": "UCT+FPU",
    "lgr": "UCT+LGR",
    "pmbp": "UCT+PMBp",
    "loaded": "Loaded",
}


def create_agent(
    agent_name: AgentName,
    seed: int | None = None,
    depth: int = 3,
    iterations: int = 400,
) -> Agent:
    if agent_name == "random":
        return RandomPlayer(seed=seed)
    if agent_name == "minimax":
        return MinimaxPlayer(depth=depth)
    if agent_name in {"uct", "fpu", "lgr", "pmbp"}:
        return _create_mcts_agent(agent_name, seed=seed, iterations=iterations)
    raise ValueError(f"unknown agent: {agent_name}")


def _create_mcts_agent(agent_name: AgentName, seed: int | None, iterations: int) -> Agent:
    # Imported lazily to avoid a circular import (training depends on runner,
    # which depends back on the players package).
    from connect4_mcts.training import train_fpu, train_lgr, train_pmbp, train_uct

    if agent_name == "uct":
        return train_uct(iterations=iterations, seed=seed)
    if agent_name == "fpu":
        return train_fpu(iterations=iterations, seed=seed)
    if agent_name == "lgr":
        return train_lgr(iterations=iterations, seed=seed)
    return train_pmbp(iterations=iterations, seed=seed)


def format_agent_name(agent_name: AgentName) -> str:
    return _AGENT_LABELS.get(agent_name, agent_name)
