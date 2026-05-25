"""Agent construction helpers shared by UI and experiments."""

from __future__ import annotations

from connect4_mcts.players.base import Agent
from connect4_mcts.players.minimax import MinimaxPlayer
from connect4_mcts.players.random import RandomPlayer


AgentName = str
AGENT_CHOICES = ("random", "minimax")


def create_agent(agent_name: AgentName, seed: int | None = None, depth: int = 3) -> Agent:
    if agent_name == "random":
        return RandomPlayer(seed=seed)
    if agent_name == "minimax":
        return MinimaxPlayer(depth=depth)
    raise ValueError(f"unknown agent: {agent_name}")


def format_agent_name(agent_name: AgentName) -> str:
    if agent_name == "random":
        return "Random"
    if agent_name == "minimax":
        return "Minimax"
    return agent_name
