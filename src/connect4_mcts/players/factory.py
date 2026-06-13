"""Agent construction helpers shared by UI and experiments."""

from __future__ import annotations

import math

from connect4_mcts.players.base import Agent
from connect4_mcts.players.minimax import MinimaxPlayer
from connect4_mcts.players.random import RandomPlayer


AgentName = str
# Built-in, fully offline agents shown in the GUI/CLI menus.
AGENT_CHOICES = ("random", "minimax", "uct", "fpu", "lgr", "pmbp")

DEFAULT_EXPLORATION = math.sqrt(2.0)
DEFAULT_FPU = 1.0
DEFAULT_POWER_MEAN_P = 2.0

_AGENT_LABELS = {
    "random": "Random",
    "minimax": "Minimax",
    "uct": "UCT",
    "fpu": "UCT+FPU",
    "lgr": "UCT+LGR",
    "pmbp": "UCT+PMBp",
    "llm": "LLM",
}


def create_agent(
    agent_name: AgentName,
    seed: int | None = None,
    depth: int = 3,
    iterations: int = 400,
    exploration: float = DEFAULT_EXPLORATION,
    fpu: float = DEFAULT_FPU,
    power_mean_p: float = DEFAULT_POWER_MEAN_P,
) -> Agent:
    if agent_name == "random":
        return RandomPlayer(seed=seed)
    if agent_name == "minimax":
        return MinimaxPlayer(depth=depth)
    if agent_name in {"uct", "fpu", "lgr", "pmbp"}:
        return _create_mcts_agent(
            agent_name,
            seed=seed,
            iterations=iterations,
            exploration=exploration,
            fpu=fpu,
            power_mean_p=power_mean_p,
        )
    if agent_name == "llm":
        return _create_llm_agent(seed=seed)
    raise ValueError(f"unknown agent: {agent_name}")


def _create_mcts_agent(
    agent_name: AgentName,
    seed: int | None,
    iterations: int,
    exploration: float,
    fpu: float,
    power_mean_p: float,
) -> Agent:
    # Imported lazily to avoid a circular import (training depends on runner,
    # which depends back on the players package).
    from connect4_mcts.training import train_fpu, train_lgr, train_pmbp, train_uct

    if agent_name == "uct":
        return train_uct(iterations=iterations, exploration=exploration, seed=seed)
    if agent_name == "fpu":
        return train_fpu(iterations=iterations, exploration=exploration, fpu=fpu, seed=seed)
    if agent_name == "lgr":
        return train_lgr(iterations=iterations, exploration=exploration, seed=seed)
    return train_pmbp(
        iterations=iterations,
        exploration=exploration,
        power_mean_p=power_mean_p,
        seed=seed,
    )


def _create_llm_agent(seed: int | None) -> Agent:
    # Imported lazily so the optional ``openai`` dependency is only required when
    # an LLM agent is actually requested. The model is taken from the
    # ``OPENAI_MODEL`` environment variable (default: gpt-4o-mini).
    import os

    from connect4_mcts.players.llm import LLMPlayer, OpenAIClient

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    return LLMPlayer(OpenAIClient(model=model), model_label=model, seed=seed)


def format_agent_name(agent_name: AgentName) -> str:
    return _AGENT_LABELS.get(agent_name, agent_name)
