"""Backward-compatible helpers; prefer :mod:`connect4_mcts.experiment_config`."""

from __future__ import annotations

from connect4_mcts.experiment_config import (
    ExperimentConfig,
    PlayerSpec,
    instantiate_player,
    load_experiment_config,
    player_kind,
)

__all__ = [
    "ExperimentConfig",
    "PlayerSpec",
    "instantiate_player",
    "load_experiment_config",
    "player_kind",
]
