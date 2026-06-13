"""Tests for experiment config loading and player instantiation."""

from __future__ import annotations

import pytest

from connect4_mcts.experiment_config import (
    ORACLE_TAG,
    instantiate_player,
    load_experiment_config,
    player_kind,
)
from connect4_mcts.players.minimax import MinimaxPlayer
from connect4_mcts.players.mcts import MCTSPlayer
from connect4_mcts.players.random import RandomPlayer


def test_load_experiment_config_global_fields(tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
seed = 42
output_dir = "results/test"
games_per_pair = 3
blunder_threshold = 0.25
blunder_sample_every = 2

[[players]]
id = "random"
type = "random"
""".strip(),
        encoding="utf-8",
    )
    config = load_experiment_config(config_path)
    assert config.seed == 42
    assert config.output_dir == "results/test"
    assert config.games_per_pair == 3
    assert config.blunder_threshold == 0.25
    assert config.blunder_sample_every == 2
    assert config.oracle_player() is None


def test_load_experiment_config_rejects_multiple_oracles(tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
seed = 0
output_dir = "out"
games_per_pair = 1

[[players]]
id = "a"
type = "uct"
tags = ["ORACLE"]

[[players]]
id = "b"
type = "uct"
tags = ["ORACLE"]
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="at most one player"):
        load_experiment_config(config_path)


def test_legacy_kind_mcts_algorithm_parsing(tmp_path) -> None:
    config_path = tmp_path / "legacy.toml"
    config_path.write_text(
        """
seed = 0
output_dir = "out"
games_per_pair = 1

[[players]]
id = "fpu-old"
kind = "mcts"
algorithm = "fpu"
play_iterations = 500
fpu = 0.8
""".strip(),
        encoding="utf-8",
    )
    spec = load_experiment_config(config_path).players[0]
    assert spec.type == "fpu"
    assert spec.params["play_iterations"] == 500


def test_instantiate_players(tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
seed = 0
output_dir = "out"
games_per_pair = 1

[[players]]
id = "random"
type = "random"

[[players]]
id = "mm"
type = "minimax"
depth = 3

[[players]]
id = "uct"
type = "uct"
iterations = 10
""".strip(),
        encoding="utf-8",
    )
    config = load_experiment_config(config_path)
    assert isinstance(instantiate_player(config.players[0]), RandomPlayer)
    assert isinstance(instantiate_player(config.players[1]), MinimaxPlayer)
    mcts = instantiate_player(config.players[2])
    assert isinstance(mcts, MCTSPlayer)
    assert mcts.iterations == 10


def test_player_kind_categories() -> None:
    from connect4_mcts.experiment_config import PlayerSpec

    assert player_kind(PlayerSpec("r", "random", frozenset(), {})) == "builtin"
    assert player_kind(PlayerSpec("u", "uct", frozenset(), {})) == "mcts"
    assert player_kind(PlayerSpec("l", "llm", frozenset(), {"model": "x"})) == "llm"
    assert ORACLE_TAG == "ORACLE"
