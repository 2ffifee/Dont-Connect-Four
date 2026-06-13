"""Load experiment/tournament configs and build players from TOML definitions."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from connect4_mcts.players import create_agent
from connect4_mcts.players.base import Agent
from connect4_mcts.players.factory import DEFAULT_EXPLORATION, DEFAULT_FPU, DEFAULT_POWER_MEAN_P
from connect4_mcts.players.llm import create_llm_player


PLAYER_TYPES = frozenset({"random", "minimax", "uct", "fpu", "lgr", "pmbp", "llm"})
ORACLE_TAG = "ORACLE"


@dataclass(frozen=True, slots=True)
class PlayerSpec:
    id: str
    type: str
    tags: frozenset[str]
    params: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    seed: int
    output_dir: str
    games_per_pair: int
    blunder_threshold: float
    blunder_sample_every: int
    players: tuple[PlayerSpec, ...]
    source_path: str | None = None

    @property
    def player_ids(self) -> tuple[str, ...]:
        return tuple(player.id for player in self.players)

    def oracle_player(self) -> PlayerSpec | None:
        for player in self.players:
            if ORACLE_TAG in player.tags:
                return player
        return None


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("rb") as file:
        raw = tomllib.load(file)

    players = tuple(_parse_player(entry) for entry in raw.get("players", []))
    if not players:
        raise ValueError("experiment config must define at least one [[players]] entry")

    oracle_players = [player for player in players if ORACLE_TAG in player.tags]
    if len(oracle_players) > 1:
        ids = ", ".join(player.id for player in oracle_players)
        raise ValueError(f"at most one player may have tag {ORACLE_TAG!r}; found: {ids}")

    return ExperimentConfig(
        seed=int(raw.get("seed", 0)),
        output_dir=str(raw.get("output_dir", "results/tournament")),
        games_per_pair=int(raw.get("games_per_pair", 10)),
        blunder_threshold=float(raw.get("blunder_threshold", 0.3)),
        blunder_sample_every=int(raw.get("blunder_sample_every", 1)),
        players=players,
        source_path=str(config_path),
    )


def instantiate_player(spec: PlayerSpec, *, game_seed: int | None = None) -> Agent:
    params = spec.params
    player_type = spec.type

    if player_type == "random":
        seed = params.get("seed", game_seed)
        return create_agent("random", seed=seed)

    if player_type == "minimax":
        return create_agent("minimax", depth=int(params.get("depth", 4)))

    if player_type in {"uct", "fpu", "lgr", "pmbp"}:
        iterations = int(params.get("iterations", params.get("play_iterations", 1000)))
        return create_agent(
            player_type,
            seed=game_seed,
            iterations=iterations,
            exploration=float(params.get("exploration", DEFAULT_EXPLORATION)),
            fpu=float(params.get("fpu", DEFAULT_FPU)),
            power_mean_p=float(params.get("power_mean_p", DEFAULT_POWER_MEAN_P)),
        )

    if player_type == "llm":
        model = str(params["model"])
        base_url = str(params.get("base_url", "") or "") or None
        api_key = str(params.get("api_key", "") or "") or None
        temperature = params.get("temperature")
        max_tokens = params.get("max_tokens")
        return create_llm_player(
            model,
            api_key=api_key,
            base_url=base_url,
            seed=game_seed,
            timeout=float(params.get("timeout", 300.0)),
            temperature=None if temperature is None else float(temperature),
            max_tokens=None if max_tokens is None else int(max_tokens),
        )

    raise ValueError(f"unknown player type: {player_type!r}")


def player_kind(spec: PlayerSpec) -> str:
    """Return a coarse agent category used in tournament metrics."""
    if spec.type == "llm":
        return "llm"
    if spec.type == "minimax":
        return "builtin"
    if spec.type == "random":
        return "builtin"
    return "mcts"


def _parse_player(entry: dict[str, Any]) -> PlayerSpec:
    player_id = str(entry["id"])
    player_type = str(entry.get("type") or entry.get("kind", "")).lower()
    if not player_type:
        raise ValueError(f"player {player_id!r} must define type")

    if player_type == "builtin":
        player_type = str(entry.get("builtin", "random")).lower()
    elif player_type == "mcts":
        player_type = str(entry.get("algorithm", "uct")).lower()

    if player_type not in PLAYER_TYPES:
        allowed = ", ".join(sorted(PLAYER_TYPES))
        raise ValueError(f"player {player_id!r} has unsupported type {player_type!r}; expected one of: {allowed}")

    reserved = {"id", "type", "kind", "builtin", "algorithm", "tags"}
    params = {key: value for key, value in entry.items() if key not in reserved}
    if player_type == "minimax" and "depth" not in params:
        params["depth"] = 4

    return PlayerSpec(
        id=player_id,
        type=player_type,
        tags=_parse_tags(entry.get("tags")),
        params=params,
    )


def _parse_tags(raw: object) -> frozenset[str]:
    if raw is None:
        return frozenset()
    if isinstance(raw, str):
        return frozenset(part.strip().upper() for part in raw.split(",") if part.strip())
    if isinstance(raw, list):
        return frozenset(str(part).strip().upper() for part in raw if str(part).strip())
    raise ValueError("tags must be a string or list of strings")
