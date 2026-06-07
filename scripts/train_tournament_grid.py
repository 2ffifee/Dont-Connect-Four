"""Train all MCTS agents listed in ``configs/tournament_grid.toml``.

Each agent grows its persistent tree by self-play until an in-RAM node cap derived
from ``memory_gb`` (7 GB for tournament agents, 16 GB for the oracle entry).
Untrained agents use ``memory_gb = 0`` and are saved with an empty tree.

Usage::

    python scripts/train_tournament_grid.py
    python scripts/train_tournament_grid.py --only oracle,fpu-1.0-m7
    python scripts/train_tournament_grid.py --resume
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tomllib
from collections.abc import Sequence
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from connect4_mcts.players.mcts import MCTSPlayer
from connect4_mcts.training import (
    estimate_tree_ram_gb,
    grow_player_to_memory_cap,
    load_player,
    max_nodes_from_memory_gb,
    save_player,
)


def _optional_float(value: object) -> float | None:
    if value is None or (isinstance(value, str) and value.strip().lower() in {"none", ""}):
        return None
    return float(value)


def _player_get(entry: dict[str, Any], defaults: dict[str, Any], key: str, fallback: Any = None) -> Any:
    if key in entry:
        return entry[key]
    if key in defaults:
        return defaults[key]
    return fallback


def _build_mcts_player(entry: dict[str, Any], defaults: dict[str, Any]) -> MCTSPlayer:
    algorithm = str(_player_get(entry, defaults, "algorithm", "uct"))
    rollout_policy = str(_player_get(entry, defaults, "rollout_policy", "random"))
    if algorithm == "lgr":
        rollout_policy = "lgr"

    fpu = _optional_float(_player_get(entry, defaults, "fpu", "none"))
    return MCTSPlayer(
        iterations=int(_player_get(entry, defaults, "build_iterations", 1000)),
        exploration=float(_player_get(entry, defaults, "exploration", math.sqrt(2.0))),
        power_mean_p=float(_player_get(entry, defaults, "power_mean_p", 2.0 if algorithm == "pmbp" else 1.0)),
        fpu=fpu,
        rollout_policy=rollout_policy,
        final_move=str(_player_get(entry, defaults, "final_move", "robust")),
        max_rollout_moves=None,
        seed=int(_player_get(entry, defaults, "seed", 0)),
    )


def _format_bytes(num_bytes: int) -> str:
    if num_bytes >= 1024**3:
        return f"{num_bytes / 1024**3:.2f} GB"
    if num_bytes >= 1024**2:
        return f"{num_bytes / 1024**2:.1f} MB"
    return f"{num_bytes / 1024:.1f} KB"


def _train_one(
    entry: dict[str, Any],
    defaults: dict[str, Any],
    *,
    resume: bool,
) -> None:
    player_id = str(entry["id"])
    output_dir = str(defaults.get("output_dir", "models/tournament"))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{player_id}.pkl")

    memory_gb = float(_player_get(entry, defaults, "memory_gb", 7.0))
    bytes_per_node = float(defaults.get("bytes_per_node", 2867.0))
    safety_fraction = float(defaults.get("safety_fraction", 0.85))
    max_nodes = max_nodes_from_memory_gb(
        memory_gb,
        bytes_per_node=bytes_per_node,
        safety_fraction=safety_fraction,
    )
    play_iterations = int(_player_get(entry, defaults, "play_iterations", 1000))
    build_iterations = int(_player_get(entry, defaults, "build_iterations", 1000))
    temperature = float(defaults.get("selfplay_temperature", 1.0))
    max_games = int(defaults.get("max_selfplay_games", 100_000))
    max_minutes = float(defaults.get("max_minutes", 480.0))
    checkpoint_every = int(defaults.get("checkpoint_every_games", 25))

    if resume and os.path.exists(output_path):
        player = load_player(output_path)
        player.iterations = build_iterations
        print(f"[{player_id}] resume from {output_path} ({player.tree_size:,} states)", flush=True)
    else:
        player = _build_mcts_player(entry, defaults)
        if max_nodes <= 0:
            print(f"[{player_id}] untrained (memory_gb=0)", flush=True)
        else:
            cap_gb = estimate_tree_ram_gb(max_nodes, bytes_per_node=bytes_per_node)
            print(
                f"[{player_id}] training to ~{max_nodes:,} states (est. RAM cap ~{cap_gb:.2f} GB)",
                flush=True,
            )

    def on_progress(games: int, tree_size: int, elapsed_minutes: float) -> None:
        mem_gb = estimate_tree_ram_gb(tree_size, bytes_per_node=bytes_per_node)
        print(
            f"  [{player_id}] games {games:>6} | states {tree_size:>12,} "
            f"(est. RAM ~{mem_gb:.2f} GB) | {elapsed_minutes:6.1f} min",
            flush=True,
        )
        if checkpoint_every > 0 and games % checkpoint_every == 0:
            player.iterations = play_iterations
            save_player(player, output_path)
            player.iterations = build_iterations
            print(f"  [{player_id}] checkpoint {_format_bytes(os.path.getsize(output_path))}", flush=True)

    if max_nodes > 0 and player.tree_size < max_nodes:
        grow_player_to_memory_cap(
            player,
            max_nodes=max_nodes,
            temperature=temperature,
            max_games=max_games,
            max_minutes=max_minutes,
            on_progress=on_progress,
        )

    player.iterations = play_iterations
    save_player(player, output_path)
    final_gb = estimate_tree_ram_gb(player.tree_size, bytes_per_node=bytes_per_node)
    print(
        f"[{player_id}] done: {player.tree_size:,} states, est. RAM ~{final_gb:.2f} GB, "
        f"pickle {_format_bytes(os.path.getsize(output_path))}, play budget {play_iterations}",
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train tournament-grid MCTS agents.")
    parser.add_argument("--config", default="configs/tournament_grid.toml")
    parser.add_argument("--only", default="", help="Comma-separated player ids to train.")
    parser.add_argument("--resume", action="store_true", help="Continue from existing pickles.")
    args = parser.parse_args(argv)

    with open(args.config, "rb") as config_file:
        config = tomllib.load(config_file)

    defaults = config.get("defaults", {})
    only = {part.strip() for part in args.only.split(",") if part.strip()}
    entries = config.get("players", [])

    mcts_entries = [entry for entry in entries if entry.get("kind") == "mcts"]
    if only:
        mcts_entries = [entry for entry in mcts_entries if entry["id"] in only]

    builtin_count = sum(1 for entry in entries if entry.get("kind") == "builtin")
    print(
        f"Tournament grid: {len(entries)} players ({builtin_count} built-in at runtime, "
        f"{len(mcts_entries)} MCTS to train).",
        flush=True,
    )

    for entry in mcts_entries:
        _train_one(entry, defaults, resume=args.resume)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
