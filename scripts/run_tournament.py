"""Round-robin tournament for agents defined in ``configs/tournament_grid.toml``.

Each pair plays ``games_per_pair`` games. Game ``i`` uses seed ``base_seed + offset``
where ``offset`` is unique per pairing and game index, so there is no need for
duplicate players with different training seeds.

LLM contestants are listed in ``configs/tournament_llm.toml`` (or another file
passed via ``--llm-config``). They join the same round-robin as MCTS/builtin
agents.

Usage::

    python scripts/run_tournament.py --games-per-pair 10 --base-seed 0
    python scripts/run_tournament.py --llm-config configs/tournament_llm.toml
    python scripts/run_tournament.py --no-llm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from connect4_mcts.experiments import play_timed_game, summarize_games
from connect4_mcts.game import GameState, Player
from connect4_mcts.players.base import Agent
from connect4_mcts.players.llm import create_llm_player
from connect4_mcts.players.minimax import MinimaxPlayer
from connect4_mcts.players.random import RandomPlayer
from connect4_mcts.runner import prepare_agents_for_game
from connect4_mcts.tournament_config import load_llm_tournament_entries
from connect4_mcts.training import estimate_tree_ram_gb, load_player


@dataclass(frozen=True, slots=True)
class TournamentEntry:
    player_id: str
    kind: str
    entry: dict[str, Any]
    est_ram_gb: float


def _load_entries(config: dict[str, Any], llm_config: dict[str, Any] | None) -> list[TournamentEntry]:
    defaults = config.get("defaults", {})
    output_dir = str(defaults.get("output_dir", "models/tournament"))
    bytes_per_node = float(defaults.get("bytes_per_node", 2867.0))
    loaded: list[TournamentEntry] = []

    raw_entries = list(config.get("players", []))
    if llm_config is not None:
        raw_entries.extend(load_llm_tournament_entries(llm_config))

    for entry in raw_entries:
        player_id = str(entry["id"])
        kind = str(entry.get("kind", "mcts"))
        est_ram = 0.0

        if kind == "mcts":
            path = os.path.join(output_dir, f"{player_id}.pkl")
            if not os.path.exists(path):
                raise FileNotFoundError(f"missing trained player pickle: {path}")
            agent = load_player(path)
            est_ram = estimate_tree_ram_gb(agent.tree_size, bytes_per_node=bytes_per_node)
            del agent

        loaded.append(TournamentEntry(player_id=player_id, kind=kind, entry=entry, est_ram_gb=est_ram))

    return loaded


def _instantiate_agent(
    entry: TournamentEntry,
    defaults: dict[str, Any],
    game_seed: int,
    *,
    agent_cache: dict[str, Agent],
) -> Agent:
    cached = agent_cache.get(entry.player_id)
    if cached is not None:
        return cached

    if entry.kind == "builtin":
        builtin = str(entry.entry.get("builtin", "random"))
        if builtin == "random":
            agent: Agent = RandomPlayer(seed=game_seed)
        elif builtin == "minimax":
            agent = MinimaxPlayer(depth=int(entry.entry.get("depth", 4)))
        else:
            raise ValueError(f"unknown builtin agent: {builtin}")
    elif entry.kind == "mcts":
        output_dir = str(defaults.get("output_dir", "models/tournament"))
        agent = load_player(os.path.join(output_dir, f"{entry.player_id}.pkl"))
    elif entry.kind == "llm":
        model_entry = entry.entry
        base_url = str(model_entry.get("base_url", "") or "") or None
        api_key = str(model_entry.get("api_key", "") or "") or None
        agent = create_llm_player(
            str(model_entry["model"]),
            api_key=api_key,
            base_url=base_url,
            seed=game_seed,
            timeout=float(model_entry.get("timeout", 300.0)),
        )
    else:
        raise ValueError(f"unknown player kind: {entry.kind}")

    if entry.kind in {"mcts", "llm"}:
        agent_cache[entry.player_id] = agent
    return agent


def _pair_seed(base_seed: int, left_index: int, right_index: int, game_index: int) -> int:
    pair_key = left_index * 1000 + right_index
    return base_seed + pair_key * 100 + game_index


def _read_toml(path: str) -> dict[str, Any]:
    with open(path, "rb") as config_file:
        return tomllib.load(config_file)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a round-robin tournament.")
    parser.add_argument("--config", default="configs/tournament_grid.toml")
    parser.add_argument(
        "--llm-config",
        default="configs/tournament_llm.toml",
        help="TOML file with LLM server settings and model list (empty file section = no LLMs).",
    )
    parser.add_argument("--no-llm", action="store_true", help="Do not load LLM contestants.")
    parser.add_argument("--games-per-pair", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--output", default="", help="Optional JSON summary path.")
    args = parser.parse_args(argv)

    if args.games_per_pair < 1:
        raise SystemExit("--games-per-pair must be at least 1")

    config = _read_toml(args.config)

    llm_config: dict[str, Any] | None = None
    if not args.no_llm and os.path.exists(args.llm_config):
        llm_config = _read_toml(args.llm_config)
        llm_count = len(llm_config.get("models", []))
        if llm_count:
            server = llm_config.get("server", {})
            print(
                f"LLM config: {args.llm_config} ({llm_count} model(s), "
                f"base_url={server.get('base_url', '')!r})",
                flush=True,
            )
        else:
            llm_config = None
    elif not args.no_llm:
        print(f"No LLM config at {args.llm_config}; running without LLM players.", flush=True)

    defaults = config.get("defaults", {})
    entries = _load_entries(config, llm_config)
    player_ids = [entry.player_id for entry in entries]

    print(f"Round-robin: {len(entries)} players, {args.games_per_pair} games/pair.")
    max_ram = max(entry.est_ram_gb for entry in entries)
    if max_ram > 0:
        print(f"Largest MCTS tree: est. RAM ~{max_ram:.2f} GB (two agents loaded per game).")

    all_games = []
    pair_summaries = []

    for i, left in enumerate(entries):
        for j in range(i + 1, len(entries)):
            right = entries[j]
            pair_cache: dict[str, Agent] = {}
            pair_games = []
            for game_index in range(args.games_per_pair):
                seed = _pair_seed(args.base_seed, i, j, game_index)
                swap = game_index % 2 == 1
                red_entry = right if swap else left
                yellow_entry = left if swap else right
                red = _instantiate_agent(red_entry, defaults, seed, agent_cache=pair_cache)
                yellow = _instantiate_agent(yellow_entry, defaults, seed + 1, agent_cache=pair_cache)
                prepare_agents_for_game(red, yellow)
                game = play_timed_game(
                    red_agent_name=red_entry.player_id,
                    yellow_agent_name=yellow_entry.player_id,
                    red=red,
                    yellow=yellow,
                    initial_state=GameState.new(first_player=Player.RED),
                    seed=seed,
                )
                pair_games.append(game)
                all_games.append(game)
            pair_cache.clear()

            summary = summarize_games(tuple(pair_games), agent_names=(left.player_id, right.player_id))
            pair_summaries.append(
                {
                    "left": left.player_id,
                    "right": right.player_id,
                    "games": args.games_per_pair,
                    "wins": summary.wins,
                    "draws": summary.draws,
                }
            )
            print(
                f"{left.player_id:>16} vs {right.player_id:<16} | "
                f"{summary.wins[left.player_id]} - {summary.wins[right.player_id]} "
                f"(draws {summary.draws})",
                flush=True,
            )

    overall = summarize_games(tuple(all_games), agent_names=tuple(player_ids))
    standings = sorted(
        player_ids,
        key=lambda pid: (overall.wins.get(pid, 0), -overall.move_counts.get(pid, 0)),
        reverse=True,
    )

    print("\nStandings (wins across all pairings):")
    for rank, player_id in enumerate(standings, start=1):
        wins = overall.wins.get(player_id, 0)
        print(f"  {rank:>2}. {player_id:<16} {wins} wins")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        payload = {
            "players": player_ids,
            "games_per_pair": args.games_per_pair,
            "base_seed": args.base_seed,
            "llm_config": None if llm_config is None else args.llm_config,
            "standings": [
                {"rank": rank, "player_id": player_id, "wins": overall.wins.get(player_id, 0)}
                for rank, player_id in enumerate(standings, start=1)
            ],
            "pairs": pair_summaries,
        }
        with open(args.output, "w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, indent=2)
        print(f"\nSaved summary to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
