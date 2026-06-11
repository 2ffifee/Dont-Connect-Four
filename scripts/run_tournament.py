"""Round-robin tournament for agents defined in TOML configs.

Each pair plays ``games_per_pair`` games. Game ``i`` uses seed
``base_seed + offset`` where ``offset`` is unique per pairing and game index, so
there is no need for duplicate players with different evaluation seeds.

The script writes report-oriented outputs to ``--output-dir``:

* ``games.csv`` - one row per game;
* ``moves.jsonl`` - one JSON object per move, including the state before the
  move for later oracle/Blunder Rate scoring;
* ``pair_summary.csv`` - one row per matchup;
* ``standings.csv`` - aggregate standings per player;
* ``run_metadata.json`` - config and run metadata.

Usage::

    python scripts/run_tournament.py --games-per-pair 10 --base-seed 0
    python scripts/run_tournament.py --llm-config configs/tournament_llm.toml
    python scripts/run_tournament.py --no-llm
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import tomllib
import traceback
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from connect4_mcts.experiments import SimulatedGame, TimedMoveRecord, summarize_games
from connect4_mcts.game import GameState, GameStatus, IllegalMoveError, Move, Player
from connect4_mcts.players.base import Agent
from connect4_mcts.players.llm import create_llm_player
from connect4_mcts.players.minimax import MinimaxPlayer
from connect4_mcts.players.random import RandomPlayer
from connect4_mcts.runner import GameRunnerError, prepare_agents_for_game
from connect4_mcts.tournament_config import load_llm_tournament_entries
from connect4_mcts.tournament_reporting import (
    TournamentFailureContext,
    format_failure_banner,
    write_failure_report,
)
from connect4_mcts.training import estimate_tree_ram_gb, load_player


LLM_COUNTER_FIELDS = ("requests", "unparseable", "illegal", "fallbacks", "moves")


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


def _play_instrumented_game(
    *,
    game_id: str,
    pair_id: str,
    pair_index: int,
    game_index: int,
    seed: int,
    red_agent_name: str,
    yellow_agent_name: str,
    red: Agent,
    yellow: Agent,
    max_moves: int | None = None,
) -> tuple[SimulatedGame, list[dict[str, Any]]]:
    state = GameState.new(first_player=Player.RED)
    agents = {
        Player.RED: (red_agent_name, red),
        Player.YELLOW: (yellow_agent_name, yellow),
    }
    records: list[TimedMoveRecord] = []
    move_rows: list[dict[str, Any]] = []

    while state.status is not GameStatus.FINISHED:
        if max_moves is not None and len(records) >= max_moves:
            raise GameRunnerError(f"game exceeded max_moves={max_moves}")

        player = state.current_player
        agent_name, agent = agents[player]
        legal_moves = state.legal_moves()
        state_before = _serialize_state(state)
        move_number = state.move_count + 1

        started_at = time.perf_counter()
        try:
            move = agent.choose_move(state)
        except Exception as exc:
            exc.move_context = {  # type: ignore[attr-defined]
                "move_number": move_number,
                "player": player.value,
                "agent": agent_name,
                "legal_move_count": len(legal_moves),
            }
            raise
        elapsed = time.perf_counter() - started_at
        if not state.is_legal_move(move):
            raise IllegalMoveError(
                f"{player.value} agent {agent_name!r} returned illegal move {move!r} "
                f"on move #{move_number} (seed={seed}, {len(legal_moves)} legal moves)"
            )

        records.append(
            TimedMoveRecord(
                player=player,
                agent_name=agent_name,
                move=move,
                move_number=move_number,
                decision_time_seconds=elapsed,
            )
        )
        move_rows.append(
            {
                "game_id": game_id,
                "pair_id": pair_id,
                "pair_index": pair_index,
                "game_index": game_index,
                "seed": seed,
                "move_number": move_number,
                "player": player.value,
                "agent": agent_name,
                "move_type": move.move_type.value,
                "column": move.column,
                "decision_time_seconds": elapsed,
                "legal_move_count": len(legal_moves),
                "legal_moves": [_serialize_move(candidate) for candidate in legal_moves],
                "state_before": state_before,
            }
        )
        state = state.apply_move(move)

    return (
        SimulatedGame(
            red_agent=red_agent_name,
            yellow_agent=yellow_agent_name,
            final_state=state,
            moves=tuple(records),
            seed=seed,
        ),
        move_rows,
    )


def _failure_context_from_exc(
    exc: BaseException,
    *,
    pair_index: int,
    pair_total: int,
    pair_id: str,
    left_player: str,
    right_player: str,
    game_index: int,
    games_per_pair: int,
    game_id: str,
    seed: int,
    red_agent: str,
    yellow_agent: str,
) -> TournamentFailureContext:
    move_number: int | None = None
    current_player: str | None = None
    move_context = getattr(exc, "move_context", None)
    if isinstance(move_context, dict):
        raw_move_number = move_context.get("move_number")
        if isinstance(raw_move_number, int):
            move_number = raw_move_number
        player = move_context.get("player")
        if isinstance(player, str):
            current_player = player
    return TournamentFailureContext(
        pair_index=pair_index,
        pair_total=pair_total,
        pair_id=pair_id,
        left_player=left_player,
        right_player=right_player,
        game_index=game_index,
        games_per_pair=games_per_pair,
        game_id=game_id,
        seed=seed,
        red_agent=red_agent,
        yellow_agent=yellow_agent,
        move_number=move_number,
        current_player=current_player,
    )


def _report_and_raise_game_failure(
    exc: BaseException,
    *,
    output_dir: str,
    pair_index: int,
    pair_total: int,
    pair_id: str,
    left_player: str,
    right_player: str,
    game_index: int,
    games_per_pair: int,
    game_id: str,
    seed: int,
    red_agent: str,
    yellow_agent: str,
) -> None:
    context = _failure_context_from_exc(
        exc,
        pair_index=pair_index,
        pair_total=pair_total,
        pair_id=pair_id,
        left_player=left_player,
        right_player=right_player,
        game_index=game_index,
        games_per_pair=games_per_pair,
        game_id=game_id,
        seed=seed,
        red_agent=red_agent,
        yellow_agent=yellow_agent,
    )
    report_path = write_failure_report(output_dir, context=context, exc=exc)
    print(format_failure_banner(context, exc), file=sys.stderr, flush=True)
    print(traceback.format_exc(), file=sys.stderr, flush=True)
    print(f"Failure report written to {report_path}", file=sys.stderr, flush=True)
    raise exc


def _pair_seed(base_seed: int, left_index: int, right_index: int, game_index: int) -> int:
    pair_key = left_index * 1000 + right_index
    return base_seed + pair_key * 100 + game_index


def _read_toml(path: str) -> dict[str, Any]:
    with open(path, "rb") as config_file:
        return tomllib.load(config_file)


def _serialize_move(move: Move) -> dict[str, Any]:
    return {"move_type": move.move_type.value, "column": move.column}


def _serialize_state(state: GameState) -> dict[str, Any]:
    return {
        "board": [
            "".join("." if cell is None else cell.value[0].upper() for cell in row)
            for row in state.board
        ],
        "current_player": state.current_player.value,
        "first_player": state.first_player.value,
        "status": state.status.value,
        "move_count": state.move_count,
        "protected_segments": [
            [[row, column] for row, column in segment]
            for segment in sorted(state.protected_segments)
        ],
    }


def _agent_counters(agent: Agent) -> dict[str, int]:
    counters: dict[str, int] = {}
    for field in LLM_COUNTER_FIELDS:
        value = getattr(agent, field, 0)
        counters[field] = int(value) if isinstance(value, int) else 0
    return counters


def _counter_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {field: after.get(field, 0) - before.get(field, 0) for field in LLM_COUNTER_FIELDS}


def _tree_size(agent: Agent) -> int | None:
    value = getattr(agent, "tree_size", None)
    return value if isinstance(value, int) else None


def _game_metrics_row(
    game: SimulatedGame,
    *,
    game_id: str,
    pair_id: str,
    pair_index: int,
    game_index: int,
    seed: int,
    red_kind: str,
    yellow_kind: str,
    red_counters: dict[str, int],
    yellow_counters: dict[str, int],
    red_tree_size_before: int | None,
    red_tree_size_after: int | None,
    yellow_tree_size_before: int | None,
    yellow_tree_size_after: int | None,
) -> dict[str, Any]:
    result = game.final_state.result
    if result is None:
        raise ValueError("simulated game ended without result")

    red_moves = [move for move in game.moves if move.player is Player.RED]
    yellow_moves = [move for move in game.moves if move.player is Player.YELLOW]
    red_total = sum(move.decision_time_seconds for move in red_moves)
    yellow_total = sum(move.decision_time_seconds for move in yellow_moves)
    red_invalid = red_counters["unparseable"] + red_counters["illegal"]
    yellow_invalid = yellow_counters["unparseable"] + yellow_counters["illegal"]

    return {
        "game_id": game_id,
        "pair_id": pair_id,
        "pair_index": pair_index,
        "game_index": game_index,
        "seed": seed,
        "red_agent": game.red_agent,
        "yellow_agent": game.yellow_agent,
        "red_kind": red_kind,
        "yellow_kind": yellow_kind,
        "winner_agent": game.winner_agent or "",
        "winner_color": "" if result.winner is None else result.winner.value,
        "is_draw": result.winner is None,
        "red_lines": result.red_lines,
        "yellow_lines": result.yellow_lines,
        "moves": len(game.moves),
        "red_decisions": len(red_moves),
        "yellow_decisions": len(yellow_moves),
        "red_total_decision_seconds": red_total,
        "yellow_total_decision_seconds": yellow_total,
        "red_avg_decision_seconds": red_total / len(red_moves) if red_moves else 0.0,
        "yellow_avg_decision_seconds": yellow_total / len(yellow_moves) if yellow_moves else 0.0,
        "red_tree_size_before": "" if red_tree_size_before is None else red_tree_size_before,
        "red_tree_size_after": "" if red_tree_size_after is None else red_tree_size_after,
        "yellow_tree_size_before": "" if yellow_tree_size_before is None else yellow_tree_size_before,
        "yellow_tree_size_after": "" if yellow_tree_size_after is None else yellow_tree_size_after,
        "red_llm_requests": red_counters["requests"],
        "red_llm_unparseable": red_counters["unparseable"],
        "red_llm_illegal": red_counters["illegal"],
        "red_llm_fallbacks": red_counters["fallbacks"],
        "red_llm_moves": red_counters["moves"],
        "red_llm_invalid_response_rate": red_invalid / red_counters["requests"] if red_counters["requests"] else 0.0,
        "red_llm_fallback_rate": red_counters["fallbacks"] / red_counters["moves"] if red_counters["moves"] else 0.0,
        "yellow_llm_requests": yellow_counters["requests"],
        "yellow_llm_unparseable": yellow_counters["unparseable"],
        "yellow_llm_illegal": yellow_counters["illegal"],
        "yellow_llm_fallbacks": yellow_counters["fallbacks"],
        "yellow_llm_moves": yellow_counters["moves"],
        "yellow_llm_invalid_response_rate": (
            yellow_invalid / yellow_counters["requests"] if yellow_counters["requests"] else 0.0
        ),
        "yellow_llm_fallback_rate": (
            yellow_counters["fallbacks"] / yellow_counters["moves"] if yellow_counters["moves"] else 0.0
        ),
    }


def _pair_summary_row(left: str, right: str, games: tuple[SimulatedGame, ...], games_per_pair: int) -> dict[str, Any]:
    summary = summarize_games(games, agent_names=(left, right))
    left_wins = summary.wins[left]
    right_wins = summary.wins[right]
    return {
        "pair_id": f"{left}_vs_{right}",
        "left": left,
        "right": right,
        "games": games_per_pair,
        "left_wins": left_wins,
        "right_wins": right_wins,
        "draws": summary.draws,
        "left_win_rate": left_wins / games_per_pair,
        "right_win_rate": right_wins / games_per_pair,
        "draw_rate": summary.draws / games_per_pair,
        "avg_moves": summary.average_moves,
        "left_avg_decision_seconds": summary.average_decision_time(left),
        "right_avg_decision_seconds": summary.average_decision_time(right),
    }


def _standings_rows(games: tuple[SimulatedGame, ...], player_ids: Sequence[str]) -> list[dict[str, Any]]:
    overall = summarize_games(games, agent_names=tuple(player_ids))
    rows: list[dict[str, Any]] = []
    for player_id in player_ids:
        player_games = [
            game for game in games if game.red_agent == player_id or game.yellow_agent == player_id
        ]
        draws = sum(1 for game in player_games if game.winner_agent is None)
        wins = overall.wins.get(player_id, 0)
        total = len(player_games)
        losses = total - wins - draws
        rows.append(
            {
                "player_id": player_id,
                "games": total,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "win_rate": wins / total if total else 0.0,
                "loss_rate": losses / total if total else 0.0,
                "draw_rate": draws / total if total else 0.0,
                "moves": overall.move_counts.get(player_id, 0),
                "avg_decision_seconds": overall.average_decision_time(player_id),
            }
        )

    rows.sort(key=lambda row: (row["wins"], row["draws"], -row["avg_decision_seconds"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _write_csv(path: str, rows: Iterable[dict[str, Any]]) -> None:
    row_list = list(rows)
    if not row_list:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fieldnames = list(row_list[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row_list)


def _write_jsonl(path: str, rows: Iterable[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, separators=(",", ":")) + "\n")


def _write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)


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
    parser.add_argument("--max-moves", type=int, default=None, help="Optional safety cap per game.")
    parser.add_argument("--output", default="", help="Optional legacy JSON summary path.")
    parser.add_argument("--output-dir", default="results/tournament", help="Directory for tournament metric files.")
    parser.add_argument(
        "--verbose-games",
        action="store_true",
        help="Print one line before each individual game (useful when diagnosing hangs or crashes).",
    )
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
    if not entries:
        raise SystemExit("tournament config must define at least one player")

    print(f"Round-robin: {len(entries)} players, {args.games_per_pair} games/pair.")
    max_ram = max(entry.est_ram_gb for entry in entries)
    if max_ram > 0:
        print(f"Largest MCTS tree: est. RAM ~{max_ram:.2f} GB (two agents loaded per game).")

    all_games: list[SimulatedGame] = []
    all_game_rows: list[dict[str, Any]] = []
    all_move_rows: list[dict[str, Any]] = []
    pair_summaries: list[dict[str, Any]] = []

    total_pairs = len(entries) * (len(entries) - 1) // 2
    pair_index = 0
    try:
        for i, left in enumerate(entries):
            for j in range(i + 1, len(entries)):
                right = entries[j]
                pair_id = f"{left.player_id}_vs_{right.player_id}"
                print(
                    f"\nPair {pair_index + 1}/{total_pairs}: {left.player_id} vs {right.player_id}",
                    flush=True,
                )
                pair_cache: dict[str, Agent] = {}
                pair_games: list[SimulatedGame] = []
                for game_index in range(args.games_per_pair):
                    seed = _pair_seed(args.base_seed, i, j, game_index)
                    swap = game_index % 2 == 1
                    red_entry = right if swap else left
                    yellow_entry = left if swap else right
                    red = _instantiate_agent(red_entry, defaults, seed, agent_cache=pair_cache)
                    yellow = _instantiate_agent(yellow_entry, defaults, seed + 1, agent_cache=pair_cache)

                    red_counters_before = _agent_counters(red)
                    yellow_counters_before = _agent_counters(yellow)
                    red_tree_before = _tree_size(red)
                    yellow_tree_before = _tree_size(yellow)

                    prepare_agents_for_game(red, yellow)
                    game_id = f"{pair_id}_g{game_index:04d}"
                    if args.verbose_games:
                        print(
                            f"  game {game_index + 1}/{args.games_per_pair}: {game_id} "
                            f"seed={seed} red={red_entry.player_id} yellow={yellow_entry.player_id}",
                            flush=True,
                        )
                    try:
                        game, move_rows = _play_instrumented_game(
                            game_id=game_id,
                            pair_id=pair_id,
                            pair_index=pair_index,
                            game_index=game_index,
                            seed=seed,
                            red_agent_name=red_entry.player_id,
                            yellow_agent_name=yellow_entry.player_id,
                            red=red,
                            yellow=yellow,
                            max_moves=args.max_moves,
                        )
                    except Exception as exc:
                        _report_and_raise_game_failure(
                            exc,
                            output_dir=args.output_dir,
                            pair_index=pair_index,
                            pair_total=total_pairs,
                            pair_id=pair_id,
                            left_player=left.player_id,
                            right_player=right.player_id,
                            game_index=game_index,
                            games_per_pair=args.games_per_pair,
                            game_id=game_id,
                            seed=seed,
                            red_agent=red_entry.player_id,
                            yellow_agent=yellow_entry.player_id,
                        )

                    red_counters = _counter_delta(red_counters_before, _agent_counters(red))
                    yellow_counters = _counter_delta(yellow_counters_before, _agent_counters(yellow))
                    all_game_rows.append(
                        _game_metrics_row(
                            game,
                            game_id=game_id,
                            pair_id=pair_id,
                            pair_index=pair_index,
                            game_index=game_index,
                            seed=seed,
                            red_kind=red_entry.kind,
                            yellow_kind=yellow_entry.kind,
                            red_counters=red_counters,
                            yellow_counters=yellow_counters,
                            red_tree_size_before=red_tree_before,
                            red_tree_size_after=_tree_size(red),
                            yellow_tree_size_before=yellow_tree_before,
                            yellow_tree_size_after=_tree_size(yellow),
                        )
                    )
                    all_move_rows.extend(move_rows)
                    pair_games.append(game)
                    all_games.append(game)
                pair_cache.clear()

                pair_summary = _pair_summary_row(
                    left.player_id,
                    right.player_id,
                    tuple(pair_games),
                    args.games_per_pair,
                )
                pair_summaries.append(pair_summary)
                print(
                    f"  result {left.player_id:>16} vs {right.player_id:<16} | "
                    f"{pair_summary['left_wins']} - {pair_summary['right_wins']} "
                    f"(draws {pair_summary['draws']})",
                    flush=True,
                )
                pair_index += 1
    except Exception:
        print(
            f"\nTournament aborted after completing {pair_index}/{total_pairs} pairings.",
            file=sys.stderr,
            flush=True,
        )
        raise

    standings = _standings_rows(tuple(all_games), player_ids)

    print("\nStandings (wins across all pairings):")
    for row in standings:
        print(f"  {row['rank']:>2}. {row['player_id']:<16} {row['wins']} wins")

    metadata = {
        "config": args.config,
        "llm_config": None if llm_config is None else args.llm_config,
        "players": player_ids,
        "games_per_pair": args.games_per_pair,
        "base_seed": args.base_seed,
        "max_moves": args.max_moves,
        "output_dir": args.output_dir,
        "files": {
            "games": "games.csv",
            "moves": "moves.jsonl",
            "pair_summary": "pair_summary.csv",
            "standings": "standings.csv",
        },
    }
    _write_csv(os.path.join(args.output_dir, "games.csv"), all_game_rows)
    _write_jsonl(os.path.join(args.output_dir, "moves.jsonl"), all_move_rows)
    _write_csv(os.path.join(args.output_dir, "pair_summary.csv"), pair_summaries)
    _write_csv(os.path.join(args.output_dir, "standings.csv"), standings)
    _write_json(os.path.join(args.output_dir, "run_metadata.json"), metadata)
    print(f"\nSaved tournament metrics to {args.output_dir}")

    legacy_payload = {
        "players": player_ids,
        "games_per_pair": args.games_per_pair,
        "base_seed": args.base_seed,
        "llm_config": None if llm_config is None else args.llm_config,
        "standings": standings,
        "pairs": pair_summaries,
    }
    if args.output:
        _write_json(args.output, legacy_payload)
        print(f"Saved legacy summary to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
