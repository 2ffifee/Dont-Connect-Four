"""Round-robin tournament for agents defined in a TOML experiment config.

Each pair plays ``games_per_pair`` games. Game ``i`` uses seed
``base_seed + offset`` where ``offset`` is unique per pairing and game index.

Outputs under ``output_dir`` (from config, overridable via CLI):

* ``games.csv`` - one row per game;
* ``moves.jsonl`` - one JSON object per move (state before move for blunder scoring);
* ``pair_summary.csv`` - one row per matchup;
* ``standings.csv`` - aggregate standings per player;
* ``run_metadata.json`` - config and run metadata.

Usage::

    python scripts/run_tournament.py --config configs/experiments/main_final.toml
    python scripts/run_tournament.py --config configs/experiments/smoke.toml --games-per-pair 2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from connect4_mcts.experiment_config import (
    ORACLE_TAG,
    ExperimentConfig,
    PlayerSpec,
    instantiate_player,
    load_experiment_config,
    player_kind,
)
from connect4_mcts.experiments import SimulatedGame, TimedMoveRecord, summarize_games
from connect4_mcts.game import GameState, GameStatus, IllegalMoveError, Move, Player
from connect4_mcts.players.base import Agent
from connect4_mcts.runner import GameRunnerError, prepare_agents_for_game
from connect4_mcts.tournament_progress import (
    build_checkpoint_payload,
    can_resume_tournament,
    flush_tournament_progress,
    load_checkpoint,
    load_saved_tournament_rows,
    tournament_is_complete,
    validate_checkpoint,
    write_csv as _progress_write_csv,
    write_json as _progress_write_json,
)
from connect4_mcts.tournament_reporting import (
    TournamentFailureContext,
    format_failure_banner,
    write_failure_report,
)


LLM_COUNTER_FIELDS = ("requests", "unparseable", "illegal", "fallbacks", "moves")
CACHEABLE_TYPES = frozenset({"uct", "fpu", "lgr", "pmbp", "llm"})


@dataclass(frozen=True, slots=True)
class TournamentEntry:
    spec: PlayerSpec


def _tournament_players(config: ExperimentConfig) -> tuple[PlayerSpec, ...]:
    """Return players that participate in the round-robin (excludes ORACLE-only evaluators)."""
    return tuple(player for player in config.players if ORACLE_TAG not in player.tags)


def _load_entries(config: ExperimentConfig) -> list[TournamentEntry]:
    players = _tournament_players(config)
    if not players:
        raise ValueError("experiment config must define at least one non-ORACLE player for the tournament")
    return [TournamentEntry(spec=player) for player in players]


def _instantiate_agent(
    entry: TournamentEntry,
    game_seed: int,
    *,
    agent_cache: dict[str, Agent],
) -> Agent:
    cached = agent_cache.get(entry.spec.id)
    if cached is not None:
        return cached

    agent = instantiate_player(entry.spec, game_seed=game_seed)
    if entry.spec.type in CACHEABLE_TYPES:
        agent_cache[entry.spec.id] = agent
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
        "red_line_total": state.red_line_total,
        "yellow_line_total": state.yellow_line_total,
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


def _format_game_result(
    game: SimulatedGame,
    *,
    game_id: str,
    game_index: int,
    games_per_pair: int,
) -> str:
    result = game.final_state.result
    if result is None:
        raise ValueError("simulated game ended without result")

    move_count = len(game.moves)
    lines = f"lines red={result.red_lines} yellow={result.yellow_lines}"

    if game.winner_agent is None:
        outcome = f"draw | {game.red_agent} (red) vs {game.yellow_agent} (yellow)"
    elif game.winner_agent == game.red_agent:
        outcome = f"{game.red_agent} (red) beat {game.yellow_agent} (yellow)"
    else:
        outcome = f"{game.yellow_agent} (yellow) beat {game.red_agent} (red)"

    seed_note = "" if game.seed is None else f" | seed={game.seed}"
    return (
        f"  game {game_index + 1}/{games_per_pair} finished: {game_id} | "
        f"{outcome} | {move_count} moves | {lines}{seed_note}"
    )


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


def _standings_rows_from_metrics(
    game_rows: Sequence[dict[str, Any]],
    player_ids: Sequence[str],
) -> list[dict[str, Any]]:
    wins = {player_id: 0 for player_id in player_ids}
    move_counts = {player_id: 0 for player_id in player_ids}
    decision_time_seconds = {player_id: 0.0 for player_id in player_ids}

    for row in game_rows:
        winner = str(row.get("winner_agent", "") or "")
        if winner:
            if winner in wins:
                wins[winner] += 1

        red_agent = str(row["red_agent"])
        yellow_agent = str(row["yellow_agent"])
        red_decisions = int(row.get("red_decisions", 0) or 0)
        yellow_decisions = int(row.get("yellow_decisions", 0) or 0)
        if red_agent in move_counts:
            move_counts[red_agent] += red_decisions
            decision_time_seconds[red_agent] += float(row.get("red_total_decision_seconds", 0.0) or 0.0)
        if yellow_agent in move_counts:
            move_counts[yellow_agent] += yellow_decisions
            decision_time_seconds[yellow_agent] += float(row.get("yellow_total_decision_seconds", 0.0) or 0.0)

    rows: list[dict[str, Any]] = []
    for player_id in player_ids:
        player_games = [
            row
            for row in game_rows
            if row["red_agent"] == player_id or row["yellow_agent"] == player_id
        ]
        draws = sum(1 for row in player_games if not str(row.get("winner_agent", "") or ""))
        total = len(player_games)
        player_wins = wins.get(player_id, 0)
        losses = total - player_wins - draws
        moves = move_counts.get(player_id, 0)
        avg_decision = decision_time_seconds.get(player_id, 0.0) / moves if moves else 0.0
        rows.append(
            {
                "player_id": player_id,
                "games": total,
                "wins": player_wins,
                "losses": losses,
                "draws": draws,
                "win_rate": player_wins / total if total else 0.0,
                "loss_rate": losses / total if total else 0.0,
                "draw_rate": draws / total if total else 0.0,
                "moves": moves,
                "avg_decision_seconds": avg_decision,
            }
        )

    rows.sort(key=lambda row: (row["wins"], row["draws"], -row["avg_decision_seconds"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


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
    parser.add_argument("--config", default="configs/experiments/main_final.toml")
    parser.add_argument("--games-per-pair", type=int, default=None, help="Override games_per_pair from config.")
    parser.add_argument("--base-seed", type=int, default=None, help="Override seed from config.")
    parser.add_argument("--max-moves", type=int, default=None, help="Optional safety cap per game.")
    parser.add_argument("--output", default="", help="Optional legacy JSON summary path.")
    parser.add_argument("--output-dir", default=None, help="Override output_dir from config.")
    parser.add_argument(
        "--verbose-games",
        action="store_true",
        help="Print one line before each individual game.",
    )
    parser.add_argument(
        "--game-verbose",
        action="store_true",
        help="Print the result of each game when it finishes.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from tournament_checkpoint.json and saved metric files in output dir.",
    )
    args = parser.parse_args(argv)

    experiment = load_experiment_config(args.config)
    games_per_pair = args.games_per_pair if args.games_per_pair is not None else experiment.games_per_pair
    base_seed = args.base_seed if args.base_seed is not None else experiment.seed
    output_dir = args.output_dir if args.output_dir is not None else experiment.output_dir

    if games_per_pair < 1:
        raise SystemExit("games_per_pair must be at least 1")

    entries = _load_entries(experiment)
    player_ids = [entry.spec.id for entry in entries]
    oracle = experiment.oracle_player()
    if oracle is not None:
        print(f"Oracle player {oracle.id!r} (tag {ORACLE_TAG}) is excluded from round-robin; used for blunder scoring.")

    print(f"Round-robin: {len(entries)} players, {games_per_pair} games/pair (online search).")

    total_pairs = len(entries) * (len(entries) - 1) // 2
    pair_ids: list[str] = []
    for i, left in enumerate(entries):
        for j in range(i + 1, len(entries)):
            pair_ids.append(f"{left.spec.id}_vs_{entries[j].spec.id}")

    if args.resume and tournament_is_complete(output_dir):
        print(f"Tournament already complete in {output_dir}; nothing to do.")
        return 0

    all_game_rows: list[dict[str, Any]] = []
    all_move_rows: list[dict[str, Any]] = []
    pair_summaries: list[dict[str, Any]] = []
    start_pair_index = 0

    if args.resume and can_resume_tournament(output_dir):
        checkpoint = load_checkpoint(output_dir)
        if checkpoint is None:
            raise SystemExit("resume requested but tournament_checkpoint.json is missing")
        try:
            validate_checkpoint(
                checkpoint,
                config_path=args.config,
                base_seed=base_seed,
                games_per_pair=games_per_pair,
                player_ids=player_ids,
            )
        except ValueError as exc:
            raise SystemExit(f"cannot resume tournament: {exc}") from exc
        saved_games, saved_moves, saved_pairs = load_saved_tournament_rows(output_dir)
        all_game_rows = [dict(row) for row in saved_games]
        all_move_rows = saved_moves
        pair_summaries = [dict(row) for row in saved_pairs]
        start_pair_index = int(checkpoint.get("completed_pairs", 0))
        print(
            f"Resuming tournament from pair {start_pair_index + 1}/{total_pairs} "
            f"({len(all_game_rows)} games saved).",
            flush=True,
        )
    elif args.resume:
        print("No in-progress tournament checkpoint found; starting from scratch.", flush=True)

    os.makedirs(output_dir, exist_ok=True)
    pair_index = 0
    try:
        for i, left in enumerate(entries):
            for j in range(i + 1, len(entries)):
                right = entries[j]
                pair_id = f"{left.spec.id}_vs_{right.spec.id}"
                if pair_index < start_pair_index:
                    pair_index += 1
                    continue

                print(
                    f"\nPair {pair_index + 1}/{total_pairs}: {left.spec.id} vs {right.spec.id}",
                    flush=True,
                )
                pair_cache: dict[str, Agent] = {}
                pair_games: list[SimulatedGame] = []
                for game_index in range(games_per_pair):
                    seed = _pair_seed(base_seed, i, j, game_index)
                    swap = game_index % 2 == 1
                    red_entry = right if swap else left
                    yellow_entry = left if swap else right
                    red = _instantiate_agent(red_entry, seed, agent_cache=pair_cache)
                    yellow = _instantiate_agent(yellow_entry, seed + 1, agent_cache=pair_cache)

                    red_counters_before = _agent_counters(red)
                    yellow_counters_before = _agent_counters(yellow)
                    red_tree_before = _tree_size(red)
                    yellow_tree_before = _tree_size(yellow)

                    prepare_agents_for_game(red, yellow)
                    game_id = f"{pair_id}_g{game_index:04d}"
                    if args.verbose_games:
                        print(
                            f"  game {game_index + 1}/{games_per_pair}: {game_id} "
                            f"seed={seed} red={red_entry.spec.id} yellow={yellow_entry.spec.id}",
                            flush=True,
                        )
                    try:
                        game, move_rows = _play_instrumented_game(
                            game_id=game_id,
                            pair_id=pair_id,
                            pair_index=pair_index,
                            game_index=game_index,
                            seed=seed,
                            red_agent_name=red_entry.spec.id,
                            yellow_agent_name=yellow_entry.spec.id,
                            red=red,
                            yellow=yellow,
                            max_moves=args.max_moves,
                        )
                    except Exception as exc:
                        _report_and_raise_game_failure(
                            exc,
                            output_dir=output_dir,
                            pair_index=pair_index,
                            pair_total=total_pairs,
                            pair_id=pair_id,
                            left_player=left.spec.id,
                            right_player=right.spec.id,
                            game_index=game_index,
                            games_per_pair=games_per_pair,
                            game_id=game_id,
                            seed=seed,
                            red_agent=red_entry.spec.id,
                            yellow_agent=yellow_entry.spec.id,
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
                            red_kind=player_kind(red_entry.spec),
                            yellow_kind=player_kind(yellow_entry.spec),
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
                    if args.game_verbose:
                        print(
                            _format_game_result(
                                game,
                                game_id=game_id,
                                game_index=game_index,
                                games_per_pair=games_per_pair,
                            ),
                            flush=True,
                        )
                pair_cache.clear()

                pair_summary = _pair_summary_row(
                    left.spec.id,
                    right.spec.id,
                    tuple(pair_games),
                    games_per_pair,
                )
                pair_summaries.append(pair_summary)
                print(
                    f"  result {left.spec.id:>16} vs {right.spec.id:<16} | "
                    f"{pair_summary['left_wins']} - {pair_summary['right_wins']} "
                    f"(draws {pair_summary['draws']})",
                    flush=True,
                )
                pair_index += 1
                flush_tournament_progress(
                    output_dir,
                    game_rows=all_game_rows,
                    move_rows=all_move_rows,
                    pair_summaries=pair_summaries,
                    checkpoint=build_checkpoint_payload(
                        config_path=args.config,
                        base_seed=base_seed,
                        games_per_pair=games_per_pair,
                        player_ids=player_ids,
                        completed_pairs=pair_index,
                        total_pairs=total_pairs,
                        pair_ids=pair_ids,
                        status="in_progress",
                    ),
                )
    except Exception:
        print(
            f"\nTournament aborted after completing {pair_index}/{total_pairs} pairings.",
            file=sys.stderr,
            flush=True,
        )
        if pair_index > start_pair_index:
            flush_tournament_progress(
                output_dir,
                game_rows=all_game_rows,
                move_rows=all_move_rows,
                pair_summaries=pair_summaries,
                checkpoint=build_checkpoint_payload(
                    config_path=args.config,
                    base_seed=base_seed,
                    games_per_pair=games_per_pair,
                    player_ids=player_ids,
                    completed_pairs=pair_index,
                    total_pairs=total_pairs,
                    pair_ids=pair_ids,
                    status="in_progress",
                ),
            )
            print(f"Progress saved to {output_dir} (resume with --resume).", file=sys.stderr, flush=True)
        raise

    standings = _standings_rows_from_metrics(all_game_rows, player_ids)

    print("\nStandings (wins across all pairings):")
    for row in standings:
        print(f"  {row['rank']:>2}. {row['player_id']:<16} {row['wins']} wins")

    metadata = {
        "config": args.config,
        "players": player_ids,
        "oracle_player": None if oracle is None else oracle.id,
        "games_per_pair": games_per_pair,
        "base_seed": base_seed,
        "max_moves": args.max_moves,
        "mcts_mode": "online_search",
        "output_dir": output_dir,
        "files": {
            "games": "games.csv",
            "moves": "moves.jsonl",
            "pair_summary": "pair_summary.csv",
            "standings": "standings.csv",
        },
    }
    _progress_write_csv(os.path.join(output_dir, "games.csv"), all_game_rows)
    _write_jsonl(os.path.join(output_dir, "moves.jsonl"), all_move_rows)
    _progress_write_csv(os.path.join(output_dir, "pair_summary.csv"), pair_summaries)
    _progress_write_csv(os.path.join(output_dir, "standings.csv"), standings)
    _progress_write_json(os.path.join(output_dir, "run_metadata.json"), metadata)
    flush_tournament_progress(
        output_dir,
        game_rows=all_game_rows,
        move_rows=all_move_rows,
        pair_summaries=pair_summaries,
        checkpoint=build_checkpoint_payload(
            config_path=args.config,
            base_seed=base_seed,
            games_per_pair=games_per_pair,
            player_ids=player_ids,
            completed_pairs=pair_index,
            total_pairs=total_pairs,
            pair_ids=pair_ids,
            status="complete",
        ),
    )
    print(f"\nSaved tournament metrics to {output_dir}")

    if args.output:
        legacy_payload = {
            "players": player_ids,
            "games_per_pair": games_per_pair,
            "base_seed": base_seed,
            "standings": standings,
            "pairs": pair_summaries,
        }
        _write_json(args.output, legacy_payload)
        print(f"Saved legacy summary to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
