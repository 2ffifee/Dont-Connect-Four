"""Score tournament moves with an oracle and compute Blunder Rate.

The script reads ``moves.jsonl`` produced by ``scripts/run_tournament.py``. For
each selected move it reconstructs the pre-move game state, runs the oracle MCTS
player in online search mode, and writes regret / blunder flags.

The oracle is defined in the experiment config as the player tagged ``ORACLE``.

Usage::

    python scripts/score_blunders.py --config configs/experiments/main_final.toml --input-dir results/main_final
    python scripts/score_blunders.py --config configs/experiments/main_final.toml --moves results/main_final/moves.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from connect4_mcts.experiment_config import load_experiment_config, instantiate_player
from connect4_mcts.game import Board, GameState, GameStatus, Move, MoveType, Player
from connect4_mcts.players.mcts import SearchEvaluation


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score tournament moves with an oracle.")
    parser.add_argument(
        "--config",
        default="",
        help="Experiment config with one player tagged ORACLE (required unless testing with mocks).",
    )
    parser.add_argument("--input-dir", default="", help="Tournament output directory with moves.jsonl.")
    parser.add_argument("--moves", default="", help="Path to moves.jsonl. Overrides --input-dir.")
    parser.add_argument("--output", default="", help="Path to blunders.csv. Defaults to input dir.")
    parser.add_argument(
        "--summary-output",
        default="",
        help="Path to blunder_summary.csv. Defaults next to --output.",
    )
    parser.add_argument("--threshold", type=float, default=None, help="Regret threshold; defaults to config value.")
    parser.add_argument("--max-positions", type=int, default=0, help="Maximum scored rows; 0 means all selected rows.")
    parser.add_argument("--sample-every", type=int, default=None, help="Score every Nth move row; defaults to config.")
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N scored positions.")
    args = parser.parse_args(argv)

    if not args.config:
        raise SystemExit("--config is required")

    experiment = load_experiment_config(args.config)
    oracle_spec = experiment.oracle_player()
    if oracle_spec is None:
        raise SystemExit("experiment config must define exactly one player with tag ORACLE for blunder scoring")

    threshold = args.threshold if args.threshold is not None else experiment.blunder_threshold
    sample_every = args.sample_every if args.sample_every is not None else experiment.blunder_sample_every

    if threshold < 0:
        raise SystemExit("--threshold must be non-negative")
    if args.max_positions < 0:
        raise SystemExit("--max-positions cannot be negative")
    if sample_every < 1:
        raise SystemExit("--sample-every must be at least 1")

    moves_path = _resolve_moves_path(args.input_dir, args.moves)
    output_path = _resolve_output_path(args.input_dir, args.output, "blunders.csv")
    summary_path = _resolve_summary_path(output_path, args.summary_output)

    oracle = instantiate_player(oracle_spec, game_seed=experiment.seed)
    if not hasattr(oracle, "evaluate"):
        raise SystemExit(
            f"oracle player {oracle_spec.id!r} (type={oracle_spec.type!r}) does not expose evaluate(...)"
        )

    rows, evaluated, cache_hits = score_moves(
        oracle,
        _iter_jsonl(moves_path),
        threshold=threshold,
        max_positions=args.max_positions,
        sample_every=sample_every,
        run_search=True,
        progress_every=args.progress_every,
    )
    summary_rows = summarize_blunders(rows)

    _write_csv(output_path, rows)
    _write_csv(summary_path, summary_rows)

    print(f"Oracle player: {oracle_spec.id}")
    print(f"Scored moves: {len(rows)}")
    print(f"Oracle evaluations: {evaluated}")
    print(f"Cache hits: {cache_hits}")
    print(f"Wrote: {output_path}")
    print(f"Wrote: {summary_path}")
    return 0


def score_moves(
    oracle: Any,
    moves: Iterable[dict[str, Any]],
    *,
    threshold: float = 0.3,
    max_positions: int = 0,
    sample_every: int = 1,
    run_search: bool = True,
    progress_every: int = 100,
) -> tuple[list[dict[str, Any]], int, int]:
    evaluation_cache: dict[tuple[Any, ...], SearchEvaluation] = {}
    rows: list[dict[str, Any]] = []
    evaluated = 0
    cache_hits = 0
    initial_tree_size = _oracle_tree_size(oracle)

    for source_index, move_row in enumerate(moves):
        if source_index % sample_every != 0:
            continue
        if max_positions and len(rows) >= max_positions:
            break

        state = state_from_payload(move_row["state_before"])
        chosen = move_from_payload(move_row)
        cache_key = _state_cache_key(move_row["state_before"])
        evaluation = evaluation_cache.get(cache_key)
        if evaluation is None:
            evaluation = oracle.evaluate(state, run_search=run_search, retain_tree=False)
            evaluation_cache[cache_key] = evaluation
            evaluated += 1
        else:
            cache_hits += 1

        rows.append(_score_row(move_row, chosen, evaluation, threshold))
        if progress_every > 0 and len(rows) % progress_every == 0:
            tree_growth = _oracle_tree_size(oracle) - initial_tree_size
            growth_note = f", oracle tree +{tree_growth} nodes" if tree_growth else ""
            print(
                f"  scored {len(rows)} moves ({evaluated} oracle lookups, {cache_hits} repeated states{growth_note})",
                flush=True,
            )

    return rows, evaluated, cache_hits


def _state_cache_key(payload: dict[str, Any]) -> tuple[Any, ...]:
    protected = payload.get("protected_segments", [])
    return (
        tuple(payload["board"]),
        str(payload["current_player"]),
        str(payload["first_player"]),
        str(payload.get("status", "")),
        int(payload["move_count"]),
        int(payload.get("red_line_total", 0)),
        int(payload.get("yellow_line_total", 0)),
        tuple(
            tuple((int(row), int(column)) for row, column in segment)
            for segment in protected
        ),
    )


def _oracle_tree_size(oracle: Any) -> int:
    tree_size = getattr(oracle, "tree_size", None)
    if isinstance(tree_size, int):
        return tree_size
    tree = getattr(oracle, "tree", None)
    if isinstance(tree, dict):
        return len(tree)
    return 0


def summarize_blunders(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row["agent"])].append(row)

    summaries = []
    for agent, agent_rows in sorted(buckets.items()):
        total = len(agent_rows)
        blunders = sum(1 for row in agent_rows if _is_true(row["is_blunder"]))
        known_value_rows = [row for row in agent_rows if row["chosen_value"] != ""]
        summaries.append(
            {
                "agent": agent,
                "scored_moves": total,
                "moves_with_known_value": len(known_value_rows),
                "blunders": blunders,
                "blunder_rate": _rate(blunders, total),
                "avg_regret": _mean(_float(row["regret"]) for row in known_value_rows),
                "avg_chosen_value": _mean(_float(row["chosen_value"]) for row in known_value_rows),
                "avg_best_value": _mean(_float(row["best_value"]) for row in known_value_rows),
                "avg_oracle_best_visits": _mean(_int(row["oracle_best_visits"]) for row in agent_rows),
                "avg_chosen_visits": _mean(_int(row["chosen_visits"]) for row in agent_rows),
            }
        )
    summaries.sort(key=lambda row: (row["blunder_rate"], -row["scored_moves"]))
    return summaries


def state_from_payload(payload: dict[str, Any]) -> GameState:
    board = board_from_rows(payload["board"])
    protected_segments = frozenset(
        tuple((int(row), int(column)) for row, column in segment)
        for segment in payload.get("protected_segments", [])
    )
    status_raw = str(payload["status"])
    if status_raw == "fair_turn":
        status_raw = "ongoing"
    return GameState(
        board=board,
        current_player=Player(str(payload["current_player"])),
        first_player=Player(str(payload["first_player"])),
        status=GameStatus(status_raw),
        move_count=int(payload["move_count"]),
        result=None,
        red_line_total=int(payload.get("red_line_total", 0)),
        yellow_line_total=int(payload.get("yellow_line_total", 0)),
        protected_segments=protected_segments,
    )


def board_from_rows(rows: list[str]) -> Board:
    player_by_symbol = {
        ".": None,
        "R": Player.RED,
        "Y": Player.YELLOW,
    }
    return tuple(tuple(player_by_symbol[symbol] for symbol in row) for row in rows)


def move_from_payload(payload: dict[str, Any]) -> Move:
    return Move(MoveType(str(payload["move_type"])), int(payload["column"]))


def _score_row(
    move_row: dict[str, Any],
    chosen: Move,
    evaluation: SearchEvaluation,
    threshold: float,
) -> dict[str, Any]:
    chosen_value = evaluation.value_of(chosen)
    regret = evaluation.regret_of(chosen)
    best_visits = evaluation.move_visits.get(evaluation.best_move, 0)
    chosen_visits = evaluation.move_visits.get(chosen, 0)
    return {
        "game_id": move_row.get("game_id", ""),
        "pair_id": move_row.get("pair_id", ""),
        "pair_index": move_row.get("pair_index", ""),
        "game_index": move_row.get("game_index", ""),
        "seed": move_row.get("seed", ""),
        "move_number": move_row.get("move_number", ""),
        "player": move_row.get("player", ""),
        "agent": move_row.get("agent", ""),
        "chosen_move_type": chosen.move_type.value,
        "chosen_column": chosen.column,
        "oracle_best_move_type": evaluation.best_move.move_type.value,
        "oracle_best_column": evaluation.best_move.column,
        "chosen_value": "" if chosen_value is None else chosen_value,
        "best_value": evaluation.root_value,
        "regret": "" if regret is None else regret,
        "is_blunder": evaluation.is_blunder(chosen, threshold=threshold),
        "threshold": threshold,
        "chosen_visits": chosen_visits,
        "oracle_best_visits": best_visits,
        "evaluated_move_count": len(evaluation.move_values),
    }


def _resolve_moves_path(input_dir: str, moves_path: str) -> str:
    path = moves_path or os.path.join(input_dir, "moves.jsonl")
    if not path or not os.path.exists(path):
        raise SystemExit(f"missing moves file: {path}")
    return path


def _resolve_output_path(input_dir: str, output_path: str, filename: str) -> str:
    if output_path:
        return output_path
    if input_dir:
        return os.path.join(input_dir, filename)
    return filename


def _resolve_summary_path(output_path: str, summary_path: str) -> str:
    if summary_path:
        return summary_path
    return os.path.join(os.path.dirname(os.path.abspath(output_path)), "blunder_summary.csv")


def _iter_jsonl(path: str) -> Iterable[dict[str, Any]]:
    with open(path, encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def _write_csv(path: str, rows: Iterable[dict[str, Any]]) -> None:
    row_list = list(rows)
    if not row_list:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fieldnames = list(row_list[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row_list)


def _mean(values: Iterable[int | float]) -> float:
    total = 0.0
    count = 0
    for value in values:
        total += value
        count += 1
    return total / count if count else 0.0


def _rate(value: int | float, total: int | float) -> float:
    return value / total if total else 0.0


def _float(value: object) -> float:
    if value in (None, ""):
        return 0.0
    return float(str(value))


def _int(value: object) -> int:
    if value in (None, ""):
        return 0
    return int(float(str(value)))


def _is_true(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


if __name__ == "__main__":
    raise SystemExit(main())
