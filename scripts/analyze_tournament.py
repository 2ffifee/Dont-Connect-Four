"""Aggregate tournament metric files into report-ready CSV summaries.

Input is the directory produced by ``scripts/run_tournament.py``. The script is
deliberately dependency-free, so it can run on the training machine immediately
after a long tournament without requiring pandas or notebooks.

Usage::

    python scripts/analyze_tournament.py --input-dir results/main_final
    python scripts/analyze_tournament.py --input-dir results/main_final --output-dir results/main_final/analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any


LLM_COUNTER_FIELDS = (
    "llm_requests",
    "llm_unparseable",
    "llm_illegal",
    "llm_fallbacks",
    "llm_moves",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate tournament metrics.")
    parser.add_argument("--input-dir", default="results/tournament", help="Directory with games.csv and moves.jsonl.")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for summary CSV files. Defaults to --input-dir.",
    )
    args = parser.parse_args(argv)

    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir
    games_path = os.path.join(input_dir, "games.csv")
    moves_path = os.path.join(input_dir, "moves.jsonl")
    if not os.path.exists(games_path):
        raise SystemExit(f"missing tournament games file: {games_path}")

    games = _read_csv(games_path)
    moves = _read_jsonl(moves_path) if os.path.exists(moves_path) else []
    if not games:
        raise SystemExit(f"no game rows found in {games_path}")

    agent_rows = aggregate_agents(games, moves)
    matchup_rows = aggregate_matchups(games)
    llm_rows = aggregate_llm(games)

    _write_csv(os.path.join(output_dir, "agent_summary.csv"), agent_rows)
    _write_csv(os.path.join(output_dir, "matchup_summary.csv"), matchup_rows)
    _write_csv(os.path.join(output_dir, "llm_summary.csv"), llm_rows)

    print(f"Read games: {len(games)}")
    print(f"Read moves: {len(moves)}")
    print(f"Wrote: {os.path.join(output_dir, 'agent_summary.csv')}")
    print(f"Wrote: {os.path.join(output_dir, 'matchup_summary.csv')}")
    print(f"Wrote: {os.path.join(output_dir, 'llm_summary.csv')}")
    return 0


def aggregate_agents(games: list[dict[str, str]], moves: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[tuple[dict[str, str], str]]] = defaultdict(list)
    for game in games:
        buckets[game["red_agent"]].append((game, "red"))
        buckets[game["yellow_agent"]].append((game, "yellow"))

    move_stats = _move_stats_by_agent(moves)
    rows = []
    for agent, agent_games in sorted(buckets.items()):
        total = len(agent_games)
        wins = sum(1 for game, _ in agent_games if game.get("winner_agent") == agent)
        draws = sum(1 for game, _ in agent_games if _is_true(game.get("is_draw", "")))
        losses = total - wins - draws
        red_games = [(game, side) for game, side in agent_games if side == "red"]
        yellow_games = [(game, side) for game, side in agent_games if side == "yellow"]

        decisions = 0
        decision_seconds = 0.0
        tree_growth = 0
        own_lines = 0
        opponent_lines = 0
        llm = _empty_llm_counters()
        kinds = set()
        opponents = set()

        for game, side in agent_games:
            other_side = "yellow" if side == "red" else "red"
            decisions += _int(game[f"{side}_decisions"])
            decision_seconds += _float(game[f"{side}_total_decision_seconds"])
            own_lines += _int(game[f"{side}_lines"])
            opponent_lines += _int(game[f"{other_side}_lines"])
            tree_growth += _tree_growth_for_side(game, side)
            kinds.add(game.get(f"{side}_kind", ""))
            opponents.add(game[f"{other_side}_agent"])
            _add_llm_counters(llm, game, side)

        move_count = move_stats.get(agent, {}).get("moves", 0)
        legal_total = move_stats.get(agent, {}).get("legal_move_total", 0)
        row = {
            "agent": agent,
            "kind": "/".join(sorted(kind for kind in kinds if kind)),
            "opponents": "/".join(sorted(opponents)),
            "games": total,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": _rate(wins, total),
            "loss_rate": _rate(losses, total),
            "draw_rate": _rate(draws, total),
            "games_as_red": len(red_games),
            "wins_as_red": sum(1 for game, _ in red_games if game.get("winner_agent") == agent),
            "win_rate_as_red": _rate(
                sum(1 for game, _ in red_games if game.get("winner_agent") == agent),
                len(red_games),
            ),
            "games_as_yellow": len(yellow_games),
            "wins_as_yellow": sum(1 for game, _ in yellow_games if game.get("winner_agent") == agent),
            "win_rate_as_yellow": _rate(
                sum(1 for game, _ in yellow_games if game.get("winner_agent") == agent),
                len(yellow_games),
            ),
            "avg_moves_per_game": _mean(_int(game["moves"]) for game, _ in agent_games),
            "decisions": decisions,
            "avg_decision_seconds": _rate(decision_seconds, decisions),
            "avg_own_lines": _rate(own_lines, total),
            "avg_opponent_lines": _rate(opponent_lines, total),
            "tree_growth": tree_growth,
            "avg_legal_moves": _rate(legal_total, move_count),
            "llm_requests": llm["llm_requests"],
            "llm_unparseable": llm["llm_unparseable"],
            "llm_illegal": llm["llm_illegal"],
            "llm_fallbacks": llm["llm_fallbacks"],
            "llm_moves": llm["llm_moves"],
            "llm_invalid_response_rate": _rate(llm["llm_unparseable"] + llm["llm_illegal"], llm["llm_requests"]),
            "llm_fallback_rate": _rate(llm["llm_fallbacks"], llm["llm_moves"]),
        }
        rows.append(row)

    rows.sort(key=lambda row: (row["wins"], row["draws"], -row["avg_decision_seconds"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def aggregate_matchups(games: list[dict[str, str]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    for game in games:
        buckets[game["pair_id"]].append(game)

    rows = []
    for pair_id, pair_games in sorted(buckets.items()):
        left, right = _pair_agents(pair_id, pair_games)
        total = len(pair_games)
        left_wins = sum(1 for game in pair_games if game.get("winner_agent") == left)
        right_wins = sum(1 for game in pair_games if game.get("winner_agent") == right)
        draws = sum(1 for game in pair_games if _is_true(game.get("is_draw", "")))
        left_decisions, left_seconds = _decisions_for_agent(pair_games, left)
        right_decisions, right_seconds = _decisions_for_agent(pair_games, right)

        rows.append(
            {
                "pair_id": pair_id,
                "left": left,
                "right": right,
                "games": total,
                "left_wins": left_wins,
                "right_wins": right_wins,
                "draws": draws,
                "left_win_rate": _rate(left_wins, total),
                "right_win_rate": _rate(right_wins, total),
                "draw_rate": _rate(draws, total),
                "avg_moves": _mean(_int(game["moves"]) for game in pair_games),
                "left_avg_decision_seconds": _rate(left_seconds, left_decisions),
                "right_avg_decision_seconds": _rate(right_seconds, right_decisions),
                "left_games_as_red": sum(1 for game in pair_games if game["red_agent"] == left),
                "left_wins_as_red": sum(
                    1 for game in pair_games if game["red_agent"] == left and game.get("winner_agent") == left
                ),
                "right_games_as_red": sum(1 for game in pair_games if game["red_agent"] == right),
                "right_wins_as_red": sum(
                    1 for game in pair_games if game["red_agent"] == right and game.get("winner_agent") == right
                ),
            }
        )
    return rows


def aggregate_llm(games: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for row in aggregate_agents(games, []):
        requests = int(row["llm_requests"])
        moves = int(row["llm_moves"])
        if requests == 0 and moves == 0:
            continue
        rows.append(
            {
                "agent": row["agent"],
                "games": row["games"],
                "llm_requests": requests,
                "llm_unparseable": row["llm_unparseable"],
                "llm_illegal": row["llm_illegal"],
                "llm_invalid_responses": int(row["llm_unparseable"]) + int(row["llm_illegal"]),
                "llm_fallbacks": row["llm_fallbacks"],
                "llm_moves": moves,
                "llm_invalid_response_rate": row["llm_invalid_response_rate"],
                "llm_fallback_rate": row["llm_fallback_rate"],
                "avg_decision_seconds": row["avg_decision_seconds"],
                "win_rate": row["win_rate"],
            }
        )
    return rows


def _read_csv(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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


def _move_stats_by_agent(moves: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"moves": 0, "legal_move_total": 0})
    for move in moves:
        agent = str(move.get("agent", ""))
        stats[agent]["moves"] += 1
        stats[agent]["legal_move_total"] += int(move.get("legal_move_count", 0))
    return stats


def _empty_llm_counters() -> dict[str, int]:
    return {field: 0 for field in LLM_COUNTER_FIELDS}


def _add_llm_counters(counters: dict[str, int], game: dict[str, str], side: str) -> None:
    for field in LLM_COUNTER_FIELDS:
        source = f"{side}_{field}"
        counters[field] += _int(game.get(source, "0"))


def _tree_growth_for_side(game: dict[str, str], side: str) -> int:
    before = _optional_int(game.get(f"{side}_tree_size_before", ""))
    after = _optional_int(game.get(f"{side}_tree_size_after", ""))
    if before is None or after is None:
        return 0
    return max(0, after - before)


def _pair_agents(pair_id: str, games: list[dict[str, str]]) -> tuple[str, str]:
    if "_vs_" in pair_id:
        left, right = pair_id.split("_vs_", 1)
        return left, right
    agents = sorted({game["red_agent"] for game in games} | {game["yellow_agent"] for game in games})
    if len(agents) != 2:
        raise ValueError(f"cannot infer pair agents for {pair_id}")
    return agents[0], agents[1]


def _decisions_for_agent(games: list[dict[str, str]], agent: str) -> tuple[int, float]:
    decisions = 0
    seconds = 0.0
    for game in games:
        if game["red_agent"] == agent:
            decisions += _int(game["red_decisions"])
            seconds += _float(game["red_total_decision_seconds"])
        if game["yellow_agent"] == agent:
            decisions += _int(game["yellow_decisions"])
            seconds += _float(game["yellow_total_decision_seconds"])
    return decisions, seconds


def _mean(values: Iterable[int | float]) -> float:
    total = 0.0
    count = 0
    for value in values:
        total += value
        count += 1
    return total / count if count else 0.0


def _rate(value: int | float, total: int | float) -> float:
    return value / total if total else 0.0


def _int(value: object) -> int:
    if value in (None, ""):
        return 0
    return int(float(str(value)))


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    return int(float(str(value)))


def _float(value: object) -> float:
    if value in (None, ""):
        return 0.0
    return float(str(value))


def _is_true(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


if __name__ == "__main__":
    raise SystemExit(main())
