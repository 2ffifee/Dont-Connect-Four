"""Incremental save and resume for long round-robin tournaments."""

from __future__ import annotations

import csv
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CHECKPOINT_VERSION = 1
CHECKPOINT_FILENAME = "tournament_checkpoint.json"


def checkpoint_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / CHECKPOINT_FILENAME


def load_checkpoint(output_dir: str | Path) -> dict[str, Any] | None:
    path = checkpoint_path(output_dir)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def tournament_is_complete(output_dir: str | Path) -> bool:
    checkpoint = load_checkpoint(output_dir)
    return checkpoint is not None and checkpoint.get("status") == "complete"


def can_resume_tournament(output_dir: str | Path) -> bool:
    checkpoint = load_checkpoint(output_dir)
    if checkpoint is None:
        return False
    return checkpoint.get("status") != "complete"


def validate_checkpoint(
    checkpoint: dict[str, Any],
    *,
    config_path: str,
    base_seed: int,
    games_per_pair: int,
    player_ids: list[str],
) -> None:
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported tournament checkpoint version")
    if checkpoint.get("config") != config_path:
        raise ValueError("tournament checkpoint config path does not match this run")
    if int(checkpoint.get("base_seed", -1)) != base_seed:
        raise ValueError("tournament checkpoint base_seed does not match this run")
    if int(checkpoint.get("games_per_pair", -1)) != games_per_pair:
        raise ValueError("tournament checkpoint games_per_pair does not match this run")
    saved_players = checkpoint.get("players")
    if saved_players != player_ids:
        raise ValueError("tournament checkpoint player list does not match this run")


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    if not Path(path).is_file():
        return []
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def read_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    if not Path(path).is_file():
        return []
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_saved_tournament_rows(output_dir: str | Path) -> tuple[list[dict[str, str]], list[dict[str, Any]], list[dict[str, str]]]:
    root = Path(output_dir)
    games = read_csv_rows(root / "games.csv")
    moves = read_jsonl_rows(root / "moves.jsonl")
    pairs = read_csv_rows(root / "pair_summary.csv")
    return games, moves, pairs


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, separators=(",", ":")) + "\n")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def flush_tournament_progress(
    output_dir: str | Path,
    *,
    game_rows: list[dict[str, Any]],
    move_rows: list[dict[str, Any]],
    pair_summaries: list[dict[str, Any]],
    checkpoint: dict[str, Any],
) -> None:
    """Rewrite metric files and the checkpoint after each completed pairing."""
    root = Path(output_dir)
    write_csv(root / "games.csv", game_rows)
    write_jsonl(root / "moves.jsonl", move_rows)
    write_csv(root / "pair_summary.csv", pair_summaries)
    write_json(checkpoint_path(root), checkpoint)


def build_checkpoint_payload(
    *,
    config_path: str,
    llm_config_path: str | None,
    base_seed: int,
    games_per_pair: int,
    player_ids: list[str],
    completed_pairs: int,
    total_pairs: int,
    pair_ids: list[str],
    status: str,
) -> dict[str, Any]:
    return {
        "version": CHECKPOINT_VERSION,
        "status": status,
        "updated_at": datetime.now(UTC).isoformat(),
        "config": config_path,
        "llm_config": llm_config_path,
        "base_seed": base_seed,
        "games_per_pair": games_per_pair,
        "players": player_ids,
        "completed_pairs": completed_pairs,
        "total_pairs": total_pairs,
        "pair_ids": pair_ids,
        "files": {
            "games": "games.csv",
            "moves": "moves.jsonl",
            "pair_summary": "pair_summary.csv",
        },
    }
