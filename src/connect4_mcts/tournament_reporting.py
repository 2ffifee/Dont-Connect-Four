"""Failure reporting helpers for tournament scripts."""

from __future__ import annotations

import json
import signal
import traceback
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TournamentFailureContext:
    pair_index: int
    pair_total: int
    pair_id: str
    left_player: str
    right_player: str
    game_index: int
    games_per_pair: int
    game_id: str
    seed: int
    red_agent: str
    yellow_agent: str
    move_number: int | None = None
    current_player: str | None = None


def describe_exit_code(returncode: int) -> str:
    """Return a human-readable hint for a child process exit code."""
    if returncode == 0:
        return "success"
    if returncode < 0:
        try:
            name = signal.Signals(-returncode).name
        except (ValueError, AttributeError):
            name = str(-returncode)
        if returncode in (-9, -signal.SIGKILL):
            return f"killed by signal {name} (often OOM — try fewer parallel jobs or smaller MCTS trees)"
        return f"terminated by signal {name}"

    hints = {
        1: "unhandled error (see traceback above)",
        137: "SIGKILL — often out-of-memory",
        139: "SIGSEGV — segmentation fault (native extension or memory corruption)",
        143: "SIGTERM — process was terminated externally",
    }
    hint = hints.get(returncode, "non-zero exit")
    return f"exit code {returncode} ({hint})"


def format_failure_banner(context: TournamentFailureContext, exc: BaseException) -> str:
    """Format a multi-line failure summary for stderr."""
    move_line = ""
    if context.move_number is not None:
        player = context.current_player or "?"
        move_line = f"\n  during move : #{context.move_number} ({player} to play)"
    return (
        f"\n{'=' * 72}\n"
        f"Tournament game failed\n"
        f"  pair        : {context.pair_index + 1}/{context.pair_total} — {context.pair_id}\n"
        f"  matchup     : {context.left_player} vs {context.right_player}\n"
        f"  game        : {context.game_index + 1}/{context.games_per_pair} ({context.game_id})\n"
        f"  seed        : {context.seed}\n"
        f"  colors      : red={context.red_agent}, yellow={context.yellow_agent}"
        f"{move_line}\n"
        f"  error type  : {type(exc).__name__}\n"
        f"  error       : {exc}\n"
        f"{'=' * 72}\n"
    )


def write_failure_report(
    output_dir: str | Path,
    *,
    context: TournamentFailureContext,
    exc: BaseException,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persist failure context for post-mortem analysis."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    report_path = path / "tournament_failure.json"
    payload: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "context": asdict(context),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }
    if extra:
        payload["extra"] = extra
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return report_path
