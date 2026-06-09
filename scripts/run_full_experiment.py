"""Run the full non-LLM experiment pipeline.

This script is a thin cross-platform orchestrator around the existing project
scripts. It can run everything from training to final CSV summaries, or skip
training when the player pickles already exist.

Examples:

    python scripts/run_full_experiment.py
    python scripts/run_full_experiment.py --skip-training
    python scripts/run_full_experiment.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class Step:
    label: str
    command: list[str] | None = None
    action: Callable[[], None] | None = None
    dry_run_text: str = ""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the full non-LLM experiment pipeline.")
    parser.add_argument("--oracle-config", default="configs/oracle.toml")
    parser.add_argument("--config", default="configs/experiments/main_final.toml", help="Tournament config.")
    parser.add_argument("--output-dir", default="results/main_final")
    parser.add_argument("--games-per-pair", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--max-moves", type=int, default=0, help="Optional safety cap per game; 0 disables it.")
    parser.add_argument("--resume-training", action="store_true", help="Resume oracle/player training from pickles.")
    parser.add_argument("--skip-training", action="store_true", help="Skip both oracle and tournament-player training.")
    parser.add_argument("--skip-oracle-training", action="store_true", help="Skip only oracle training.")
    parser.add_argument("--skip-player-training", action="store_true", help="Skip only tournament-player training.")
    parser.add_argument("--skip-blunders", action="store_true", help="Do not run oracle Blunder Rate scoring.")
    parser.add_argument("--blunder-threshold", type=float, default=None, help="Override oracle config threshold.")
    parser.add_argument("--blunder-max-positions", type=int, default=0, help="0 means score all selected moves.")
    parser.add_argument("--blunder-sample-every", type=int, default=1)
    parser.add_argument("--blunder-progress-every", type=int, default=100)
    parser.add_argument("--with-llm", action="store_true", help="Include LLM entries from --llm-config.")
    parser.add_argument("--llm-config", default="configs/tournament_llm.toml")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for child scripts.")
    args = parser.parse_args(argv)

    oracle_config = _project_path(args.oracle_config)
    tournament_config = _project_path(args.config)
    output_dir = _project_path(args.output_dir)
    llm_config = _project_path(args.llm_config)
    oracle_path = _oracle_output_path(oracle_config)
    tournament_defaults, tournament_players = _tournament_config(tournament_config)
    tournament_oracle = _find_player(tournament_players, "oracle")
    tournament_output_dir = _project_path(str(tournament_defaults.get("output_dir", "models/tournament")))
    tournament_oracle_path = tournament_output_dir / "oracle.pkl"
    threshold = args.blunder_threshold
    if threshold is None:
        threshold = _oracle_blunder_threshold(oracle_config)

    steps: list[Step] = []
    skip_oracle_training = args.skip_training or args.skip_oracle_training
    skip_player_training = args.skip_training or args.skip_player_training

    if not skip_oracle_training:
        command = [args.python, "scripts/train_oracle.py", "--config", str(oracle_config)]
        if args.resume_training:
            command.append("--resume")
        steps.append(Step("train oracle", command=command))

    if tournament_oracle is not None:
        play_iterations = int(_player_get(tournament_oracle, tournament_defaults, "play_iterations", 1000))
        steps.append(
            Step(
                "prepare tournament oracle",
                action=lambda: _prepare_tournament_oracle(oracle_path, tournament_oracle_path, play_iterations),
                dry_run_text=(
                    f"<internal> copy {_display_path(oracle_path)} -> "
                    f"{_display_path(tournament_oracle_path)} with iterations={play_iterations}"
                ),
            )
        )

    if not skip_player_training:
        command = [args.python, "scripts/train_tournament_grid.py", "--config", str(tournament_config)]
        train_ids = [
            str(entry["id"])
            for entry in tournament_players
            if entry.get("kind") == "mcts" and str(entry["id"]) != "oracle"
        ]
        if tournament_oracle is not None:
            command.extend(["--only", ",".join(train_ids)])
        if args.resume_training:
            command.append("--resume")
        if train_ids:
            steps.append(Step("train tournament players", command=command))

    tournament_command = [
        args.python,
        "scripts/run_tournament.py",
        "--config",
        str(tournament_config),
        "--games-per-pair",
        str(args.games_per_pair),
        "--base-seed",
        str(args.base_seed),
        "--output-dir",
        str(output_dir),
    ]
    if args.max_moves > 0:
        tournament_command.extend(["--max-moves", str(args.max_moves)])
    if args.with_llm:
        tournament_command.extend(["--llm-config", str(llm_config)])
    else:
        tournament_command.append("--no-llm")
    steps.append(Step("run tournament", command=tournament_command))

    steps.append(
        Step("analyze tournament", command=[args.python, "scripts/analyze_tournament.py", "--input-dir", str(output_dir)])
    )

    if not args.skip_blunders:
        blunder_command = [
            args.python,
            "scripts/score_blunders.py",
            "--oracle",
            str(oracle_path),
            "--input-dir",
            str(output_dir),
            "--threshold",
            str(threshold),
            "--sample-every",
            str(args.blunder_sample_every),
            "--progress-every",
            str(args.blunder_progress_every),
        ]
        if args.blunder_max_positions > 0:
            blunder_command.extend(["--max-positions", str(args.blunder_max_positions)])
        steps.append(Step("score blunders", command=blunder_command))

    print("Full experiment pipeline:")
    print(f"  oracle config     : {_display_path(oracle_config)}")
    print(f"  tournament config : {_display_path(tournament_config)}")
    print(f"  output dir        : {_display_path(output_dir)}")
    print(f"  oracle pickle     : {_display_path(oracle_path)}")
    print(f"  llm contestants   : {'yes' if args.with_llm else 'no'}")
    print()

    total_start = time.perf_counter()
    for step in steps:
        if _run_step(step, dry_run=args.dry_run) != 0:
            return 1

    if args.dry_run:
        print("\nDry run complete. No commands were executed.")
    else:
        elapsed = (time.perf_counter() - total_start) / 60.0
        print(f"\nFull experiment finished in {elapsed:.1f} min.")
        print(f"Results: {_display_path(output_dir)}")
    return 0


def _run_step(step: Step, *, dry_run: bool) -> int:
    print(f"==> {step.label}")
    if step.command is not None:
        print(_format_command(step.command))
    elif step.dry_run_text:
        print(step.dry_run_text)
    if dry_run:
        print()
        return 0

    started_at = time.perf_counter()
    if step.command is not None:
        completed = subprocess.run(step.command, cwd=PROJECT_ROOT, check=False)
        returncode = completed.returncode
    elif step.action is not None:
        try:
            step.action()
            returncode = 0
        except Exception as exc:  # noqa: BLE001 - surface internal preparation failures cleanly
            print(f"\nStep failed: {exc}", file=sys.stderr)
            returncode = 1
    else:
        returncode = 0
    elapsed = (time.perf_counter() - started_at) / 60.0
    if returncode != 0:
        print(f"\nStep failed after {elapsed:.1f} min: {step.label}", file=sys.stderr)
        return returncode
    print(f"<== {step.label} done in {elapsed:.1f} min\n")
    return 0


def _project_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _read_toml(path: Path) -> dict:
    with path.open("rb") as file:
        return tomllib.load(file)


def _tournament_config(config_path: Path) -> tuple[dict, list[dict]]:
    config = _read_toml(config_path)
    return config.get("defaults", {}), list(config.get("players", []))


def _find_player(players: list[dict], player_id: str) -> dict | None:
    for player in players:
        if str(player.get("id", "")) == player_id:
            return player
    return None


def _player_get(entry: dict, defaults: dict, key: str, fallback: object) -> object:
    if key in entry:
        return entry[key]
    if key in defaults:
        return defaults[key]
    return fallback


def _oracle_output_path(config_path: Path) -> Path:
    config = _read_toml(config_path)
    output = str(config.get("training", {}).get("output", "models/oracle_uct.pkl"))
    return _project_path(output)


def _oracle_blunder_threshold(config_path: Path) -> float:
    config = _read_toml(config_path)
    return float(config.get("evaluation", {}).get("blunder_threshold", 0.3))


def _prepare_tournament_oracle(oracle_path: Path, output_path: Path, play_iterations: int) -> None:
    if not oracle_path.exists():
        raise FileNotFoundError(f"missing oracle pickle: {oracle_path}")

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from connect4_mcts.training import load_player, save_player

    player = load_player(str(oracle_path))
    player.iterations = play_iterations
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_player(player, str(output_path))
    print(f"Prepared tournament oracle: {_display_path(output_path)} (iterations={play_iterations})")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _format_command(command: list[str]) -> str:
    return " ".join(_quote(part) for part in command)


def _quote(part: str) -> str:
    if not part:
        return '""'
    if any(char.isspace() for char in part):
        return f'"{part}"'
    return part


if __name__ == "__main__":
    raise SystemExit(main())
