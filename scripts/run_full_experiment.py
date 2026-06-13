"""Run the full experiment pipeline: tournament, analysis, optional blunder scoring.

Examples::

    python scripts/run_full_experiment.py
    python scripts/run_full_experiment.py --config configs/experiments/smoke.toml
    python scripts/run_full_experiment.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from connect4_mcts.experiment_config import load_experiment_config
from connect4_mcts.experiment_pipeline import (
    load_pipeline_state,
    mark_stage_complete,
    parse_skip_stages,
    stages_to_skip,
)
from connect4_mcts.tournament_progress import can_resume_tournament


@dataclass(frozen=True, slots=True)
class Step:
    stage_id: str | None
    label: str
    command: list[str] | None = None
    action: Callable[[], None] | None = None
    dry_run_text: str = ""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the full experiment pipeline.")
    parser.add_argument("--config", default="configs/experiments/main_final.toml", help="Experiment config.")
    parser.add_argument("--output-dir", default=None, help="Override output_dir from config.")
    parser.add_argument("--games-per-pair", type=int, default=None, help="Override games_per_pair from config.")
    parser.add_argument("--base-seed", type=int, default=None, help="Override seed from config.")
    parser.add_argument("--max-moves", type=int, default=0, help="Optional safety cap per game; 0 disables it.")
    parser.add_argument("--skip-blunders", action="store_true", help="Do not run oracle Blunder Rate scoring.")
    parser.add_argument("--blunder-threshold", type=float, default=None, help="Override blunder_threshold from config.")
    parser.add_argument("--blunder-max-positions", type=int, default=0, help="0 means score all selected moves.")
    parser.add_argument("--blunder-sample-every", type=int, default=None, help="Override blunder_sample_every from config.")
    parser.add_argument("--blunder-progress-every", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for child scripts.")
    parser.add_argument(
        "--verbose-games",
        action="store_true",
        help="Pass --verbose-games to run_tournament.py.",
    )
    parser.add_argument(
        "--skip-stage",
        "--skip-etap",
        action="append",
        dest="skip_stage",
        default=[],
        metavar="STAGE",
        help=(
            "Skip a pipeline stage (repeatable, comma-separated). "
            "Stages: run-tournament, analyze-tournament, score-blunders. "
            "Aliases: tournament, analyze, blunders."
        ),
    )
    parser.add_argument(
        "--resume-pipeline",
        action="store_true",
        help="Skip stages already marked complete in output-dir/experiment_pipeline.json.",
    )
    args = parser.parse_args(argv)

    experiment = load_experiment_config(_project_path(args.config))
    output_dir = _project_path(args.output_dir) if args.output_dir else _project_path(experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    games_per_pair = args.games_per_pair if args.games_per_pair is not None else experiment.games_per_pair
    base_seed = args.base_seed if args.base_seed is not None else experiment.seed
    blunder_threshold = (
        args.blunder_threshold if args.blunder_threshold is not None else experiment.blunder_threshold
    )
    blunder_sample_every = (
        args.blunder_sample_every if args.blunder_sample_every is not None else experiment.blunder_sample_every
    )
    oracle_player = experiment.oracle_player()

    try:
        explicit_skip = parse_skip_stages(args.skip_stage)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    pipeline_state = load_pipeline_state(output_dir)
    skip_stages = stages_to_skip(
        explicit_skip=explicit_skip,
        resume_pipeline=args.resume_pipeline,
        state=pipeline_state,
    )
    if args.skip_blunders:
        skip_stages.add("score-blunders")
    if oracle_player is None:
        skip_stages.add("score-blunders")

    steps: list[Step] = []

    if "run-tournament" in skip_stages:
        print("Skipping stage: run-tournament")
    else:
        tournament_command = [
            args.python,
            "scripts/run_tournament.py",
            "--config",
            str(_project_path(args.config)),
            "--games-per-pair",
            str(games_per_pair),
            "--base-seed",
            str(base_seed),
            "--output-dir",
            str(output_dir),
        ]
        if args.max_moves > 0:
            tournament_command.extend(["--max-moves", str(args.max_moves)])
        if args.verbose_games:
            tournament_command.append("--verbose-games")
        if args.resume_pipeline and can_resume_tournament(output_dir):
            tournament_command.append("--resume")
        steps.append(Step("run-tournament", "run tournament", command=tournament_command))

    if "analyze-tournament" in skip_stages:
        print("Skipping stage: analyze-tournament")
    else:
        steps.append(
            Step(
                "analyze-tournament",
                "analyze tournament",
                command=[args.python, "scripts/analyze_tournament.py", "--input-dir", str(output_dir)],
            )
        )

    if "score-blunders" in skip_stages:
        if oracle_player is None and not args.skip_blunders:
            print("Skipping stage: score-blunders (no player with ORACLE tag in config)")
        else:
            print("Skipping stage: score-blunders")
    else:
        blunder_command = [
            args.python,
            "scripts/score_blunders.py",
            "--config",
            str(_project_path(args.config)),
            "--input-dir",
            str(output_dir),
            "--threshold",
            str(blunder_threshold),
            "--sample-every",
            str(blunder_sample_every),
            "--progress-every",
            str(args.blunder_progress_every),
        ]
        if args.blunder_max_positions > 0:
            blunder_command.extend(["--max-positions", str(args.blunder_max_positions)])
        steps.append(Step("score-blunders", "score blunders", command=blunder_command))

    print("Full experiment pipeline:")
    print(f"  config            : {_display_path(_project_path(args.config))}")
    print(f"  output dir        : {_display_path(output_dir)}")
    print(f"  games per pair    : {games_per_pair}")
    print(f"  oracle player     : {oracle_player.id if oracle_player else '(none)'}")
    if skip_stages:
        ordered = [stage for stage in skip_stages if stage]
        print(f"  skipped stages    : {', '.join(sorted(ordered))}")
    if args.resume_pipeline and pipeline_state.completed_stages:
        print(f"  pipeline resume   : {', '.join(pipeline_state.completed_stages)}")
    print()

    total_start = time.perf_counter()
    for step in steps:
        if _run_step(step, dry_run=args.dry_run, output_dir=output_dir) != 0:
            return 1

    if args.dry_run:
        print("\nDry run complete. No commands were executed.")
    else:
        elapsed = (time.perf_counter() - total_start) / 60.0
        print(f"\nFull experiment finished in {elapsed:.1f} min.")
        print(f"Results: {_display_path(output_dir)}")
    return 0


def _run_step(step: Step, *, dry_run: bool, output_dir: Path) -> int:
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
        if returncode != 0:
            _print_subprocess_failure(step.label, step.command, returncode)
    elif step.action is not None:
        try:
            step.action()
            returncode = 0
        except Exception as exc:  # noqa: BLE001
            print(f"\nStep failed: {exc}", file=sys.stderr)
            returncode = 1
    else:
        returncode = 0
    elapsed = (time.perf_counter() - started_at) / 60.0
    if returncode != 0:
        print(f"\nStep failed after {elapsed:.1f} min: {step.label}", file=sys.stderr)
        return returncode
    if step.stage_id is not None:
        state_path = mark_stage_complete(output_dir, step.stage_id)
        print(f"Pipeline checkpoint updated: {_display_path(state_path)}")
    print(f"<== {step.label} done in {elapsed:.1f} min\n")
    return 0


def _project_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _format_command(command: list[str]) -> str:
    return " ".join(_quote(part) for part in command)


def _failure_report_path(command: list[str]) -> Path | None:
    if "--output-dir" not in command:
        return None
    index = command.index("--output-dir")
    if index + 1 >= len(command):
        return None
    return _project_path(command[index + 1]) / "tournament_failure.json"


def _print_subprocess_failure(label: str, command: list[str], returncode: int) -> None:
    from connect4_mcts.tournament_reporting import describe_exit_code

    print(f"\nSubprocess '{label}' failed: {describe_exit_code(returncode)}", file=sys.stderr)
    print(f"Command: {_format_command(command)}", file=sys.stderr)
    failure_report = _failure_report_path(command)
    if failure_report is not None and failure_report.is_file():
        print(f"See failure report: {_display_path(failure_report)}", file=sys.stderr)
    else:
        print(
            "No tournament_failure.json found — if the process was killed (OOM), there may be "
            "no Python traceback. Re-run with --verbose-games to see the last started game.",
            file=sys.stderr,
        )


def _quote(part: str) -> str:
    if not part:
        return '""'
    if any(char.isspace() for char in part):
        return f'"{part}"'
    return part


if __name__ == "__main__":
    raise SystemExit(main())
