"""Train the Blunder Rate reference engine (the "oracle").

The oracle is a plain-UCT MCTS player whose persistent transposition tree is
grown by self-play and then reused as a value cache when scoring tournament
positions for the Blunder Rate metric.

The run is bounded by a memory budget (5 GB by default, translated into a node
cap), a maximum number of self-play games and a wall-clock limit. Progress is
printed and checkpoints are written periodically so a long run is never lost.

Usage (after setting up the environment)::

    python scripts/train_oracle.py
    python scripts/train_oracle.py --config configs/oracle.toml
    python scripts/train_oracle.py --resume          # continue growing models/oracle_uct.pkl

Configuration values live in ``configs/oracle.toml``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import tomllib
from collections.abc import Sequence

# Allow running directly from the repository without installation.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from connect4_mcts.game import GameState, GameStatus, Player
from connect4_mcts.players.mcts import MCTSPlayer
from connect4_mcts.training import load_player, save_player


def _optional_float(value: object) -> float | None:
    if value is None or (isinstance(value, str) and value.strip().lower() in {"none", ""}):
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None or (isinstance(value, str) and value.strip().lower() in {"none", ""}):
        return None
    return int(value)


def _max_nodes_from_memory(memory: dict) -> int:
    limit_gb = float(memory.get("limit_gb", 5.0))
    bytes_per_node = float(memory.get("bytes_per_node_estimate", 2867))
    safety = float(memory.get("safety_fraction", 0.85))
    return int(limit_gb * (1024**3) * safety / bytes_per_node)


def _build_player(oracle: dict, build_iterations: int) -> MCTSPlayer:
    algorithm = str(oracle.get("algorithm", "uct"))
    if algorithm != "uct":
        print(f"WARNING: oracle.algorithm = '{algorithm}'. A neutral oracle should use plain UCT.")

    return MCTSPlayer(
        iterations=build_iterations,
        exploration=float(oracle.get("exploration", 1.0)),
        power_mean_p=float(oracle.get("power_mean_p", 1.0)),
        fpu=_optional_float(oracle.get("fpu", "none")),
        rollout_policy=str(oracle.get("rollout_policy", "random")),
        final_move=str(oracle.get("final_move", "robust")),
        max_rollout_moves=_optional_int(oracle.get("max_rollout_moves", "none")),
        seed=_optional_int(oracle.get("seed", 0)),
    )


def _save_oracle(player: MCTSPlayer, path: str, eval_iterations: int, build_iterations: int) -> None:
    # Persist with the evaluation budget so the saved oracle is ready to score
    # positions, then restore the (smaller) build budget to continue training.
    player.iterations = eval_iterations
    save_player(player, path)
    player.iterations = build_iterations


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the Blunder Rate oracle by self-play.")
    parser.add_argument("--config", default="configs/oracle.toml", help="Path to the oracle TOML config.")
    parser.add_argument("--resume", action="store_true", help="Continue from the existing output file.")
    args = parser.parse_args(argv)

    with open(args.config, "rb") as config_file:
        config = tomllib.load(config_file)

    oracle_cfg = config.get("oracle", {})
    training_cfg = config.get("training", {})
    memory_cfg = config.get("memory", {})
    evaluation_cfg = config.get("evaluation", {})

    build_iterations = int(training_cfg.get("selfplay_iterations", 1000))
    eval_iterations = int(evaluation_cfg.get("iterations_per_position", 20000))
    temperature = float(training_cfg.get("selfplay_temperature", 1.0))
    max_games = int(training_cfg.get("max_selfplay_games", 100000))
    max_minutes = float(training_cfg.get("max_minutes", 480))
    checkpoint_every = int(training_cfg.get("checkpoint_every_games", 25))
    output = str(training_cfg.get("output", "models/oracle_uct.pkl"))
    bytes_per_node = float(memory_cfg.get("bytes_per_node_estimate", 2867))
    max_nodes = _max_nodes_from_memory(memory_cfg)

    output_dir = os.path.dirname(os.path.abspath(output))
    os.makedirs(output_dir, exist_ok=True)

    if args.resume and os.path.exists(output):
        player = load_player(output)
        player.iterations = build_iterations
        print(f"Resuming from {output} with {player.tree_size:,} cached states.")
    else:
        player = _build_player(oracle_cfg, build_iterations)

    estimated_gb = max_nodes * bytes_per_node / (1024**3)
    print("Oracle training plan:")
    print(f"  algorithm        : plain UCT (C={player.exploration}, p={player.power_mean_p}, fpu={player.fpu})")
    print(f"  build budget     : {build_iterations} iterations/move (temperature {temperature})")
    print(f"  eval budget      : {eval_iterations} iterations/position (stored in saved player)")
    print(f"  memory cap       : ~{max_nodes:,} nodes (~{estimated_gb:.2f} GB at {bytes_per_node:.0f} B/node)")
    print(f"  stop conditions  : nodes>=cap OR games>={max_games} OR minutes>={max_minutes}")
    print(f"  output           : {output}")
    print(flush=True)

    start_time = time.perf_counter()
    games_played = 0

    while player.tree_size < max_nodes and games_played < max_games:
        elapsed_minutes = (time.perf_counter() - start_time) / 60.0
        if elapsed_minutes >= max_minutes:
            print(f"Reached wall-clock limit ({max_minutes} min).")
            break

        first_player = Player.RED if games_played % 2 == 0 else Player.YELLOW
        state = GameState.new(first_player=first_player)
        while state.status is not GameStatus.FINISHED:
            move = player.sample_move(state, temperature=temperature)
            state = state.apply_move(move)
        games_played += 1

        if games_played % 5 == 0 or player.tree_size >= max_nodes:
            mem_gb = player.tree_size * bytes_per_node / (1024**3)
            print(
                f"  games {games_played:>6} | states {player.tree_size:>12,} "
                f"(~{mem_gb:.2f} GB) | {elapsed_minutes:6.1f} min",
                flush=True,
            )

        if checkpoint_every > 0 and games_played % checkpoint_every == 0:
            _save_oracle(player, output, eval_iterations, build_iterations)
            print(f"  checkpoint saved to {output}", flush=True)

    _save_oracle(player, output, eval_iterations, build_iterations)
    total_minutes = (time.perf_counter() - start_time) / 60.0
    final_gb = player.tree_size * bytes_per_node / (1024**3)
    print(
        f"Done. {games_played} games, {player.tree_size:,} states (~{final_gb:.2f} GB) "
        f"in {total_minutes:.1f} min. Saved to {output} (eval budget {eval_iterations})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
