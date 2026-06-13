"""Train an example MCTS player by self-play and save it to a pickle file.

Usage (after setting up the environment)::

    python scripts/train_example_player.py
    python scripts/train_example_player.py --algorithm pmbp --selfplay-games 100

The resulting ``.pkl`` file can be loaded programmatically with
``connect4_mcts.load_player`` / ``load_player_for_play`` (e.g. for tournaments).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

# Allow running the script directly from the repository without installation.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from connect4_mcts.training import save_player, train_fpu, train_lgr, train_pmbp, train_uct


_TRAINERS = {
    "uct": train_uct,
    "fpu": train_fpu,
    "lgr": train_lgr,
    "pmbp": train_pmbp,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and save an example MCTS player.")
    parser.add_argument("--algorithm", choices=tuple(_TRAINERS), default="uct", help="Algorithm to train.")
    parser.add_argument("--output", default="models/uct_example.pkl", help="Where to write the pickled player.")
    parser.add_argument("--iterations", type=int, default=400, help="Per-move search budget for play.")
    parser.add_argument("--selfplay-games", type=int, default=40, help="Number of self-play training games.")
    parser.add_argument(
        "--selfplay-iterations",
        type=int,
        default=150,
        help="Per-move search budget used during self-play training.",
    )
    parser.add_argument(
        "--max-rollout-moves",
        type=int,
        default=None,
        help="Optional cap on rollout length (speeds up training).",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    args = parser.parse_args(argv)

    trainer = _TRAINERS[args.algorithm]
    print(
        f"Training {args.algorithm} player: {args.selfplay_games} self-play games "
        f"({args.selfplay_iterations} iters/move), play budget {args.iterations} iters/move ..."
    )

    player = trainer(
        iterations=args.iterations,
        selfplay_games=args.selfplay_games,
        selfplay_iterations=args.selfplay_iterations,
        max_rollout_moves=args.max_rollout_moves,
        seed=args.seed,
    )

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    save_player(player, args.output)

    print(f"Done. Tree size: {player.tree_size} states. Saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
