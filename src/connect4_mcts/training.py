"""Training/construction helpers for MCTS players.

"Training" here means letting a player **play games against itself** and thereby
grow its persistent search tree (and, for LGR, its reply memory). A trained
player is one that already carries a partially built tree; how that tree is
shaped depends on the hyperparameters of the chosen algorithm.

Each ``train_*`` function fixes the hyperparameters of one algorithm from the
project report, optionally runs ``selfplay_games`` of self-play to grow the
tree, and returns a ready-to-use :class:`MCTSPlayer` (an object implementing
``choose_move``) that plugs straight into the existing game mechanism.

Trained players - including the grown tree, the learned LGR memory and the RNG
state - can be persisted with :func:`save_player` and restored with
:func:`load_player`.
"""

from __future__ import annotations

import math
import os
import pickle
import time
from collections.abc import Callable
from typing import Any

from connect4_mcts.game import GameState, GameStatus, Player
from connect4_mcts.players.base import Agent
from connect4_mcts.players.mcts import LGRMemory, MCTSPlayer


DEFAULT_ITERATIONS = 1000
DEFAULT_EXPLORATION = math.sqrt(2.0)
DEFAULT_FPU = 1.0
DEFAULT_POWER_MEAN_P = 2.0
DEFAULT_SELFPLAY_TEMPERATURE = 1.0
DEFAULT_BYTES_PER_NODE = 2867.0
DEFAULT_MEMORY_SAFETY = 0.85


def max_nodes_from_memory_gb(
    limit_gb: float,
    *,
    bytes_per_node: float = DEFAULT_BYTES_PER_NODE,
    safety_fraction: float = DEFAULT_MEMORY_SAFETY,
) -> int:
    """Translate a RAM budget into a node cap for persistent MCTS trees."""
    if limit_gb <= 0:
        return 0
    return int(limit_gb * (1024**3) * safety_fraction / bytes_per_node)


def estimate_tree_ram_gb(
    tree_size: int,
    *,
    bytes_per_node: float = DEFAULT_BYTES_PER_NODE,
) -> float:
    return tree_size * bytes_per_node / (1024**3)


def grow_player_to_memory_cap(
    player: MCTSPlayer,
    *,
    max_nodes: int,
    temperature: float = DEFAULT_SELFPLAY_TEMPERATURE,
    max_games: int = 100_000,
    max_minutes: float = 480.0,
    progress_every: int = 5,
    on_progress: Callable[[int, int, float], None] | None = None,
) -> int:
    """Grow ``player``'s tree by self-play until the node cap or a safety stop.

    Returns the number of self-play games completed.
    """
    if max_nodes <= 0:
        return 0

    start_time = time.perf_counter()
    games_played = 0

    while player.tree_size < max_nodes and games_played < max_games:
        elapsed_minutes = (time.perf_counter() - start_time) / 60.0
        if elapsed_minutes >= max_minutes:
            break

        first_player = Player.RED if games_played % 2 == 0 else Player.YELLOW
        state = GameState.new(first_player=first_player)
        while state.status is not GameStatus.FINISHED:
            move = player.sample_move(state, temperature=temperature)
            state = state.apply_move(move)
        games_played += 1

        if on_progress is not None and (
            games_played % progress_every == 0 or player.tree_size >= max_nodes
        ):
            on_progress(games_played, player.tree_size, elapsed_minutes)

    return games_played


def selfplay_train(
    player: MCTSPlayer,
    games: int,
    *,
    temperature: float = DEFAULT_SELFPLAY_TEMPERATURE,
    iterations: int | None = None,
    max_moves: int | None = None,
) -> MCTSPlayer:
    """Grow ``player``'s tree by playing ``games`` self-play games.

    The player controls both sides; every ply runs the player's MCTS search
    (extending the shared, persistent tree) and a move is sampled with the given
    ``temperature`` to keep the training games diverse. ``iterations`` overrides
    the per-move search budget during training only. The same ``player`` object
    is returned for convenience.
    """
    if games < 0:
        raise ValueError("games cannot be negative")
    if games == 0:
        return player

    original_iterations = player.iterations
    if iterations is not None:
        if iterations < 1:
            raise ValueError("iterations must be at least 1 when set")
        player.iterations = iterations

    try:
        for game_index in range(games):
            first_player = Player.RED if game_index % 2 == 0 else Player.YELLOW
            state = GameState.new(first_player=first_player)
            moves_played = 0
            while state.status is not GameStatus.FINISHED:
                if max_moves is not None and moves_played >= max_moves:
                    break
                move = player.sample_move(state, temperature=temperature)
                state = state.apply_move(move)
                moves_played += 1
    finally:
        player.iterations = original_iterations

    return player


def train_uct(
    *,
    iterations: int = DEFAULT_ITERATIONS,
    exploration: float = DEFAULT_EXPLORATION,
    final_move: str = "robust",
    max_rollout_moves: int | None = None,
    selfplay_games: int = 0,
    selfplay_iterations: int | None = None,
    selfplay_temperature: float = DEFAULT_SELFPLAY_TEMPERATURE,
    seed: int | None = None,
) -> MCTSPlayer:
    """Build (and optionally self-train) a baseline UCT player.

    ``iterations`` is the per-move search budget and ``exploration`` is the UCT
    constant ``C``. ``selfplay_games`` grows the tree up front via self-play.
    """
    player = MCTSPlayer(
        iterations=iterations,
        exploration=exploration,
        rollout_policy="random",
        final_move=final_move,
        max_rollout_moves=max_rollout_moves,
        seed=seed,
    )
    return selfplay_train(
        player,
        selfplay_games,
        temperature=selfplay_temperature,
        iterations=selfplay_iterations,
    )


def train_fpu(
    *,
    iterations: int = DEFAULT_ITERATIONS,
    exploration: float = DEFAULT_EXPLORATION,
    fpu: float = DEFAULT_FPU,
    final_move: str = "robust",
    max_rollout_moves: int | None = None,
    selfplay_games: int = 0,
    selfplay_iterations: int | None = None,
    selfplay_temperature: float = DEFAULT_SELFPLAY_TEMPERATURE,
    seed: int | None = None,
) -> MCTSPlayer:
    """Build (and optionally self-train) a UCT player with First Play Urgency.

    ``fpu`` is the fixed value assigned to unvisited children during selection.
    Higher values push the search to try every move before exploiting; lower
    values let it revisit promising branches sooner.
    """
    player = MCTSPlayer(
        iterations=iterations,
        exploration=exploration,
        fpu=fpu,
        rollout_policy="random",
        final_move=final_move,
        max_rollout_moves=max_rollout_moves,
        seed=seed,
    )
    return selfplay_train(
        player,
        selfplay_games,
        temperature=selfplay_temperature,
        iterations=selfplay_iterations,
    )


def train_pmbp(
    *,
    iterations: int = DEFAULT_ITERATIONS,
    exploration: float = DEFAULT_EXPLORATION,
    power_mean_p: float = DEFAULT_POWER_MEAN_P,
    final_move: str = "robust",
    max_rollout_moves: int | None = None,
    selfplay_games: int = 0,
    selfplay_iterations: int | None = None,
    selfplay_temperature: float = DEFAULT_SELFPLAY_TEMPERATURE,
    seed: int | None = None,
) -> MCTSPlayer:
    """Build (and optionally self-train) a UCT player with Power-Mean Backprop.

    ``power_mean_p`` controls how simulation rewards are aggregated into node
    values. ``p == 1`` is the arithmetic mean (plain UCT); ``p > 1`` shifts the
    estimate toward the best observed outcomes.
    """
    player = MCTSPlayer(
        iterations=iterations,
        exploration=exploration,
        power_mean_p=power_mean_p,
        rollout_policy="random",
        final_move=final_move,
        max_rollout_moves=max_rollout_moves,
        seed=seed,
    )
    return selfplay_train(
        player,
        selfplay_games,
        temperature=selfplay_temperature,
        iterations=selfplay_iterations,
    )


def train_lgr(
    *,
    iterations: int = DEFAULT_ITERATIONS,
    exploration: float = DEFAULT_EXPLORATION,
    final_move: str = "robust",
    max_rollout_moves: int | None = None,
    selfplay_games: int = 0,
    selfplay_iterations: int | None = None,
    selfplay_temperature: float = DEFAULT_SELFPLAY_TEMPERATURE,
    memory: LGRMemory | None = None,
    seed: int | None = None,
) -> MCTSPlayer:
    """Build (and optionally self-train) a UCT player with Last Good Reply.

    Self-play grows both the search tree and the persistent :class:`LGRMemory`
    that the rollout policy relies on.
    """
    player = MCTSPlayer(
        iterations=iterations,
        exploration=exploration,
        rollout_policy="lgr",
        lgr_memory=memory if memory is not None else LGRMemory(),
        final_move=final_move,
        max_rollout_moves=max_rollout_moves,
        seed=seed,
    )
    return selfplay_train(
        player,
        selfplay_games,
        temperature=selfplay_temperature,
        iterations=selfplay_iterations,
    )


def save_player(player: Agent, path: str | os.PathLike[str]) -> None:
    """Persist a trained player (tree and learned memory included) to ``path``."""
    with open(os.fspath(path), "wb") as file:
        pickle.dump(player, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_player(path: str | os.PathLike[str]) -> Any:
    """Load a player previously stored with :func:`save_player`."""
    with open(os.fspath(path), "rb") as file:
        return pickle.load(file)
