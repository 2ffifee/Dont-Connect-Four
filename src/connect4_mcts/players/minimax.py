"""Minimax move selection agent with alpha-beta pruning."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from connect4_mcts.evaluation import evaluate_position
from connect4_mcts.game import GameState, Move, Player
from connect4_mcts.players.base import MoveSelectionError


Evaluator = Callable[[GameState, Player], int]


@dataclass(frozen=True, slots=True)
class MinimaxPlayer:
    depth: int = 3
    evaluator: Evaluator = evaluate_position

    def __post_init__(self) -> None:
        if self.depth < 0:
            raise ValueError("depth cannot be negative")

    def choose_move(self, state: GameState) -> Move:
        legal_moves = state.legal_moves()
        if not legal_moves:
            raise MoveSelectionError("cannot choose a move when no legal moves are available")

        root_player = state.current_player
        best_move = legal_moves[0]
        best_value = float("-inf")
        alpha = float("-inf")
        beta = float("inf")

        for move in legal_moves:
            next_state = state.apply_move(move)
            value = self._minimax(
                next_state,
                remaining_depth=max(0, self.depth - 1),
                root_player=root_player,
                alpha=alpha,
                beta=beta,
            )
            if value > best_value:
                best_value = value
                best_move = move
            alpha = max(alpha, best_value)

        return best_move

    def _minimax(
        self,
        state: GameState,
        remaining_depth: int,
        root_player: Player,
        alpha: float,
        beta: float,
    ) -> int:
        legal_moves = state.legal_moves()
        if remaining_depth == 0 or not legal_moves:
            return self.evaluator(state, root_player)

        if state.current_player is root_player:
            value = float("-inf")
            for move in legal_moves:
                value = max(
                    value,
                    self._minimax(
                        state.apply_move(move),
                        remaining_depth=remaining_depth - 1,
                        root_player=root_player,
                        alpha=alpha,
                        beta=beta,
                    ),
                )
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return int(value)

        value = float("inf")
        for move in legal_moves:
            value = min(
                value,
                self._minimax(
                    state.apply_move(move),
                    remaining_depth=remaining_depth - 1,
                    root_player=root_player,
                    alpha=alpha,
                    beta=beta,
                ),
            )
            beta = min(beta, value)
            if alpha >= beta:
                break
        return int(value)
