"""Random move selection agent."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from connect4_mcts.game import GameState, Move
from connect4_mcts.players.base import MoveSelectionError


@dataclass(slots=True)
class RandomPlayer:
    seed: int | None = None
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def choose_move(self, state: GameState) -> Move:
        legal_moves = state.legal_moves()
        if not legal_moves:
            raise MoveSelectionError("cannot choose a move when no legal moves are available")

        return self._rng.choice(legal_moves)
