"""Shared player interfaces and errors."""

from __future__ import annotations

from typing import Protocol

from connect4_mcts.game import GameState, Move


class MoveSelectionError(ValueError):
    """Raised when a player cannot select a move for the given state."""


class Agent(Protocol):
    def choose_move(self, state: GameState) -> Move:
        """Return a legal move for the given game state."""
