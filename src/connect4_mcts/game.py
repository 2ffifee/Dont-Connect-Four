"""Core domain model for the modified Connect4 game."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


ROWS = 6
COLUMNS = 8


class Player(Enum):
    RED = "red"
    YELLOW = "yellow"

    @property
    def opponent(self) -> "Player":
        return Player.YELLOW if self is Player.RED else Player.RED


class MoveType(Enum):
    DROP = "drop"
    PUSH = "push"


class GameStatus(Enum):
    ONGOING = "ongoing"
    FAIR_TURN = "fair_turn"
    FINISHED = "finished"


@dataclass(frozen=True, slots=True)
class Move:
    move_type: MoveType
    column: int

    def __post_init__(self) -> None:
        if not 0 <= self.column < COLUMNS:
            raise ValueError(f"column must be between 0 and {COLUMNS - 1}")


Cell = Player | None
Board = tuple[tuple[Cell, ...], ...]


def empty_board() -> Board:
    return tuple(tuple(None for _ in range(COLUMNS)) for _ in range(ROWS))


@dataclass(frozen=True, slots=True)
class GameState:
    board: Board
    current_player: Player = Player.RED
    status: GameStatus = GameStatus.ONGOING
    move_count: int = 0

    @classmethod
    def new(cls, first_player: Player = Player.RED) -> "GameState":
        return cls(board=empty_board(), current_player=first_player)

    def __post_init__(self) -> None:
        if len(self.board) != ROWS:
            raise ValueError(f"board must have {ROWS} rows")

        for row in self.board:
            if len(row) != COLUMNS:
                raise ValueError(f"each board row must have {COLUMNS} columns")

            invalid_cells = [cell for cell in row if cell is not None and not isinstance(cell, Player)]
            if invalid_cells:
                raise ValueError("board cells must be Player values or None")

        if self.move_count < 0:
            raise ValueError("move_count cannot be negative")

        if not isinstance(self.current_player, Player):
            raise ValueError("current_player must be a Player")

        if not isinstance(self.status, GameStatus):
            raise ValueError("status must be a GameStatus")
