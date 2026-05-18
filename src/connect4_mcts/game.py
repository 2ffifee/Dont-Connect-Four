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


class IllegalMoveError(ValueError):
    """Raised when a move cannot be applied to the current board."""


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

    def is_column_full(self, column: int) -> bool:
        self._validate_column(column)
        return self.board[0][column] is not None

    def legal_moves(self) -> tuple[Move, ...]:
        if self.status is GameStatus.FINISHED:
            return ()

        moves: list[Move] = []
        for column in range(COLUMNS):
            if not self.is_column_full(column):
                moves.append(Move(MoveType.DROP, column))
                moves.append(Move(MoveType.PUSH, column))
        return tuple(moves)

    def is_legal_move(self, move: Move) -> bool:
        return move in self.legal_moves()

    def apply_move(self, move: Move) -> "GameState":
        if not self.is_legal_move(move):
            raise IllegalMoveError(f"illegal move: {move.move_type.value} in column {move.column}")

        if move.move_type is MoveType.DROP:
            next_board = self._apply_drop(move.column)
        elif move.move_type is MoveType.PUSH:
            next_board = self._apply_push(move.column)
        else:
            raise IllegalMoveError(f"unsupported move type: {move.move_type}")

        return GameState(
            board=next_board,
            current_player=self.current_player.opponent,
            status=self.status,
            move_count=self.move_count + 1,
        )

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

    def _apply_drop(self, column: int) -> Board:
        rows = [list(row) for row in self.board]

        for row_index in range(ROWS - 1, -1, -1):
            if rows[row_index][column] is None:
                rows[row_index][column] = self.current_player
                return _freeze_board(rows)

        raise IllegalMoveError(f"column {column} is full")

    def _apply_push(self, column: int) -> Board:
        rows = [list(row) for row in self.board]

        # A push inserts a token from the bottom, moving existing tokens upward.
        for row_index in range(ROWS - 1):
            rows[row_index][column] = rows[row_index + 1][column]
        rows[ROWS - 1][column] = self.current_player

        return _freeze_board(rows)

    @staticmethod
    def _validate_column(column: int) -> None:
        if not 0 <= column < COLUMNS:
            raise ValueError(f"column must be between 0 and {COLUMNS - 1}")


def _freeze_board(rows: list[list[Cell]]) -> Board:
    return tuple(tuple(row) for row in rows)
