"""Core domain model for the modified Connect4 game.

Performance notes
-----------------
``apply_move`` is on the hot path of MCTS rollouts and tournament play, so it is
optimized to:

* check legality in O(1) instead of materializing ``legal_moves()``;
* rebuild only the affected board cells (sharing the immutable row tuples that
  do not change);
* detect a finishing line **incrementally** from the cells touched by the move
  (an ``ONGOING`` board has no four-in-a-row, so any new line must pass through
  those cells), instead of rescanning the whole board on every move;
* count all lines (the expensive full scan) only when a terminal state is
  actually reached, i.e. to build the :class:`GameResult`;
* construct the successor state without re-running ``__post_init__`` validation
  (the successor is guaranteed valid by construction).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


ROWS = 6
COLUMNS = 8

_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


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
class GameResult:
    winner: Player | None
    red_lines: int
    yellow_lines: int

    @classmethod
    def from_line_counts(cls, line_counts: dict[Player, int]) -> "GameResult":
        red_lines = line_counts[Player.RED]
        yellow_lines = line_counts[Player.YELLOW]

        if red_lines > yellow_lines:
            winner = Player.YELLOW
        elif yellow_lines > red_lines:
            winner = Player.RED
        else:
            winner = None

        return cls(winner=winner, red_lines=red_lines, yellow_lines=yellow_lines)

    @property
    def is_draw(self) -> bool:
        return self.winner is None

    def lines_for(self, player: Player) -> int:
        if player is Player.RED:
            return self.red_lines
        if player is Player.YELLOW:
            return self.yellow_lines
        raise ValueError("player must be a Player")

    def __post_init__(self) -> None:
        if self.winner is not None and not isinstance(self.winner, Player):
            raise ValueError("winner must be a Player or None")

        if self.red_lines < 0 or self.yellow_lines < 0:
            raise ValueError("line counts cannot be negative")


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


# Pre-built, shareable move objects so the hot ``legal_moves`` path does not
# allocate new ``Move`` instances on every call.
_DROP_MOVES = tuple(Move(MoveType.DROP, column) for column in range(COLUMNS))
_PUSH_MOVES = tuple(Move(MoveType.PUSH, column) for column in range(COLUMNS))


@dataclass(frozen=True, slots=True)
class GameState:
    board: Board
    current_player: Player = Player.RED
    first_player: Player = Player.RED
    status: GameStatus = GameStatus.ONGOING
    move_count: int = 0
    result: GameResult | None = None

    @classmethod
    def new(cls, first_player: Player = Player.RED) -> "GameState":
        return cls(board=empty_board(), current_player=first_player, first_player=first_player)

    def is_column_full(self, column: int) -> bool:
        self._validate_column(column)
        return self.board[0][column] is not None

    def legal_moves(self) -> tuple[Move, ...]:
        if self.status is GameStatus.FINISHED:
            return ()

        top_row = self.board[0]
        moves: list[Move] = []
        for column in range(COLUMNS):
            if top_row[column] is None:
                moves.append(_DROP_MOVES[column])
                moves.append(_PUSH_MOVES[column])
        return tuple(moves)

    def is_legal_move(self, move: Move) -> bool:
        if self.status is GameStatus.FINISHED:
            return False
        if not 0 <= move.column < COLUMNS:
            return False
        return self.board[0][move.column] is None

    def count_lines(self, player: Player) -> int:
        if not isinstance(player, Player):
            raise ValueError("player must be a Player")

        return _count_lines(self.board, player)

    def line_counts(self) -> dict[Player, int]:
        return _line_counts(self.board)

    def apply_move(self, move: Move) -> "GameState":
        column = move.column
        if (
            self.status is GameStatus.FINISHED
            or not 0 <= column < COLUMNS
            or self.board[0][column] is not None
        ):
            raise IllegalMoveError(f"illegal move: {move.move_type.value} in column {column}")

        mover = self.current_player
        if move.move_type is MoveType.DROP:
            next_board, affected = _apply_drop(self.board, column, mover)
        elif move.move_type is MoveType.PUSH:
            next_board, affected = _apply_push(self.board, column, mover)
        else:
            raise IllegalMoveError(f"unsupported move type: {move.move_type}")

        next_status, result = self._status_and_result(next_board, mover, affected)
        return self._successor(next_board, mover.opponent, next_status, result)

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

        if not isinstance(self.first_player, Player):
            raise ValueError("first_player must be a Player")

        if not isinstance(self.status, GameStatus):
            raise ValueError("status must be a GameStatus")

        if self.result is not None and not isinstance(self.result, GameResult):
            raise ValueError("result must be a GameResult or None")

        if self.status is GameStatus.FINISHED and self.result is None:
            raise ValueError("finished game must have a result")

        if self.status is not GameStatus.FINISHED and self.result is not None:
            raise ValueError("unfinished game cannot have a result")

    def _status_and_result(
        self,
        board: Board,
        player_making_move: Player,
        affected: tuple[tuple[int, int], ...],
    ) -> tuple[GameStatus, GameResult | None]:
        if self.status is GameStatus.FAIR_TURN:
            return GameStatus.FINISHED, GameResult.from_line_counts(_line_counts(board))

        has_any_line = _line_exists_through(board, affected)
        if has_any_line and player_making_move is self.first_player:
            return GameStatus.FAIR_TURN, None
        if has_any_line or _is_board_full(board):
            return GameStatus.FINISHED, GameResult.from_line_counts(_line_counts(board))
        return GameStatus.ONGOING, None

    def _successor(
        self,
        board: Board,
        current_player: Player,
        status: GameStatus,
        result: GameResult | None,
    ) -> "GameState":
        # Build the successor without re-running ``__post_init__``: it is valid
        # by construction and this avoids per-move validation overhead.
        successor = object.__new__(GameState)
        object.__setattr__(successor, "board", board)
        object.__setattr__(successor, "current_player", current_player)
        object.__setattr__(successor, "first_player", self.first_player)
        object.__setattr__(successor, "status", status)
        object.__setattr__(successor, "move_count", self.move_count + 1)
        object.__setattr__(successor, "result", result)
        return successor

    @staticmethod
    def _validate_column(column: int) -> None:
        if not 0 <= column < COLUMNS:
            raise ValueError(f"column must be between 0 and {COLUMNS - 1}")


def _apply_drop(board: Board, column: int, mover: Player) -> tuple[Board, tuple[tuple[int, int], ...]]:
    for row_index in range(ROWS - 1, -1, -1):
        if board[row_index][column] is None:
            old_row = board[row_index]
            new_row = old_row[:column] + (mover,) + old_row[column + 1:]
            new_board = board[:row_index] + (new_row,) + board[row_index + 1:]
            return new_board, ((row_index, column),)

    raise IllegalMoveError(f"column {column} is full")


def _apply_push(board: Board, column: int, mover: Player) -> tuple[Board, tuple[tuple[int, int], ...]]:
    # A push inserts a token at the bottom of the column, shifting the existing
    # tokens in that column up by one. Only ``column`` changes in each row.
    new_rows = []
    for row_index in range(ROWS):
        value = mover if row_index == ROWS - 1 else board[row_index + 1][column]
        old_row = board[row_index]
        new_rows.append(old_row[:column] + (value,) + old_row[column + 1:])

    affected = tuple((row_index, column) for row_index in range(ROWS))
    return tuple(new_rows), affected


def _line_exists_through(board: Board, cells: tuple[tuple[int, int], ...]) -> bool:
    """Return whether any four-in-a-row passes through one of ``cells``.

    Uses a run-length count in both directions from each affected cell, so it is
    independent of which player owns the cell.
    """
    for row, column in cells:
        cell = board[row][column]
        if cell is None:
            continue

        for row_step, column_step in _DIRECTIONS:
            count = 1

            r, c = row + row_step, column + column_step
            while 0 <= r < ROWS and 0 <= c < COLUMNS and board[r][c] is cell:
                count += 1
                r += row_step
                c += column_step

            r, c = row - row_step, column - column_step
            while 0 <= r < ROWS and 0 <= c < COLUMNS and board[r][c] is cell:
                count += 1
                r -= row_step
                c -= column_step

            if count >= 4:
                return True

    return False


def _is_board_full(board: Board) -> bool:
    return all(cell is not None for cell in board[0])


def _line_counts(board: Board) -> dict[Player, int]:
    return {
        Player.RED: _count_lines(board, Player.RED),
        Player.YELLOW: _count_lines(board, Player.YELLOW),
    }


def _count_lines(board: Board, player: Player) -> int:
    count = 0
    for row in range(ROWS):
        board_row = board[row]
        for column in range(COLUMNS):
            if board_row[column] is not player:
                continue
            count += _count_lines_from_cell(board, player, row, column)
    return count


def _count_lines_from_cell(board: Board, player: Player, row: int, column: int) -> int:
    return sum(
        1
        for row_step, column_step in _DIRECTIONS
        if _has_line(board, player, row, column, row_step, column_step)
    )


def _has_line(board: Board, player: Player, row: int, column: int, row_step: int, column_step: int) -> bool:
    end_row = row + row_step * 3
    end_column = column + column_step * 3
    if not (0 <= end_row < ROWS and 0 <= end_column < COLUMNS):
        return False

    return all(board[row + row_step * offset][column + column_step * offset] is player for offset in range(4))
