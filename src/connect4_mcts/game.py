"""Core domain model for the modified Connect4 game.

Performance notes
-----------------
``apply_move`` is on the hot path of MCTS rollouts and tournament play, so it is
optimized to:

* check legality in O(1) instead of materializing ``legal_moves()``;
* rebuild only the affected board cells (sharing the immutable row tuples that
  do not change);
* detect newly completed four-in-a-row segments incrementally by diffing the
  board before and after the move;
* construct the successor state without re-running ``__post_init__`` validation
  (the successor is guaranteed valid by construction).

Line scoring
------------
Each distinct four-in-a-row segment counts as one line. A run of length ``n``
(``n >= 4``) contributes ``n - 3`` segments. Completed segments are recorded
cumulatively: destroying a line on the board does not reduce a player's total,
and rebuilding the same segment at the same coordinates counts again.

Immediate loss
--------------
If any four-in-a-row segment is completed on a move, the owner of that segment
loses immediately (regardless of who played the move). If segments for both
players are completed on the same move, the game is a draw. Otherwise play
continues until the board is full (then line totals decide, with equal counts as
a draw).
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
    FINISHED = "finished"


class IllegalMoveError(ValueError):
    """Raised when a move cannot be applied to the current board."""


@dataclass(frozen=True, slots=True)
class GameResult:
    winner: Player | None
    red_lines: int
    yellow_lines: int

    @classmethod
    def from_board(cls, board: Board) -> "GameResult":
        counts = _line_counts(board)
        return cls.from_line_counts(counts)

    @classmethod
    def from_line_counts(cls, line_counts: dict[Player, int]) -> "GameResult":
        return cls(
            winner=_winner_from_counts(line_counts),
            red_lines=line_counts[Player.RED],
            yellow_lines=line_counts[Player.YELLOW],
        )

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


_DROP_MOVES = tuple(Move(MoveType.DROP, column) for column in range(COLUMNS))
_PUSH_MOVES = tuple(Move(MoveType.PUSH, column) for column in range(COLUMNS))


LineSegment = tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class GameState:
    board: Board
    current_player: Player = Player.RED
    first_player: Player = Player.RED
    status: GameStatus = GameStatus.ONGOING
    move_count: int = 0
    result: GameResult | None = None
    red_line_total: int = 0
    yellow_line_total: int = 0
    protected_segments: frozenset[LineSegment] = frozenset()

    @classmethod
    def new(cls, first_player: Player = Player.RED) -> "GameState":
        return cls(board=empty_board(), current_player=first_player, first_player=first_player)

    def is_column_full(self, column: int) -> bool:
        self._validate_column(column)
        return self.board[0][column] is not None

    def legal_moves(self) -> tuple[Move, ...]:
        if self.status is GameStatus.FINISHED:
            return ()
        return self._moves_into_open_columns()

    def is_legal_move(self, move: Move) -> bool:
        if self.status is GameStatus.FINISHED:
            return False
        if not 0 <= move.column < COLUMNS:
            return False
        return self.board[0][move.column] is None

    def _moves_into_open_columns(self) -> tuple[Move, ...]:
        top_row = self.board[0]
        moves: list[Move] = []
        for column in range(COLUMNS):
            if top_row[column] is None:
                moves.append(_DROP_MOVES[column])
                moves.append(_PUSH_MOVES[column])
        return tuple(moves)

    def count_lines(self, player: Player) -> int:
        """Count four-in-a-row segments currently visible on the board."""
        if not isinstance(player, Player):
            raise ValueError("player must be a Player")
        return _count_lines(self.board, player)

    def line_counts(self) -> dict[Player, int]:
        """Return cumulative line totals (segments ever completed)."""
        return {
            Player.RED: self.red_line_total,
            Player.YELLOW: self.yellow_line_total,
        }

    def apply_move(self, move: Move) -> "GameState":
        if not self.is_legal_move(move):
            raise IllegalMoveError(
                f"illegal move: {move.move_type.value} in column {move.column}"
            )

        old_board = self.board
        column = move.column
        mover = self.current_player
        if move.move_type is MoveType.DROP:
            next_board, _ = _apply_drop(old_board, column, mover)
        elif move.move_type is MoveType.PUSH:
            next_board, _ = _apply_push(old_board, column, mover)
        else:
            raise IllegalMoveError(f"unsupported move type: {move.move_type}")

        formed_segments = _newly_formed_segments(old_board, next_board)
        red_total = self.red_line_total
        yellow_total = self.yellow_line_total
        for segment in formed_segments:
            owner = _segment_owner(next_board, segment)
            if owner is Player.RED:
                red_total += 1
            else:
                yellow_total += 1

        next_status, result = self._status_and_result(
            next_board,
            red_total,
            yellow_total,
            formed_segments=formed_segments,
        )
        display_segments = _all_line_segments(next_board)
        return self._successor(
            next_board,
            mover.opponent,
            next_status,
            result,
            red_total,
            yellow_total,
            display_segments,
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

        if self.red_line_total < 0 or self.yellow_line_total < 0:
            raise ValueError("line totals cannot be negative")

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

        if (
            self.move_count == 0
            and self.red_line_total == 0
            and self.yellow_line_total == 0
        ):
            counts = _line_counts(self.board)
            if counts[Player.RED] or counts[Player.YELLOW]:
                object.__setattr__(self, "red_line_total", counts[Player.RED])
                object.__setattr__(self, "yellow_line_total", counts[Player.YELLOW])

    def _status_and_result(
        self,
        board: Board,
        red_total: int,
        yellow_total: int,
        *,
        formed_segments: frozenset[LineSegment] | tuple[LineSegment, ...],
    ) -> tuple[GameStatus, GameResult | None]:
        totals = {Player.RED: red_total, Player.YELLOW: yellow_total}

        if formed_segments:
            red_line_formed = any(
                _segment_owner(board, segment) is Player.RED for segment in formed_segments
            )
            yellow_line_formed = any(
                _segment_owner(board, segment) is Player.YELLOW for segment in formed_segments
            )
            result = GameResult(
                winner=None,
                red_lines=red_total,
                yellow_lines=yellow_total,
            )
            if red_line_formed and yellow_line_formed:
                return GameStatus.FINISHED, result
            if red_line_formed:
                return GameStatus.FINISHED, GameResult(
                    winner=Player.YELLOW,
                    red_lines=red_total,
                    yellow_lines=yellow_total,
                )
            if yellow_line_formed:
                return GameStatus.FINISHED, GameResult(
                    winner=Player.RED,
                    red_lines=red_total,
                    yellow_lines=yellow_total,
                )

        if _is_board_full(board):
            return GameStatus.FINISHED, GameResult.from_line_counts(totals)

        return GameStatus.ONGOING, None

    def _successor(
        self,
        board: Board,
        current_player: Player,
        status: GameStatus,
        result: GameResult | None,
        red_line_total: int,
        yellow_line_total: int,
        protected_segments: frozenset[LineSegment],
    ) -> "GameState":
        successor = object.__new__(GameState)
        object.__setattr__(successor, "board", board)
        object.__setattr__(successor, "current_player", current_player)
        object.__setattr__(successor, "first_player", self.first_player)
        object.__setattr__(successor, "status", status)
        object.__setattr__(successor, "move_count", self.move_count + 1)
        object.__setattr__(successor, "result", result)
        object.__setattr__(successor, "red_line_total", red_line_total)
        object.__setattr__(successor, "yellow_line_total", yellow_line_total)
        object.__setattr__(successor, "protected_segments", protected_segments)
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
    new_rows = []
    for row_index in range(ROWS):
        value = mover if row_index == ROWS - 1 else board[row_index + 1][column]
        old_row = board[row_index]
        new_rows.append(old_row[:column] + (value,) + old_row[column + 1:])

    affected = tuple((row_index, column) for row_index in range(ROWS))
    return tuple(new_rows), affected


def _is_board_full(board: Board) -> bool:
    return all(cell is not None for cell in board[0])


def _line_counts(board: Board) -> dict[Player, int]:
    return {
        Player.RED: _count_lines(board, Player.RED),
        Player.YELLOW: _count_lines(board, Player.YELLOW),
    }


def _segment_positions(
    row: int,
    column: int,
    row_step: int,
    column_step: int,
) -> LineSegment:
    return tuple(
        (row + row_step * offset, column + column_step * offset)
        for offset in range(4)
    )


def _all_line_segments(board: Board) -> frozenset[LineSegment]:
    segments: set[LineSegment] = set()
    for player in (Player.RED, Player.YELLOW):
        for row in range(ROWS):
            board_row = board[row]
            for column in range(COLUMNS):
                if board_row[column] is not player:
                    continue
                for row_step, column_step in _DIRECTIONS:
                    if _has_line(board, player, row, column, row_step, column_step):
                        segments.add(_segment_positions(row, column, row_step, column_step))
    return frozenset(segments)


def _newly_formed_segments(old_board: Board, new_board: Board) -> frozenset[LineSegment]:
    return _all_line_segments(new_board) - _all_line_segments(old_board)


def _segment_owner(board: Board, segment: LineSegment) -> Player:
    player = board[segment[0][0]][segment[0][1]]
    if player is None:
        raise ValueError("line segment must belong to a player")
    return player


def _winner_from_counts(counts: dict[Player, int]) -> Player | None:
    """Fewer completed four-in-a-row segments wins; equal counts are a draw."""
    red_count = counts[Player.RED]
    yellow_count = counts[Player.YELLOW]
    if red_count < yellow_count:
        return Player.RED
    if yellow_count < red_count:
        return Player.YELLOW
    return None


def _count_lines(board: Board, player: Player) -> int:
    """Count every distinct four-in-a-row segment (overlapping segments count separately)."""
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
