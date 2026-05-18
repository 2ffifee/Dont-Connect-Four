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

        moves: list[Move] = []
        for column in range(COLUMNS):
            if not self.is_column_full(column):
                moves.append(Move(MoveType.DROP, column))
                moves.append(Move(MoveType.PUSH, column))
        return tuple(moves)

    def is_legal_move(self, move: Move) -> bool:
        return move in self.legal_moves()

    def count_lines(self, player: Player) -> int:
        if not isinstance(player, Player):
            raise ValueError("player must be a Player")

        return _count_lines(self.board, player)

    def line_counts(self) -> dict[Player, int]:
        return {
            Player.RED: self.count_lines(Player.RED),
            Player.YELLOW: self.count_lines(Player.YELLOW),
        }

    def apply_move(self, move: Move) -> "GameState":
        if not self.is_legal_move(move):
            raise IllegalMoveError(f"illegal move: {move.move_type.value} in column {move.column}")

        player_making_move = self.current_player
        if move.move_type is MoveType.DROP:
            next_board = self._apply_drop(move.column)
        elif move.move_type is MoveType.PUSH:
            next_board = self._apply_push(move.column)
        else:
            raise IllegalMoveError(f"unsupported move type: {move.move_type}")

        return GameState(
            board=next_board,
            current_player=player_making_move.opponent,
            first_player=self.first_player,
            status=self._status_after_move(next_board, player_making_move),
            move_count=self.move_count + 1,
            result=self._result_after_move(next_board, player_making_move),
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

    def _status_after_move(self, board: Board, player_making_move: Player) -> GameStatus:
        if self.status is GameStatus.FAIR_TURN:
            return GameStatus.FINISHED

        line_counts = self._line_counts_for_board(board)
        has_any_line = any(count > 0 for count in line_counts.values())
        if has_any_line and player_making_move is self.first_player:
            return GameStatus.FAIR_TURN
        if has_any_line or self._is_board_full(board):
            return GameStatus.FINISHED
        return GameStatus.ONGOING

    def _result_after_move(self, board: Board, player_making_move: Player) -> GameResult | None:
        next_status = self._status_after_move(board, player_making_move)
        if next_status is not GameStatus.FINISHED:
            return None

        return GameResult.from_line_counts(self._line_counts_for_board(board))

    def _line_counts_for_board(self, board: Board) -> dict[Player, int]:
        return {
            Player.RED: _count_lines(board, Player.RED),
            Player.YELLOW: _count_lines(board, Player.YELLOW),
        }

    @staticmethod
    def _is_board_full(board: Board) -> bool:
        return all(cell is not None for cell in board[0])


def _freeze_board(rows: list[list[Cell]]) -> Board:
    return tuple(tuple(row) for row in rows)


def _count_lines(board: Board, player: Player) -> int:
    count = 0
    for row in range(ROWS):
        for column in range(COLUMNS):
            count += _count_lines_from_cell(board, player, row, column)
    return count


def _count_lines_from_cell(board: Board, player: Player, row: int, column: int) -> int:
    directions = ((0, 1), (1, 0), (1, 1), (1, -1))
    return sum(1 for row_step, column_step in directions if _has_line(board, player, row, column, row_step, column_step))


def _has_line(board: Board, player: Player, row: int, column: int, row_step: int, column_step: int) -> bool:
    end_row = row + row_step * 3
    end_column = column + column_step * 3
    if not (0 <= end_row < ROWS and 0 <= end_column < COLUMNS):
        return False

    return all(board[row + row_step * offset][column + column_step * offset] is player for offset in range(4))
