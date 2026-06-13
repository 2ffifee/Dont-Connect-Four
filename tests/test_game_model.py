import pytest

from connect4_mcts.game import (
    COLUMNS,
    ROWS,
    GameState,
    GameResult,
    GameStatus,
    IllegalMoveError,
    Move,
    MoveType,
    Player,
    _winner_from_counts,
    empty_board,
)


def test_empty_board_has_expected_dimensions_and_empty_cells() -> None:
    board = empty_board()

    assert len(board) == ROWS
    assert all(len(row) == COLUMNS for row in board)
    assert all(cell is None for row in board for cell in row)


def test_new_game_starts_with_red_player_on_empty_board() -> None:
    state = GameState.new()

    assert state.board == empty_board()
    assert state.current_player is Player.RED
    assert state.first_player is Player.RED
    assert state.status is GameStatus.ONGOING
    assert state.move_count == 0
    assert state.result is None


def test_new_game_can_start_with_yellow_player() -> None:
    state = GameState.new(first_player=Player.YELLOW)

    assert state.current_player is Player.YELLOW
    assert state.first_player is Player.YELLOW


def test_players_have_opponents() -> None:
    assert Player.RED.opponent is Player.YELLOW
    assert Player.YELLOW.opponent is Player.RED


def test_move_rejects_column_outside_board() -> None:
    with pytest.raises(ValueError, match="column"):
        Move(MoveType.DROP, COLUMNS)

    with pytest.raises(ValueError, match="column"):
        Move(MoveType.PUSH, -1)


def test_game_state_rejects_invalid_board_shape() -> None:
    board_with_missing_column = tuple(tuple(None for _ in range(COLUMNS - 1)) for _ in range(ROWS))

    with pytest.raises(ValueError, match="columns"):
        GameState(board=board_with_missing_column)


def test_new_game_has_drop_and_push_moves_for_each_column() -> None:
    state = GameState.new()

    assert state.legal_moves() == tuple(
        Move(move_type, column)
        for column in range(COLUMNS)
        for move_type in (MoveType.DROP, MoveType.PUSH)
    )


def test_apply_drop_places_token_in_lowest_empty_cell() -> None:
    state = GameState.new()

    after_red = state.apply_move(Move(MoveType.DROP, 2))
    after_yellow = after_red.apply_move(Move(MoveType.DROP, 2))

    assert after_red.board[ROWS - 1][2] is Player.RED
    assert after_yellow.board[ROWS - 2][2] is Player.YELLOW
    assert after_yellow.board[ROWS - 1][2] is Player.RED
    assert after_yellow.current_player is Player.RED
    assert after_yellow.move_count == 2


def test_apply_push_inserts_token_at_bottom_and_shifts_column_up() -> None:
    state = GameState.new()
    state = state.apply_move(Move(MoveType.DROP, 3))
    state = state.apply_move(Move(MoveType.DROP, 3))

    after_push = state.apply_move(Move(MoveType.PUSH, 3))

    assert after_push.board[ROWS - 3][3] is Player.YELLOW
    assert after_push.board[ROWS - 2][3] is Player.RED
    assert after_push.board[ROWS - 1][3] is Player.RED
    assert after_push.current_player is Player.YELLOW
    assert after_push.move_count == 3


def test_full_column_has_no_legal_drop_or_push_moves() -> None:
    state = GameState.new()
    for _ in range(ROWS):
        state = state.apply_move(Move(MoveType.DROP, 0))

    assert state.is_column_full(0)
    assert Move(MoveType.DROP, 0) not in state.legal_moves()
    assert Move(MoveType.PUSH, 0) not in state.legal_moves()


def test_apply_move_rejects_moves_in_full_column() -> None:
    state = GameState.new()
    for _ in range(ROWS):
        state = state.apply_move(Move(MoveType.DROP, 0))

    with pytest.raises(IllegalMoveError, match="illegal move"):
        state.apply_move(Move(MoveType.DROP, 0))

    with pytest.raises(IllegalMoveError, match="illegal move"):
        state.apply_move(Move(MoveType.PUSH, 0))


def test_finished_game_has_no_legal_moves() -> None:
    state = GameState(board=empty_board(), status=GameStatus.FINISHED, result=GameResult(None, 0, 0))

    assert state.legal_moves() == ()


def test_count_lines_returns_zero_on_empty_board() -> None:
    state = GameState.new()

    assert state.count_lines(Player.RED) == 0
    assert state.count_lines(Player.YELLOW) == 0
    assert state.line_counts() == {Player.RED: 0, Player.YELLOW: 0}


def test_count_lines_detects_horizontal_line() -> None:
    board = board_from_rows(
        "........",
        "........",
        "........",
        "........",
        "........",
        "RRRR....",
    )

    assert GameState(board=board).count_lines(Player.RED) == 1


def test_count_lines_detects_vertical_line() -> None:
    board = board_from_rows(
        "........",
        "........",
        "Y.......",
        "Y.......",
        "Y.......",
        "Y.......",
    )

    assert GameState(board=board).count_lines(Player.YELLOW) == 1


def test_count_lines_detects_diagonal_down_right_line() -> None:
    board = board_from_rows(
        "........",
        "........",
        ".R......",
        "..R.....",
        "...R....",
        "....R...",
    )

    assert GameState(board=board).count_lines(Player.RED) == 1


def test_count_lines_detects_diagonal_down_left_line() -> None:
    board = board_from_rows(
        "........",
        "........",
        "....Y...",
        "...Y....",
        "..Y.....",
        ".Y......",
    )

    assert GameState(board=board).count_lines(Player.YELLOW) == 1


def test_count_lines_counts_overlapping_lines() -> None:
    board = board_from_rows(
        "........",
        "........",
        "........",
        "........",
        "........",
        "RRRRR...",
    )

    assert GameState(board=board).count_lines(Player.RED) == 2


def test_line_counts_include_both_players() -> None:
    board = board_from_rows(
        "........",
        "........",
        "Y.......",
        "Y.......",
        "Y.......",
        "YRRRR...",
    )

    assert GameState(board=board).line_counts() == {
        Player.RED: 1,
        Player.YELLOW: 1,
    }


def test_own_line_completes_immediate_loss() -> None:
    state = GameState(
        board=board_from_rows(
            "........",
            "........",
            "........",
            "........",
            "........",
            "RRR.....",
        ),
        current_player=Player.RED,
        first_player=Player.RED,
    )

    after_move = state.apply_move(Move(MoveType.DROP, 3))

    assert after_move.status is GameStatus.FINISHED
    assert after_move.result is not None
    assert after_move.result.winner is Player.YELLOW
    assert after_move.line_counts()[Player.RED] == 1
    assert after_move.protected_segments


def test_opponent_line_formed_is_immediate_loss_for_line_owner() -> None:
    state = GameState(
        board=board_from_rows(
            "........",
            ".Y......",
            "Y.......",
            "Y.......",
            "Y.......",
            "Y.......",
        ),
        current_player=Player.RED,
        first_player=Player.RED,
    )

    after = state.apply_move(Move(MoveType.PUSH, 0))

    assert after.status is GameStatus.FINISHED
    assert after.result is not None
    assert after.result.winner is Player.RED
    assert after.line_counts()[Player.YELLOW] == 2


def test_count_lines_six_in_a_row_counts_as_three_segments() -> None:
    board = board_from_rows(
        "........",
        "........",
        "........",
        "........",
        "........",
        "RRRRRR..",
    )

    assert GameState(board=board).count_lines(Player.RED) == 3


def test_equal_line_counts_is_draw() -> None:
    assert _winner_from_counts({Player.RED: 3, Player.YELLOW: 3}) is None


def test_game_continues_when_move_does_not_complete_a_line() -> None:
    state = GameState(
        board=board_from_rows(
            "........",
            "........",
            "........",
            "........",
            "........",
            "RRRRYYYY",
        ),
        current_player=Player.YELLOW,
        first_player=Player.RED,
    )

    after = state.apply_move(Move(MoveType.DROP, 0))

    assert after.status is GameStatus.ONGOING
    assert after.result is None
    assert after.protected_segments


def test_second_player_own_line_is_immediate_loss() -> None:
    state = GameState(
        board=board_from_rows(
            "........",
            "........",
            "........",
            "........",
            "........",
            "YYY.....",
        ),
        current_player=Player.YELLOW,
        first_player=Player.RED,
    )

    finished = state.apply_move(Move(MoveType.DROP, 3))

    assert finished.status is GameStatus.FINISHED
    assert finished.result.winner is Player.RED
    assert finished.result.red_lines == 0
    assert finished.result.yellow_lines == 1


def test_finished_game_requires_result() -> None:
    with pytest.raises(ValueError, match="result"):
        GameState(board=empty_board(), status=GameStatus.FINISHED)


def test_unfinished_game_rejects_result() -> None:
    with pytest.raises(ValueError, match="result"):
        GameState(board=empty_board(), result=GameResult(None, 0, 0))


def test_game_result_rejects_negative_line_counts() -> None:
    with pytest.raises(ValueError, match="negative"):
        GameResult(winner=None, red_lines=-1, yellow_lines=0)
def test_cumulative_line_total_persists_after_line_is_broken() -> None:
    state = GameState(
        board=board_from_rows(
            "........",
            "........",
            "........",
            "........",
            "R.......",
            "RRR.....",
        ),
        red_line_total=1,
        move_count=2,
        current_player=Player.YELLOW,
    )

    assert state.count_lines(Player.RED) == 0
    assert state.line_counts()[Player.RED] == 1


def test_rebuilt_line_increments_cumulative_total_again() -> None:
    state = GameState(
        board=board_from_rows(
            "........",
            "........",
            "........",
            "........",
            "........",
            "RRR.....",
        ),
        red_line_total=1,
        move_count=3,
        current_player=Player.RED,
        first_player=Player.RED,
    )

    rebuilt = state.apply_move(Move(MoveType.DROP, 3))

    assert rebuilt.count_lines(Player.RED) == 1
    assert rebuilt.line_counts()[Player.RED] == 2


def board_from_rows(*rows: str):
    player_by_symbol = {
        ".": None,
        "R": Player.RED,
        "Y": Player.YELLOW,
    }
    return tuple(tuple(player_by_symbol[symbol] for symbol in row) for row in rows)
