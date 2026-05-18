import pytest

from connect4_mcts.game import (
    COLUMNS,
    ROWS,
    GameState,
    GameStatus,
    IllegalMoveError,
    Move,
    MoveType,
    Player,
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
    assert state.status is GameStatus.ONGOING
    assert state.move_count == 0


def test_new_game_can_start_with_yellow_player() -> None:
    state = GameState.new(first_player=Player.YELLOW)

    assert state.current_player is Player.YELLOW


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
    state = GameState(board=empty_board(), status=GameStatus.FINISHED)

    assert state.legal_moves() == ()
