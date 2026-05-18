import pytest

from connect4_mcts.game import COLUMNS, ROWS, GameState, GameStatus, Move, MoveType, Player, empty_board


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
