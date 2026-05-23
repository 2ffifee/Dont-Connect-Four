import pytest

from connect4_mcts.cli import format_move, parse_move, render_board, result_message, status_message
from connect4_mcts.game import GameResult, GameState, GameStatus, Move, MoveType, Player


def test_parse_move_accepts_short_and_long_move_types() -> None:
    assert parse_move("d 1") == Move(MoveType.DROP, 0)
    assert parse_move("drop 8") == Move(MoveType.DROP, 7)
    assert parse_move("p 2") == Move(MoveType.PUSH, 1)
    assert parse_move("push 7") == Move(MoveType.PUSH, 6)


def test_parse_move_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="Move"):
        parse_move("x 1")

    with pytest.raises(ValueError, match="Column"):
        parse_move("d 9")

    with pytest.raises(ValueError, match="Enter"):
        parse_move("d")


def test_render_board_shows_cells_and_column_numbers() -> None:
    state = GameState.new()
    state = state.apply_move(Move(MoveType.DROP, 0))
    state = state.apply_move(Move(MoveType.DROP, 1))

    rendered = render_board(state)

    assert "|R|Y|.|.|.|.|.|.|" in rendered
    assert "1 2 3 4 5 6 7 8" in rendered


def test_status_message_marks_fair_turn() -> None:
    state = GameState(board=GameState.new().board, current_player=Player.YELLOW, status=GameStatus.FAIR_TURN)

    assert status_message(state) == "Turn: yellow (fair turn)"


def test_result_message_describes_winner_and_line_counts() -> None:
    result = GameResult(winner=Player.RED, red_lines=0, yellow_lines=2)

    assert result_message(result) == "Winner: red. Lines: red=0, yellow=2"


def test_result_message_describes_draw() -> None:
    result = GameResult(winner=None, red_lines=1, yellow_lines=1)

    assert result_message(result) == "Draw. Lines: red=1, yellow=1"


def test_format_move_uses_one_based_columns() -> None:
    assert format_move(Move(MoveType.DROP, 0)) == "d 1"
    assert format_move(Move(MoveType.PUSH, 7)) == "p 8"
