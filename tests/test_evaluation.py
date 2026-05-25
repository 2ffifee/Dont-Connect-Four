import pytest

from connect4_mcts.evaluation import evaluate_position
from connect4_mcts.game import COLUMNS, ROWS, GameResult, GameState, GameStatus, Move, MoveType, Player


def test_terminal_win_is_scored_positive_for_winner() -> None:
    state = GameState(
        board=empty_board(),
        status=GameStatus.FINISHED,
        result=GameResult(winner=Player.RED, red_lines=0, yellow_lines=1),
    )

    assert evaluate_position(state, Player.RED) > 0


def test_terminal_loss_is_scored_negative_for_loser() -> None:
    state = GameState(
        board=empty_board(),
        status=GameStatus.FINISHED,
        result=GameResult(winner=Player.YELLOW, red_lines=1, yellow_lines=0),
    )

    assert evaluate_position(state, Player.RED) < 0


def test_terminal_draw_is_scored_neutral() -> None:
    state = GameState(
        board=empty_board(),
        status=GameStatus.FINISHED,
        result=GameResult(winner=None, red_lines=1, yellow_lines=1),
    )

    assert evaluate_position(state, Player.RED) == 0
    assert evaluate_position(state, Player.YELLOW) == 0


def test_terminal_score_uses_line_count_margin() -> None:
    narrow_win = GameState(
        board=empty_board(),
        status=GameStatus.FINISHED,
        result=GameResult(winner=Player.RED, red_lines=0, yellow_lines=1),
    )
    larger_win = GameState(
        board=empty_board(),
        status=GameStatus.FINISHED,
        result=GameResult(winner=Player.RED, red_lines=0, yellow_lines=3),
    )

    assert evaluate_position(larger_win, Player.RED) > evaluate_position(narrow_win, Player.RED)
    assert evaluate_position(larger_win, Player.YELLOW) < evaluate_position(narrow_win, Player.YELLOW)


def test_fair_turn_score_resolves_required_response() -> None:
    state = GameState(
        board=board_from_rows(
            "........",
            "........",
            "........",
            "........",
            "........",
            "RRR.YYY.",
        ),
        current_player=Player.RED,
        first_player=Player.RED,
    )

    fair_turn = state.apply_move(Move(MoveType.DROP, 3))

    assert fair_turn.status is GameStatus.FAIR_TURN
    assert fair_turn.result is None
    assert evaluate_position(fair_turn, Player.RED) < 0
    assert evaluate_position(fair_turn, Player.YELLOW) > 0


def test_own_three_token_window_is_worse_than_empty_position() -> None:
    risky_state = GameState(
        board=board_from_rows(
            "........",
            "........",
            "........",
            "........",
            "........",
            "RRR.....",
        )
    )
    empty_state = GameState.new()

    assert evaluate_position(risky_state, Player.RED) < evaluate_position(empty_state, Player.RED)


def test_opponent_three_token_window_is_better_than_empty_position() -> None:
    favorable_state = GameState(
        board=board_from_rows(
            "........",
            "........",
            "........",
            "........",
            "........",
            "YYY.....",
        )
    )
    empty_state = GameState.new()

    assert evaluate_position(favorable_state, Player.RED) > evaluate_position(empty_state, Player.RED)


def test_blocked_window_does_not_score_like_open_window() -> None:
    open_state = GameState(
        board=board_from_rows(
            "........",
            "........",
            "........",
            "........",
            "........",
            "RRR.....",
        )
    )
    blocked_state = GameState(
        board=board_from_rows(
            "........",
            "........",
            "........",
            "........",
            "........",
            "RRRY....",
        )
    )

    assert evaluate_position(blocked_state, Player.RED) > evaluate_position(open_state, Player.RED)


def test_evaluate_position_rejects_non_player_perspective() -> None:
    with pytest.raises(ValueError, match="player"):
        evaluate_position(GameState.new(), "red")  # type: ignore[arg-type]


def empty_board():
    return tuple(tuple(None for _ in range(COLUMNS)) for _ in range(ROWS))


def board_from_rows(*rows: str):
    player_by_symbol = {
        ".": None,
        "R": Player.RED,
        "Y": Player.YELLOW,
    }
    return tuple(tuple(player_by_symbol[symbol] for symbol in row) for row in rows)
