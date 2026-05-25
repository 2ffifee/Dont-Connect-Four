import pytest

from connect4_mcts.game import COLUMNS, ROWS, GameResult, GameState, GameStatus, Move, MoveType, Player
from connect4_mcts.players import MoveSelectionError, RandomPlayer
from connect4_mcts.players.minimax import MinimaxPlayer


def test_random_player_returns_legal_move() -> None:
    state = GameState.new()
    player = RandomPlayer(seed=1)

    move = player.choose_move(state)

    assert move in state.legal_moves()


def test_random_player_uses_reproducible_sequence_for_seed() -> None:
    state = GameState.new()
    first_player = RandomPlayer(seed=42)
    second_player = RandomPlayer(seed=42)

    first_sequence = [first_player.choose_move(state) for _ in range(10)]
    second_sequence = [second_player.choose_move(state) for _ in range(10)]

    assert first_sequence == second_sequence


def test_random_player_handles_restricted_legal_moves() -> None:
    state = GameState.new()
    for _ in range(6):
        state = state.apply_move(Move(MoveType.DROP, 0))

    player = RandomPlayer(seed=7)

    for _ in range(20):
        assert player.choose_move(state) in state.legal_moves()


def test_random_player_rejects_terminal_state_without_moves() -> None:
    state = GameState(
        board=GameState.new().board,
        status=GameStatus.FINISHED,
        result=GameResult(winner=None, red_lines=0, yellow_lines=0),
    )
    player = RandomPlayer(seed=1)

    with pytest.raises(MoveSelectionError, match="no legal moves"):
        player.choose_move(state)


def test_minimax_player_returns_legal_move() -> None:
    state = GameState.new()
    player = MinimaxPlayer(depth=1)

    move = player.choose_move(state)

    assert move in state.legal_moves()


def test_minimax_player_rejects_terminal_state_without_moves() -> None:
    state = GameState(
        board=GameState.new().board,
        status=GameStatus.FINISHED,
        result=GameResult(winner=None, red_lines=0, yellow_lines=0),
    )
    player = MinimaxPlayer(depth=1)

    with pytest.raises(MoveSelectionError, match="no legal moves"):
        player.choose_move(state)


def test_minimax_player_rejects_negative_depth() -> None:
    with pytest.raises(ValueError, match="depth"):
        MinimaxPlayer(depth=-1)


def test_minimax_player_depth_zero_uses_evaluator_on_candidate_positions() -> None:
    seen_states: list[GameState] = []

    def evaluator(state: GameState, player: Player) -> int:
        seen_states.append(state)
        return 0

    player = MinimaxPlayer(depth=0, evaluator=evaluator)

    move = player.choose_move(GameState.new())

    assert move == Move(MoveType.DROP, 0)
    assert len(seen_states) == COLUMNS * 2
    assert all(state.move_count == 1 for state in seen_states)


def test_minimax_player_avoids_obvious_immediate_loss() -> None:
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
        first_player=Player.YELLOW,
    )
    player = MinimaxPlayer(depth=1)

    move = player.choose_move(state)

    assert move.column != 3


def test_minimax_player_selects_immediate_winning_move() -> None:
    state = GameState(
        board=board_from_rows(
            "........",
            "........",
            "........",
            "........",
            "YYY.....",
            "...Y....",
        ),
        current_player=Player.RED,
        first_player=Player.YELLOW,
    )
    player = MinimaxPlayer(depth=1)

    move = player.choose_move(state)

    assert move == Move(MoveType.PUSH, 3)


def board_from_rows(*rows: str):
    player_by_symbol = {
        ".": None,
        "R": Player.RED,
        "Y": Player.YELLOW,
    }
    return tuple(tuple(player_by_symbol[symbol] for symbol in row) for row in rows)
