import pytest

from connect4_mcts.game import GameResult, GameState, GameStatus, Move, MoveType
from connect4_mcts.players import MoveSelectionError, RandomPlayer


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
