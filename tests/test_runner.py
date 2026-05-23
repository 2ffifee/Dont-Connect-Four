import pytest

from connect4_mcts.game import GameState, GameStatus, IllegalMoveError, Move, MoveType, Player
from connect4_mcts.players import RandomPlayer
from connect4_mcts.runner import GameRunnerError, play_game


def test_play_game_runs_random_players_to_terminal_state() -> None:
    played_game = play_game(RandomPlayer(seed=1), RandomPlayer(seed=2))

    assert played_game.final_state.status is GameStatus.FINISHED
    assert played_game.final_state.result is not None
    assert len(played_game.moves) == played_game.final_state.move_count


def test_play_game_records_players_and_move_numbers() -> None:
    played_game = play_game(RandomPlayer(seed=10), RandomPlayer(seed=11), max_moves=200)

    assert played_game.moves[0].player is Player.RED
    assert played_game.moves[0].move_number == 1
    assert [record.move_number for record in played_game.moves] == list(range(1, len(played_game.moves) + 1))


def test_play_game_uses_initial_state_current_player() -> None:
    initial_state = GameState.new(first_player=Player.YELLOW)

    played_game = play_game(RandomPlayer(seed=4), RandomPlayer(seed=5), initial_state=initial_state)

    assert played_game.moves[0].player is Player.YELLOW


def test_play_game_rejects_illegal_agent_move() -> None:
    class IllegalAgent:
        def choose_move(self, state: GameState) -> Move:
            return Move(MoveType.DROP, 0)

    state = GameState.new()
    for _ in range(6):
        state = state.apply_move(Move(MoveType.DROP, 0))

    with pytest.raises(IllegalMoveError, match="illegal move"):
        play_game(IllegalAgent(), RandomPlayer(seed=1), initial_state=state)


def test_play_game_respects_max_moves() -> None:
    with pytest.raises(GameRunnerError, match="max_moves"):
        play_game(RandomPlayer(seed=1), RandomPlayer(seed=2), max_moves=1)


def test_play_game_records_moves_that_were_legal_when_played() -> None:
    played_game = play_game(RandomPlayer(seed=20), RandomPlayer(seed=21), max_moves=200)
    state = GameState.new()

    for record in played_game.moves:
        assert record.player is state.current_player
        assert record.move in state.legal_moves()
        state = state.apply_move(record.move)

    assert state == played_game.final_state
