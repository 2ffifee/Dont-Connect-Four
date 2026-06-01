import pytest

from connect4_mcts.game import GameResult, GameState, GameStatus, Move, MoveType, Player
from connect4_mcts.players import MoveSelectionError
from connect4_mcts.players.mcts import LGRMemory, MCTSPlayer


def board_from_rows(*rows: str):
    player_by_symbol = {".": None, "R": Player.RED, "Y": Player.YELLOW}
    return tuple(tuple(player_by_symbol[symbol] for symbol in row) for row in rows)


def test_mcts_player_returns_legal_move() -> None:
    state = GameState.new()
    player = MCTSPlayer(iterations=80, seed=1)

    move = player.choose_move(state)

    assert move in state.legal_moves()


def test_mcts_player_is_reproducible_for_seed() -> None:
    state = GameState.new()

    first = MCTSPlayer(iterations=120, seed=21).choose_move(state)
    second = MCTSPlayer(iterations=120, seed=21).choose_move(state)

    assert first == second


def test_mcts_player_keeps_persistent_tree_across_moves() -> None:
    player = MCTSPlayer(iterations=60, seed=3)
    state = GameState.new()
    assert player.tree_size == 0

    move = player.choose_move(state)
    after_first = player.tree_size
    assert after_first > 0

    # The state reached after playing this move is already known to the tree, so
    # the next search reuses and extends the existing statistics.
    next_state = state.apply_move(move)
    assert next_state in player.tree

    player.choose_move(next_state)
    assert player.tree_size >= after_first


def test_mcts_player_rejects_terminal_state_without_moves() -> None:
    state = GameState(
        board=GameState.new().board,
        status=GameStatus.FINISHED,
        result=GameResult(winner=None, red_lines=0, yellow_lines=0),
    )
    player = MCTSPlayer(iterations=10, seed=1)

    with pytest.raises(MoveSelectionError, match="no legal moves"):
        player.choose_move(state)


def test_mcts_player_returns_legal_move_with_restricted_options() -> None:
    state = GameState.new()
    # Fill the first column completely so it offers no legal moves.
    for _ in range(6):
        state = state.apply_move(Move(MoveType.DROP, 0))

    player = MCTSPlayer(iterations=60, seed=2)
    move = player.choose_move(state)

    assert move in state.legal_moves()
    assert move.column != 0


def test_mcts_player_avoids_immediate_losing_move() -> None:
    # RED is the second player; completing its own 4-line ends the game with
    # RED holding more lines, which means RED loses the suicide variant.
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
    player = MCTSPlayer(iterations=200, final_move="max_value", seed=1)

    move = player.choose_move(state)

    assert move.column != 3


def test_mcts_player_validates_hyperparameters() -> None:
    with pytest.raises(ValueError, match="iterations"):
        MCTSPlayer(iterations=0)
    with pytest.raises(ValueError, match="power_mean_p"):
        MCTSPlayer(power_mean_p=0)
    with pytest.raises(ValueError, match="rollout_policy"):
        MCTSPlayer(rollout_policy="greedy")
    with pytest.raises(ValueError, match="final_move"):
        MCTSPlayer(final_move="best")


def test_lgr_policy_creates_memory_by_default() -> None:
    player = MCTSPlayer(iterations=10, rollout_policy="lgr", seed=1)

    assert isinstance(player.lgr_memory, LGRMemory)


def test_lgr_memory_records_only_winner_replies() -> None:
    memory = LGRMemory()
    history = [
        (Player.RED, None, Move(MoveType.DROP, 0)),
        (Player.YELLOW, Move(MoveType.DROP, 0), Move(MoveType.PUSH, 1)),
        (Player.RED, Move(MoveType.PUSH, 1), Move(MoveType.DROP, 2)),
    ]

    memory.update_from_game(history, winner=Player.RED)

    assert memory.reply_for(Player.RED, None) == Move(MoveType.DROP, 0)
    assert memory.reply_for(Player.RED, Move(MoveType.PUSH, 1)) == Move(MoveType.DROP, 2)
    assert memory.reply_for(Player.YELLOW, Move(MoveType.DROP, 0)) is None


def test_lgr_memory_update_ignores_draws() -> None:
    memory = LGRMemory()
    history = [(Player.RED, None, Move(MoveType.DROP, 0))]

    memory.update_from_game(history, winner=None)

    assert len(memory) == 0
