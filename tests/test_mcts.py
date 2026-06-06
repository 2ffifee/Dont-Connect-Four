import pytest

from connect4_mcts.game import GameResult, GameState, GameStatus, Move, MoveType, Player
from connect4_mcts.players import MoveSelectionError
from connect4_mcts.players.mcts import LGRMemory, MCTSPlayer, SearchEvaluation


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


def test_evaluate_reports_root_and_move_values() -> None:
    state = GameState.new()
    player = MCTSPlayer(iterations=200, seed=7)

    evaluation = player.evaluate(state)

    assert isinstance(evaluation, SearchEvaluation)
    assert evaluation.player_to_move is state.current_player
    legal = set(state.legal_moves())
    assert set(evaluation.move_values).issubset(legal)
    assert evaluation.best_move in evaluation.move_values
    assert all(0.0 <= value <= 1.0 for value in evaluation.move_values.values())
    # The root value is the value of the best move under search.
    assert evaluation.root_value == max(evaluation.move_values.values())
    assert evaluation.regret_of(evaluation.best_move) == 0.0


def test_evaluate_flags_losing_move_as_blunder() -> None:
    # RED (second player) completing its own 4-line loses the suicide variant,
    # so dropping into column 3 should be valued far below the best move.
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
    player = MCTSPlayer(iterations=400, seed=1)

    evaluation = player.evaluate(state)
    losing_move = Move(MoveType.DROP, 3)

    # Completing the line is an immediate, certain loss for RED.
    assert evaluation.value_of(losing_move) is not None
    assert evaluation.value_of(losing_move) < 0.1
    assert evaluation.value_of(losing_move) < evaluation.root_value
    assert evaluation.best_move != losing_move


def test_search_evaluation_blunder_and_regret() -> None:
    best = Move(MoveType.DROP, 0)
    mediocre = Move(MoveType.DROP, 1)
    losing = Move(MoveType.PUSH, 2)
    evaluation = SearchEvaluation(
        player_to_move=Player.RED,
        root_value=0.9,
        best_move=best,
        move_values={best: 0.9, mediocre: 0.7, losing: 0.2},
        move_visits={best: 100, mediocre: 40, losing: 5},
    )

    assert evaluation.regret_of(best) == 0.0
    assert evaluation.regret_of(mediocre) == pytest.approx(0.2)
    assert evaluation.regret_of(Move(MoveType.DROP, 7)) is None
    assert not evaluation.is_blunder(mediocre, threshold=0.3)
    assert evaluation.is_blunder(losing, threshold=0.3)


def test_evaluate_rejects_state_without_moves() -> None:
    state = GameState(
        board=GameState.new().board,
        status=GameStatus.FINISHED,
        result=GameResult(winner=None, red_lines=0, yellow_lines=0),
    )
    player = MCTSPlayer(iterations=10, seed=1)

    with pytest.raises(MoveSelectionError, match="no legal moves"):
        player.evaluate(state)


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
