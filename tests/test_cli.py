import pytest

import connect4_mcts.cli as cli
from connect4_mcts.cli import create_agent, format_move, parse_move, render_board, result_message, status_message
from connect4_mcts.game import GameResult, GameState, GameStatus, Move, MoveType, Player
from connect4_mcts.players import MinimaxPlayer, RandomPlayer


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


def test_status_message_describes_current_player() -> None:
    state = GameState(board=GameState.new().board, current_player=Player.YELLOW)

    assert status_message(state) == "Turn: yellow"


def test_result_message_describes_winner_and_line_counts() -> None:
    result = GameResult(winner=Player.RED, red_lines=0, yellow_lines=2)

    assert result_message(result) == "Winner: red. Lines: red=0, yellow=2"


def test_result_message_describes_draw() -> None:
    result = GameResult(winner=None, red_lines=1, yellow_lines=1)

    assert result_message(result) == "Draw. Lines: red=1, yellow=1"


def test_format_move_uses_one_based_columns() -> None:
    assert format_move(Move(MoveType.DROP, 0)) == "d 1"
    assert format_move(Move(MoveType.PUSH, 7)) == "p 8"


def test_create_agent_builds_random_player() -> None:
    agent = create_agent("random", seed=1)

    assert isinstance(agent, RandomPlayer)


def test_create_agent_builds_minimax_player() -> None:
    agent = create_agent("minimax", depth=2)

    assert isinstance(agent, MinimaxPlayer)
    assert agent.depth == 2


def test_create_agent_rejects_unknown_agent() -> None:
    with pytest.raises(ValueError, match="unknown agent"):
        create_agent("unknown")


def test_main_passes_interactive_agent_options(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_run_human_vs_agent(human: Player, agent_name: str, seed: int | None, depth: int) -> GameState:
        calls.append((human, agent_name, seed, depth))
        return GameState.new()

    monkeypatch.setattr(cli, "run_human_vs_agent", fake_run_human_vs_agent)

    assert cli.main(["--human", "yellow", "--agent", "minimax", "--depth", "2", "--seed", "7"]) == 0
    assert calls == [(Player.YELLOW, "minimax", 7, 2)]


def test_main_passes_demo_agent_options(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_run_demo(red_agent: str, yellow_agent: str, seed: int | None, depth: int) -> GameState:
        calls.append((red_agent, yellow_agent, seed, depth))
        return GameState.new()

    monkeypatch.setattr(cli, "run_demo", fake_run_demo)

    assert cli.main(["--demo", "--red", "minimax", "--yellow", "random", "--depth", "1", "--seed", "5"]) == 0
    assert calls == [("minimax", "random", 5, 1)]
