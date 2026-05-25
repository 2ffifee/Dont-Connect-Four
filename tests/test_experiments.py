import pytest

import connect4_mcts.experiments as experiments
from connect4_mcts.experiments import (
    SimulatedGame,
    TimedMoveRecord,
    format_match_summary,
    play_timed_game,
    run_match,
    summarize_games,
)
from connect4_mcts.game import COLUMNS, ROWS, GameResult, GameState, GameStatus, Move, MoveType, Player


class FixedMoveAgent:
    def __init__(self, move: Move) -> None:
        self.move = move

    def choose_move(self, state: GameState) -> Move:
        return self.move


def test_play_timed_game_records_decision_time(monkeypatch: pytest.MonkeyPatch) -> None:
    times = iter([10.0, 10.125])
    monkeypatch.setattr(experiments.time, "perf_counter", lambda: next(times))
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

    game = play_timed_game(
        red_agent_name="fixed-red",
        yellow_agent_name="fixed-yellow",
        red=FixedMoveAgent(Move(MoveType.DROP, 3)),
        yellow=FixedMoveAgent(Move(MoveType.DROP, 0)),
        initial_state=state,
    )

    assert game.final_state.status is GameStatus.FINISHED
    assert game.winner_agent == "fixed-yellow"
    assert len(game.moves) == 1
    assert game.moves[0].decision_time_seconds == pytest.approx(0.125)


def test_run_match_alternates_sides_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    pairs = []
    terminal_draw = GameState(
        board=empty_board(),
        status=GameStatus.FINISHED,
        result=GameResult(winner=None, red_lines=0, yellow_lines=0),
    )

    def fake_create_agent(agent_name: str, seed: int | None = None, depth: int = 3):
        return FixedMoveAgent(Move(MoveType.DROP, 0))

    def fake_play_timed_game(red_agent_name, yellow_agent_name, red, yellow, **kwargs):
        pairs.append((red_agent_name, yellow_agent_name, kwargs["seed"]))
        return SimulatedGame(
            red_agent=red_agent_name,
            yellow_agent=yellow_agent_name,
            final_state=terminal_draw,
            moves=(),
            seed=kwargs["seed"],
        )

    monkeypatch.setattr(experiments, "create_agent", fake_create_agent)
    monkeypatch.setattr(experiments, "play_timed_game", fake_play_timed_game)

    summary = run_match("minimax", "random", games=3, seed=100, depth=2, swap_sides=True)

    assert summary.game_count == 3
    assert pairs == [
        ("minimax", "random", 100),
        ("random", "minimax", 102),
        ("minimax", "random", 104),
    ]


def test_run_match_rejects_non_positive_game_count() -> None:
    with pytest.raises(ValueError, match="games"):
        run_match("minimax", "random", games=0)


def test_summarize_games_counts_wins_draws_and_average_decision_time() -> None:
    red_win = GameState(
        board=empty_board(),
        status=GameStatus.FINISHED,
        result=GameResult(winner=Player.RED, red_lines=0, yellow_lines=1),
    )
    draw = GameState(
        board=empty_board(),
        status=GameStatus.FINISHED,
        result=GameResult(winner=None, red_lines=1, yellow_lines=1),
    )
    games = (
        SimulatedGame(
            red_agent="minimax",
            yellow_agent="random",
            final_state=red_win,
            moves=(
                TimedMoveRecord(Player.RED, "minimax", Move(MoveType.DROP, 0), 1, 0.2),
                TimedMoveRecord(Player.YELLOW, "random", Move(MoveType.DROP, 1), 2, 0.1),
            ),
        ),
        SimulatedGame(
            red_agent="random",
            yellow_agent="minimax",
            final_state=draw,
            moves=(
                TimedMoveRecord(Player.RED, "random", Move(MoveType.DROP, 0), 1, 0.3),
                TimedMoveRecord(Player.YELLOW, "minimax", Move(MoveType.DROP, 1), 2, 0.4),
            ),
        ),
    )

    summary = summarize_games(games, agent_names=("minimax", "random"))

    assert summary.wins == {"minimax": 1, "random": 0}
    assert summary.draws == 1
    assert summary.average_moves == 2
    assert summary.average_decision_time("minimax") == pytest.approx(0.3)
    assert summary.average_decision_time("random") == pytest.approx(0.2)


def test_format_match_summary_includes_key_metrics() -> None:
    terminal_draw = GameState(
        board=empty_board(),
        status=GameStatus.FINISHED,
        result=GameResult(winner=None, red_lines=0, yellow_lines=0),
    )
    summary = summarize_games(
        (
            SimulatedGame(
                red_agent="minimax",
                yellow_agent="random",
                final_state=terminal_draw,
                moves=(TimedMoveRecord(Player.RED, "minimax", Move(MoveType.DROP, 0), 1, 0.001),),
            ),
        ),
        agent_names=("minimax", "random"),
    )

    formatted = format_match_summary(summary)

    assert "Games: 1" in formatted
    assert "Minimax wins: 0 (0.0%)" in formatted
    assert "Minimax avg decision: 1.000 ms" in formatted
    assert "Draws: 1 (100.0%)" in formatted
    assert "Average moves: 1.00" in formatted


def test_main_prints_match_summary(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    terminal_draw = GameState(
        board=empty_board(),
        status=GameStatus.FINISHED,
        result=GameResult(winner=None, red_lines=0, yellow_lines=0),
    )
    summary = summarize_games(
        (SimulatedGame("minimax", "random", terminal_draw, moves=()),),
        agent_names=("minimax", "random"),
    )

    monkeypatch.setattr(experiments, "run_match", lambda **kwargs: summary)

    assert experiments.main(["--red", "minimax", "--yellow", "random", "--games", "1"]) == 0
    assert "Games: 1" in capsys.readouterr().out


def empty_board():
    return tuple(tuple(None for _ in range(COLUMNS)) for _ in range(ROWS))


def board_from_rows(*rows: str):
    player_by_symbol = {
        ".": None,
        "R": Player.RED,
        "Y": Player.YELLOW,
    }
    return tuple(tuple(player_by_symbol[symbol] for symbol in row) for row in rows)
