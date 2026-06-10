import csv
import json

import pytest

import scripts.score_blunders as score_blunders
from connect4_mcts.game import Move, MoveType, Player
from connect4_mcts.players.mcts import SearchEvaluation


class FakeOracle:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, state, run_search: bool = True) -> SearchEvaluation:
        self.calls += 1
        assert run_search is False
        best = Move(MoveType.DROP, 0)
        blunder = Move(MoveType.PUSH, 1)
        return SearchEvaluation(
            player_to_move=state.current_player,
            root_value=0.9,
            best_move=best,
            move_values={best: 0.9, blunder: 0.4},
            move_visits={best: 100, blunder: 20},
        )


def test_score_blunders_writes_move_and_summary_outputs(monkeypatch, tmp_path) -> None:
    oracle = FakeOracle()
    monkeypatch.setattr(score_blunders, "load_player", lambda path: oracle)
    input_dir = tmp_path / "tournament"
    input_dir.mkdir()
    moves_path = input_dir / "moves.jsonl"
    _write_moves(moves_path)

    result = score_blunders.main(
        [
            "--oracle",
            "fake-oracle.pkl",
            "--input-dir",
            str(input_dir),
            "--cache-only",
            "--threshold",
            "0.3",
        ]
    )

    assert result == 0
    assert oracle.calls == 1

    blunders = _read_csv(input_dir / "blunders.csv")
    summary = _read_csv(input_dir / "blunder_summary.csv")

    assert len(blunders) == 2
    assert blunders[0]["agent"] == "uct"
    assert blunders[0]["oracle_best_move_type"] == "drop"
    assert blunders[0]["oracle_best_column"] == "0"
    assert float(blunders[0]["chosen_value"]) == pytest.approx(0.9)
    assert float(blunders[0]["regret"]) == pytest.approx(0.0)
    assert blunders[0]["is_blunder"] == "False"

    assert blunders[1]["agent"] == "llm"
    assert float(blunders[1]["chosen_value"]) == pytest.approx(0.4)
    assert float(blunders[1]["regret"]) == pytest.approx(0.5)
    assert blunders[1]["is_blunder"] == "True"

    by_agent = {row["agent"]: row for row in summary}
    assert by_agent["uct"]["blunders"] == "0"
    assert by_agent["llm"]["blunders"] == "1"
    assert float(by_agent["llm"]["blunder_rate"]) == pytest.approx(1.0)


def test_state_from_payload_restores_game_state() -> None:
    payload = {
        "board": [
            "........",
            "........",
            "........",
            "........",
            "........",
            "RY......",
        ],
        "current_player": "yellow",
        "first_player": "red",
        "status": "ongoing",
        "move_count": 2,
        "protected_segments": [],
    }

    state = score_blunders.state_from_payload(payload)

    assert state.current_player is Player.YELLOW
    assert state.first_player is Player.RED
    assert state.move_count == 2
    assert state.board[5][0] is Player.RED
    assert state.board[5][1] is Player.YELLOW


def _write_moves(path) -> None:
    state_before = {
        "board": ["........"] * 6,
        "current_player": "red",
        "first_player": "red",
        "status": "ongoing",
        "move_count": 0,
        "protected_segments": [],
    }
    rows = [
        {
            "game_id": "g1",
            "pair_id": "uct_vs_llm",
            "pair_index": 0,
            "game_index": 0,
            "seed": 1,
            "move_number": 1,
            "player": "red",
            "agent": "uct",
            "move_type": "drop",
            "column": 0,
            "state_before": state_before,
        },
        {
            "game_id": "g1",
            "pair_id": "uct_vs_llm",
            "pair_index": 0,
            "game_index": 0,
            "seed": 1,
            "move_number": 2,
            "player": "red",
            "agent": "llm",
            "move_type": "push",
            "column": 1,
            "state_before": state_before,
        },
    ]
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))
