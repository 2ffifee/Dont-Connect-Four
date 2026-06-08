from connect4_mcts.game import GameState
from connect4_mcts.players import MockLLMClient, RandomPlayer
from connect4_mcts.players.llm import LLMPlayer
from connect4_mcts.runner import play_game, prepare_agents_for_game
from connect4_mcts.tournament_config import load_llm_tournament_entries


def test_prepare_agents_for_game_runs_llm_rules_briefing_for_both_colors() -> None:
    calls: list[str] = []

    class _RecordingClient(MockLLMClient):
        def complete(self, messages, *, timeout=...):
            calls.append(messages[-1]["content"])
            return "Understood."

    red = LLMPlayer(_RecordingClient(responses=["ok"]), model_label="red")
    yellow = LLMPlayer(_RecordingClient(responses=["ok"]), model_label="yellow")

    prepare_agents_for_game(red, yellow)

    assert len(calls) == 2
    assert "FIRST player" in calls[0]
    assert "SECOND player" in calls[1]
    assert red.rules_acknowledged
    assert yellow.rules_acknowledged


def test_play_game_invokes_prepare_agents_for_game() -> None:
    from unittest.mock import patch

    llm = LLMPlayer(MockLLMClient(responses=["ok", '{"move_type": "drop", "column": 0}']))

    with patch("connect4_mcts.runner.prepare_agents_for_game") as prepare:
        play_game(llm, RandomPlayer(seed=1), max_moves=200)

    prepare.assert_called_once()


def test_load_llm_tournament_entries_uses_server_defaults() -> None:
    entries = load_llm_tournament_entries(
        {
            "server": {"base_url": "http://localhost:11434/v1", "api_key": "", "timeout": 120.0},
            "models": [{"id": "test", "model": "llama3.2"}],
        }
    )

    assert len(entries) == 1
    assert entries[0]["id"] == "llm-test"
    assert entries[0]["kind"] == "llm"
    assert entries[0]["model"] == "llama3.2"
    assert entries[0]["base_url"] == "http://localhost:11434/v1"
    assert entries[0]["timeout"] == 120.0
