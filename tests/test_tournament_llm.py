from connect4_mcts.game import GameState
from connect4_mcts.players import MockLLMClient, RandomPlayer
from connect4_mcts.players.llm import LLMPlayer
from connect4_mcts.runner import play_game, prepare_agents_for_game


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


def test_experiment_config_parses_llm_player(tmp_path) -> None:
    from connect4_mcts.experiment_config import load_experiment_config

    config_path = tmp_path / "llm.toml"
    config_path.write_text(
        """
seed = 0
output_dir = "results"
games_per_pair = 1

[[players]]
id = "llm-test"
type = "llm"
model = "llama3.2"
base_url = "http://localhost:11434/v1"
timeout = 120.0
""".strip(),
        encoding="utf-8",
    )
    config = load_experiment_config(config_path)
    llm = config.players[0]
    assert llm.id == "llm-test"
    assert llm.type == "llm"
    assert llm.params["model"] == "llama3.2"
    assert llm.params["base_url"] == "http://localhost:11434/v1"
    assert llm.params["timeout"] == 120.0
