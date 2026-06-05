"""Tests for the LLM-backed agent (offline, via MockLLMClient)."""

from __future__ import annotations

import re

import pytest

from connect4_mcts.game import GameState, GameStatus, Move, MoveType, Player
from connect4_mcts.players import MockLLMClient, RandomPlayer
from connect4_mcts.players.base import MoveSelectionError
from connect4_mcts.players.llm import (
    DEFAULT_SYSTEM_PROMPT,
    LLMPlayer,
    OpenAIClient,
    parse_move,
    render_turn,
    resolve_openai_credentials,
)
from connect4_mcts.runner import play_game


def _first_legal_responder(messages):
    """A 'competent' mock LLM: replies with the first legal move it is offered."""
    user_text = messages[-1]["content"]
    match = re.search(r"-\s*(drop|push) column (\d+)", user_text)
    assert match, "turn prompt should list legal moves"
    return f'{{"move_type": "{match.group(1)}", "column": {match.group(2)}}}'


def test_parse_move_accepts_plain_json():
    assert parse_move('{"move_type": "drop", "column": 3}') == Move(MoveType.DROP, 3)


def test_parse_move_accepts_json_in_code_fence_with_prose():
    text = 'Sure, I think this avoids a line:\n```json\n{"move_type": "push", "column": 5}\n```'
    assert parse_move(text) == Move(MoveType.PUSH, 5)


def test_parse_move_accepts_loose_text():
    assert parse_move("I will drop in column 2.") == Move(MoveType.DROP, 2)


def test_parse_move_rejects_out_of_range_column():
    assert parse_move('{"move_type": "drop", "column": 99}') is None


def test_parse_move_rejects_garbage():
    assert parse_move("no idea") is None
    assert parse_move("") is None


def test_parse_move_rejects_boolean_column():
    assert parse_move('{"move_type": "drop", "column": true}') is None


def test_system_prompt_stresses_inverted_goal():
    assert "INVERTED" in DEFAULT_SYSTEM_PROMPT
    assert "FEWER" in DEFAULT_SYSTEM_PROMPT


def test_turn_prompt_lists_legal_moves_and_board():
    state = GameState.new()
    legal = state.legal_moves()
    prompt = render_turn(state, legal)
    assert "row 0:" in prompt
    assert "Legal moves:" in prompt
    for move in legal:
        assert f"{move.move_type.value} column {move.column}" in prompt


def test_llm_player_returns_legal_move_from_valid_reply():
    client = MockLLMClient(responses=['{"move_type": "drop", "column": 0}'])
    player = LLMPlayer(client)
    state = GameState.new()

    move = player.choose_move(state)

    assert move == Move(MoveType.DROP, 0)
    assert state.is_legal_move(move)
    assert player.requests == 1
    assert player.invalid_responses == 0
    assert player.fallbacks == 0


def _state_with_full_first_column() -> GameState:
    state = GameState.new()
    for _ in range(6):
        state = state.apply_move(Move(MoveType.DROP, 0))
    assert state.is_column_full(0)
    assert state.status is GameStatus.ONGOING
    return state


def test_llm_player_retries_after_invalid_then_succeeds():
    client = MockLLMClient(
        responses=[
            "I have no idea what to do",            # unparseable
            '{"move_type": "drop", "column": 0}',   # well-formed but illegal (column full)
            '{"move_type": "push", "column": 1}',   # finally legal
        ]
    )
    player = LLMPlayer(client, max_attempts=3)
    state = _state_with_full_first_column()

    move = player.choose_move(state)

    assert move == Move(MoveType.PUSH, 1)
    assert player.requests == 3
    assert player.unparseable == 1
    assert player.illegal == 1
    assert player.fallbacks == 0
    # Each retry should have fed corrective feedback back to the model.
    assert any("not a valid, legal move" in m["content"] for call in client.calls for m in call)


def test_llm_player_falls_back_to_random_after_exhausting_attempts():
    client = MockLLMClient(responses=["nope"])
    player = LLMPlayer(client, max_attempts=2, on_failure="random", seed=0)
    state = GameState.new()

    move = player.choose_move(state)

    assert state.is_legal_move(move)
    assert player.requests == 2
    assert player.fallbacks == 1
    assert player.illegal_move_rate == 1.0


def test_llm_player_can_raise_on_failure():
    client = MockLLMClient(responses=["nope"])
    player = LLMPlayer(client, max_attempts=1, on_failure="raise")

    with pytest.raises(MoveSelectionError):
        player.choose_move(GameState.new())


def test_llm_player_rejects_invalid_config():
    client = MockLLMClient(responses=["x"])
    with pytest.raises(ValueError):
        LLMPlayer(client, max_attempts=0)
    with pytest.raises(ValueError):
        LLMPlayer(client, on_failure="explode")


def test_llm_player_raises_without_legal_moves():
    finished = GameState.new().apply_move(Move(MoveType.DROP, 0))
    # Force a finished state so there are no legal moves.
    from connect4_mcts.game import GameResult

    state = GameState(
        board=finished.board,
        current_player=Player.YELLOW,
        first_player=Player.RED,
        status=GameStatus.FINISHED,
        move_count=finished.move_count,
        result=GameResult(winner=Player.RED, red_lines=0, yellow_lines=0),
    )
    player = LLMPlayer(MockLLMClient(responses=["x"]))
    with pytest.raises(MoveSelectionError):
        player.choose_move(state)


def test_resolve_credentials_uses_placeholder_for_local_server_without_key():
    key, base = resolve_openai_credentials(api_key="", base_url="http://localhost:11434/v1")

    assert key == "ollama"
    assert base == "http://localhost:11434/v1"


def test_resolve_credentials_keeps_explicit_key():
    key, base = resolve_openai_credentials(api_key="sk-test", base_url="http://localhost:11434/v1")

    assert key == "sk-test"
    assert base == "http://localhost:11434/v1"


def test_resolve_credentials_requires_key_for_default_openai_endpoint(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    with pytest.raises(ValueError, match="API key required"):
        resolve_openai_credentials(api_key="", base_url="")


def test_adapt_params_swaps_max_tokens_for_reasoning_models():
    params = {"model": "gpt-5.4-mini", "messages": [], "max_tokens": 256}
    exc = Exception("Unsupported parameter: 'max_tokens' is not supported; use 'max_completion_tokens'.")

    adapted = OpenAIClient._adapt_params(params, exc)

    assert adapted is not None
    assert "max_tokens" not in adapted
    assert adapted["max_completion_tokens"] == 256


def test_adapt_params_drops_unsupported_temperature():
    params = {"model": "gpt-5.5", "messages": [], "temperature": 0.0}
    exc = Exception("Unsupported value: 'temperature' does not support 0.0 with this model.")

    adapted = OpenAIClient._adapt_params(params, exc)

    assert adapted is not None
    assert "temperature" not in adapted


def test_adapt_params_returns_none_for_unrelated_errors():
    params = {"model": "gpt-5.5", "messages": [], "max_tokens": 256}
    exc = Exception("Connection error: could not reach host")

    assert OpenAIClient._adapt_params(params, exc) is None


def test_llm_player_plays_full_game_through_runner():
    llm = LLMPlayer(MockLLMClient(responder=_first_legal_responder), model_label="mock")
    opponent = RandomPlayer(seed=1)

    game = play_game(red=llm, yellow=opponent)

    assert game.final_state.status is GameStatus.FINISHED
    # A competent mock never needs the random fallback.
    assert llm.fallbacks == 0
    assert llm.moves > 0
