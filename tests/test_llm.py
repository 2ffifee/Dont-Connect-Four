"""Tests for the LLM-backed agent (offline, via MockLLMClient)."""

from __future__ import annotations

import json
import re

import pytest

from connect4_mcts.game import GameState, GameStatus, Move, MoveType, Player
from connect4_mcts.players import MockLLMClient, RandomPlayer
from connect4_mcts.players.base import MoveSelectionError
from connect4_mcts.players.llm import (
    DEFAULT_SYSTEM_PROMPT,
    LLMPlayer,
    OpenAIClient,
    _USE_CLIENT_TIMEOUT,
    detect_llm_provider,
    extract_thinking,
    gemini_supports_visible_thoughts,
    normalize_llm_endpoint,
    parse_move,
    openrouter_should_request_reasoning,
    cot_streams_live,
    model_exposes_thinking,
    prefers_blocking_completion,
    render_rules_briefing,
    render_turn,
    resolve_llm_credentials,
    resolve_openai_credentials,
    uses_structured_reasoning,
    _extract_reasoning_details,
    _read_reasoning_from_part,
)
from connect4_mcts.runner import play_game


def _first_legal_responder(messages):
    """A 'competent' mock LLM: rules ack + first legal move from the turn prompt."""
    user_text = messages[-1]["content"]
    if "Do NOT reply with JSON" in user_text:
        return "I understand the rules and will wait for my turn."
    match = re.search(r"-\s*(drop|push) column (\d+)", user_text)
    assert match, "turn prompt should list legal moves"
    return f'{{"move_type": "{match.group(1)}", "column": {match.group(2)}}}'


def test_llm_player_forwards_streaming_thinking_updates() -> None:
    class StreamingMock(MockLLMClient):
        def __init__(self) -> None:
            super().__init__(responses=[""])

        def complete(self, messages, *, timeout=_USE_CLIENT_TIMEOUT, on_thinking_update=None):
            del messages, timeout
            if on_thinking_update is not None:
                on_thinking_update("part one")
                on_thinking_update("part one part two")
            return '{"move_type": "drop", "column": 0}'

    player = LLMPlayer(StreamingMock())
    player.send_rules_briefing(Player.YELLOW)

    assert player.last_thinking == "part one part two"

    player = LLMPlayer(StreamingMock())
    captured: list[str] = []

    def capture(text: str) -> None:
        captured.append(text)

    player._emit_thinking = capture  # type: ignore[method-assign]
    player.choose_move(GameState.new())

    assert captured == ["part one", "part one part two"]


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


def test_rules_briefing_has_no_request_timeout():
    recorded: list[float | None | object] = []

    class _TimeoutRecordingClient(MockLLMClient):
        def complete(self, messages, *, timeout=_USE_CLIENT_TIMEOUT):
            recorded.append(timeout)
            return super().complete(messages, timeout=timeout)

    client = _TimeoutRecordingClient(responses=["ok"])
    player = LLMPlayer(client)
    player.send_rules_briefing(Player.YELLOW)
    assert recorded == [None]


def test_rules_briefing_asks_for_acknowledgment_not_a_move():
    client = MockLLMClient(responses=["I understand the rules and will wait for RED."])
    player = LLMPlayer(client)

    reply = player.send_rules_briefing(Player.YELLOW)

    assert reply == "I understand the rules and will wait for RED."
    assert player.rules_acknowledged
    assert player.moves == 0
    assert len(client.calls) == 1
    messages = client.calls[0]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "SECOND player" in messages[1]["content"]
    assert "YELLOW" in messages[1]["content"]
    assert "Do NOT reply with JSON" in messages[1]["content"]
    assert player._conversation[2]["role"] == "assistant"


def test_rules_briefing_for_red_first_player():
    client = MockLLMClient(responses=["Ready to play as RED."])
    player = LLMPlayer(client)

    reply = player.send_rules_briefing(Player.RED)

    assert "Ready" in reply
    assert "FIRST player" in client.calls[0][1]["content"]
    assert "RED" in client.calls[0][1]["content"]


def test_extract_thinking_splits_tagged_reasoning():
    thinking, remainder = extract_thinking(
        "Reasoning here\n"
        '{"move_type": "drop", "column": 2}'
    )
    assert thinking is not None
    assert "Reasoning here" in thinking
    assert '"move_type"' in remainder


def test_extract_thinking_splits_think_tags():
    tagged = "<think>Hidden reasoning</think>\n" + '{"move_type": "drop", "column": 1}'
    thinking, remainder = extract_thinking(tagged)
    assert thinking == "Hidden reasoning"
    assert parse_move(remainder) == Move(MoveType.DROP, 1)


def test_extract_thinking_splits_gemini_thought_tags():
    tagged = '<thought>Consider column 3.</thought>{"move_type": "drop", "column": 3}'
    thinking, remainder = extract_thinking(tagged)
    assert thinking == "Consider column 3."
    assert parse_move(remainder) == Move(MoveType.DROP, 3)


def test_extract_thinking_joins_multiple_tag_blocks():
    tagged = (
        "<think>First</think>"
        "<think>Second</think>"
        '{"move_type": "drop", "column": 1}'
    )
    thinking, remainder = extract_thinking(tagged)
    assert thinking == "First\n\nSecond"
    assert parse_move(remainder) == Move(MoveType.DROP, 1)


def test_extract_thinking_ignores_markdown_fence_before_json():
    text = '```json\n{"move_type": "drop", "column": 2}'
    thinking, remainder = extract_thinking(text)
    assert thinking is None
    assert parse_move(remainder) == Move(MoveType.DROP, 2)


def test_extract_thinking_keeps_prose_and_strips_fence_marker():
    text = 'Column 4 looks safest.\n```json\n{"move_type": "push", "column": 4}\n```'
    thinking, remainder = extract_thinking(text)
    assert thinking == "Column 4 looks safest."
    assert parse_move(remainder) == Move(MoveType.PUSH, 4)


def test_extract_thinking_ignores_single_quote_fence_marker():
    text = "'''json\n{\"move_type\": \"drop\", \"column\": 1}"
    thinking, remainder = extract_thinking(text)
    assert thinking is None
    assert parse_move(remainder) == Move(MoveType.DROP, 1)


def test_normalize_thinking_text_drops_fence_only_reasoning():
    from connect4_mcts.players.llm import _normalize_thinking_text

    assert _normalize_thinking_text("```json") is None
    assert _normalize_thinking_text("'''json") is None
    assert _normalize_thinking_text("Actual reasoning") == "Actual reasoning"


def test_extract_thinking_splits_ministral_think_tags():
    tagged = (
        "[THINK]Column 3 avoids creating a new segment.[/THINK]\n"
        '```json\n{"move_type": "drop", "column": 3}\n```'
    )
    thinking, remainder = extract_thinking(tagged)
    assert thinking == "Column 3 avoids creating a new segment."
    assert parse_move(remainder) == Move(MoveType.DROP, 3)


def test_read_reasoning_from_part_supports_openrouter_reasoning_details() -> None:
    message = type(
        "Message",
        (),
        {
            "content": '{"move_type": "drop", "column": 1}',
            "reasoning_details": [
                {"type": "reasoning.text", "text": "Scan legal moves first."},
                {"type": "reasoning.text", "text": "Prefer a safe drop."},
            ],
        },
    )()
    assert _read_reasoning_from_part(message) == "Scan legal moves first.\n\nPrefer a safe drop."


def test_openrouter_should_request_reasoning_for_ministral() -> None:
    assert openrouter_should_request_reasoning("mistralai/ministral-14b-2512")
    assert openrouter_should_request_reasoning("deepseek/deepseek-r1")
    assert not openrouter_should_request_reasoning("openai/gpt-4o-mini")


def test_uses_structured_reasoning_detects_common_model_ids() -> None:
    assert uses_structured_reasoning("o3-mini")
    assert uses_structured_reasoning("deepseek-reasoner")
    assert uses_structured_reasoning("anthropic/claude-3.7-sonnet:thinking")
    assert not uses_structured_reasoning("gpt-4o-mini")


def test_prefers_blocking_for_cloud_reasoning_models_not_local() -> None:
    assert not prefers_blocking_completion("http://localhost:11434/v1", "llama3")
    assert prefers_blocking_completion("https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-2.5-flash")
    assert prefers_blocking_completion("https://api.openai.com/v1", "o3-mini")
    assert prefers_blocking_completion(
        "https://openrouter.ai/api/v1",
        "mistralai/ministral-14b-2512",
    )
    assert not prefers_blocking_completion("https://api.openai.com/v1", "gpt-4o-mini")


def test_cot_streams_live_only_for_local_reasoning_models() -> None:
    assert cot_streams_live("http://localhost:11434/v1", "qwen3:8b")
    assert cot_streams_live("http://127.0.0.1:1234/v1", "deepseek-r1:8b")
    assert not cot_streams_live("http://localhost:11434/v1", "llama3")
    assert not cot_streams_live("https://api.openai.com/v1", "gpt-4o-mini")
    assert not cot_streams_live("https://openrouter.ai/api/v1", "mistralai/ministral-14b-2512")


def test_model_exposes_thinking_detects_supported_models() -> None:
    assert model_exposes_thinking("https://openrouter.ai/api/v1", "mistralai/ministral-14b-2512")
    assert model_exposes_thinking(
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "gemini-2.5-flash",
    )
    assert not model_exposes_thinking("https://api.openai.com/v1", "o3-mini")
    assert model_exposes_thinking("http://localhost:11434/v1", "qwen3:8b")
    assert not model_exposes_thinking("https://api.openai.com/v1", "gpt-4o-mini")
    assert not model_exposes_thinking("http://localhost:11434/v1", "llama3")


def test_read_reasoning_from_part_uses_model_dump_payload() -> None:
    class Message:
        content = '{"move_type": "drop", "column": 2}'

        @staticmethod
        def model_dump() -> dict[str, object]:
            return {
                "content": '{"move_type": "drop", "column": 2}',
                "reasoning": "Hidden in raw payload",
            }

    assert _read_reasoning_from_part(Message()) == "Hidden in raw payload"


def test_read_reasoning_from_part_supports_reasoning_content() -> None:
    message = type("Message", (), {"reasoning_content": "Plan the move", "content": '{"move_type": "drop", "column": 2}'})()
    assert _read_reasoning_from_part(message) == "Plan the move"


def test_openai_client_blocking_reasoning_field_updates_thinking(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeMessage:
        content = '{"move_type": "drop", "column": 4}'
        reasoning_content = "Evaluate columns 0-7."

        @staticmethod
        def model_dump() -> dict[str, object]:
            return {
                "content": '{"move_type": "drop", "column": 4}',
                "reasoning_content": "Evaluate columns 0-7.",
            }

    class FakeCompletions:
        @staticmethod
        def create(**kwargs: object) -> object:
            captured.update(kwargs)
            message = FakeMessage()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice]})()

    class FakeChat:
        completions = FakeCompletions

    class FakeOpenAI:
        def __init__(self, *, api_key: str, base_url: str | None, default_headers=None) -> None:
            del default_headers
            self.chat = FakeChat()

    monkeypatch.setitem(__import__("sys").modules, "openai", type("openai", (), {"OpenAI": FakeOpenAI}))

    client = OpenAIClient(
        model="deepseek-reasoner",
        api_key="test-key",
        base_url="https://api.deepseek.com/v1",
    )
    updates: list[str] = []
    content = client.complete(
        [{"role": "user", "content": "pick a move"}],
        on_thinking_update=updates.append,
    )

    assert not captured.get("stream")
    assert client.last_thinking == "Evaluate columns 0-7."
    assert updates == ["Evaluate columns 0-7."]
    assert parse_move(content) == Move(MoveType.DROP, 4)


def test_openai_client_local_server_streams_thinking_live(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeDelta:
        def __init__(self, content: str = "", reasoning_content: str = "") -> None:
            self.content = content
            self.reasoning_content = reasoning_content

    class FakeChunk:
        def __init__(self, delta: FakeDelta) -> None:
            self.choices = [type("Choice", (), {"delta": delta})()]

    class FakeCompletions:
        @staticmethod
        def create(**kwargs: object) -> object:
            calls.append(dict(kwargs))
            if kwargs.get("stream"):
                yield FakeChunk(FakeDelta(reasoning_content="Local plan"))
                yield FakeChunk(FakeDelta(content='{"move_type": "push", "column": 1}'))
                return
            message = type(
                "Message",
                (),
                {"content": '{"move_type": "push", "column": 1}'},
            )()
            return type("Response", (), {"choices": [type("Choice", (), {"message": message})()]})()

    class FakeChat:
        completions = FakeCompletions

    class FakeOpenAI:
        def __init__(self, *, api_key: str, base_url: str | None, default_headers=None) -> None:
            del default_headers
            self.chat = FakeChat()

    monkeypatch.setitem(__import__("sys").modules, "openai", type("openai", (), {"OpenAI": FakeOpenAI}))

    client = OpenAIClient(
        model="qwen3:8b",
        api_key="test-key",
        base_url="http://localhost:11434/v1",
    )
    updates: list[str] = []
    content = client.complete(
        [{"role": "user", "content": "pick a move"}],
        on_thinking_update=updates.append,
    )

    assert any(call.get("stream") for call in calls)
    assert client.last_thinking == "Local plan"
    assert updates[-1] == "Local plan"
    assert parse_move(content) == Move(MoveType.PUSH, 1)


def test_openrouter_client_requests_reasoning_for_ministral(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeCompletions:
        @staticmethod
        def create(**kwargs: object) -> object:
            captured.update(kwargs)
            message = type(
                "Message",
                (),
                {
                    "content": '[THINK]Plan[/THINK]{"move_type": "drop", "column": 0}',
                    "reasoning": "Separate reasoning field",
                },
            )()
            return type("Response", (), {"choices": [type("Choice", (), {"message": message})()]})()

    class FakeChat:
        completions = FakeCompletions

    class FakeOpenAI:
        def __init__(self, *, api_key: str, base_url: str | None, default_headers=None) -> None:
            del default_headers
            self.chat = FakeChat()

    monkeypatch.setitem(__import__("sys").modules, "openai", type("openai", (), {"OpenAI": FakeOpenAI}))

    client = OpenAIClient(
        model="mistralai/ministral-14b-2512",
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
    )
    updates: list[str] = []
    content = client.complete(
        [{"role": "user", "content": "pick a move"}],
        on_thinking_update=updates.append,
    )

    extra_body = captured.get("extra_body")
    assert isinstance(extra_body, dict)
    assert extra_body.get("reasoning") == {"enabled": True}
    assert client.last_thinking in {"Separate reasoning field", "Plan", "Separate reasoning field\n\nPlan"}
    assert parse_move(content) == Move(MoveType.DROP, 0)


def test_openrouter_client_reads_reasoning_from_raw_http_payload(monkeypatch) -> None:
    class FakeMessage:
        content = '{"move_type": "push", "column": 5}'

    class FakeRaw:
        text = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"move_type": "push", "column": 5}',
                            "reasoning": "Prefer a safe push.",
                        }
                    }
                ]
            }
        )

        @staticmethod
        def parse() -> object:
            message = FakeMessage()
            return type("Response", (), {"choices": [type("Choice", (), {"message": message})()]})()

    class FakeRawWrapper:
        @staticmethod
        def create(**kwargs: object) -> FakeRaw:
            return FakeRaw()

    class FakeCompletions:
        with_raw_response = FakeRawWrapper()

        @staticmethod
        def create(**kwargs: object) -> object:
            message = FakeMessage()
            return type("Response", (), {"choices": [type("Choice", (), {"message": message})()]})()

    class FakeChat:
        completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, *, api_key: str, base_url: str | None, default_headers=None) -> None:
            del default_headers
            self.chat = FakeChat()

    monkeypatch.setitem(__import__("sys").modules, "openai", type("openai", (), {"OpenAI": FakeOpenAI}))

    client = OpenAIClient(
        model="mistralai/ministral-14b-2512",
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
    )
    content = client.complete([{"role": "user", "content": "pick a move"}])

    assert client.last_thinking == "Prefer a safe push."
    assert parse_move(content) == Move(MoveType.PUSH, 5)


def test_openrouter_client_requests_reasoning_for_reasoning_models(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeCompletions:
        @staticmethod
        def create(**kwargs: object) -> object:
            captured.update(kwargs)
            message = type("Message", (), {"content": '{"move_type": "drop", "column": 0}'})()
            return type("Response", (), {"choices": [type("Choice", (), {"message": message})()]})()

    class FakeChat:
        completions = FakeCompletions

    class FakeOpenAI:
        def __init__(self, *, api_key: str, base_url: str | None, default_headers=None) -> None:
            del default_headers
            self.chat = FakeChat()

    monkeypatch.setitem(__import__("sys").modules, "openai", type("openai", (), {"OpenAI": FakeOpenAI}))

    client = OpenAIClient(
        model="deepseek/deepseek-r1",
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
    )
    client.complete([{"role": "user", "content": "hi"}])

    extra_body = captured.get("extra_body")
    assert isinstance(extra_body, dict)
    assert extra_body.get("reasoning") == {"enabled": True}


def test_gemini_client_requests_include_thoughts_for_thinking_models(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeCompletions:
        @staticmethod
        def create(**kwargs: object) -> object:
            captured.update(kwargs)
            message = type("Message", (), {"content": '{"move_type": "drop", "column": 0}'})()
            return type("Response", (), {"choices": [type("Choice", (), {"message": message})()]})()

    class FakeChat:
        completions = FakeCompletions

    class FakeOpenAI:
        def __init__(self, *, api_key: str, base_url: str | None) -> None:
            self.chat = FakeChat()

    monkeypatch.setitem(__import__("sys").modules, "openai", type("openai", (), {"OpenAI": FakeOpenAI}))

    client = OpenAIClient(
        model="gemini-2.5-flash",
        api_key="test-key",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    client.complete([{"role": "user", "content": "hi"}])

    extra_body = captured.get("extra_body")
    assert isinstance(extra_body, dict)
    google = extra_body.get("extra_body", extra_body).get("google")
    assert isinstance(google, dict)
    assert google["thinking_config"]["include_thoughts"] is True


def test_gemini_client_skips_include_thoughts_for_older_models(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeCompletions:
        @staticmethod
        def create(**kwargs: object) -> object:
            captured.update(kwargs)
            message = type("Message", (), {"content": '{"move_type": "drop", "column": 0}'})()
            return type("Response", (), {"choices": [type("Choice", (), {"message": message})()]})()

    class FakeChat:
        completions = FakeCompletions

    class FakeOpenAI:
        def __init__(self, *, api_key: str, base_url: str | None) -> None:
            self.chat = FakeChat()

    monkeypatch.setitem(__import__("sys").modules, "openai", type("openai", (), {"OpenAI": FakeOpenAI}))

    client = OpenAIClient(
        model="gemini-2.0-flash",
        api_key="test-key",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    client.complete([{"role": "user", "content": "hi"}])

    assert "extra_body" not in captured


def test_gemini_supports_visible_thoughts() -> None:
    assert gemini_supports_visible_thoughts("gemini-2.5-flash")
    assert gemini_supports_visible_thoughts("models/gemini-3.5-flash")
    assert not gemini_supports_visible_thoughts("gemini-2.0-flash")


def test_gemini_client_uses_blocking_not_streaming(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeCompletions:
        @staticmethod
        def create(**kwargs: object) -> object:
            calls.append(dict(kwargs))
            message = type(
                "Message",
                (),
                {"content": '<thought>Plan</thought>{"move_type": "drop", "column": 2}'},
            )()
            return type("Response", (), {"choices": [type("Choice", (), {"message": message})()]})()

    class FakeChat:
        completions = FakeCompletions

    class FakeOpenAI:
        def __init__(self, *, api_key: str, base_url: str | None) -> None:
            self.chat = FakeChat()

    monkeypatch.setitem(__import__("sys").modules, "openai", type("openai", (), {"OpenAI": FakeOpenAI}))

    client = OpenAIClient(
        model="gemini-2.5-flash",
        api_key="test-key",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    updates: list[str] = []
    content = client.complete(
        [{"role": "user", "content": "pick a move"}],
        on_thinking_update=updates.append,
    )

    assert all(not call.get("stream") for call in calls)
    assert client.last_thinking == "Plan"
    assert parse_move(content) == Move(MoveType.DROP, 2)
    assert updates == ["Plan"]


def test_openai_client_falls_back_to_blocking_when_streaming_fails(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeCompletions:
        @staticmethod
        def create(**kwargs: object) -> object:
            calls.append(dict(kwargs))
            if kwargs.get("stream"):
                raise RuntimeError("streaming unsupported")
            message = type(
                "Message",
                (),
                {"content": '<thought>Plan</thought>{"move_type": "drop", "column": 2}'},
            )()
            return type("Response", (), {"choices": [type("Choice", (), {"message": message})()]})()

    class FakeChat:
        completions = FakeCompletions

    class FakeOpenAI:
        def __init__(self, *, api_key: str, base_url: str | None) -> None:
            self.chat = FakeChat()

    monkeypatch.setitem(__import__("sys").modules, "openai", type("openai", (), {"OpenAI": FakeOpenAI}))

    client = OpenAIClient(
        model="qwen3:8b",
        api_key="test-key",
        base_url="http://localhost:11434/v1",
    )
    updates: list[str] = []
    content = client.complete(
        [{"role": "user", "content": "pick a move"}],
        on_thinking_update=updates.append,
    )

    assert any(call.get("stream") for call in calls)
    assert any(not call.get("stream") for call in calls)
    assert client.last_thinking == "Plan"
    assert parse_move(content) == Move(MoveType.DROP, 2)


def test_choose_move_after_briefing_reuses_opening_conversation():
    client = MockLLMClient(
        responses=[
            "Understood, waiting for RED.",
            '{"move_type": "drop", "column": 1}',
        ]
    )
    player = LLMPlayer(client)
    player.send_rules_briefing(Player.YELLOW)
    state = GameState.new()
    state = state.apply_move(Move(MoveType.DROP, 0))

    move = player.choose_move(state)

    assert move == Move(MoveType.DROP, 1)
    assert len(client.calls) == 2
    second_call = client.calls[1]
    assert second_call[0]["role"] == "system"
    assert second_call[1]["role"] == "user"
    assert "SECOND player" in second_call[1]["content"]
    assert second_call[2]["role"] == "assistant"
    assert second_call[3]["role"] == "user"
    assert "Legal moves:" in second_call[3]["content"]


def test_begin_new_game_resets_rules_acknowledged():
    client = MockLLMClient(responses=["ok"])
    player = LLMPlayer(client)
    player.send_rules_briefing(Player.YELLOW)
    player.begin_new_game()
    assert not player.rules_acknowledged


def test_conversation_keeps_prior_moves_within_same_game():
    client = MockLLMClient(
        responses=[
            '{"move_type": "drop", "column": 0}',
            '{"move_type": "drop", "column": 1}',
        ]
    )
    player = LLMPlayer(client)
    first_state = GameState.new()
    second_state = first_state.apply_move(Move(MoveType.DROP, 0))

    player.choose_move(first_state)
    player.choose_move(second_state)

    assert len(client.calls) == 2
    first_call = client.calls[0]
    second_call = client.calls[1]

    assert first_call[0]["role"] == "system"
    assert first_call[1]["role"] == "user"
    assert len(first_call) == 2

    assert second_call[0]["role"] == "system"
    assert second_call[1]["role"] == "user"
    assert second_call[2]["role"] == "assistant"
    assert second_call[3]["role"] == "user"


def test_begin_new_game_clears_conversation_for_next_game():
    client = MockLLMClient(
        responses=[
            '{"move_type": "drop", "column": 0}',
            '{"move_type": "drop", "column": 1}',
        ]
    )
    player = LLMPlayer(client)

    player.choose_move(GameState.new())
    player.begin_new_game()
    player.choose_move(GameState.new())

    assert player.moves == 1
    assert player.games_played == 1
    assert len(client.calls) == 2
    assert len(client.calls[1]) == 2
    assert client.calls[1][0]["role"] == "system"
    assert client.calls[1][1]["role"] == "user"


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
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="API key required"):
        resolve_openai_credentials(api_key="", base_url="")


def test_detect_gemini_provider_from_url() -> None:
    assert detect_llm_provider("") == "openai"
    assert detect_llm_provider("https://generativelanguage.googleapis.com/v1beta/openai/") == "gemini"
    assert detect_llm_provider("generativelanguage.googleapis.com/v1beta") == "gemini"
    assert detect_llm_provider("http://localhost:11434/v1") == "openai_compatible"


def test_normalize_gemini_endpoint_uses_openai_compat_base() -> None:
    assert (
        normalize_llm_endpoint("https://generativelanguage.googleapis.com/v1beta/")
        == "https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    assert normalize_llm_endpoint("http://localhost:11434/v1") == "http://localhost:11434/v1"


def test_resolve_credentials_uses_gemini_env_key(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    key, base = resolve_llm_credentials(
        api_key="",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    assert key == "gemini-test-key"
    assert base == "https://generativelanguage.googleapis.com/v1beta/openai/"


def test_resolve_credentials_requires_gemini_key_without_env(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="Google Gemini"):
        resolve_llm_credentials(
            api_key="",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )


@pytest.mark.parametrize(
    ("base_url", "env_name", "env_value"),
    [
        ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "or-test"),
        ("https://api.x.ai/v1", "XAI_API_KEY", "xai-test"),
        ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "groq-test"),
        ("https://api.mistral.ai/v1", "MISTRAL_API_KEY", "mistral-test"),
        ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY", "deepseek-test"),
        ("https://api.together.xyz/v1", "TOGETHER_API_KEY", "together-test"),
    ],
)
def test_resolve_credentials_uses_provider_env_keys(monkeypatch, base_url, env_name, env_value) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv(env_name, env_value)

    key, resolved_base = resolve_llm_credentials(api_key="", base_url=base_url)

    assert key == env_value
    assert resolved_base == base_url


def test_resolve_credentials_requires_provider_specific_key(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="Groq"):
        resolve_llm_credentials(api_key="", base_url="https://api.groq.com/openai/v1")


def test_openai_client_normalizes_gemini_base_url(monkeypatch) -> None:
    created: dict[str, object] = {}

    class FakeOpenAI:
        def __init__(self, *, api_key: str, base_url: str | None, default_headers=None) -> None:
            created["api_key"] = api_key
            created["base_url"] = base_url
            created["default_headers"] = default_headers

    fake_openai = type("openai", (), {"OpenAI": FakeOpenAI})
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai)

    client = OpenAIClient(
        model="gemini-2.0-flash",
        api_key="test-key",
        base_url="https://generativelanguage.googleapis.com/v1beta/",
    )

    assert client.provider == "gemini"
    assert client.base_url == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert created["base_url"] == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert created["api_key"] == "test-key"


def test_openai_client_adds_openrouter_headers(monkeypatch) -> None:
    created: dict[str, object] = {}

    class FakeOpenAI:
        def __init__(self, *, api_key: str, base_url: str | None, default_headers=None) -> None:
            created["default_headers"] = default_headers

    fake_openai = type("openai", (), {"OpenAI": FakeOpenAI})
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai)

    OpenAIClient(
        model="anthropic/claude-3.5-sonnet",
        api_key="or-test",
        base_url="https://openrouter.ai/api/v1",
    )

    headers = created["default_headers"]
    assert isinstance(headers, dict)
    assert headers["HTTP-Referer"]
    assert headers["X-Title"]


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
    assert llm.requests > 0
    # A new play_game clears in-game LLM history before the next match starts.
    assert llm.games_played == 1
