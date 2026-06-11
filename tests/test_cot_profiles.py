"""Tests for provider-specific chain-of-thought profile resolution."""

from __future__ import annotations

from connect4_mcts.cot_profiles import (
    CoTStrategy,
    extract_mistral_thinking_chunks,
    extract_thinking_for_strategy,
    flatten_assistant_content,
    humanize_reasoning_text,
    normalize_thinking_text,
    resolve_cot_profile,
)


def test_openai_gpt4o_has_no_cot() -> None:
    profile = resolve_cot_profile("https://api.openai.com/v1", "gpt-4o-mini")
    assert profile.strategy is CoTStrategy.NONE
    assert not profile.display


def test_openai_o3_hides_cot_in_chat_api() -> None:
    profile = resolve_cot_profile(None, "o3-mini")
    assert profile.strategy is CoTStrategy.OPENAI_HIDDEN
    assert not profile.display


def test_gemini_25_requests_thoughts() -> None:
    url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    profile = resolve_cot_profile(url, "gemini-2.5-flash")
    assert profile.strategy is CoTStrategy.GEMINI_INCLUDE_THOUGHTS
    assert profile.display
    assert profile.blocking
    assert profile.request_gemini_thoughts


def test_openrouter_ministral_uses_openrouter_reasoning() -> None:
    profile = resolve_cot_profile(
        "https://openrouter.ai/api/v1",
        "mistralai/ministral-14b-2512",
    )
    assert profile.strategy is CoTStrategy.OPENROUTER_REASONING
    assert profile.request_openrouter_reasoning
    assert profile.use_openrouter_raw_response


def test_deepseek_reasoner_uses_reasoning_content() -> None:
    profile = resolve_cot_profile("https://api.deepseek.com/v1", "deepseek-reasoner")
    assert profile.strategy is CoTStrategy.DEEPSEEK_REASONING_CONTENT
    assert profile.display


def test_mistral_ministral_uses_thinking_chunks() -> None:
    profile = resolve_cot_profile("https://api.mistral.ai/v1", "ministral-8b-latest")
    assert profile.strategy is CoTStrategy.MISTRAL_THINKING_CHUNKS
    assert profile.request_mistral_reasoning_effort == "high"


def test_xai_grok4_exposes_reasoning_content() -> None:
    profile = resolve_cot_profile("https://api.x.ai/v1", "grok-4")
    assert profile.strategy is CoTStrategy.XAI_REASONING_CONTENT


def test_groq_has_no_visible_cot() -> None:
    profile = resolve_cot_profile("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile")
    assert profile.strategy is CoTStrategy.NONE


def test_local_qwen_streams_live() -> None:
    profile = resolve_cot_profile("http://localhost:11434/v1", "qwen3:8b")
    assert profile.strategy is CoTStrategy.LOCAL_STREAM
    assert profile.streams_live
    assert not profile.blocking


def test_local_llama_has_no_cot() -> None:
    profile = resolve_cot_profile("http://localhost:11434/v1", "llama3")
    assert profile.strategy is CoTStrategy.NONE


def test_extract_mistral_thinking_chunks() -> None:
    content = [
        {"type": "thinking", "thinking": [{"type": "text", "text": "Plan the column."}]},
        {"type": "text", "text": '{"move_type": "drop", "column": 1}'},
    ]
    assert extract_mistral_thinking_chunks(content) == "Plan the column."
    assert flatten_assistant_content(content) == '{"move_type": "drop", "column": 1}'


def test_extract_thinking_for_openrouter_reasoning_field() -> None:
    message = type(
        "Message",
        (),
        {
            "content": '{"move_type": "drop", "column": 0}',
            "reasoning": "Evaluate center columns.",
        },
    )()
    thinking = extract_thinking_for_strategy(
        CoTStrategy.OPENROUTER_REASONING,
        message=message,
        raw_message=None,
        content=message.content,
    )
    assert thinking == "Evaluate center columns."


def test_humanize_reasoning_text_decodes_sentencepiece_literals() -> None:
    raw = "\u010a\u010aThe\u0120best\u0120move\u0120requires\u0120analyzing"
    assert humanize_reasoning_text(raw) == "The best move requires analyzing"

    openrouter_sample = (
        "\u010aOkay,\u0120I\u0120need\u0120to\u0120figure\u0120out\u0120the\u0120best\u0120move"
        "\u0120for\u0120Yellow."
    )
    cleaned = normalize_thinking_text(openrouter_sample)
    assert cleaned is not None
    assert "Okay, I need to figure out the best move" in cleaned
    assert "\u0120" not in cleaned
    assert "\u010a" not in cleaned


def test_extract_thinking_for_openrouter_deepseek_r1_style_reasoning() -> None:
    reasoning = (
        "\u010aOkay,\u0120I\u0120need\u0120to\u0120pick\u0120a\u0120column.\u010a"
        "First,\u0120check\u0120legal\u0120moves."
    )
    message = type(
        "Message",
        (),
        {
            "content": '{"move_type": "drop", "column": 3}',
            "reasoning": reasoning,
        },
    )()
    thinking = extract_thinking_for_strategy(
        CoTStrategy.OPENROUTER_REASONING,
        message=message,
        raw_message=None,
        content=message.content,
    )
    assert thinking is not None
    assert "Okay, I need to pick a column." in thinking
    assert "First, check legal moves." in thinking


def test_extract_thinking_merges_inline_tags_for_ministral() -> None:
    content = '[THINK]Tag plan[/THINK]{"move_type": "drop", "column": 2}'
    thinking = extract_thinking_for_strategy(
        CoTStrategy.OPENROUTER_REASONING,
        message=type("Message", (), {"content": content})(),
        raw_message=None,
        content=content,
    )
    assert "Tag plan" in (thinking or "")
