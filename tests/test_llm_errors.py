"""Tests for user-facing LLM API error formatting."""

from __future__ import annotations

from connect4_mcts.players.llm import format_llm_api_error, is_concerning_llm_error


def test_is_concerning_llm_error_detects_quota_and_auth_failures() -> None:
    assert is_concerning_llm_error(Exception("Error code: 429 - quota exceeded"))
    assert is_concerning_llm_error(Exception("401 Unauthorized"))
    assert not is_concerning_llm_error(Exception("agent returned illegal move"))


def test_format_llm_api_error_summarizes_gemini_quota() -> None:
    exc = Exception(
        "Error code: 429 - [{'error': {'message': 'Quota exceeded for metric: "
        "generativelanguage.googleapis.com/generate_content_free_tier_requests, "
        "limit: 0, model: gemini-2.5-pro'}}]"
    )
    text = format_llm_api_error(exc)
    assert "quota" in text.lower()
    assert "gemini-2.5-pro" in text
