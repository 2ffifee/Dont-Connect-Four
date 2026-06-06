"""Tests for remembered LLM endpoint storage."""

from __future__ import annotations

from connect4_mcts.llm_settings import load_endpoints, remember_endpoint, suggest_endpoints


def test_load_endpoints_returns_defaults_when_missing(tmp_path) -> None:
    path = tmp_path / "llm_endpoints.json"

    endpoints = load_endpoints(path)

    assert "http://localhost:11434/v1" in endpoints


def test_remember_endpoint_persists_and_promotes_most_recent(tmp_path) -> None:
    path = tmp_path / "llm_endpoints.json"

    remember_endpoint("http://localhost:8000/v1", path=path)
    remember_endpoint("http://localhost:11434/v1", path=path)

    assert load_endpoints(path)[:2] == [
        "http://localhost:11434/v1",
        "http://localhost:8000/v1",
    ]


def test_suggest_endpoints_filters_by_partial_match(tmp_path) -> None:
    path = tmp_path / "llm_endpoints.json"
    remember_endpoint("http://localhost:11434/v1", path=path)
    remember_endpoint("http://localhost:1234/v1", path=path)
    remember_endpoint("http://192.168.0.5:11434/v1", path=path)

    assert suggest_endpoints("11434", path=path) == [
        "http://192.168.0.5:11434/v1",
        "http://localhost:11434/v1",
    ]
    assert suggest_endpoints("1234", path=path) == ["http://localhost:1234/v1"]


def test_remember_blank_endpoint_keeps_openai_default_choice(tmp_path) -> None:
    path = tmp_path / "llm_endpoints.json"

    remember_endpoint("", path=path)

    assert load_endpoints(path)[0] == ""
