"""Tests for remembered LLM endpoint storage and provider metadata."""

from __future__ import annotations

import pytest

from connect4_mcts.llm_settings import (
    DEFAULT_ENDPOINT_PRESETS,
    api_key_hint,
    display_for_url,
    endpoint_dropdown_values,
    endpoint_provider_label,
    format_preset_display,
    load_endpoints,
    match_endpoint_preset,
    prefer_models_for_endpoint,
    provider_default_headers,
    remember_endpoint,
    resolve_endpoint_input,
    suggest_endpoints,
)


def test_load_endpoints_returns_defaults_when_missing(tmp_path) -> None:
    path = tmp_path / "llm_endpoints.json"

    endpoints = load_endpoints(path)

    assert "https://generativelanguage.googleapis.com/v1beta/openai/" in endpoints
    assert "http://localhost:11434/v1" in endpoints


def test_endpoint_dropdown_values_include_named_presets(tmp_path) -> None:
    path = tmp_path / "llm_endpoints.json"

    values = endpoint_dropdown_values(path=path)

    assert format_preset_display(DEFAULT_ENDPOINT_PRESETS[0]) in values
    assert format_preset_display(DEFAULT_ENDPOINT_PRESETS[1]) in values
    assert any("Google Gemini" in value for value in values)
    assert any("OpenAI" in value for value in values)
    assert any("Grok" in value for value in values)


def test_match_endpoint_preset_recognizes_all_defaults() -> None:
    for preset in DEFAULT_ENDPOINT_PRESETS:
        matched = match_endpoint_preset(preset.url)
        assert matched.provider_id == preset.provider_id
        assert matched.env_keys == preset.env_keys
        assert matched.local == preset.local


def test_match_endpoint_preset_recognizes_provider_hosts() -> None:
    assert match_endpoint_preset("https://api.groq.com/openai/v1").provider_id == "groq"
    assert match_endpoint_preset("https://openrouter.ai/api/v1").provider_id == "openrouter"
    assert match_endpoint_preset("https://api.x.ai/v1").provider_id == "xai"
    assert match_endpoint_preset("https://api.mistral.ai/v1").provider_id == "mistral"
    assert match_endpoint_preset("https://api.deepseek.com/v1").provider_id == "deepseek"
    assert match_endpoint_preset("https://api.together.xyz/v1").provider_id == "together"
    assert match_endpoint_preset("https://api.openai.com/v1").provider_id == "openai"


def test_api_key_hint_mentions_provider_env_vars() -> None:
    groq = next(preset for preset in DEFAULT_ENDPOINT_PRESETS if preset.provider_id == "groq")
    assert "GROQ_API_KEY" in api_key_hint(groq.url)
    assert "Optional" in api_key_hint("http://localhost:11434/v1")


def test_provider_default_headers_for_openrouter() -> None:
    headers = provider_default_headers("https://openrouter.ai/api/v1")
    assert headers["HTTP-Referer"]
    assert headers["X-Title"]


def test_prefer_models_for_endpoint_sorts_by_provider() -> None:
    gemini_url = DEFAULT_ENDPOINT_PRESETS[1].url
    ordered = prefer_models_for_endpoint(
        gemini_url,
        ["other", "gemini-2.5-flash", "gemini-2.0-flash"],
    )
    assert ordered[:2] == ["gemini-2.5-flash", "gemini-2.0-flash"]

    openrouter_url = next(preset.url for preset in DEFAULT_ENDPOINT_PRESETS if preset.provider_id == "openrouter")
    ordered = prefer_models_for_endpoint(
        openrouter_url,
        ["z-model", "anthropic/claude-3.5-sonnet", "openai/gpt-4o-mini"],
    )
    assert ordered[0].startswith("anthropic/")


def test_endpoint_provider_label_uses_preset_name() -> None:
    assert endpoint_provider_label("") == "OpenAI (default endpoint)"
    assert endpoint_provider_label("https://api.groq.com/openai/v1") == "Groq"


def test_resolve_endpoint_input_maps_preset_labels() -> None:
    gemini = DEFAULT_ENDPOINT_PRESETS[1]
    display = format_preset_display(gemini)

    assert resolve_endpoint_input(display) == gemini.url
    assert resolve_endpoint_input("OpenAI (default api.openai.com)") == ""
    assert resolve_endpoint_input("https://example.com/v1") == "https://example.com/v1"
    assert resolve_endpoint_input("xAI Grok — https://api.x.ai/v1") == "https://api.x.ai/v1"


def test_display_for_url_round_trips_presets() -> None:
    for preset in DEFAULT_ENDPOINT_PRESETS:
        assert display_for_url(preset.url) == format_preset_display(preset)


def test_remember_endpoint_persists_and_promotes_most_recent(tmp_path) -> None:
    path = tmp_path / "llm_endpoints.json"

    remember_endpoint("http://localhost:8000/v1", path=path)
    remember_endpoint("http://localhost:11434/v1", path=path)

    assert load_endpoints(path)[:2] == [
        "http://localhost:11434/v1",
        "http://localhost:8000/v1",
    ]


def test_remember_endpoint_stores_raw_url_not_display_label(tmp_path) -> None:
    path = tmp_path / "llm_endpoints.json"
    gemini = format_preset_display(DEFAULT_ENDPOINT_PRESETS[1])

    remember_endpoint(gemini, path=path)

    assert load_endpoints(path)[0] == DEFAULT_ENDPOINT_PRESETS[1].url


def test_suggest_endpoints_filters_by_partial_match(tmp_path) -> None:
    path = tmp_path / "llm_endpoints.json"
    remember_endpoint("http://localhost:11434/v1", path=path)
    remember_endpoint("http://localhost:1234/v1", path=path)
    remember_endpoint("http://192.168.0.5:11434/v1", path=path)

    matches = suggest_endpoints("11434", path=path)

    assert format_preset_display(DEFAULT_ENDPOINT_PRESETS[-2]) in matches
    assert "http://192.168.0.5:11434/v1" in matches
    assert "http://localhost:1234/v1" not in matches

    assert suggest_endpoints("1234", path=path) == [format_preset_display(DEFAULT_ENDPOINT_PRESETS[-1])]


def test_suggest_endpoints_filters_by_provider_label(tmp_path) -> None:
    path = tmp_path / "llm_endpoints.json"

    matches = suggest_endpoints("gemini", path=path)

    assert any("Google Gemini" in value for value in matches)


def test_remember_blank_endpoint_keeps_openai_default_choice(tmp_path) -> None:
    path = tmp_path / "llm_endpoints.json"

    remember_endpoint("", path=path)

    assert load_endpoints(path)[0] == ""
