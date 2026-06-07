"""Local persistence for LLM endpoint URLs used by the GUI."""

from __future__ import annotations

import json
import os
from pathlib import Path


_DEFAULT_ENDPOINTS = (
    "https://generativelanguage.googleapis.com/v1beta/openai/",
    "http://localhost:11434/v1",
    "http://localhost:1234/v1",
    "http://127.0.0.1:11434/v1",
)
_MAX_ENDPOINTS = 20


def settings_path() -> Path:
    """Return the JSON file used to store remembered endpoints."""
    config_home = os.environ.get("XDG_CONFIG_HOME")
    if config_home:
        base = Path(config_home)
    else:
        base = Path.home() / ".config"
    return base / "connect4-mcts" / "llm_endpoints.json"


def _read_saved_endpoints(path: Path) -> list[str]:
    if not path.is_file():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    endpoints = data.get("endpoints", [])
    if not isinstance(endpoints, list):
        return []

    return _dedupe(str(endpoint).strip() for endpoint in endpoints)


def load_endpoints(path: Path | None = None) -> list[str]:
    """Return saved endpoint URLs, most recently used first."""
    file_path = path or settings_path()
    cleaned = _read_saved_endpoints(file_path)
    return cleaned or list(_DEFAULT_ENDPOINTS)


def suggest_endpoints(partial: str, *, path: Path | None = None) -> list[str]:
    """Return endpoints matching ``partial`` (case-insensitive substring)."""
    partial = partial.strip().lower()
    endpoints = load_endpoints(path)
    if not partial:
        return endpoints
    return [endpoint for endpoint in endpoints if partial in endpoint.lower()]


def remember_endpoint(endpoint: str, *, path: Path | None = None) -> None:
    """Move ``endpoint`` to the front of the saved list (blank = OpenAI default)."""
    file_path = path or settings_path()
    normalized = endpoint.strip()
    current = [item for item in _read_saved_endpoints(file_path) if item != normalized]
    if normalized:
        current.insert(0, normalized)
    else:
        current.insert(0, "")

    payload = {"endpoints": current[:_MAX_ENDPOINTS]}
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except OSError:
        return


def _dedupe(endpoints) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for endpoint in endpoints:
        if endpoint in seen:
            continue
        seen.add(endpoint)
        ordered.append(endpoint)
    return ordered
