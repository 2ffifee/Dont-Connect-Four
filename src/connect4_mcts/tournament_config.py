"""Helpers for loading tournament player definitions from TOML configs."""

from __future__ import annotations

from typing import Any


def load_llm_tournament_entries(llm_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand ``configs/tournament_llm.toml`` into tournament player records."""
    server = llm_config.get("server", {})
    base_url = str(server.get("base_url", "") or "")
    api_key = str(server.get("api_key", "") or "")
    timeout = float(server.get("timeout", 300.0))

    entries: list[dict[str, Any]] = []
    for model_entry in llm_config.get("models", []):
        model_id = str(model_entry["id"])
        entries.append(
            {
                "id": f"llm-{model_id}",
                "kind": "llm",
                "model": str(model_entry["model"]),
                "base_url": str(model_entry.get("base_url", base_url) or ""),
                "api_key": str(model_entry.get("api_key", api_key) or ""),
                "timeout": float(model_entry.get("timeout", timeout)),
                "label": str(model_entry.get("label", model_entry["model"])),
            }
        )
    return entries
