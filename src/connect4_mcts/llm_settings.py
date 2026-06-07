"""Local persistence and provider metadata for LLM endpoints used by the GUI."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True, slots=True)
class EndpointPreset:
    """A built-in OpenAI-compatible API provider shown in the GUI dropdown."""

    label: str
    url: str
    provider_id: str = "openai_compatible"
    env_keys: tuple[str, ...] = ()
    local: bool = False


# OpenAI-compatible providers (blank url = OpenAI SDK default, api.openai.com).
DEFAULT_ENDPOINT_PRESETS: tuple[EndpointPreset, ...] = (
    EndpointPreset("OpenAI", "", provider_id="openai", env_keys=("OPENAI_API_KEY",)),
    EndpointPreset(
        "Google Gemini",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        provider_id="gemini",
        env_keys=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    ),
    EndpointPreset(
        "OpenRouter (Anthropic, Meta, …)",
        "https://openrouter.ai/api/v1",
        provider_id="openrouter",
        env_keys=("OPENROUTER_API_KEY",),
    ),
    EndpointPreset(
        "xAI Grok",
        "https://api.x.ai/v1",
        provider_id="xai",
        env_keys=("XAI_API_KEY",),
    ),
    EndpointPreset(
        "Groq",
        "https://api.groq.com/openai/v1",
        provider_id="groq",
        env_keys=("GROQ_API_KEY",),
    ),
    EndpointPreset(
        "Mistral",
        "https://api.mistral.ai/v1",
        provider_id="mistral",
        env_keys=("MISTRAL_API_KEY",),
    ),
    EndpointPreset(
        "DeepSeek",
        "https://api.deepseek.com/v1",
        provider_id="deepseek",
        env_keys=("DEEPSEEK_API_KEY",),
    ),
    EndpointPreset(
        "Together AI",
        "https://api.together.xyz/v1",
        provider_id="together",
        env_keys=("TOGETHER_API_KEY",),
    ),
    EndpointPreset(
        "Ollama (local)",
        "http://localhost:11434/v1",
        provider_id="local",
        local=True,
    ),
    EndpointPreset(
        "LM Studio (local)",
        "http://localhost:1234/v1",
        provider_id="local",
        local=True,
    ),
)

_CUSTOM_ENDPOINT = EndpointPreset(
    "Custom OpenAI-compatible",
    "",
    provider_id="openai_compatible",
    env_keys=("OPENAI_API_KEY",),
)

_PROVIDER_HOST_MARKERS: tuple[tuple[str, EndpointPreset], ...] = tuple(
    (urlparse(preset.url).netloc.lower(), preset)
    for preset in DEFAULT_ENDPOINT_PRESETS
    if preset.url
)

_MODEL_PREFIXES_BY_PROVIDER: dict[str, tuple[str, ...]] = {
    "openai": ("gpt-", "gpt", "o1", "o3", "o4", "chatgpt"),
    "gemini": ("gemini-", "models/gemini-"),
    "openrouter": (
        "anthropic/",
        "openai/",
        "google/",
        "meta-llama/",
        "mistralai/",
        "x-ai/",
        "deepseek/",
        "qwen/",
    ),
    "xai": ("grok-", "grok"),
    "groq": ("llama", "mixtral", "gemma", "qwen", "deepseek", "gpt-oss", "kimi"),
    "mistral": ("mistral-", "ministral", "codestral", "pixtral", "devstral"),
    "deepseek": ("deepseek-",),
    "together": ("meta-llama/", "mistralai/", "Qwen/", "deepseek-ai/", "google/"),
}

_OPENROUTER_DEFAULT_HEADERS = {
    "HTTP-Referer": "https://github.com/connect4-mcts",
    "X-Title": "Connect4 MCTS",
}

_PRESET_URLS = {preset.url for preset in DEFAULT_ENDPOINT_PRESETS}
_DEFAULT_ENDPOINTS = tuple(preset.url for preset in DEFAULT_ENDPOINT_PRESETS if preset.url)
_MAX_ENDPOINTS = 20
_DISPLAY_SEPARATOR = " — "
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_OPENAI_HOSTS = frozenset({"api.openai.com"})


def settings_path() -> Path:
    """Return the JSON file used to store remembered endpoints."""
    config_home = os.environ.get("XDG_CONFIG_HOME")
    if config_home:
        base = Path(config_home)
    else:
        base = Path.home() / ".config"
    return base / "connect4-mcts" / "llm_endpoints.json"


def normalize_endpoint_url(url: str) -> str:
    """Return a canonical form used to match presets and saved URLs."""
    return url.strip().rstrip("/")


def _endpoint_host(url: str) -> str:
    text = url.strip()
    if not text:
        return ""
    if "://" not in text:
        text = f"http://{text}"
    return urlparse(text).netloc.lower().split(":")[0]


def is_local_endpoint(url: str | None) -> bool:
    """Return whether ``url`` points at a local OpenAI-compatible server."""
    host = _endpoint_host(url or "")
    return host in _LOCAL_HOSTS


def match_endpoint_preset(base_url: str | None) -> EndpointPreset:
    """Resolve dropdown / URL metadata for ``base_url``."""
    normalized = normalize_endpoint_url(base_url or "")
    if not normalized:
        return DEFAULT_ENDPOINT_PRESETS[0]

    for preset in DEFAULT_ENDPOINT_PRESETS:
        if normalize_endpoint_url(preset.url) == normalized:
            return preset

    host = _endpoint_host(normalized)
    if host in _OPENAI_HOSTS:
        return DEFAULT_ENDPOINT_PRESETS[0]
    if host in _LOCAL_HOSTS:
        return EndpointPreset(
            "Local server",
            normalized if normalized.startswith(("http://", "https://")) else f"http://{normalized}",
            provider_id="local",
            local=True,
        )

    for marker_host, preset in _PROVIDER_HOST_MARKERS:
        if host == marker_host or host.endswith(f".{marker_host}"):
            return preset

    return EndpointPreset(
        _CUSTOM_ENDPOINT.label,
        normalized,
        provider_id=_CUSTOM_ENDPOINT.provider_id,
        env_keys=_CUSTOM_ENDPOINT.env_keys,
    )


def endpoint_provider_label(base_url: str | None) -> str:
    """Return a short human-readable provider name for status messages."""
    preset = match_endpoint_preset(base_url)
    if preset.provider_id == "openai" and not preset.url:
        return "OpenAI (default endpoint)"
    if preset.url and preset in DEFAULT_ENDPOINT_PRESETS:
        return preset.label
    if preset.url:
        return preset.url
    return preset.label


def api_key_hint(base_url: str | None) -> str:
    """Return dialog helper text for the API key field."""
    preset = match_endpoint_preset(base_url)
    if preset.local or is_local_endpoint(base_url):
        return "Optional for local servers (Ollama, LM Studio)."
    if preset.env_keys:
        env_hint = " or ".join(preset.env_keys)
        return f"Required for {preset.label}. Enter a key or set {env_hint}."
    return "Enter your provider API key."


def provider_default_headers(base_url: str | None) -> dict[str, str]:
    """Return optional HTTP headers for provider-specific OpenAI clients."""
    preset = match_endpoint_preset(base_url)
    if preset.provider_id == "openrouter":
        return dict(_OPENROUTER_DEFAULT_HEADERS)
    return {}


def prefer_models_for_endpoint(base_url: str | None, models: list[str]) -> list[str]:
    """Surface likely chat models first for the selected provider."""
    preset = match_endpoint_preset(base_url)
    prefixes = _MODEL_PREFIXES_BY_PROVIDER.get(preset.provider_id, ())
    if not prefixes:
        return list(models)

    lowered = [(model, model.lower()) for model in models]
    preferred = [model for model, text in lowered if text.startswith(prefixes)]
    if not preferred:
        return list(models)
    preferred_set = set(preferred)
    others = [model for model in models if model not in preferred_set]
    return preferred + others


def format_preset_display(preset: EndpointPreset) -> str:
    """Return a dropdown label for a built-in provider preset."""
    if not preset.url:
        return f"{preset.label} (default api.openai.com)"
    return f"{preset.label}{_DISPLAY_SEPARATOR}{preset.url}"


def display_for_url(url: str) -> str:
    """Map a stored URL to its preset label, or return the raw URL."""
    normalized = normalize_endpoint_url(url)
    for preset in DEFAULT_ENDPOINT_PRESETS:
        if normalize_endpoint_url(preset.url) == normalized:
            return format_preset_display(preset)
    return url.strip()


def resolve_endpoint_input(text: str) -> str:
    """Convert a dropdown selection or typed URL into a base URL."""
    normalized = text.strip()
    if not normalized:
        return ""

    for preset in DEFAULT_ENDPOINT_PRESETS:
        if normalized == format_preset_display(preset):
            return preset.url

    if _DISPLAY_SEPARATOR in normalized:
        candidate = normalized.rsplit(_DISPLAY_SEPARATOR, 1)[-1].strip()
        if candidate.startswith(("http://", "https://")):
            return candidate

    return normalized


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


def endpoint_dropdown_values(*, path: Path | None = None) -> list[str]:
    """Return preset labels plus any custom URLs remembered locally."""
    saved_urls = _read_saved_endpoints(path or settings_path())
    values = [format_preset_display(preset) for preset in DEFAULT_ENDPOINT_PRESETS]
    for url in saved_urls:
        if url and url not in _PRESET_URLS and url not in values:
            values.append(url)
    return values


def suggest_endpoints(partial: str, *, path: Path | None = None) -> list[str]:
    """Return dropdown entries matching ``partial`` (label or URL substring)."""
    partial = partial.strip().lower()
    values = endpoint_dropdown_values(path=path)
    if not partial:
        return values
    return [value for value in values if partial in value.lower()]


def remember_endpoint(endpoint: str, *, path: Path | None = None) -> None:
    """Move ``endpoint`` to the front of the saved list (blank = OpenAI default)."""
    file_path = path or settings_path()
    normalized = resolve_endpoint_input(endpoint)
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
