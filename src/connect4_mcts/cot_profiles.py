"""Chain-of-thought profiles for supported LLM providers and models.

Each OpenAI-compatible endpoint in the GUI is mapped to a :class:`CoTStrategy`
that documents *how* the provider returns reasoning and which request knobs we
must set.  See provider docs:

* OpenAI Chat Completions — reasoning models (o1/o3/gpt-5) hide CoT text; only
  token counts are exposed.  Visible summaries require the Responses API.
* Google Gemini — ``thinking_config.include_thoughts`` via ``extra_body.google``.
* OpenRouter — ``reasoning.enabled``; response fields ``reasoning`` and
  ``reasoning_details`` (raw JSON required — OpenAI SDK strips them).
* Mistral — ``reasoning_effort``; ``message.content`` is a list of
  ``type: thinking`` / ``type: text`` chunks.
* DeepSeek — ``deepseek-reasoner`` exposes ``reasoning_content``.
* xAI Grok — Chat Completions expose ``reasoning_content`` (summaries on grok-4).
* Groq — fast inference; no visible CoT in chat responses.
* Local (Ollama/LM Studio) — server-dependent ``reasoning_content`` and/or
  inline ``[THINK]`` or ``think`` XML tags; streamed live in the GUI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from connect4_mcts.llm_settings import is_local_endpoint, match_endpoint_preset

# Re-exported model-name heuristics (used by tests and legacy imports).
_STRUCTURED_REASONING_MODEL_MARKERS = (
    "o1-",
    "o1/",
    "/o1",
    "o3-",
    "o3/",
    "/o3",
    "o4-",
    "o4/",
    "/o4",
    "deepseek-r",
    "deepseek-reasoner",
    "deepseek-reason",
    "qwen3",
    "qwq",
    ":thinking",
    "/thinking",
    "thinking-",
    "-thinking",
    "/r1",
    "r1-",
    "reasoner",
    "gpt-5",
)

_GEMINI_THINKING_MODEL_MARKERS = ("gemini-2.5", "gemini-3", "gemini-3.5")

_LOCAL_THINKING_MODEL_MARKERS = (
    "think",
    "reason",
    "r1",
    "ministral",
    "magistral",
    "mistral",
    "qwen",
    "qwq",
    "deepseek",
    "o1",
    "o3",
    "glm",
    "kimi",
    "nemotron",
    "intellect",
    "mimo",
    "pixtral",
)

_THINKING_TAG_PATTERNS = (
    re.compile(r"<\s*think\s*>(.*?)\s*<\s*/\s*think\s*>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<\s*thought\s*>(.*?)\s*<\s*/\s*thought\s*>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<\s*redacted_thinking\s*>(.*?)\s*<\s*/\s*redacted_thinking\s*>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<\s*thinking\s*>(.*?)\s*<\s*/\s*thinking\s*>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<\s*reasoning\s*>(.*?)\s*<\s*/\s*reasoning\s*>", re.DOTALL | re.IGNORECASE),
    re.compile(r"\[\s*THINK\s*\](.*?)\[\s*/\s*THINK\s*\]", re.DOTALL),
)

_FENCE_MARKER_PATTERN = re.compile(r"^[`'\"]{3,}(?:json|JSON)?\s*$")


class CoTStrategy(str, Enum):
    """Extraction and transport strategy for a provider + model pair."""

    NONE = "none"
    """No visible chain-of-thought (panel hidden, no request extras)."""

    OPENAI_HIDDEN = "openai_hidden"
    """OpenAI reasoning models via Chat Completions — CoT not exposed as text."""

    OPENROUTER_REASONING = "openrouter_reasoning"
    """OpenRouter ``reasoning`` + ``reasoning_details`` (raw JSON capture)."""

    GEMINI_INCLUDE_THOUGHTS = "gemini_include_thoughts"
    """Gemini 2.5+ with ``include_thoughts``; thoughts may appear as XML in content."""

    DEEPSEEK_REASONING_CONTENT = "deepseek_reasoning_content"
    """DeepSeek ``message.reasoning_content`` / ``delta.reasoning_content``."""

    MISTRAL_THINKING_CHUNKS = "mistral_thinking_chunks"
    """Mistral API structured ``content`` chunks (``type: thinking``)."""

    XAI_REASONING_CONTENT = "xai_reasoning_content"
    """xAI Grok ``reasoning_content`` field on chat completions."""

    REASONING_CONTENT = "reasoning_content"
    """Generic ``reasoning_content`` / ``reasoning`` API fields (Together, etc.)."""

    INLINE_CONTENT_TAGS = "inline_content_tags"
    """Reasoning only inside assistant ``content`` (``[THINK]``, XML tags)."""

    LOCAL_STREAM = "local_stream"
    """Local server: stream ``reasoning_content`` and inline tags live."""


@dataclass(frozen=True, slots=True)
class CoTProfile:
    """Resolved chain-of-thought behaviour for one endpoint + model."""

    strategy: CoTStrategy
    provider_id: str
    model: str

    @property
    def display(self) -> bool:
        """Whether the GUI should show the chain-of-thought side panel."""
        return self.strategy not in (CoTStrategy.NONE, CoTStrategy.OPENAI_HIDDEN)

    @property
    def streams_live(self) -> bool:
        """Whether CoT is streamed token-by-token (local servers only)."""
        return self.strategy is CoTStrategy.LOCAL_STREAM

    @property
    def blocking(self) -> bool:
        """Whether to wait for one full response instead of streaming."""
        return not self.streams_live and self.strategy not in (CoTStrategy.NONE,)

    @property
    def use_openrouter_raw_response(self) -> bool:
        return self.strategy is CoTStrategy.OPENROUTER_REASONING

    @property
    def request_openrouter_reasoning(self) -> bool:
        return self.strategy is CoTStrategy.OPENROUTER_REASONING

    @property
    def request_gemini_thoughts(self) -> bool:
        return self.strategy is CoTStrategy.GEMINI_INCLUDE_THOUGHTS

    @property
    def request_mistral_reasoning_effort(self) -> str | None:
        if self.strategy is CoTStrategy.MISTRAL_THINKING_CHUNKS:
            return "high"
        return None


def uses_structured_reasoning(model: str) -> bool:
    """Return whether a model id suggests API reasoning fields (o-series, R1, …)."""
    name = model.lower().removeprefix("models/")
    return any(marker in name for marker in _STRUCTURED_REASONING_MODEL_MARKERS)


def gemini_supports_visible_thoughts(model: str) -> bool:
    """Return whether Gemini should be asked for ``include_thoughts``."""
    name = model.lower().removeprefix("models/")
    return any(marker in name for marker in _GEMINI_THINKING_MODEL_MARKERS)


def openrouter_should_request_reasoning(model: str) -> bool:
    """Return whether OpenRouter should receive ``reasoning.enabled``."""
    name = model.lower()
    if uses_structured_reasoning(model):
        return True
    return any(
        marker in name
        for marker in (
            "ministral",
            "mistral",
            "magistral",
            "pixtral",
            "deepseek",
            "qwen",
            "glm",
            "kimi",
            "nemotron",
            "intellect",
            "mimo",
            "anthropic",
            "claude",
        )
    )


def _normalize_model_name(model: str) -> str:
    return model.lower().removeprefix("models/")


def _resolve_openai(model: str) -> CoTStrategy:
    """OpenAI Chat Completions: reasoning text is not returned (Responses API only)."""
    if uses_structured_reasoning(model):
        return CoTStrategy.OPENAI_HIDDEN
    return CoTStrategy.NONE


def _resolve_gemini(model: str) -> CoTStrategy:
    if gemini_supports_visible_thoughts(model):
        return CoTStrategy.GEMINI_INCLUDE_THOUGHTS
    return CoTStrategy.NONE


def _resolve_openrouter(model: str) -> CoTStrategy:
    name = _normalize_model_name(model)
    if not openrouter_should_request_reasoning(model):
        return CoTStrategy.NONE
    # Ministral/Magistral often embed [THINK] in content; OpenRouter also returns
    # reasoning fields — OPENROUTER_REASONING extractor merges both.
    if any(marker in name for marker in ("ministral", "magistral", "mistralai/")):
        return CoTStrategy.OPENROUTER_REASONING
    if "anthropic/" in name and (":thinking" in name or "thinking" in name):
        return CoTStrategy.OPENROUTER_REASONING
    if "deepseek/" in name and any(m in name for m in ("reasoner", "r1", "reason")):
        return CoTStrategy.OPENROUTER_REASONING
    if "qwen/" in name and (":thinking" in name or "qwq" in name or "qwen3" in name):
        return CoTStrategy.OPENROUTER_REASONING
    if uses_structured_reasoning(model):
        return CoTStrategy.OPENROUTER_REASONING
    if any(marker in name for marker in ("deepseek", "qwen", "glm", "kimi", "nemotron")):
        return CoTStrategy.OPENROUTER_REASONING
    return CoTStrategy.OPENROUTER_REASONING


def _resolve_mistral(model: str) -> CoTStrategy:
    name = _normalize_model_name(model)
    if any(marker in name for marker in ("ministral", "magistral", "reasoning", "pixtral")):
        return CoTStrategy.MISTRAL_THINKING_CHUNKS
    if any(marker in name for marker in ("mistral-small", "mistral-medium", "mistral-large")):
        return CoTStrategy.MISTRAL_THINKING_CHUNKS
    return CoTStrategy.NONE


def _resolve_deepseek(model: str) -> CoTStrategy:
    name = _normalize_model_name(model)
    if "reasoner" in name or name in {"deepseek-r1", "deepseek-reasoner"}:
        return CoTStrategy.DEEPSEEK_REASONING_CONTENT
    if "r1" in name and "deepseek" in name:
        return CoTStrategy.DEEPSEEK_REASONING_CONTENT
    return CoTStrategy.NONE


def _resolve_xai(model: str) -> CoTStrategy:
    name = _normalize_model_name(model)
    if "grok" not in name:
        return CoTStrategy.NONE
    if any(marker in name for marker in ("grok-3", "grok-4", "grok-4.3", "reason", "think")):
        return CoTStrategy.XAI_REASONING_CONTENT
    return CoTStrategy.NONE


def _resolve_groq(model: str) -> CoTStrategy:
    """Groq serves fast models without exposing reasoning text in chat responses."""
    del model
    return CoTStrategy.NONE


def _resolve_together(model: str) -> CoTStrategy:
    name = _normalize_model_name(model)
    if uses_structured_reasoning(model):
        return CoTStrategy.REASONING_CONTENT
    if any(
        marker in name
        for marker in (
            "deepseek-r",
            "deepseek-reasoner",
            "qwen3",
            "qwq",
            "ministral",
            "reasoning",
        )
    ):
        return CoTStrategy.REASONING_CONTENT
    return CoTStrategy.NONE


def _resolve_local(model: str) -> CoTStrategy:
    name = _normalize_model_name(model)
    if any(marker in name for marker in _LOCAL_THINKING_MODEL_MARKERS):
        return CoTStrategy.LOCAL_STREAM
    return CoTStrategy.NONE


def _resolve_openai_compatible(model: str) -> CoTStrategy:
    """Custom cloud endpoints: best-effort heuristics."""
    if uses_structured_reasoning(model):
        return CoTStrategy.REASONING_CONTENT
    if openrouter_should_request_reasoning(model):
        return CoTStrategy.REASONING_CONTENT
    name = _normalize_model_name(model)
    if any(marker in name for marker in ("think", "reason", "r1")):
        return CoTStrategy.INLINE_CONTENT_TAGS
    return CoTStrategy.NONE


_PROVIDER_RESOLVERS: dict[str, callable] = {
    "openai": _resolve_openai,
    "gemini": _resolve_gemini,
    "openrouter": _resolve_openrouter,
    "mistral": _resolve_mistral,
    "deepseek": _resolve_deepseek,
    "xai": _resolve_xai,
    "groq": _resolve_groq,
    "together": _resolve_together,
    "local": _resolve_local,
    "openai_compatible": _resolve_openai_compatible,
}


def resolve_cot_profile(base_url: str | None, model: str) -> CoTProfile:
    """Return the hard-mapped CoT profile for an endpoint URL and model id."""
    preset = match_endpoint_preset(base_url)
    provider_id = preset.provider_id
    if preset.local or is_local_endpoint(base_url):
        provider_id = "local"

    resolver = _PROVIDER_RESOLVERS.get(provider_id, _resolve_openai_compatible)
    strategy = resolver(model)
    return CoTProfile(strategy=strategy, provider_id=provider_id, model=model)


# --- Normalization and extraction helpers used by OpenAIClient ----------------

# SentencePiece-style literals leaked when providers return raw token strings.
_SPM_NEWLINE = "\u010a"  # Ċ
_SPM_SPACE = "\u0120"  # Ġ
_SPM_UNDERSCORE = "\u2581"  # ▁


def humanize_reasoning_text(text: str) -> str:
    """Turn provider-specific reasoning token literals into readable plain text.

    OpenRouter / DeepSeek R1 and similar models often expose chain-of-thought as
    undetokenized strings where spaces are ``Ġ`` (U+0120) and newlines are ``Ċ``
    (U+010A).  Gemini may use ``think`` XML blocks; Mistral returns structured
    chunks — both are normalized here after extraction.
    """
    if not text:
        return text

    cleaned = text.replace(_SPM_NEWLINE, "\n").replace(_SPM_SPACE, " ").replace(_SPM_UNDERSCORE, " ")
    cleaned = cleaned.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")

    lines: list[str] = []
    blank_run = 0
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            blank_run += 1
            if blank_run <= 1:
                lines.append("")
            continue
        blank_run = 0
        lines.append(re.sub(r" {2,}", " ", line))

    return "\n".join(lines).strip()


def normalize_thinking_text(text: str | None) -> str | None:
    """Drop markdown fence markers and other non-reasoning noise from CoT text."""
    if not text:
        return None
    cleaned = text.strip()
    if not cleaned:
        return None
    if _FENCE_MARKER_PATTERN.fullmatch(cleaned):
        return None

    cleaned = re.sub(r"^[`'\"]{3,}(?:json|JSON)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[`'\"]{3,}\s*$", "", cleaned)
    cleaned = re.sub(r"\n[`'\"]{3,}(?:json|JSON)?\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = humanize_reasoning_text(cleaned.strip())
    if not cleaned or _FENCE_MARKER_PATTERN.fullmatch(cleaned):
        return None
    return cleaned


def extract_inline_thinking_tags(text: str) -> tuple[str | None, str]:
    """Split assistant content into optional tagged CoT and the remaining text."""
    if not text:
        return None, ""

    thinking_parts: list[str] = []
    remainder = text
    for pattern in _THINKING_TAG_PATTERNS:
        while True:
            match = pattern.search(remainder)
            if not match:
                break
            segment = normalize_thinking_text(match.group(1))
            if segment:
                thinking_parts.append(segment)
            remainder = pattern.sub("", remainder, count=1).strip()

    if thinking_parts:
        return "\n\n".join(thinking_parts), remainder
    return None, text


def reasoning_text_from_value(value: object) -> str | None:
    """Coerce nested API values (reasoning_details items, etc.) to plain text."""
    if value is None:
        return None
    if isinstance(value, str):
        return normalize_thinking_text(value)
    if isinstance(value, list):
        parts = [reasoning_text_from_value(item) for item in value]
        joined = "\n\n".join(part for part in parts if part)
        return joined or None
    if isinstance(value, dict):
        kind = str(value.get("type", "")).lower()
        if "encrypt" in kind:
            return None
        for key in ("text", "summary", "content", "reasoning", "thinking"):
            nested = value.get(key)
            if nested is not None:
                text = reasoning_text_from_value(nested)
                if text:
                    return text
        return None
    for attr in ("text", "summary", "content", "reasoning", "thinking"):
        nested = getattr(value, attr, None)
        if nested is not None:
            text = reasoning_text_from_value(nested)
            if text:
                return text
    return normalize_thinking_text(str(value))


def extract_reasoning_details(details: object) -> str | None:
    if details is None:
        return None
    if isinstance(details, list):
        parts = [reasoning_text_from_value(item) for item in details]
        joined = "\n\n".join(part for part in parts if part)
        return joined or None
    return reasoning_text_from_value(details)


def read_reasoning_from_dict(data: dict[str, object]) -> str | None:
    """Read standard reasoning fields from a message/delta dict."""
    collected: list[str] = []
    for key in ("reasoning_content", "reasoning", "thinking"):
        text = reasoning_text_from_value(data.get(key))
        if text and text not in collected:
            collected.append(text)
    details = extract_reasoning_details(data.get("reasoning_details"))
    if details and details not in collected:
        collected.append(details)
    return "\n\n".join(collected) if collected else None


def flatten_assistant_content(content: object) -> str:
    """Return plain assistant text from a string or Mistral chunk list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for chunk in content:
            if isinstance(chunk, dict):
                if chunk.get("type") == "text":
                    parts.append(str(chunk.get("text", "")))
                continue
            if getattr(chunk, "type", None) == "text":
                parts.append(str(getattr(chunk, "text", "")))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def extract_mistral_thinking_chunks(content: object) -> str | None:
    """Parse Mistral ``message.content`` when it is a list of thinking/text chunks."""
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for chunk in content:
        if not isinstance(chunk, dict):
            chunk_dump = getattr(chunk, "model_dump", None)
            if callable(chunk_dump):
                try:
                    chunk = chunk_dump()
                except Exception:  # noqa: BLE001
                    continue
            else:
                chunk_type = getattr(chunk, "type", None)
                if chunk_type == "thinking":
                    thinking = getattr(chunk, "thinking", None)
                    text = reasoning_text_from_value(thinking)
                    if text:
                        parts.append(text)
                continue
        if chunk.get("type") != "thinking":
            continue
        text = reasoning_text_from_value(chunk.get("thinking"))
        if text:
            parts.append(text)
    return "\n\n".join(parts) if parts else None


def extract_thinking_for_strategy(
    strategy: CoTStrategy,
    *,
    message: object,
    raw_message: dict[str, object] | None,
    content: str,
) -> str | None:
    """Extract visible CoT using the strategy assigned to this provider + model."""
    if strategy in (CoTStrategy.NONE, CoTStrategy.OPENAI_HIDDEN):
        return None

    collected: list[str] = []

    if strategy is CoTStrategy.MISTRAL_THINKING_CHUNKS:
        raw_content = content
        if raw_message is not None:
            nested = raw_message.get("content")
            if nested is not None:
                raw_content = nested
        if not isinstance(raw_content, list):
            raw_content = getattr(message, "content", None)
        mistral = extract_mistral_thinking_chunks(raw_content)
        if mistral:
            collected.append(mistral)

    if strategy in (
        CoTStrategy.OPENROUTER_REASONING,
        CoTStrategy.DEEPSEEK_REASONING_CONTENT,
        CoTStrategy.XAI_REASONING_CONTENT,
        CoTStrategy.REASONING_CONTENT,
        CoTStrategy.LOCAL_STREAM,
        CoTStrategy.GEMINI_INCLUDE_THOUGHTS,
    ):
        for source in (message, raw_message):
            if source is None:
                continue
            if isinstance(source, dict):
                text = read_reasoning_from_dict(source)
            else:
                text = _read_reasoning_from_object(source)
            if text and text not in collected:
                collected.append(text)

    if strategy in (
        CoTStrategy.OPENROUTER_REASONING,
        CoTStrategy.INLINE_CONTENT_TAGS,
        CoTStrategy.LOCAL_STREAM,
        CoTStrategy.GEMINI_INCLUDE_THOUGHTS,
        CoTStrategy.MISTRAL_THINKING_CHUNKS,
    ):
        tagged, _ = extract_inline_thinking_tags(content)
        if tagged and tagged not in collected:
            collected.append(tagged)

    return "\n\n".join(collected) if collected else None


def _part_payload(part: object) -> dict[str, object] | None:
    model_dump = getattr(part, "model_dump", None)
    if not callable(model_dump):
        return None
    try:
        data = model_dump()
    except Exception:  # noqa: BLE001
        return None
    return data if isinstance(data, dict) else None


def _part_dict_blobs(part: object) -> list[dict[str, object]]:
    blobs: list[dict[str, object]] = []
    seen: set[int] = set()

    def add(obj: object) -> None:
        if not isinstance(obj, dict):
            return
        marker = id(obj)
        if marker in seen:
            return
        seen.add(marker)
        blobs.append(obj)

    dump = _part_payload(part)
    if dump is not None:
        add(dump)
        for key in ("message", "delta"):
            nested = dump.get(key)
            if isinstance(nested, dict):
                add(nested)

    extra = getattr(part, "model_extra", None)
    if isinstance(extra, dict):
        add(extra)

    for attr in ("message", "delta"):
        nested = getattr(part, attr, None)
        nested_dump = _part_payload(nested)
        if nested_dump is not None:
            add(nested_dump)

    raw_dict = getattr(part, "__dict__", None)
    if isinstance(raw_dict, dict):
        add(raw_dict)

    return blobs


def _read_reasoning_from_object(part: object) -> str | None:
    collected: list[str] = []
    for blob in _part_dict_blobs(part):
        text = read_reasoning_from_dict(blob)
        if text and text not in collected:
            collected.append(text)

    for attr in ("reasoning_content", "reasoning", "thinking", "reasoning_details"):
        value = getattr(part, attr, None)
        if attr == "reasoning_details":
            text = extract_reasoning_details(value)
        else:
            text = reasoning_text_from_value(value)
        if text and text not in collected:
            collected.append(text)

    if not collected:
        return None
    return "\n\n".join(collected)


def gemini_thinking_extra_body() -> dict[str, object]:
    """Build ``extra_body`` for Gemini ``include_thoughts`` via OpenAI SDK."""
    return {
        "extra_body": {
            "google": {
                "thinking_config": {
                    "include_thoughts": True,
                }
            }
        }
    }


def openrouter_reasoning_extra_body() -> dict[str, object]:
    return {"reasoning": {"enabled": True}}


# Backward-compatible aliases used by GUI and tests
def model_exposes_thinking(base_url: str | None, model: str) -> bool:
    return resolve_cot_profile(base_url, model).display


def cot_streams_live(base_url: str | None, model: str = "") -> bool:
    """Return whether CoT should stream live (local reasoning models)."""
    return resolve_cot_profile(base_url, model or "local").streams_live


def prefers_blocking_completion(base_url: str | None, model: str) -> bool:
    profile = resolve_cot_profile(base_url, model)
    if profile.strategy is CoTStrategy.NONE:
        return False
    return profile.blocking
