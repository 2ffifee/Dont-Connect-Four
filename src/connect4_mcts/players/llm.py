"""LLM-backed agent for the modified (suicide) Connect4 game.

A large language model is harnessed as a plain *policy*: at every turn it is
shown the rules, the board, and the list of legal moves, and asked to pick one
move (zero-shot). The model output is parsed into a :class:`Move`, validated
against the legal moves, and - on a bad/illegal answer - retried with corrective
feedback before falling back to a random legal move.

The model is reached through a small :class:`LLMClient` protocol so the player
is provider-agnostic:

* :class:`OpenAIClient` - any OpenAI-compatible chat endpoint (OpenAI itself,
  Google Gemini via ``generativelanguage.googleapis.com``, or a local server),
  selected via ``base_url`` (provider is inferred from the URL).
* :class:`MockLLMClient` - deterministic, offline client for tests/demos.

Because the objective here is *inverted* (forming a four-in-a-row is bad), the
prompt deliberately and repeatedly stresses that, since LLMs carry a very strong
"connect four = win" prior from standard Connect4.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
from collections.abc import Callable, Sequence
from typing import Literal, Protocol, runtime_checkable

from connect4_mcts.cot_profiles import (
    CoTProfile,
    CoTStrategy,
    cot_streams_live,
    extract_inline_thinking_tags,
    extract_reasoning_details,
    extract_thinking_for_strategy,
    flatten_assistant_content,
    gemini_supports_visible_thoughts,
    gemini_thinking_extra_body,
    model_exposes_thinking,
    normalize_thinking_text,
    openrouter_should_request_reasoning,
    prefers_blocking_completion,
    resolve_cot_profile,
    uses_structured_reasoning,
    _part_payload,
    _read_reasoning_from_object,
)

_extract_reasoning_details = extract_reasoning_details
from connect4_mcts.game import COLUMNS, ROWS, GameState, Move, MoveType, Player
from connect4_mcts.players.base import MoveSelectionError


Message = dict[str, str]

logger = logging.getLogger(__name__)

# Passed to :meth:`LLMClient.complete` to use the client's configured timeout.
_USE_CLIENT_TIMEOUT = object()


def llm_debug_enabled() -> bool:
    """Return whether verbose LLM request/response logging is enabled."""
    return os.environ.get("CONNECT4_LLM_DEBUG", "").strip().lower() in ("1", "true", "yes")


def _log_llm_debug(message: str, *args: object) -> None:
    if llm_debug_enabled():
        logger.info("[llm] " + message, *args)


def _log_llm_warning(message: str, *args: object) -> None:
    logger.warning("[llm] " + message, *args)


def is_concerning_llm_error(exc: Exception) -> bool:
    """Return whether an LLM failure should be surfaced prominently in the GUI."""
    text = str(exc).lower()
    markers = (
        "429",
        "quota",
        "rate limit",
        "resource_exhausted",
        "billing",
        "401",
        "403",
        "authentication",
        "invalid api key",
        "unauthorized",
        "timeout",
        "timed out",
        "connection error",
        "connection refused",
        "service unavailable",
        "503",
        "502",
        "500",
        "overloaded",
    )
    return any(marker in text for marker in markers)


def format_llm_api_error(exc: Exception, *, agent: object | None = None) -> str:
    """Return a short, user-facing summary of an LLM API failure."""
    text = str(exc).strip() or exc.__class__.__name__
    lowered = text.lower()

    if "429" in text or "quota" in lowered or "resource_exhausted" in lowered:
        model_match = re.search(r"model:\s*([\w./:-]+)", text, re.IGNORECASE)
        model_hint = f" ({model_match.group(1)})" if model_match else ""
        if "free_tier" in lowered or "limit: 0" in lowered:
            return f"API quota exceeded{model_hint} — free tier limit reached. Try another model or provider."
        return f"API rate limit / quota exceeded{model_hint}. Retry later or switch model."

    if any(marker in lowered for marker in ("401", "403", "unauthorized", "invalid api key", "authentication")):
        return "API authentication failed — check your API key and provider."

    if any(marker in lowered for marker in ("timeout", "timed out")):
        return "LLM request timed out — try again or use a faster model."

    if any(marker in lowered for marker in ("connection error", "connection refused", "connect")):
        return "Could not reach the LLM server — check URL and network."

    if any(marker in lowered for marker in ("503", "502", "500", "overloaded", "service unavailable")):
        return "LLM provider temporarily unavailable — retry in a moment."

    message_match = re.search(r"'message':\s*'([^']{1,240})'", text)
    if message_match:
        return message_match.group(1)

    if len(text) > 240:
        text = text[:237] + "..."

    client = getattr(agent, "client", None) if agent is not None else None
    debug = getattr(client, "last_completion_debug", None)
    if debug:
        return f"{text} | {debug}"
    return text


@runtime_checkable
class LLMClient(Protocol):
    """Minimal chat interface: turn a message list into a text completion."""

    def complete(
        self,
        messages: Sequence[Message],
        *,
        timeout: float | None | object = _USE_CLIENT_TIMEOUT,
        on_thinking_update: Callable[[str], None] | None = None,
    ) -> str: ...


_CELL_GLYPH = {None: ".", Player.RED: "R", Player.YELLOW: "Y"}

DEFAULT_SYSTEM_PROMPT = """\
You are an expert player of a MODIFIED Connect4 game. Read the rules carefully \
because they DIFFER from standard Connect4.

Board:
- The board has 6 rows and 8 columns. Rows are numbered 0 (top) to 5 (bottom),
  columns 0 (left) to 7 (right). Cells contain R (red), Y (yellow) or . (empty).
- Pieces obey gravity: a column fills from the bottom up.

Moves (on a column that is not full you may choose one of two move types):
- "drop": the piece lands on top of the existing pieces in that column
  (the lowest empty cell).
- "push": the piece is inserted at the BOTTOM of the column; every piece already
  in that column shifts UP by one row.

GOAL (THIS IS INVERTED - READ TWICE):
- Making four of YOUR OWN pieces in a row (horizontal, vertical, or diagonal) is
  BAD, not good. Connecting four does NOT win.
- At the end, count every completed four-in-a-row segment for each player.
  Overlapping segments count separately (e.g. six in a row counts as three
  segments of four). Whoever has FEWER segments WINS; whoever has MORE LOSES.
- If both players have the SAME number of segments when the board is full, the
  game is a draw.
- If any four-in-a-row segment is completed on a move, the owner of that
  segment loses immediately (even if the opponent played the move).
- If both players complete a segment on the same move, the game is a draw.
- Therefore you must AVOID completing your own lines and try to FORCE the
  opponent into completing theirs.

Line counting notes:
- Completed segments are counted cumulatively: breaking a line on the board does
  not reduce totals, and rebuilding the same segment later counts again.
- Lines may be broken at any time; only column-full restrictions apply.

When it is your turn:
- Choose exactly one move from the provided list of legal moves.
- You may reason first in [THINK]...[/THINK] tags (optional).
- Then respond with a JSON object in the form:
  {"move_type": "drop" | "push", "column": <integer 0-7>}
- Put no other prose outside the optional [THINK] block and the JSON object.
"""


def render_rules_briefing(llm_player: Player = Player.YELLOW) -> str:
    """Opening user message: assign color/order and ask for acknowledgment only."""
    if llm_player is Player.YELLOW:
        role = "YELLOW (Y), Player 2 - the SECOND player"
        opponent = "RED (R), Player 1, who moves FIRST"
        wait_for = "RED has played the first move"
    else:
        role = "RED (R), Player 1 - the FIRST player"
        opponent = "YELLOW (Y), Player 2, who moves SECOND"
        wait_for = "the game state for your first move"

    return "\n".join(
        [
            f"You are {role} in this game.",
            f"Your opponent is {opponent}.",
            "",
            "Read the rules in the system message carefully.",
            "",
            "IMPORTANT for this message only:",
            "- Do NOT output a move.",
            "- Do NOT reply with JSON.",
            "- Briefly confirm that you understand the rules and that you will wait until "
            f"{wait_for}. You will receive the board when it is your turn.",
        ]
    )


def render_board(state: GameState) -> str:
    """Return a human/LLM-readable ASCII rendering of the board."""
    header = "        " + " ".join(str(column) for column in range(COLUMNS))
    lines = [header]
    for row in range(ROWS):
        glyphs = " ".join(_CELL_GLYPH[state.board[row][column]] for column in range(COLUMNS))
        lines.append(f"row {row}:  {glyphs}")
    return "\n".join(lines)


def _move_label(move: Move) -> str:
    return f'{move.move_type.value} column {move.column}'


def render_turn(state: GameState, legal_moves: Sequence[Move], include_line_counts: bool = True) -> str:
    mover = state.current_player
    is_first = mover is state.first_player
    parts = [
        render_board(state),
        "",
        f"You are {mover.value.upper()} ({'R' if mover is Player.RED else 'Y'}) and it is your move.",
        f"You moved {'first' if is_first else 'second'} this game.",
    ]

    if include_line_counts:
        counts = state.line_counts()
        parts.append(
            f"Completed four-in-a-row segments so far - RED: {counts[Player.RED]}, "
            f"YELLOW: {counts[Player.YELLOW]} "
            "(overlapping segments count separately; fewer wins)."
        )

    parts.append("")
    parts.append("Legal moves:")
    parts.extend(f"  - {_move_label(move)}" for move in legal_moves)
    parts.append("")
    parts.append(
        'Reply with an optional [THINK]...[/THINK] reasoning block, then a JSON object like '
        '{"move_type": "drop", "column": 3} choosing one of the legal moves above.'
    )
    return "\n".join(parts)


_read_reasoning_from_part = _read_reasoning_from_object


def _debug_log_assistant_message(message: object, *, raw_message: dict[str, object] | None = None) -> None:
    if not llm_debug_enabled():
        return

    content = getattr(message, "content", None) or (raw_message or {}).get("content") or ""
    preview = str(content).replace("\n", "\\n")
    if len(preview) > 400:
        preview = preview[:397] + "..."
    _log_llm_debug("assistant content preview (%d chars): %s", len(str(content)), preview)

    for label, blob in (
        ("message", _part_payload(message)),
        ("raw", raw_message),
    ):
        if not isinstance(blob, dict):
            continue
        reasoning_keys = {
            key: blob.get(key)
            for key in ("reasoning", "reasoning_content", "reasoning_details", "thinking")
            if blob.get(key) is not None
        }
        if reasoning_keys:
            _log_llm_debug("%s reasoning keys: %s", label, reasoning_keys)


_FENCE_MARKER_PATTERN = re.compile(r"^[`'\"]{3,}(?:json|JSON)?\s*$")
_FENCED_JSON_BLOCK = re.compile(
    r"^[`'\"]{3,}(?:json|JSON)?\s*\n(\{.*?\})\s*(?:\n[`'\"]{3,}\s*)?$",
    re.DOTALL | re.IGNORECASE,
)
_MOVE_JSON_PATTERN = re.compile(r'\{[^{}]*"move_type"', re.DOTALL)


_normalize_thinking_text = normalize_thinking_text


def _extract_move_json_remainder(text: str) -> str:
    """Return the move JSON substring, stripping surrounding markdown fences."""
    stripped = text.strip()
    fenced = _FENCED_JSON_BLOCK.match(stripped)
    if fenced:
        return fenced.group(1).strip()

    json_start = _MOVE_JSON_PATTERN.search(stripped)
    if json_start is None:
        return stripped

    remainder = stripped[json_start.start() :].strip()
    remainder = re.sub(r"\n[`'\"]{3,}\s*$", "", remainder)
    return remainder


def extract_thinking(text: str) -> tuple[str | None, str]:
    """Split model output into optional chain-of-thought and the remaining text."""
    if not text:
        return None, ""

    tagged, remainder = extract_inline_thinking_tags(text)
    if tagged:
        return tagged, _extract_move_json_remainder(remainder)

    fenced = _FENCED_JSON_BLOCK.match(text.strip())
    if fenced:
        before = text[: fenced.start()].strip()
        thinking = _normalize_thinking_text(before)
        return thinking, fenced.group(1).strip()

    json_start = _MOVE_JSON_PATTERN.search(text)
    if json_start is not None:
        remainder = _extract_move_json_remainder(text)
        if json_start.start() > 0:
            thinking = _normalize_thinking_text(text[: json_start.start()])
            if thinking:
                return thinking, remainder
        return None, remainder

    return None, text.strip()


def parse_move(text: str) -> Move | None:
    """Best-effort parse of a move from free-form model output.

    Tries a JSON object first, then a loose ``drop/push <column>`` pattern.
    Returns ``None`` when nothing parseable is found.
    """
    if not text:
        return None

    move = _parse_json_move(text)
    if move is not None:
        return move

    match = re.search(r"\b(drop|push)\b\D{0,15}?(\d+)", text, re.IGNORECASE)
    if match:
        return _build_move(match.group(1), int(match.group(2)))

    return None


def _parse_json_move(text: str) -> Move | None:
    for candidate in re.findall(r"\{[^{}]*\}", text, re.DOTALL):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        move = _build_move(str(data.get("move_type", "")), data.get("column"))
        if move is not None:
            return move
    return None


def _build_move(move_type: object, column: object) -> Move | None:
    move_type = str(move_type).strip().lower()
    if move_type not in ("drop", "push"):
        return None
    if isinstance(column, bool) or not isinstance(column, int):
        return None
    if not 0 <= column < COLUMNS:
        return None
    return Move(MoveType(move_type), column)


def create_llm_player(
    model: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    seed: int | None = None,
    timeout: float = 300.0,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> LLMPlayer:
    """Build a fresh :class:`LLMPlayer` for a single game session."""
    return LLMPlayer(
        create_llm_client(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        model_label=model,
        seed=seed,
    )


def create_llm_client(
    model: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 300.0,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> OpenAIClient:
    """Build an LLM HTTP client, inferring the provider from ``base_url``."""
    return OpenAIClient(
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        temperature=temperature,
        max_tokens=max_tokens,
    )


class LLMPlayer:
    """Zero-shot LLM policy that implements the ``Agent`` protocol.

    Parameters
    ----------
    client:
        Backend used to obtain completions (see :class:`LLMClient`).
    model_label:
        Human-readable label for reporting.
    max_attempts:
        How many times to (re)query the model on a turn before falling back.
    on_failure:
        ``"random"`` (default) plays a random legal move when every attempt
        fails; ``"raise"`` raises :class:`MoveSelectionError` instead.
    include_line_counts:
        Whether to show the running line counts in the turn prompt.
    system_prompt:
        Override for the rules prompt (defaults to :data:`DEFAULT_SYSTEM_PROMPT`).
    seed:
        Seed for the random fallback, for reproducibility.

    Within a single game the player keeps an in-memory conversation: rules once in
    the system message, then each turn adds the current board and the model's
    prior replies from **this game only**. Call :meth:`begin_new_game` when a
    game ends or resets so the next game starts without history from earlier
    matches.

    The instance also records simple counters used by the experiment harness to
    compute an *illegal-move rate*: :attr:`requests`, :attr:`unparseable`,
    :attr:`illegal`, :attr:`fallbacks`, :attr:`moves`.
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        model_label: str = "llm",
        max_attempts: int = 3,
        on_failure: str = "random",
        include_line_counts: bool = True,
        system_prompt: str | None = None,
        seed: int | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if on_failure not in ("random", "raise"):
            raise ValueError("on_failure must be 'random' or 'raise'")

        self.client = client
        self.model_label = model_label
        self.max_attempts = max_attempts
        self.on_failure = on_failure
        self.include_line_counts = include_line_counts
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._rng = random.Random(seed)

        self.requests = 0
        self.unparseable = 0
        self.illegal = 0
        self.fallbacks = 0
        self.moves = 0
        self.games_played = 0
        self._conversation: list[Message] = []
        self.rules_acknowledged = False
        self.last_reply: str | None = None
        self.last_thinking: str | None = None

    def _emit_thinking(self, text: str) -> None:
        self.last_thinking = text

    def _request_completion(
        self,
        messages: Sequence[Message],
        *,
        timeout: float | None | object = _USE_CLIENT_TIMEOUT,
    ) -> str:
        complete = self.client.complete
        try:
            return complete(messages, timeout=timeout, on_thinking_update=self._emit_thinking) or ""
        except TypeError:
            return complete(messages, timeout=timeout) or ""

    def _capture_client_reply(self, reply: str) -> str:
        """Store the latest raw reply and any exposed chain-of-thought."""
        self.last_reply = reply
        client_thinking = getattr(self.client, "last_thinking", None)
        thinking, remainder = extract_thinking(reply)
        merged_thinking = client_thinking or thinking
        if merged_thinking and thinking and client_thinking and thinking not in client_thinking:
            merged_thinking = f"{client_thinking.strip()}\n\n{thinking.strip()}".strip()
        self.last_thinking = merged_thinking or self.last_thinking
        move_text = remainder or reply
        _log_llm_debug(
            "reply len=%d thinking_len=%s move_text_len=%d parseable=%s debug=%s",
            len(reply),
            len(self.last_thinking) if self.last_thinking else None,
            len(move_text),
            parse_move(move_text) is not None,
            getattr(self.client, "last_completion_debug", None),
        )
        return move_text

    def begin_new_game(self) -> None:
        """Start a new game: drop in-game conversation history."""
        self._conversation = []
        self.moves = 0
        self.games_played += 1
        self.rules_acknowledged = False
        self.last_reply = None
        self.last_thinking = None

    def send_rules_briefing(self, llm_player: Player = Player.YELLOW) -> str:
        """Send rules and wait for a non-move acknowledgment before the first turn.

        Seeds the in-game conversation with the system rules plus a briefing that
        tells the model its color and that it must not choose a move yet.
        """
        self._conversation = [{"role": "system", "content": self.system_prompt}]
        briefing_user = {"role": "user", "content": render_rules_briefing(llm_player)}
        self.requests += 1
        reply = self._request_completion(self._conversation + [briefing_user], timeout=None)
        reply = self._capture_client_reply(reply)
        self._conversation.append(briefing_user)
        self._conversation.append({"role": "assistant", "content": self.last_reply or reply})
        self.rules_acknowledged = True
        return self.last_reply or reply

    def build_turn_user_message(self, state: GameState) -> Message:
        """Build the user message for the current turn."""
        legal_moves = state.legal_moves()
        if not legal_moves:
            raise MoveSelectionError("cannot build a prompt when no legal moves are available")
        return {
            "role": "user",
            "content": render_turn(state, legal_moves, self.include_line_counts),
        }

    def build_turn_messages(self, state: GameState) -> list[Message]:
        """Build the full message list sent for ``state`` (rules + in-game history)."""
        if not self._conversation:
            self._conversation.append({"role": "system", "content": self.system_prompt})
        turn_user = self.build_turn_user_message(state)
        return [*self._conversation, turn_user]

    @property
    def invalid_responses(self) -> int:
        return self.unparseable + self.illegal

    @property
    def illegal_move_rate(self) -> float:
        """Share of model responses that were unparseable or illegal."""
        if self.requests == 0:
            return 0.0
        return self.invalid_responses / self.requests

    def choose_move(self, state: GameState) -> Move:
        legal_moves = state.legal_moves()
        if not legal_moves:
            raise MoveSelectionError("cannot choose a move when no legal moves are available")

        self.moves += 1
        turn_user = self.build_turn_user_message(state)
        messages = self.build_turn_messages(state)

        for _ in range(self.max_attempts):
            self.requests += 1
            reply = self._request_completion(messages)
            reply = self._capture_client_reply(reply)
            move = parse_move(reply)
            if move is None:
                self.unparseable += 1
            elif move not in legal_moves:
                self.illegal += 1
            else:
                self._record_turn(turn_user, reply)
                return move

            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": self._retry_message(legal_moves)})

        self.fallbacks += 1
        if self.on_failure == "raise":
            raise MoveSelectionError(
                f"LLM '{self.model_label}' failed to return a legal move after {self.max_attempts} attempts"
            )
        move = self._rng.choice(legal_moves)
        self._record_turn(turn_user, self._move_reply(move))
        return move

    def _record_turn(self, turn_user: Message, reply: str) -> None:
        if not self._conversation:
            self._conversation.append({"role": "system", "content": self.system_prompt})
        self._conversation.append(turn_user)
        self._conversation.append({"role": "assistant", "content": reply})

    @staticmethod
    def _move_reply(move: Move) -> str:
        return f'{{"move_type": "{move.move_type.value}", "column": {move.column}}}'

    @staticmethod
    def _retry_message(legal_moves: Sequence[Move]) -> str:
        options = "; ".join(_move_label(move) for move in legal_moves)
        return (
            "That was not a valid, legal move. Respond with ONLY a JSON object "
            '{"move_type": "drop" | "push", "column": <int>} and pick strictly '
            f"from these legal moves: {options}."
        )


class MockLLMClient:
    """Deterministic, offline client for tests and demos.

    Provide either a list of canned ``responses`` (consumed in order, the last
    one repeating once exhausted) or a ``responder`` callable mapping the message
    list to a reply string.
    """

    def __init__(
        self,
        responses: Sequence[str] | None = None,
        responder: Callable[[Sequence[Message]], str] | None = None,
    ) -> None:
        if responses is None and responder is None:
            raise ValueError("provide either responses or a responder")
        self._responses = list(responses) if responses is not None else None
        self._responder = responder
        self.calls: list[list[Message]] = []
        self.last_thinking: str | None = None

    def complete(
        self,
        messages: Sequence[Message],
        *,
        timeout: float | None | object = _USE_CLIENT_TIMEOUT,
        on_thinking_update: Callable[[str], None] | None = None,
    ) -> str:
        del timeout, on_thinking_update
        self.calls.append(list(messages))
        if self._responder is not None:
            return self._responder(messages)
        assert self._responses is not None
        if not self._responses:
            return ""
        if len(self._responses) == 1:
            return self._responses[0]
        return self._responses.pop(0)


_LOCAL_API_KEY_PLACEHOLDER = "ollama"
_GOOGLE_GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
_gemini_thinking_extra_body = gemini_thinking_extra_body


def _gemini_thinking_config(params: dict[str, object]) -> dict[str, object] | None:
    extra_body = params.get("extra_body")
    if not isinstance(extra_body, dict):
        return None
    google = extra_body.get("google")
    if isinstance(google, dict):
        thinking = google.get("thinking_config")
        if isinstance(thinking, dict):
            return thinking
    nested = extra_body.get("extra_body")
    if isinstance(nested, dict):
        google = nested.get("google")
        if isinstance(google, dict):
            thinking = google.get("thinking_config")
            if isinstance(thinking, dict):
                return thinking
    return None

LLMProvider = Literal["openai", "gemini", "openai_compatible"]


def detect_llm_provider(base_url: str | None) -> LLMProvider:
    """Infer the backend type from a user-supplied endpoint URL."""
    from connect4_mcts.llm_settings import match_endpoint_preset

    preset = match_endpoint_preset(base_url)
    if preset.provider_id == "gemini":
        return "gemini"
    if preset.provider_id == "openai" and not (base_url or "").strip():
        return "openai"
    if preset.provider_id == "openai":
        return "openai_compatible"
    return "openai_compatible"


def normalize_llm_endpoint(
    base_url: str | None,
    *,
    provider: LLMProvider | None = None,
) -> str | None:
    """Return a canonical endpoint URL for the inferred provider."""
    text = (base_url or "").strip()
    provider = provider or detect_llm_provider(text or None)
    if provider == "gemini":
        return _GOOGLE_GEMINI_OPENAI_BASE
    return text or None


def resolve_llm_credentials(
    api_key: str | None = None,
    base_url: str | None = None,
) -> tuple[str, str | None]:
    """Resolve API key and normalized endpoint for supported providers."""
    from connect4_mcts.llm_settings import is_local_endpoint, match_endpoint_preset

    preset = match_endpoint_preset(base_url)
    provider = detect_llm_provider(base_url)
    resolved_base = normalize_llm_endpoint(base_url, provider=provider)
    if not resolved_base and preset.provider_id == "openai":
        resolved_base = (os.environ.get("OPENAI_BASE_URL") or "").strip() or None

    resolved_key = (api_key or "").strip()
    if not resolved_key:
        for env_name in preset.env_keys:
            resolved_key = os.environ.get(env_name, "").strip()
            if resolved_key:
                break

    if resolved_key:
        return resolved_key, resolved_base

    if preset.local or is_local_endpoint(resolved_base):
        return _LOCAL_API_KEY_PLACEHOLDER, resolved_base

    env_hint = preset.env_keys[0] if preset.env_keys else "OPENAI_API_KEY"
    if preset.provider_id == "openai" and not resolved_base:
        raise ValueError(
            "API key required for OpenAI. Enter a key in the dialog or set OPENAI_API_KEY."
        )
    if preset.provider_id == "gemini":
        raise ValueError(
            "API key required for Google Gemini. Enter a key in the dialog or set "
            "GEMINI_API_KEY (or GOOGLE_API_KEY)."
        )
    raise ValueError(
        f"API key required for {preset.label}. Enter a key in the dialog or set {env_hint}."
    )


def resolve_openai_credentials(
    api_key: str | None = None,
    base_url: str | None = None,
) -> tuple[str, str | None]:
    """Backward-compatible alias for :func:`resolve_llm_credentials`."""
    return resolve_llm_credentials(api_key, base_url)


class OpenAIClient:
    """Client for OpenAI-compatible chat-completions endpoints.

    Works with OpenAI itself, Google Gemini (when ``base_url`` points at
    ``generativelanguage.googleapis.com``), and local OpenAI-compatible servers.
    The provider is inferred automatically from ``base_url``.

    Requires the optional ``openai`` package. ``api_key`` and ``base_url`` fall
    back to provider-specific environment variables, so the same client works
    against OpenAI, Gemini, or a local compatible server.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float = 300.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise ImportError(
                "OpenAIClient requires the 'openai' package. Install it with "
                "`pip install openai` (or `pip install '.[llm]'`)."
            ) from exc

        # NOTE: ``temperature`` and ``max_tokens`` default to ``None`` (omitted)
        # on purpose. Current OpenAI models (gpt-5.x, o-series) reject a custom
        # ``temperature`` and reject ``max_tokens`` (they require
        # ``max_completion_tokens``), so sending those by default makes every
        # such model 400. They are only sent when explicitly set, and
        # :meth:`complete` adapts them on a parameter error.
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.provider = detect_llm_provider(base_url)
        resolved_key, resolved_base = resolve_llm_credentials(api_key, base_url)
        self.base_url = resolved_base
        from connect4_mcts.llm_settings import match_endpoint_preset, provider_default_headers

        self.endpoint_preset = match_endpoint_preset(resolved_base)
        default_headers = provider_default_headers(resolved_base)
        client_kwargs: dict[str, object] = {"api_key": resolved_key, "base_url": resolved_base}
        if default_headers:
            client_kwargs["default_headers"] = default_headers
        self._client = OpenAI(**client_kwargs)
        self.cot_profile: CoTProfile = resolve_cot_profile(resolved_base, model)
        self.last_thinking: str | None = None
        self.last_completion_debug: str | None = None
        self._last_raw_choice_message: dict[str, object] | None = None

    def _set_completion_debug(self, summary: str) -> None:
        self.last_completion_debug = summary
        _log_llm_debug("%s", summary)

    @staticmethod
    def _summarize_params(params: dict[str, object]) -> str:
        keys = sorted(params.keys())
        mode = "stream" if params.get("stream") else "blocking"
        thinking = "yes" if _gemini_thinking_config(params) else "no"
        return f"mode={mode} keys={keys} gemini_thinking={thinking}"

    def list_models(self) -> list[str]:  # pragma: no cover - network
        """Return the sorted model ids exposed by the endpoint.

        Also serves as a connection/authentication check: it raises if the
        server is unreachable or the key is rejected.
        """
        response = self._client.models.list()
        return sorted(model.id for model in response.data)

    def complete(  # pragma: no cover - network
        self,
        messages: Sequence[Message],
        *,
        timeout: float | None | object = _USE_CLIENT_TIMEOUT,
        on_thinking_update: Callable[[str], None] | None = None,
    ) -> str:
        params = self._build_completion_params(messages)
        call_timeout = self.timeout if timeout is _USE_CLIENT_TIMEOUT else timeout
        param_variants = self._completion_param_variants(params)
        errors: list[Exception] = []
        use_streaming = on_thinking_update is not None and self.cot_profile.streams_live

        _log_llm_debug(
            "complete provider=%s model=%s cot=%s variants=%d streaming=%s blocking=%s timeout=%s",
            self.provider,
            self.model,
            self.cot_profile.strategy.value,
            len(param_variants),
            use_streaming,
            self.cot_profile.blocking,
            call_timeout,
        )

        if use_streaming:
            for index, variant in enumerate(param_variants):
                summary = self._summarize_params(variant)
                try:
                    content = self._complete_streaming(variant, call_timeout, on_thinking_update)
                except Exception as exc:  # noqa: BLE001 - fall back to blocking completion
                    errors.append(exc)
                    _log_llm_warning("streaming failed (%s): %s", summary, exc)
                    continue
                self._set_completion_debug(
                    f"{summary} variant={index} content_len={len(content)} thinking={'yes' if self.last_thinking else 'no'}"
                )
                if content.strip():
                    return content
                _log_llm_warning("streaming returned empty content (%s)", summary)

        for index, variant in enumerate(param_variants):
            summary = self._summarize_params(variant)
            try:
                content = self._complete_blocking(variant, call_timeout, on_thinking_update)
            except Exception as exc:  # noqa: BLE001 - try next param variant
                errors.append(exc)
                _log_llm_warning("blocking failed (%s): %s", summary, exc)
                continue
            self._set_completion_debug(
                f"{summary} variant={index} content_len={len(content)} thinking={'yes' if self.last_thinking else 'no'}"
            )
            if content.strip():
                return content
            _log_llm_warning("blocking returned empty content (%s)", summary)

        if errors:
            self._set_completion_debug(f"failed after {len(errors)} error(s): {errors[-1]}")
            raise errors[-1]
        self._set_completion_debug("failed: all variants returned empty content")
        return ""

    def _build_completion_params(self, messages: Sequence[Message]) -> dict[str, object]:
        params: dict[str, object] = {"model": self.model, "messages": list(messages)}
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        return self._apply_provider_params(params)

    def _completion_param_variants(self, params: dict[str, object]) -> list[dict[str, object]]:
        if not self._has_gemini_thinking(params):
            return [params]
        stripped = self._without_gemini_thinking(params)
        return [params, stripped]

    @staticmethod
    def _has_gemini_thinking(params: dict[str, object]) -> bool:
        return _gemini_thinking_config(params) is not None

    @staticmethod
    def _without_gemini_thinking(params: dict[str, object]) -> dict[str, object]:
        variant = dict(params)
        variant.pop("extra_body", None)
        return variant

    def _complete_blocking(
        self,
        params: dict[str, object],
        timeout: float | None,
        on_thinking_update: Callable[[str], None] | None,
    ) -> str:
        try:
            response = self._create_completion(params, timeout)
        except Exception as exc:  # noqa: BLE001 - adapt to model-specific parameter rules
            adapted = self._adapt_params(params, exc)
            if adapted is None:
                raise
            response = self._create_completion(adapted, timeout)

        return self._finalize_message(response.choices[0].message, on_thinking_update, response=response)

    def _publish_thinking(
        self,
        thinking: str | None,
        on_thinking_update: Callable[[str], None] | None,
    ) -> None:
        thinking = _normalize_thinking_text(thinking)
        if not thinking:
            return
        self.last_thinking = thinking
        if on_thinking_update is not None:
            on_thinking_update(thinking)

    def _complete_streaming(
        self,
        params: dict[str, object],
        timeout: float | None,
        on_thinking_update: Callable[[str], None],
    ) -> str:
        stream_params = dict(params)
        stream_params["stream"] = True
        try:
            stream = self._create_completion(stream_params, timeout)
        except Exception as exc:  # noqa: BLE001 - adapt to model-specific parameter rules
            adapted = self._adapt_params(stream_params, exc)
            if adapted is None:
                raise
            stream = self._create_completion(adapted, timeout)

        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if llm_debug_enabled():
                delta_dump = _part_payload(delta) or {}
                delta_keys = {
                    key: delta_dump.get(key)
                    for key in ("content", "reasoning", "reasoning_content", "reasoning_details", "thinking")
                    if delta_dump.get(key) is not None
                }
                if delta_keys:
                    _log_llm_debug("stream delta keys: %s", delta_keys)
            reasoning = _read_reasoning_from_part(delta)
            if reasoning:
                reasoning_parts.append(reasoning)
                self._publish_thinking("".join(reasoning_parts), on_thinking_update)
            if delta.content:
                content_parts.append(delta.content)
                content = "".join(content_parts)
                thinking, _ = extract_thinking(content)
                if thinking and not reasoning_parts:
                    self._publish_thinking(thinking, on_thinking_update)

        content = "".join(content_parts)
        thinking: str | None = None
        if reasoning_parts:
            thinking = "".join(reasoning_parts).strip() or None
        tagged_thinking, _ = extract_thinking(content)
        if tagged_thinking:
            if thinking and tagged_thinking not in thinking:
                thinking = f"{thinking.strip()}\n\n{tagged_thinking.strip()}".strip()
            elif not thinking:
                thinking = tagged_thinking
        self._publish_thinking(thinking, on_thinking_update)
        return content

    def _finalize_message(
        self,
        message: object,
        on_thinking_update: Callable[[str], None] | None,
        *,
        response: object | None = None,
    ) -> str:
        raw_message: dict[str, object] | None = getattr(self, "_last_raw_choice_message", None)
        if response is not None and raw_message is None:
            choice_dump = _part_payload(response.choices[0])
            if isinstance(choice_dump, dict):
                nested = choice_dump.get("message")
                if isinstance(nested, dict):
                    raw_message = nested

        raw_content = message.content
        if not raw_content and raw_message is not None:
            raw_content = raw_message.get("content")
        content = flatten_assistant_content(raw_content)

        _debug_log_assistant_message(message, raw_message=raw_message)

        thinking = extract_thinking_for_strategy(
            self.cot_profile.strategy,
            message=message,
            raw_message=raw_message,
            content=content,
        )
        tagged_thinking, tagged_remainder = extract_thinking(content)
        if thinking is None:
            thinking = tagged_thinking
        elif tagged_thinking and tagged_thinking not in thinking:
            thinking = _normalize_thinking_text(f"{thinking.strip()}\n\n{tagged_thinking.strip()}")

        self._publish_thinking(thinking, on_thinking_update)
        return tagged_remainder or _extract_move_json_remainder(content)

    def _apply_provider_params(self, params: dict[str, object]) -> dict[str, object]:
        """Add provider-specific request fields from the resolved CoT profile."""
        merged = dict(params)
        profile = self.cot_profile
        extra_body: dict[str, object] = {}

        if profile.request_gemini_thoughts:
            extra_body.update(gemini_thinking_extra_body())

        if profile.request_openrouter_reasoning:
            extra_body["reasoning"] = {"enabled": True}

        effort = profile.request_mistral_reasoning_effort
        if effort is not None:
            merged["reasoning_effort"] = effort

        if extra_body:
            merged["extra_body"] = extra_body
        return merged

    def _create_completion(self, params: dict[str, object], timeout: float | None) -> object:
        self._last_raw_choice_message = None
        use_raw_openrouter = self.cot_profile.use_openrouter_raw_response
        if use_raw_openrouter:
            completions = self._client.chat.completions
            raw_api = getattr(completions, "with_raw_response", None)
            if raw_api is not None:
                if callable(raw_api):
                    raw_api = raw_api()
                create = getattr(raw_api, "create", None)
            else:
                create = None
            if create is not None:
                if timeout is None:
                    raw = create(**params)
                else:
                    raw = create(timeout=timeout, **params)
                try:
                    payload = json.loads(raw.text)
                except json.JSONDecodeError:
                    return raw.parse()
                choices = payload.get("choices")
                if isinstance(choices, list) and choices:
                    first = choices[0]
                    if isinstance(first, dict):
                        message = first.get("message")
                        if isinstance(message, dict):
                            self._last_raw_choice_message = message
                return raw.parse()

        if timeout is None:
            return self._client.chat.completions.create(**params)
        return self._client.chat.completions.create(timeout=timeout, **params)

    @staticmethod
    def _adapt_params(params: dict[str, object], exc: Exception) -> dict[str, object] | None:
        """Retry-friendly fix-ups for model-specific parameter rules.

        Returns adjusted params if the error looks like an unsupported-parameter
        complaint we can work around (``max_tokens`` -> ``max_completion_tokens``,
        unsupported ``temperature``), otherwise ``None`` to re-raise.
        """
        text = str(exc).lower()
        adapted = dict(params)
        changed = False

        if "max_tokens" in adapted and ("max_completion_tokens" in text or "max_tokens" in text):
            adapted["max_completion_tokens"] = adapted.pop("max_tokens")
            changed = True
        if "temperature" in adapted and "temperature" in text:
            adapted.pop("temperature")
            changed = True
        if "extra_body" in adapted and any(
            marker in text for marker in ("extra_body", "thinking", "thought", "google", "invalid", "unknown", "reasoning")
        ):
            adapted.pop("extra_body", None)
            changed = True

        return adapted if changed else None
