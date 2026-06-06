"""LLM-backed agent for the modified (suicide) Connect4 game.

A large language model is harnessed as a plain *policy*: at every turn it is
shown the rules, the board, and the list of legal moves, and asked to pick one
move (zero-shot). The model output is parsed into a :class:`Move`, validated
against the legal moves, and - on a bad/illegal answer - retried with corrective
feedback before falling back to a random legal move.

The model is reached through a small :class:`LLMClient` protocol so the player
is provider-agnostic:

* :class:`OpenAIClient` - any OpenAI-compatible chat endpoint (OpenAI itself or
  a local server exposing the same API), selected via ``base_url``.
* :class:`MockLLMClient` - deterministic, offline client for tests/demos.

Because the objective here is *inverted* (forming a four-in-a-row is bad), the
prompt deliberately and repeatedly stresses that, since LLMs carry a very strong
"connect four = win" prior from standard Connect4.
"""

from __future__ import annotations

import json
import os
import random
import re
from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

from connect4_mcts.game import COLUMNS, ROWS, GameState, Move, MoveType, Player
from connect4_mcts.players.base import MoveSelectionError


Message = dict[str, str]


@runtime_checkable
class LLMClient(Protocol):
    """Minimal chat interface: turn a message list into a text completion."""

    def complete(self, messages: Sequence[Message]) -> str: ...


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
- At the end, whoever has FEWER completed lines of length 4+ WINS. Whoever has
  MORE lines LOSES.
- If both players have the SAME number of lines, compare the lengths of their
  lines (each line is the full run length, e.g. 4, 5, 6...). Sort each player's
  line lengths from longest to shortest and compare position by position (both
  lists have the same length because the line counts match). The player with the
  LONGER line at the first differing position LOSES. If every length matches, the
  game continues (unless the board is full, then it is a draw).
- Therefore you must AVOID completing your own lines and try to FORCE the
  opponent into completing theirs, preferably short lines (length 4).

Fair-turn rule:
- If the player who moved first completes a line, the second player gets exactly
  one more move.
- After that move: if line COUNTS are unequal, the game ends and the player with
  fewer lines wins (using the length tie-break above when counts are equal at
  other end-of-game checks).
- If that move leaves EQUAL line counts for both players, the game continues
  normally instead of ending.

When it is your turn:
- Choose exactly one move from the provided list of legal moves.
- Respond with ONLY a JSON object, no prose, in the form:
  {"move_type": "drop" | "push", "column": <integer 0-7>}
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
        f"You moved {'first' if is_first else 'second'} this game"
        + (" (fair-turn rule applies to you)." if is_first else "."),
    ]

    if include_line_counts:
        counts = state.line_counts()
        red_lengths = state.line_lengths(Player.RED)
        yellow_lengths = state.line_lengths(Player.YELLOW)
        parts.append(
            f"Completed lines so far - RED: {counts[Player.RED]} {list(red_lengths)}, "
            f"YELLOW: {counts[Player.YELLOW]} {list(yellow_lengths)} "
            "(fewer lines wins; equal counts break ties by line lengths, longer loses)."
        )

    parts.append("")
    parts.append("Legal moves:")
    parts.extend(f"  - {_move_label(move)}" for move in legal_moves)
    parts.append("")
    parts.append(
        'Reply with ONLY a JSON object like {"move_type": "drop", "column": 3} '
        "choosing one of the legal moves above."
    )
    return "\n".join(parts)


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
) -> LLMPlayer:
    """Build a fresh :class:`LLMPlayer` for a single game session."""
    return LLMPlayer(
        OpenAIClient(model=model, api_key=api_key, base_url=base_url),
        model_label=model,
        seed=seed,
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

    def begin_new_game(self) -> None:
        """Start a new game: drop in-game conversation history."""
        self._conversation = []
        self.moves = 0
        self.games_played += 1
        self.rules_acknowledged = False

    def send_rules_briefing(self, llm_player: Player = Player.YELLOW) -> str:
        """Send rules and wait for a non-move acknowledgment before the first turn.

        Seeds the in-game conversation with the system rules plus a briefing that
        tells the model its color and that it must not choose a move yet.
        """
        self._conversation = [{"role": "system", "content": self.system_prompt}]
        briefing_user = {"role": "user", "content": render_rules_briefing(llm_player)}
        self.requests += 1
        reply = self.client.complete(self._conversation + [briefing_user]) or ""
        self._conversation.append(briefing_user)
        self._conversation.append({"role": "assistant", "content": reply})
        self.rules_acknowledged = True
        return reply

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
            reply = self.client.complete(messages) or ""
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

    def complete(self, messages: Sequence[Message]) -> str:
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


def resolve_openai_credentials(
    api_key: str | None = None,
    base_url: str | None = None,
) -> tuple[str, str | None]:
    """Resolve credentials for an OpenAI-compatible client.

    The official ``openai`` Python package refuses to connect without *any*
    ``api_key``, even when the target server (Ollama, LM Studio, vLLM) does not
    validate keys. For custom ``base_url`` endpoints we therefore supply a
    harmless placeholder when no key was given.
    """
    resolved_base = (base_url or os.environ.get("OPENAI_BASE_URL") or "").strip() or None
    resolved_key = (api_key or os.environ.get("OPENAI_API_KEY") or "").strip()

    if not resolved_key:
        if resolved_base is not None:
            resolved_key = _LOCAL_API_KEY_PLACEHOLDER
        else:
            raise ValueError(
                "API key required for OpenAI. Enter a key in the dialog or set OPENAI_API_KEY."
            )

    return resolved_key, resolved_base


class OpenAIClient:
    """Client for any OpenAI-compatible chat-completions endpoint.

    Requires the optional ``openai`` package. ``api_key`` and ``base_url`` fall
    back to the ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` environment variables,
    so the same client works against OpenAI or a local compatible server.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float = 60.0,
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
        resolved_key, resolved_base = resolve_openai_credentials(api_key, base_url)
        self._client = OpenAI(api_key=resolved_key, base_url=resolved_base)

    def list_models(self) -> list[str]:  # pragma: no cover - network
        """Return the sorted model ids exposed by the endpoint.

        Also serves as a connection/authentication check: it raises if the
        server is unreachable or the key is rejected.
        """
        response = self._client.models.list()
        return sorted(model.id for model in response.data)

    def complete(self, messages: Sequence[Message]) -> str:  # pragma: no cover - network
        params: dict[str, object] = {"model": self.model, "messages": list(messages)}
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens

        try:
            response = self._client.chat.completions.create(timeout=self.timeout, **params)
        except Exception as exc:  # noqa: BLE001 - adapt to model-specific parameter rules
            adapted = self._adapt_params(params, exc)
            if adapted is None:
                raise
            response = self._client.chat.completions.create(timeout=self.timeout, **adapted)

        return response.choices[0].message.content or ""

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

        return adapted if changed else None
