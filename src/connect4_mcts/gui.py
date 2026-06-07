"""Pygame interface for playing against an agent."""

from __future__ import annotations

import argparse
import os
import random
import threading
from collections.abc import Sequence
from dataclasses import dataclass

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from connect4_mcts.game import COLUMNS, ROWS, GameResult, GameState, GameStatus, IllegalMoveError, Move, MoveType, Player
from connect4_mcts.players import AGENT_CHOICES, Agent, AgentName, create_agent, format_agent_name


CELL_SIZE = 72
BOARD_WIDTH = COLUMNS * CELL_SIZE
BOARD_HEIGHT = ROWS * CELL_SIZE
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 760
BOARD_LEFT = (WINDOW_WIDTH - BOARD_WIDTH) // 2
BOARD_TOP = 118
MARGIN = 24
TOP_BAR_HEIGHT = 104
FOOTER_RESERVE = 76
THINKING_PANEL_WIDTH = 272
THINKING_PANEL_GAP = 20
THINKING_PANEL_PADDING = 10
THINKING_LINE_HEIGHT = 20
THINKING_PANEL_MIN_WIDTH = 180
MIN_CELL_SIZE = 28
BUTTON_WIDTH = 96
BUTTON_HEIGHT = 38
BUTTON_GAP = 8
SETUP_BUTTON_WIDTH = 132
SETUP_BUTTON_HEIGHT = 42
SETUP_BUTTON_GAP = 16

BACKGROUND = (245, 247, 250)
BOARD_COLOR = (30, 91, 168)
BOARD_EDGE = (22, 61, 119)
EMPTY_CELL = (238, 242, 246)
RED_CELL = (210, 54, 68)
YELLOW_CELL = (242, 190, 65)
TEXT = (25, 32, 44)
MUTED_TEXT = (94, 108, 132)
BUTTON = (226, 232, 240)
BUTTON_ACTIVE = (47, 112, 193)
BUTTON_BORDER = (152, 164, 180)
WHITE = (255, 255, 255)
ERROR = (171, 39, 50)

ScreenMode = str


class ScrollableTextPanel:
    """Scrollable read-only text area with optional auto-follow while streaming."""

    heading = "LLM chain-of-thought"

    def __init__(self) -> None:
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.text = ""
        self.scroll_y = 0
        self._follow_bottom = True

    def clear(self) -> None:
        self.text = ""
        self.scroll_y = 0
        self._follow_bottom = True

    def set_text(self, text: str | None) -> None:
        new = text or ""
        if new == self.text:
            return
        self.text = new
        if self._follow_bottom:
            self.scroll_y = 10**9

    def handle_wheel(self, delta_y: int, font: pygame.font.Font) -> None:
        content_height = self._content_height(font)
        visible = self._content_viewport_height()
        max_scroll = max(0, content_height - visible)
        self._follow_bottom = False
        current = min(max_scroll, self.scroll_y)
        self.scroll_y = max(0, min(max_scroll, current - delta_y * THINKING_LINE_HEIGHT))
        if max_scroll > 0 and self.scroll_y >= max_scroll:
            self._follow_bottom = True

    def _content_viewport_height(self) -> int:
        heading_space = 28
        return max(0, self.rect.height - 2 * THINKING_PANEL_PADDING - heading_space)

    def _layout_lines(self, font: pygame.font.Font) -> list[str]:
        inner_width = max(1, self.rect.width - 2 * THINKING_PANEL_PADDING)
        return _wrap_text_preserve_newlines(self.text, font, inner_width)

    def _content_height(self, font: pygame.font.Font) -> int:
        if not self.text:
            return 0
        return len(self._layout_lines(font)) * THINKING_LINE_HEIGHT

    def draw(self, surface: pygame.Surface, font: pygame.font.Font, *, placeholder: str | None = None) -> None:
        if self.rect.width <= 0 or self.rect.height <= 0:
            return

        pygame.draw.rect(surface, BUTTON, self.rect, border_radius=6)
        pygame.draw.rect(surface, BUTTON_BORDER, self.rect, width=1, border_radius=6)

        heading = font.render(f"{self.heading}:", True, TEXT)
        surface.blit(heading, (self.rect.x + THINKING_PANEL_PADDING, self.rect.y + THINKING_PANEL_PADDING))

        content_top = self.rect.y + THINKING_PANEL_PADDING + 28
        content_rect = pygame.Rect(
            self.rect.x + THINKING_PANEL_PADDING,
            content_top,
            self.rect.width - 2 * THINKING_PANEL_PADDING,
            self._content_viewport_height(),
        )

        display_text = self.text or placeholder or ""
        lines = _wrap_text_preserve_newlines(display_text, font, content_rect.width) if display_text else []
        content_height = len(lines) * THINKING_LINE_HEIGHT
        visible = content_rect.height
        max_scroll = max(0, content_height - visible)
        if self._follow_bottom:
            self.scroll_y = max_scroll
        else:
            self.scroll_y = max(0, min(max_scroll, self.scroll_y))

        previous_clip = surface.get_clip()
        surface.set_clip(content_rect)
        y = content_top - self.scroll_y
        for line in lines:
            if y + THINKING_LINE_HEIGHT >= content_rect.top and y <= content_rect.bottom:
                if line:
                    line_surface = font.render(line, True, MUTED_TEXT)
                    surface.blit(line_surface, (content_rect.x, y))
            y += THINKING_LINE_HEIGHT
        surface.set_clip(previous_clip)

        if max_scroll > 0:
            track = pygame.Rect(self.rect.right - 8, content_rect.top, 4, content_rect.height)
            pygame.draw.rect(surface, BOARD_EDGE, track, border_radius=2)
            thumb_height = max(16, int(content_rect.height * visible / content_height))
            thumb_y = content_rect.top + int((content_rect.height - thumb_height) * self.scroll_y / max_scroll)
            thumb = pygame.Rect(track.x, thumb_y, track.width, thumb_height)
            pygame.draw.rect(surface, MUTED_TEXT, thumb, border_radius=2)


@dataclass(frozen=True, slots=True)
class BoardLayout:
    left: int = BOARD_LEFT
    top: int = BOARD_TOP
    cell_size: int = CELL_SIZE

    @property
    def width(self) -> int:
        return COLUMNS * self.cell_size

    @property
    def height(self) -> int:
        return ROWS * self.cell_size

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(self.left, self.top, self.width, self.height)


@dataclass(slots=True)
class GuiConfig:
    human: Player = Player.RED
    agent_name: AgentName = "random"
    seed: int | None = None
    depth: int = 3
    two_player: bool = False
    loaded_agent: Agent | None = None
    loaded_label: str | None = None
    llm_agent: Agent | None = None
    llm_model: str | None = None
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_label: str | None = None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Play Don't Connect 4 against an agent.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for the random player.")
    parser.add_argument("--human", choices=("red", "yellow"), default="red", help="Human player color.")
    parser.add_argument("--agent", choices=AGENT_CHOICES, default="random", help="Initial opponent selection.")
    parser.add_argument("--depth", type=int, default=3, help="Search depth for minimax.")
    parser.add_argument("--two-player", action="store_true", help="Start in local two-player mode.")
    parser.add_argument("--load", default=None, help="Path to a pickled trained player to use as opponent.")
    parser.add_argument("--llm", action="store_true", help="Use an LLM (OpenAI-compatible) opponent.")
    parser.add_argument("--llm-model", default="gpt-4o-mini", help="LLM model name (with --llm).")
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help="Base URL of an OpenAI-compatible server, e.g. http://localhost:11434/v1 (with --llm). "
        "API key is read from OPENAI_API_KEY.",
    )
    args = parser.parse_args(argv)

    config = GuiConfig(
        human=Player(args.human),
        agent_name=args.agent,
        seed=args.seed,
        depth=args.depth,
        two_player=args.two_player,
    )
    if args.load:
        from connect4_mcts.training import load_player

        config.loaded_agent = load_player(args.load)
        config.loaded_label = os.path.basename(args.load)
        config.agent_name = "loaded"

    if args.llm:
        from connect4_mcts.players.llm import create_llm_player

        config.llm_model = args.llm_model
        config.llm_base_url = args.llm_base_url
        config.llm_agent = create_llm_player(args.llm_model, base_url=args.llm_base_url)
        config.llm_label = f"LLM: {args.llm_model}"
        config.agent_name = "llm"

    HumanVsAgentGui(config=config).run()
    return 0


class HumanVsRandomGui:
    def __init__(self, human: Player = Player.RED, seed: int | None = None) -> None:
        self._delegate = HumanVsAgentGui(config=GuiConfig(human=human, agent_name="random", seed=seed))

    def run(self) -> None:
        self._delegate.run()


class HumanVsAgentGui:
    def __init__(self, config: GuiConfig | None = None) -> None:
        pygame.init()
        self.width = WINDOW_WIDTH
        self.height = WINDOW_HEIGHT
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption("Don't Connect 4")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Segoe UI", 24)
        self.small_font = pygame.font.SysFont("Segoe UI", 18)
        self.large_font = pygame.font.SysFont("Segoe UI", 30, bold=True)
        self.config = config or GuiConfig()
        self.mode: ScreenMode = "setup"
        self.state = GameState.new(first_player=Player.RED)
        self.agent: Agent = self._make_agent()
        self.selected_move_type = MoveType.DROP
        self.message = ""
        self.llm_awaiting_rules_ack = False
        self.llm_thinking_panel = ScrollableTextPanel()
        self._agent_busy = False
        self._async_generation = 0
        self._async_lock = threading.Lock()
        self._async_result: tuple[str, object] | None = None
        self.layout = self._compute_board_layout()

    def run(self) -> None:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    self._handle_resize(event.w, event.h)
                elif event.type == pygame.KEYDOWN:
                    self._handle_key(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)

                elif event.type == pygame.MOUSEWHEEL:
                    self._handle_wheel(event.y)

            self._poll_llm_thinking()
            self._process_async_results()
            self._draw()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

    def _handle_resize(self, width: int, height: int) -> None:
        # SDL2/pygame 2 resizes the display surface automatically. Calling
        # set_mode() here would re-request the size every event and fight the
        # window manager (notably on Wayland/Hyprland), so we only refresh our
        # cached surface and recompute the layout.
        self.width = max(1, width)
        self.height = max(1, height)
        surface = pygame.display.get_surface()
        if surface is not None:
            self.screen = surface
        self.layout = self._compute_board_layout()

    def _handle_key(self, key: int) -> None:
        if key == pygame.K_ESCAPE:
            pygame.event.post(pygame.event.Event(pygame.QUIT))
        elif self.mode == "setup":
            if key in {pygame.K_RETURN, pygame.K_KP_ENTER}:
                self._start_game()
        elif key == pygame.K_SPACE:
            self._toggle_move_type()
        elif key == pygame.K_r:
            self._reset()

    def _handle_click(self, position: tuple[int, int]) -> None:
        if self.mode == "setup":
            self._handle_setup_click(position)
            return

        if self._drop_button_rect().collidepoint(position):
            self.selected_move_type = MoveType.DROP
            return
        if self._push_button_rect().collidepoint(position):
            self.selected_move_type = MoveType.PUSH
            return
        if self._reset_button_rect().collidepoint(position):
            self._reset()
            return
        if self._menu_button_rect().collidepoint(position):
            self._invalidate_async_work()
            self.mode = "setup"
            self.message = ""
            self.llm_thinking_panel.clear()
            return

        if self.state.status is GameStatus.FINISHED:
            return
        if self._llm_rules_gate_active():
            return
        if not self.config.two_player and self.state.current_player is not self.config.human:
            return

        column = column_from_position(position, self.layout)
        if column is None:
            return

        try:
            self._apply_human_move(Move(self.selected_move_type, column))
        except IllegalMoveError:
            self.message = "Illegal move"

    def _apply_human_move(self, move: Move) -> None:
        self.state = self.state.apply_move(move)
        self.message = ""

        if self.config.two_player:
            return

        if self.state.status is not GameStatus.FINISHED and self.state.current_player is not self.config.human:
            self._schedule_agent_turn()

    def _invalidate_async_work(self) -> None:
        self._async_generation += 1
        self._agent_busy = False
        with self._async_lock:
            self._async_result = None

    def _process_async_results(self) -> None:
        with self._async_lock:
            result = self._async_result
            self._async_result = None
        if result is None:
            return

        self._agent_busy = False
        self.llm_awaiting_rules_ack = False
        kind, payload = result

        if kind == "briefing":
            ack = str(payload)
            self.message = f"LLM ready: {_shorten(ack, 72)}"
            thinking = getattr(self.agent, "last_thinking", None)
            if thinking:
                self.llm_thinking_panel.set_text(str(thinking))
            self._maybe_schedule_agent_turn()
            return

        if kind == "briefing_error":
            self.message = f"LLM rules briefing failed ({_short_error(payload)})"
            return

        if kind == "error":
            exc = payload
            legal_moves = self.state.legal_moves()
            if not legal_moves:
                self.message = f"Agent error ({_short_error(exc)})"
                return
            move = random.choice(legal_moves)
            self.state = self.state.apply_move(move)
            name = self._opponent_label()
            self.message = f"{name} error ({_short_error(exc)}) - played random {format_move(move)}"
            self._maybe_schedule_agent_turn()
            return

        move, thinking = payload
        try:
            if not self.state.is_legal_move(move):
                raise IllegalMoveError(f"agent returned illegal move: {move}")
        except IllegalMoveError as exc:
            legal_moves = self.state.legal_moves()
            if not legal_moves:
                self.message = str(exc)
                return
            move = random.choice(legal_moves)
            name = self._opponent_label()
            self.message = f"{name} illegal move - played random {format_move(move)}"
        else:
            name = self._opponent_label()
            self.message = f"{name}: {format_move(move)}"

        if thinking:
            self.llm_thinking_panel.set_text(str(thinking))

        self.state = self.state.apply_move(move)
        self._maybe_schedule_agent_turn()

    def _opponent_label(self) -> str:
        if self.config.agent_name == "loaded" and self.config.loaded_label:
            return _shorten(self.config.loaded_label, 28)
        if self.config.agent_name == "llm" and self.config.llm_label:
            return _shorten(self.config.llm_label, 28)
        return format_agent_name(self.config.agent_name)

    def _maybe_schedule_agent_turn(self) -> None:
        if self.config.two_player or self.state.status is GameStatus.FINISHED:
            return
        if self.state.current_player is self.config.human:
            return
        if self._agent_busy:
            return
        self._schedule_agent_turn()

    def _schedule_agent_turn(self) -> None:
        if self._agent_busy or self.state.status is GameStatus.FINISHED:
            return
        if self.config.two_player or self.state.current_player is self.config.human:
            return

        self._agent_busy = True
        self.message = f"{self._opponent_label()} thinking..."
        self.llm_thinking_panel.clear()
        state = self.state
        agent = self.agent
        generation = self._async_generation

        def worker() -> tuple[str, object]:
            try:
                move = agent.choose_move(state)
                thinking = getattr(agent, "last_thinking", None)
                return ("move", (move, thinking))
            except Exception as exc:  # noqa: BLE001 - report in the UI thread
                return ("error", exc)

        def run() -> None:
            result = worker()
            if generation != self._async_generation:
                return
            with self._async_lock:
                self._async_result = result

        threading.Thread(target=run, daemon=True).start()

    def _schedule_rules_briefing(self) -> None:
        send_briefing = getattr(self.agent, "send_rules_briefing", None)
        if not callable(send_briefing):
            self._maybe_schedule_agent_turn()
            return

        llm_color = self._llm_agent_color()
        self.llm_awaiting_rules_ack = True
        self.message = "Sending rules to LLM..."
        self.llm_thinking_panel.clear()
        agent = self.agent
        generation = self._async_generation

        def worker() -> tuple[str, object]:
            try:
                ack = send_briefing(llm_color)
                return ("briefing", ack)
            except Exception as exc:  # noqa: BLE001 - report in the UI thread
                return ("briefing_error", exc)

        def run() -> None:
            result = worker()
            if generation != self._async_generation:
                return
            with self._async_lock:
                self._async_result = result

        threading.Thread(target=run, daemon=True).start()

    def _refresh_display(self) -> None:
        self._draw()
        pygame.display.flip()

    def _reset(self) -> None:
        self._start_game()

    def _start_game(self) -> None:
        self._invalidate_async_work()
        self.state = GameState.new(first_player=Player.RED)
        self.agent = self._make_agent()
        self._reset_llm_conversation()
        self.selected_move_type = MoveType.DROP
        self.mode = "game"
        self.message = ""
        self.llm_thinking_panel.clear()
        self.layout = self._compute_board_layout()
        if self._needs_llm_rules_briefing():
            self._schedule_rules_briefing()
            return
        self._maybe_schedule_agent_turn()

    def _llm_agent_color(self) -> Player:
        return self.config.human.opponent

    def _needs_llm_rules_briefing(self) -> bool:
        return self.config.agent_name == "llm" and not self.config.two_player

    def _llm_rules_gate_active(self) -> bool:
        if not self._needs_llm_rules_briefing():
            return False
        if self.llm_awaiting_rules_ack:
            return True
        if not getattr(self.agent, "rules_acknowledged", True):
            # Human RED waits for YELLOW LLM briefing before the opening move.
            return self.config.human is Player.RED and self._llm_agent_color() is Player.YELLOW
        return False

    def _make_agent(self) -> Agent:
        if self.config.agent_name == "loaded" and self.config.loaded_agent is not None:
            return self.config.loaded_agent
        if self.config.agent_name == "llm":
            if self.config.llm_agent is None and self.config.llm_model is not None:
                self.config.llm_agent = self._make_llm_agent()
            if self.config.llm_agent is not None:
                return self.config.llm_agent
        return create_agent(self.config.agent_name, seed=self.config.seed, depth=self.config.depth)

    def _make_llm_agent(self) -> Agent:
        from connect4_mcts.players.llm import create_llm_player

        return create_llm_player(
            self.config.llm_model,
            api_key=self.config.llm_api_key,
            base_url=self.config.llm_base_url,
            seed=self.config.seed,
        )

    def _reset_llm_conversation(self) -> None:
        if self.config.agent_name != "llm":
            return
        begin_new_game = getattr(self.agent, "begin_new_game", None)
        if callable(begin_new_game):
            begin_new_game()

    def _load_player_from_file(self) -> None:
        path = _prompt_player_file()
        if not path:
            self.message = "Load cancelled (or no file dialog)"
            return

        from connect4_mcts.training import load_player

        try:
            agent = load_player(path)
        except Exception:  # noqa: BLE001 - surface any load failure to the user
            self.message = "Failed to load player"
            return

        self.config.loaded_agent = agent
        self.config.loaded_label = os.path.basename(path)
        self.config.agent_name = "loaded"
        self.message = ""

    def _configure_llm_opponent(self) -> None:
        connection = _prompt_llm_connection()
        if connection is None:
            self.message = "LLM setup cancelled"
            return
        base_url, api_key = connection

        try:
            from connect4_mcts.players.llm import OpenAIClient, create_llm_player
        except ImportError:
            self._notify_llm(False, "Install 'openai' to play vs LLM (pip install openai)")
            return

        endpoint = base_url or "OpenAI (default endpoint)"
        self.message = f"Connecting to {endpoint}..."
        self._refresh_display()

        try:
            client = OpenAIClient(api_key=api_key or None, base_url=base_url or None)
            models = client.list_models()
        except Exception as exc:  # noqa: BLE001 - surface any connection/auth failure
            self._notify_llm(False, f"Connection to {endpoint} failed:\n{_short_error(exc)}")
            return

        if not models:
            self._notify_llm(False, f"Connected to {endpoint}, but it returned no models.")
            return

        self._notify_llm(True, f"Connected to {endpoint}.\n{len(models)} model(s) available.")

        model = _prompt_model_choice(_chat_models_first(models))
        if not model:
            self.message = "LLM model selection cancelled"
            return

        self.config.llm_model = model
        self.config.llm_base_url = base_url or None
        self.config.llm_api_key = api_key or None
        self.config.llm_agent = create_llm_player(
            model,
            api_key=api_key or None,
            base_url=base_url or None,
            seed=self.config.seed,
        )
        from connect4_mcts.llm_settings import remember_endpoint

        remember_endpoint(base_url)
        self.config.llm_label = f"LLM: {model}"
        self.config.agent_name = "llm"
        self.message = f"LLM ready: {model}"

    def _notify_llm(self, success: bool, text: str) -> None:
        self.message = text.replace("\n", " ")
        _notify(success, "LLM connection" if success else "LLM connection failed", text)

    def _handle_setup_click(self, position: tuple[int, int]) -> None:
        rects = self._setup_rects()
        if rects["mode_single"].collidepoint(position):
            self.config.two_player = False
            return
        if rects["mode_multiplayer"].collidepoint(position):
            self.config.two_player = True
            return
        if rects["human_red"].collidepoint(position) and not self.config.two_player:
            self.config.human = Player.RED
            return
        if rects["human_yellow"].collidepoint(position) and not self.config.two_player:
            self.config.human = Player.YELLOW
            return
        if rects["agent_random"].collidepoint(position):
            self.config.agent_name = "random"
            return
        if rects["agent_minimax"].collidepoint(position):
            self.config.agent_name = "minimax"
            return
        if rects["load"].collidepoint(position):
            self._load_player_from_file()
            return
        if rects["llm"].collidepoint(position):
            self._configure_llm_opponent()
            return
        if rects["depth_minus"].collidepoint(position):
            self.config.depth = max(1, self.config.depth - 1)
            return
        if rects["depth_plus"].collidepoint(position):
            self.config.depth += 1
            return
        if rects["start"].collidepoint(position):
            self._start_game()

    def _poll_llm_thinking(self) -> None:
        if self.config.agent_name != "llm" or self.mode != "game":
            return
        if not self._agent_busy and not self.llm_awaiting_rules_ack:
            return
        thinking = getattr(self.agent, "last_thinking", None)
        if thinking:
            self.llm_thinking_panel.set_text(str(thinking))

    def _handle_wheel(self, delta_y: int) -> None:
        if self.mode != "game" or self.config.agent_name != "llm":
            return
        if not self.llm_thinking_panel.rect.collidepoint(pygame.mouse.get_pos()):
            return
        self.llm_thinking_panel.handle_wheel(delta_y, self.small_font)

    def _toggle_move_type(self) -> None:
        self.selected_move_type = MoveType.PUSH if self.selected_move_type is MoveType.DROP else MoveType.DROP

    def _thinking_panel_width(self) -> int:
        if self.config.agent_name != "llm" or self.mode != "game":
            return 0
        max_panel = max(THINKING_PANEL_MIN_WIDTH, (self.width - 2 * MARGIN) // 3)
        return min(THINKING_PANEL_WIDTH, max_panel)

    def _compute_board_layout(self) -> BoardLayout:
        available_height = self.height - TOP_BAR_HEIGHT - FOOTER_RESERVE
        panel_width = self._thinking_panel_width()
        board_area_width = self.width - 2 * MARGIN - panel_width - (THINKING_PANEL_GAP if panel_width else 0)
        cell_size = min(board_area_width // COLUMNS, available_height // ROWS)
        cell_size = max(MIN_CELL_SIZE, cell_size)
        board_width = cell_size * COLUMNS
        if panel_width:
            left = MARGIN + panel_width + THINKING_PANEL_GAP + max(0, (board_area_width - board_width) // 2)
        else:
            left = (self.width - board_width) // 2

        panel_height = available_height
        self.llm_thinking_panel.rect = pygame.Rect(
            MARGIN,
            TOP_BAR_HEIGHT,
            panel_width,
            panel_height,
        )
        return BoardLayout(left=left, top=TOP_BAR_HEIGHT, cell_size=cell_size)

    def _draw(self) -> None:
        self.screen.fill(BACKGROUND)
        if self.mode == "setup":
            self._draw_setup()
            return

        self._draw_header()
        self._draw_controls()
        self._draw_board()
        self._draw_thinking_panel()
        self._draw_footer()

    def _draw_header(self) -> None:
        title = self.large_font.render("Don't Connect 4", True, TEXT)
        self.screen.blit(title, (MARGIN, 22))

        if self._llm_rules_gate_active():
            status_line = "Waiting for LLM to acknowledge the rules..."
        elif self._agent_busy:
            status_line = f"{self._opponent_label()} thinking..."
        else:
            status_line = gui_status_message(
                self.state, self.config.human, self.config.agent_name, self.config.two_player
            )
        status = self.font.render(status_line, True, TEXT)
        self.screen.blit(status, (MARGIN, 62))

    def _draw_controls(self) -> None:
        self._draw_button(self._drop_button_rect(), "Drop", self.selected_move_type is MoveType.DROP)
        self._draw_button(self._push_button_rect(), "Push", self.selected_move_type is MoveType.PUSH)
        self._draw_button(self._reset_button_rect(), "Reset", False)
        self._draw_button(self._menu_button_rect(), "Menu", False)

    def _draw_setup(self) -> None:
        rects = self._setup_rects()
        center_x = self.width // 2

        title = self.large_font.render("Don't Connect 4", True, TEXT)
        self.screen.blit(title, title.get_rect(center=(center_x, rects["title_y"])))

        subtitle = self.font.render("Choose game setup", True, MUTED_TEXT)
        self.screen.blit(subtitle, subtitle.get_rect(center=(center_x, rects["subtitle_y"])))

        self._draw_setup_label("Mode", rects["mode_label_y"], center_x)
        self._draw_button(rects["mode_single"], "Single", not self.config.two_player)
        self._draw_button(rects["mode_multiplayer"], "2 Players", self.config.two_player)

        if not self.config.two_player:
            self._draw_setup_label("Your color (Red moves first)", rects["color_label_y"], center_x)
            self._draw_button(rects["human_red"], "Red", self.config.human is Player.RED)
            self._draw_button(rects["human_yellow"], "Yellow", self.config.human is Player.YELLOW)
        else:
            self._draw_setup_label("Local two-player (Red moves first)", rects["color_label_y"], center_x)

        self._draw_setup_label("Opponent", rects["opponent_label_y"], center_x)
        self._draw_button(rects["agent_random"], "Random", self.config.agent_name == "random")
        self._draw_button(rects["agent_minimax"], "Minimax", self.config.agent_name == "minimax")

        load_label = _shorten(self.config.loaded_label) if self.config.loaded_label else "Load player..."
        self._draw_button(rects["load"], load_label, self.config.agent_name == "loaded")

        llm_label = _shorten(self.config.llm_label) if self.config.llm_label else "Play vs LLM..."
        self._draw_button(rects["llm"], llm_label, self.config.agent_name == "llm")

        self._draw_setup_label("Minimax depth", rects["depth_label_y"], center_x)
        self._draw_button(rects["depth_minus"], "-", False)
        depth = self.font.render(str(self.config.depth), True, TEXT)
        self.screen.blit(depth, depth.get_rect(center=rects["depth_value"].center))
        self._draw_button(rects["depth_plus"], "+", False)

        self._draw_button(rects["start"], "Start", True)

        if self.message:
            color = ERROR if "fail" in self.message.lower() else MUTED_TEXT
            status = self.small_font.render(self.message, True, color)
            self.screen.blit(status, status.get_rect(center=(center_x, rects["start"].bottom + 26)))

    def _draw_setup_label(self, label: str, y: int, center_x: int) -> None:
        surface = self.font.render(label, True, TEXT)
        self.screen.blit(surface, surface.get_rect(center=(center_x, y)))

    def _draw_button(self, rect: pygame.Rect, label: str, active: bool) -> None:
        fill = BUTTON_ACTIVE if active else BUTTON
        text_color = WHITE if active else TEXT
        pygame.draw.rect(self.screen, fill, rect, border_radius=6)
        pygame.draw.rect(self.screen, BUTTON_BORDER, rect, width=1, border_radius=6)
        label_surface = self.small_font.render(label, True, text_color)
        self.screen.blit(label_surface, label_surface.get_rect(center=rect.center))

    def _draw_board(self) -> None:
        board_rect = self.layout.rect
        pygame.draw.rect(self.screen, BOARD_EDGE, board_rect.inflate(12, 12), border_radius=8)
        pygame.draw.rect(self.screen, BOARD_COLOR, board_rect, border_radius=6)

        radius = self.layout.cell_size // 2 - 8
        for row_index, row in enumerate(self.state.board):
            for column_index, cell in enumerate(row):
                center = cell_center(row_index, column_index, self.layout)
                pygame.draw.circle(self.screen, _cell_color(cell), center, radius)

        for column in range(COLUMNS):
            label = self.small_font.render(str(column + 1), True, MUTED_TEXT)
            x = self.layout.left + column * self.layout.cell_size + self.layout.cell_size // 2
            self.screen.blit(label, label.get_rect(center=(x, self.layout.top + self.layout.height + 22)))

    def _draw_thinking_panel(self) -> None:
        if self.config.agent_name != "llm":
            return
        placeholder = None
        if self._agent_busy or self.llm_awaiting_rules_ack:
            placeholder = "Waiting for response..."
        self.llm_thinking_panel.draw(self.screen, self.small_font, placeholder=placeholder)

    def _draw_footer(self) -> None:
        footer_y = self.layout.top + self.layout.height + 48
        if self.state.status is GameStatus.FINISHED:
            text = result_text(self.state.result)
            color = TEXT
        elif self._llm_rules_gate_active():
            text = "Waiting for LLM to acknowledge the rules..."
            color = MUTED_TEXT
        elif self.message:
            text = self.message
            lowered = self.message.lower()
            color = ERROR if "error" in lowered or "illegal" in lowered or "fail" in lowered else MUTED_TEXT
        else:
            text = "Space toggles move type. R resets."
            color = MUTED_TEXT

        footer = self.small_font.render(text, True, color)
        self.screen.blit(footer, footer.get_rect(center=(self.width // 2, footer_y)))

    def _control_row(self) -> tuple[int, int]:
        total_width = 4 * BUTTON_WIDTH + 3 * BUTTON_GAP
        left = self.width - MARGIN - total_width
        return left, 30

    def _drop_button_rect(self) -> pygame.Rect:
        left, top = self._control_row()
        return pygame.Rect(left, top, BUTTON_WIDTH, BUTTON_HEIGHT)

    def _push_button_rect(self) -> pygame.Rect:
        left, top = self._control_row()
        return pygame.Rect(left + (BUTTON_WIDTH + BUTTON_GAP), top, BUTTON_WIDTH, BUTTON_HEIGHT)

    def _reset_button_rect(self) -> pygame.Rect:
        left, top = self._control_row()
        return pygame.Rect(left + 2 * (BUTTON_WIDTH + BUTTON_GAP), top, BUTTON_WIDTH, BUTTON_HEIGHT)

    def _menu_button_rect(self) -> pygame.Rect:
        left, top = self._control_row()
        return pygame.Rect(left + 3 * (BUTTON_WIDTH + BUTTON_GAP), top, BUTTON_WIDTH, BUTTON_HEIGHT)

    def _setup_rects(self) -> dict[str, object]:
        center_x = self.width // 2
        pair_width = 2 * SETUP_BUTTON_WIDTH + SETUP_BUTTON_GAP
        pair_left = center_x - pair_width // 2
        pair_right_left = pair_left + SETUP_BUTTON_WIDTH + SETUP_BUTTON_GAP

        block_height = 700
        top = max(16, (self.height - block_height) // 2)
        label_to_button = 28
        row_gap = 22

        def pair(top_y: int) -> tuple[pygame.Rect, pygame.Rect]:
            left_rect = pygame.Rect(pair_left, top_y, SETUP_BUTTON_WIDTH, SETUP_BUTTON_HEIGHT)
            right_rect = pygame.Rect(pair_right_left, top_y, SETUP_BUTTON_WIDTH, SETUP_BUTTON_HEIGHT)
            return left_rect, right_rect

        title_y = top + 16
        subtitle_y = top + 52

        cursor = top + 100
        mode_label_y = cursor
        mode_single, mode_multiplayer = pair(cursor + label_to_button)
        cursor += label_to_button + SETUP_BUTTON_HEIGHT + row_gap

        color_label_y = cursor
        human_red, human_yellow = pair(cursor + label_to_button)
        cursor += label_to_button + SETUP_BUTTON_HEIGHT + row_gap

        opponent_label_y = cursor
        agent_random, agent_minimax = pair(cursor + label_to_button)
        cursor += label_to_button + SETUP_BUTTON_HEIGHT + 10
        load = pygame.Rect(pair_left, cursor, pair_width, SETUP_BUTTON_HEIGHT)
        cursor += SETUP_BUTTON_HEIGHT + 10
        llm = pygame.Rect(pair_left, cursor, pair_width, SETUP_BUTTON_HEIGHT)
        cursor += SETUP_BUTTON_HEIGHT + row_gap

        depth_label_y = cursor
        depth_top = cursor + label_to_button
        depth_minus = pygame.Rect(center_x - 92, depth_top, 54, SETUP_BUTTON_HEIGHT)
        depth_value = pygame.Rect(center_x - 27, depth_top, 54, SETUP_BUTTON_HEIGHT)
        depth_plus = pygame.Rect(center_x + 38, depth_top, 54, SETUP_BUTTON_HEIGHT)
        cursor = depth_top + SETUP_BUTTON_HEIGHT + row_gap

        start = pygame.Rect(center_x - 90, cursor, 180, 50)

        return {
            "title_y": title_y,
            "subtitle_y": subtitle_y,
            "mode_label_y": mode_label_y,
            "mode_single": mode_single,
            "mode_multiplayer": mode_multiplayer,
            "color_label_y": color_label_y,
            "human_red": human_red,
            "human_yellow": human_yellow,
            "opponent_label_y": opponent_label_y,
            "agent_random": agent_random,
            "agent_minimax": agent_minimax,
            "load": load,
            "llm": llm,
            "depth_label_y": depth_label_y,
            "depth_minus": depth_minus,
            "depth_value": depth_value,
            "depth_plus": depth_plus,
            "start": start,
        }


def _prompt_player_file() -> str | None:
    """Open a native file dialog to pick a pickled player.

    Uses tkinter (standard library). Returns ``None`` if the user cancels or no
    dialog backend is available, in which case ``--load`` can be used instead.
    """
    try:
        import tkinter
        from tkinter import filedialog
    except Exception:  # noqa: BLE001 - tkinter may be missing on some systems
        return None

    try:
        root = tkinter.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Load trained player",
            filetypes=[("Pickled player", "*.pkl"), ("All files", "*.*")],
        )
        root.destroy()
        return path or None
    except Exception:  # noqa: BLE001 - dialog can fail on headless/odd setups
        return None


def _prompt_llm_connection() -> tuple[str, str] | None:
    """Prompt for the LLM endpoint and API key.

    Endpoint URLs are remembered locally and suggested with autocomplete as the
    user types. API keys are **not** saved to disk.
    """
    try:
        import tkinter
        from tkinter import ttk
    except Exception:  # noqa: BLE001 - tkinter may be missing on some systems
        return None

    from connect4_mcts.llm_settings import load_endpoints, suggest_endpoints

    saved_endpoints = load_endpoints()
    chosen: dict[str, tuple[str, str] | None] = {"value": None}

    try:
        root = tkinter.Tk()
        root.title("LLM connection")
        root.geometry("520x210")
        root.resizable(False, False)

        tkinter.Label(
            root,
            text="Base URL (blank = OpenAI; start typing to filter saved addresses):",
            anchor="w",
        ).pack(fill="x", padx=14, pady=(14, 4))

        url_var = tkinter.StringVar(value=saved_endpoints[0] if saved_endpoints else "")
        url_combo = ttk.Combobox(root, textvariable=url_var, values=saved_endpoints)
        url_combo.pack(fill="x", padx=14)

        def refresh_endpoint_suggestions(_event: object | None = None) -> None:
            url_combo["values"] = suggest_endpoints(url_var.get())

        url_combo.bind("<KeyRelease>", refresh_endpoint_suggestions)
        url_combo.bind("<Button-1>", refresh_endpoint_suggestions)

        tkinter.Label(
            root,
            text="API key (blank = OK for local server; for OpenAI use your key or OPENAI_API_KEY):",
            anchor="w",
        ).pack(fill="x", padx=14, pady=(12, 4))

        api_key_var = tkinter.StringVar()
        api_key_entry = tkinter.Entry(root, textvariable=api_key_var, show="*")
        api_key_entry.pack(fill="x", padx=14)

        def confirm() -> None:
            chosen["value"] = (url_var.get().strip(), api_key_var.get().strip())
            root.destroy()

        def cancel() -> None:
            chosen["value"] = None
            root.destroy()

        buttons = tkinter.Frame(root)
        buttons.pack(pady=16)
        tkinter.Button(buttons, text="Connect", width=10, command=confirm).pack(side="left", padx=8)
        tkinter.Button(buttons, text="Cancel", width=10, command=cancel).pack(side="left", padx=8)
        root.protocol("WM_DELETE_WINDOW", cancel)
        url_combo.focus_set()
        root.mainloop()
        return chosen["value"]
    except Exception:  # noqa: BLE001 - dialog can fail on headless/odd setups
        return None


def _prompt_model_choice(models: Sequence[str]) -> str | None:
    """Show a dropdown of ``models`` and return the chosen id (or ``None``)."""
    if not models:
        return None
    try:
        import tkinter
        from tkinter import ttk
    except Exception:  # noqa: BLE001 - tkinter may be missing on some systems
        return None

    chosen: dict[str, str | None] = {"value": None}
    try:
        root = tkinter.Tk()
        root.title("Choose LLM model")
        root.geometry("420x150")

        tkinter.Label(root, text="Select a model to play against:").pack(padx=14, pady=(16, 6))
        selected = tkinter.StringVar(value=models[0])
        # Editable so an advanced user can still type a model not in the list.
        combo = ttk.Combobox(root, textvariable=selected, values=list(models))
        combo.pack(fill="x", padx=14)

        def confirm() -> None:
            chosen["value"] = selected.get().strip() or None
            root.destroy()

        def cancel() -> None:
            chosen["value"] = None
            root.destroy()

        buttons = tkinter.Frame(root)
        buttons.pack(pady=16)
        tkinter.Button(buttons, text="Play", width=10, command=confirm).pack(side="left", padx=8)
        tkinter.Button(buttons, text="Cancel", width=10, command=cancel).pack(side="left", padx=8)
        root.protocol("WM_DELETE_WINDOW", cancel)
        combo.focus_set()
        root.mainloop()
        return chosen["value"]
    except Exception:  # noqa: BLE001 - dialog can fail on headless/odd setups
        return None


def _notify(success: bool, title: str, message: str) -> None:
    """Pop a native info/error dialog (best effort, headless-safe)."""
    try:
        import tkinter
        from tkinter import messagebox
    except Exception:  # noqa: BLE001 - tkinter may be missing on some systems
        return

    try:
        root = tkinter.Tk()
        root.withdraw()
        try:
            show = messagebox.showinfo if success else messagebox.showerror
            show(title, message, parent=root)
        finally:
            root.destroy()
    except Exception:  # noqa: BLE001 - dialog can fail on headless/odd setups
        return


_CHAT_MODEL_PREFIXES = ("gpt-", "gpt", "o1", "o3", "o4", "chatgpt")


def _chat_models_first(models: Sequence[str]) -> list[str]:
    """Surface likely chat models first; keep the rest available below them."""
    chat = [model for model in models if model.lower().startswith(_CHAT_MODEL_PREFIXES)]
    if not chat:
        return list(models)
    others = [model for model in models if model not in set(chat)]
    return chat + others


def _wrap_text_preserve_newlines(text: str, font: pygame.font.Font, max_width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.splitlines():
        stripped = paragraph.strip()
        if not stripped:
            lines.append("")
            continue
        lines.extend(_wrap_text(stripped, font, max_width))
    return lines


def _wrap_text(text: str, font: pygame.font.Font, max_width: int) -> list[str]:
    words = text.replace("\n", " ").split()
    if not words:
        return []

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if font.size(candidate)[0] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _short_error(exc: Exception, max_length: int = 200) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    if len(text) > max_length:
        text = text[: max_length - 3] + "..."
    return text


def _shorten(text: str, max_length: int = 22) -> str:
    if len(text) <= max_length:
        return text
    return "..." + text[-(max_length - 3):]


def column_from_position(position: tuple[int, int], layout: BoardLayout) -> int | None:
    x, y = position
    if not layout.rect.collidepoint(x, y):
        return None
    return (x - layout.left) // layout.cell_size


def cell_center(row: int, column: int, layout: BoardLayout) -> tuple[int, int]:
    return (
        layout.left + column * layout.cell_size + layout.cell_size // 2,
        layout.top + row * layout.cell_size + layout.cell_size // 2,
    )


def gui_status_message(
    state: GameState,
    human: Player,
    agent_name: AgentName = "random",
    two_player: bool = False,
) -> str:
    if state.status is GameStatus.FINISHED:
        return "Game finished"

    if two_player:
        actor = f"{state.current_player.value.capitalize()} turn"
    elif state.current_player is human:
        actor = "Your turn"
    else:
        actor = f"{format_agent_name(agent_name)} turn"

    if state.status is GameStatus.FAIR_TURN:
        return f"{actor} - fair turn"
    return actor


def result_text(result: GameResult | None) -> str:
    if result is None:
        return "Game finished"
    if result.is_draw:
        outcome = "Draw"
    else:
        outcome = f"Winner: {result.winner.value}"
    return f"{outcome}. Lines red={result.red_lines}, yellow={result.yellow_lines}"


def format_move(move: Move) -> str:
    prefix = "drop" if move.move_type is MoveType.DROP else "push"
    return f"{prefix} {move.column + 1}"


def _cell_color(cell: Player | None) -> tuple[int, int, int]:
    if cell is Player.RED:
        return RED_CELL
    if cell is Player.YELLOW:
        return YELLOW_CELL
    return EMPTY_CELL


if __name__ == "__main__":
    raise SystemExit(main())
