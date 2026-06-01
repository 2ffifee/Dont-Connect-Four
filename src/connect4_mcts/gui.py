"""Pygame interface for playing against an agent."""

from __future__ import annotations

import argparse
import os
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Play Don't Connect 4 against an agent.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for the random player.")
    parser.add_argument("--human", choices=("red", "yellow"), default="red", help="Human player color.")
    parser.add_argument("--agent", choices=AGENT_CHOICES, default="random", help="Initial opponent selection.")
    parser.add_argument("--depth", type=int, default=3, help="Search depth for minimax.")
    parser.add_argument("--two-player", action="store_true", help="Start in local two-player mode.")
    parser.add_argument("--load", default=None, help="Path to a pickled trained player to use as opponent.")
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
            self.mode = "setup"
            self.message = ""
            return

        if self.state.status is GameStatus.FINISHED:
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
            self._refresh_display()
            self._play_agent_turn()

    def _refresh_display(self) -> None:
        self._draw()
        pygame.display.flip()

    def _play_agent_turn(self) -> None:
        if self.state.status is GameStatus.FINISHED:
            return

        move = self.agent.choose_move(self.state)
        self.state = self.state.apply_move(move)
        self.message = f"{format_agent_name(self.config.agent_name)}: {format_move(move)}"

    def _reset(self) -> None:
        self._start_game()

    def _start_game(self) -> None:
        self.state = GameState.new(first_player=Player.RED)
        self.agent = self._make_agent()
        self.selected_move_type = MoveType.DROP
        self.mode = "game"
        self.message = ""
        self.layout = self._compute_board_layout()
        if not self.config.two_player and self.state.current_player is not self.config.human:
            self._play_agent_turn()

    def _make_agent(self) -> Agent:
        if self.config.agent_name == "loaded" and self.config.loaded_agent is not None:
            return self.config.loaded_agent
        return create_agent(self.config.agent_name, seed=self.config.seed, depth=self.config.depth)

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

    def _handle_setup_click(self, position: tuple[int, int]) -> None:
        rects = self._setup_rects()
        if rects["mode_single"].collidepoint(position):
            self.config.two_player = False
            return
        if rects["mode_multiplayer"].collidepoint(position):
            self.config.two_player = True
            return
        if rects["human_red"].collidepoint(position):
            self.config.human = Player.RED
            return
        if rects["human_yellow"].collidepoint(position):
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
        if rects["depth_minus"].collidepoint(position):
            self.config.depth = max(1, self.config.depth - 1)
            return
        if rects["depth_plus"].collidepoint(position):
            self.config.depth += 1
            return
        if rects["start"].collidepoint(position):
            self._start_game()

    def _toggle_move_type(self) -> None:
        self.selected_move_type = MoveType.PUSH if self.selected_move_type is MoveType.DROP else MoveType.DROP

    def _compute_board_layout(self) -> BoardLayout:
        available_width = self.width - 2 * MARGIN
        available_height = self.height - TOP_BAR_HEIGHT - FOOTER_RESERVE
        cell_size = min(available_width // COLUMNS, available_height // ROWS)
        cell_size = max(MIN_CELL_SIZE, cell_size)
        board_width = cell_size * COLUMNS
        left = (self.width - board_width) // 2
        return BoardLayout(left=left, top=TOP_BAR_HEIGHT, cell_size=cell_size)

    def _draw(self) -> None:
        self.screen.fill(BACKGROUND)
        if self.mode == "setup":
            self._draw_setup()
            return

        self._draw_header()
        self._draw_controls()
        self._draw_board()
        self._draw_footer()

    def _draw_header(self) -> None:
        title = self.large_font.render("Don't Connect 4", True, TEXT)
        self.screen.blit(title, (MARGIN, 22))

        status = self.font.render(
            gui_status_message(self.state, self.config.human, self.config.agent_name, self.config.two_player),
            True,
            TEXT,
        )
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

        self._draw_setup_label("Your color", rects["color_label_y"], center_x)
        self._draw_button(rects["human_red"], "Red", self.config.human is Player.RED)
        self._draw_button(rects["human_yellow"], "Yellow", self.config.human is Player.YELLOW)

        self._draw_setup_label("Opponent", rects["opponent_label_y"], center_x)
        self._draw_button(rects["agent_random"], "Random", self.config.agent_name == "random")
        self._draw_button(rects["agent_minimax"], "Minimax", self.config.agent_name == "minimax")

        load_label = _shorten(self.config.loaded_label) if self.config.loaded_label else "Load player..."
        self._draw_button(rects["load"], load_label, self.config.agent_name == "loaded")

        self._draw_setup_label("Minimax depth", rects["depth_label_y"], center_x)
        self._draw_button(rects["depth_minus"], "-", False)
        depth = self.font.render(str(self.config.depth), True, TEXT)
        self.screen.blit(depth, depth.get_rect(center=rects["depth_value"].center))
        self._draw_button(rects["depth_plus"], "+", False)

        self._draw_button(rects["start"], "Start", True)

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

    def _draw_footer(self) -> None:
        footer_y = self.layout.top + self.layout.height + 48
        if self.state.status is GameStatus.FINISHED:
            text = result_text(self.state.result)
            color = TEXT
        elif self.message:
            text = self.message
            color = ERROR if self.message == "Illegal move" else MUTED_TEXT
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

        block_height = 640
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
