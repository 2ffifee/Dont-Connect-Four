"""Pygame interface for playing against an agent."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from dataclasses import dataclass

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from connect4_mcts.cli import AGENT_CHOICES, create_agent, format_agent_name
from connect4_mcts.game import COLUMNS, ROWS, GameResult, GameState, GameStatus, IllegalMoveError, Move, MoveType, Player
from connect4_mcts.players import Agent


CELL_SIZE = 72
BOARD_WIDTH = COLUMNS * CELL_SIZE
BOARD_HEIGHT = ROWS * CELL_SIZE
WINDOW_WIDTH = 760
WINDOW_HEIGHT = 680
BOARD_LEFT = (WINDOW_WIDTH - BOARD_WIDTH) // 2
BOARD_TOP = 118
BUTTON_WIDTH = 96
BUTTON_HEIGHT = 38
SETUP_BUTTON_WIDTH = 132
SETUP_BUTTON_HEIGHT = 42

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
AgentName = str


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Play Don't Connect 4 against an agent.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for the random player.")
    parser.add_argument("--human", choices=("red", "yellow"), default="red", help="Human player color.")
    parser.add_argument("--agent", choices=AGENT_CHOICES, default="random", help="Initial opponent selection.")
    parser.add_argument("--depth", type=int, default=3, help="Search depth for minimax.")
    args = parser.parse_args(argv)

    HumanVsAgentGui(
        config=GuiConfig(human=Player(args.human), agent_name=args.agent, seed=args.seed, depth=args.depth)
    ).run()
    return 0


class HumanVsRandomGui:
    def __init__(self, human: Player = Player.RED, seed: int | None = None) -> None:
        self._delegate = HumanVsAgentGui(config=GuiConfig(human=human, agent_name="random", seed=seed))

    def run(self) -> None:
        self._delegate.run()


class HumanVsAgentGui:
    def __init__(self, config: GuiConfig | None = None) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Don't Connect 4")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Segoe UI", 24)
        self.small_font = pygame.font.SysFont("Segoe UI", 18)
        self.large_font = pygame.font.SysFont("Segoe UI", 30, bold=True)
        self.layout = BoardLayout()
        self.config = config or GuiConfig()
        self.mode: ScreenMode = "setup"
        self.state = GameState.new(first_player=Player.RED)
        self.agent: Agent = create_agent(self.config.agent_name, seed=self.config.seed, depth=self.config.depth)
        self.selected_move_type = MoveType.DROP
        self.message = ""

    def run(self) -> None:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    self._handle_key(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)

            self._draw()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

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
        if self.state.current_player is not self.config.human:
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
        self.agent = create_agent(self.config.agent_name, seed=self.config.seed, depth=self.config.depth)
        self.selected_move_type = MoveType.DROP
        self.mode = "game"
        self.message = ""
        if self.state.current_player is not self.config.human:
            self._play_agent_turn()

    def _handle_setup_click(self, position: tuple[int, int]) -> None:
        if self._human_red_button_rect().collidepoint(position):
            self.config.human = Player.RED
            return
        if self._human_yellow_button_rect().collidepoint(position):
            self.config.human = Player.YELLOW
            return
        if self._agent_random_button_rect().collidepoint(position):
            self.config.agent_name = "random"
            return
        if self._agent_minimax_button_rect().collidepoint(position):
            self.config.agent_name = "minimax"
            return
        if self._depth_minus_button_rect().collidepoint(position):
            self.config.depth = max(1, self.config.depth - 1)
            return
        if self._depth_plus_button_rect().collidepoint(position):
            self.config.depth += 1
            return
        if self._start_button_rect().collidepoint(position):
            self._start_game()

    def _toggle_move_type(self) -> None:
        self.selected_move_type = MoveType.PUSH if self.selected_move_type is MoveType.DROP else MoveType.DROP

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
        self.screen.blit(title, (BOARD_LEFT, 26))

        status = self.font.render(gui_status_message(self.state, self.config.human, self.config.agent_name), True, TEXT)
        self.screen.blit(status, (BOARD_LEFT, 66))

    def _draw_controls(self) -> None:
        self._draw_button(self._drop_button_rect(), "Drop", self.selected_move_type is MoveType.DROP)
        self._draw_button(self._push_button_rect(), "Push", self.selected_move_type is MoveType.PUSH)
        self._draw_button(self._reset_button_rect(), "Reset", False)
        self._draw_button(self._menu_button_rect(), "Menu", False)

    def _draw_setup(self) -> None:
        title = self.large_font.render("Don't Connect 4", True, TEXT)
        self.screen.blit(title, (BOARD_LEFT, 58))

        subtitle = self.font.render("Choose game setup", True, MUTED_TEXT)
        self.screen.blit(subtitle, (BOARD_LEFT, 96))

        self._draw_setup_label("Your color", 168)
        self._draw_button(self._human_red_button_rect(), "Red", self.config.human is Player.RED)
        self._draw_button(self._human_yellow_button_rect(), "Yellow", self.config.human is Player.YELLOW)

        self._draw_setup_label("Opponent", 256)
        self._draw_button(self._agent_random_button_rect(), "Random", self.config.agent_name == "random")
        self._draw_button(self._agent_minimax_button_rect(), "Minimax", self.config.agent_name == "minimax")

        self._draw_setup_label("Minimax depth", 344)
        self._draw_button(self._depth_minus_button_rect(), "-", False)
        depth = self.font.render(str(self.config.depth), True, TEXT)
        self.screen.blit(depth, depth.get_rect(center=(BOARD_LEFT + 92, 424)))
        self._draw_button(self._depth_plus_button_rect(), "+", False)

        self._draw_button(self._start_button_rect(), "Start", True)

    def _draw_setup_label(self, label: str, y: int) -> None:
        surface = self.font.render(label, True, TEXT)
        self.screen.blit(surface, (BOARD_LEFT, y))

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

        for row_index, row in enumerate(self.state.board):
            for column_index, cell in enumerate(row):
                center = cell_center(row_index, column_index, self.layout)
                pygame.draw.circle(self.screen, _cell_color(cell), center, CELL_SIZE // 2 - 8)

        for column in range(COLUMNS):
            label = self.small_font.render(str(column + 1), True, MUTED_TEXT)
            x = self.layout.left + column * self.layout.cell_size + self.layout.cell_size // 2
            self.screen.blit(label, label.get_rect(center=(x, self.layout.top + self.layout.height + 24)))

    def _draw_footer(self) -> None:
        footer_y = self.layout.top + self.layout.height + 54
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
        self.screen.blit(footer, (BOARD_LEFT, footer_y))

    def _drop_button_rect(self) -> pygame.Rect:
        return pygame.Rect(WINDOW_WIDTH - 330, 34, BUTTON_WIDTH, BUTTON_HEIGHT)

    def _push_button_rect(self) -> pygame.Rect:
        return pygame.Rect(WINDOW_WIDTH - 224, 34, BUTTON_WIDTH, BUTTON_HEIGHT)

    def _reset_button_rect(self) -> pygame.Rect:
        return pygame.Rect(WINDOW_WIDTH - 118, 34, BUTTON_WIDTH, BUTTON_HEIGHT)

    def _menu_button_rect(self) -> pygame.Rect:
        return pygame.Rect(WINDOW_WIDTH - 118, 80, BUTTON_WIDTH, BUTTON_HEIGHT)

    def _human_red_button_rect(self) -> pygame.Rect:
        return pygame.Rect(BOARD_LEFT, 204, SETUP_BUTTON_WIDTH, SETUP_BUTTON_HEIGHT)

    def _human_yellow_button_rect(self) -> pygame.Rect:
        return pygame.Rect(BOARD_LEFT + 148, 204, SETUP_BUTTON_WIDTH, SETUP_BUTTON_HEIGHT)

    def _agent_random_button_rect(self) -> pygame.Rect:
        return pygame.Rect(BOARD_LEFT, 292, SETUP_BUTTON_WIDTH, SETUP_BUTTON_HEIGHT)

    def _agent_minimax_button_rect(self) -> pygame.Rect:
        return pygame.Rect(BOARD_LEFT + 148, 292, SETUP_BUTTON_WIDTH, SETUP_BUTTON_HEIGHT)

    def _depth_minus_button_rect(self) -> pygame.Rect:
        return pygame.Rect(BOARD_LEFT, 400, 54, SETUP_BUTTON_HEIGHT)

    def _depth_plus_button_rect(self) -> pygame.Rect:
        return pygame.Rect(BOARD_LEFT + 130, 400, 54, SETUP_BUTTON_HEIGHT)

    def _start_button_rect(self) -> pygame.Rect:
        return pygame.Rect(BOARD_LEFT, 500, 180, 50)


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


def gui_status_message(state: GameState, human: Player, agent_name: AgentName = "random") -> str:
    if state.status is GameStatus.FINISHED:
        return "Game finished"

    actor = "Your turn" if state.current_player is human else f"{format_agent_name(agent_name)} turn"
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
