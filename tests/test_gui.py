import pygame
import pytest

import connect4_mcts.gui as gui
from connect4_mcts.game import GameResult, GameState, GameStatus, Move, MoveType, Player
from connect4_mcts.gui import BoardLayout, cell_center, column_from_position, format_move, gui_status_message, result_text
from connect4_mcts.players.factory import DEFAULT_EXPLORATION, DEFAULT_FPU, DEFAULT_POWER_MEAN_P


def test_column_from_position_returns_column_inside_board() -> None:
    layout = BoardLayout(left=10, top=20, cell_size=50)

    assert column_from_position((10, 20), layout) == 0
    assert column_from_position((59, 69), layout) == 0
    assert column_from_position((60, 20), layout) == 1
    assert column_from_position((409, 20), layout) == 7


def test_column_from_position_returns_none_outside_board() -> None:
    layout = BoardLayout(left=10, top=20, cell_size=50)

    assert column_from_position((9, 20), layout) is None
    assert column_from_position((410, 20), layout) is None
    assert column_from_position((10, 19), layout) is None
    assert column_from_position((10, 320), layout) is None


def test_cell_center_uses_layout_geometry() -> None:
    layout = BoardLayout(left=10, top=20, cell_size=50)

    assert cell_center(0, 0, layout) == (35, 45)
    assert cell_center(2, 3, layout) == (185, 145)


def test_gui_status_message_describes_turns() -> None:
    red_turn = GameState.new(first_player=Player.RED)
    yellow_turn = red_turn.apply_move(Move(MoveType.DROP, 0))

    assert gui_status_message(red_turn, human=Player.RED) == "Your turn"
    assert gui_status_message(yellow_turn, human=Player.RED) == "Random turn"


def test_gui_status_message_uses_selected_agent_name() -> None:
    red_turn = GameState.new(first_player=Player.RED)

    assert gui_status_message(red_turn, human=Player.YELLOW, agent_name="minimax") == "Minimax turn"


def test_result_text_describes_winner_and_draw() -> None:
    assert result_text(GameResult(winner=Player.RED, red_lines=0, yellow_lines=1)) == "Winner: red. Lines red=0, yellow=1"
    assert result_text(GameResult(winner=None, red_lines=1, yellow_lines=1)) == "Draw. Lines red=1, yellow=1"


def test_format_move_uses_human_columns() -> None:
    assert format_move(Move(MoveType.DROP, 0)) == "drop 1"
    assert format_move(Move(MoveType.PUSH, 7)) == "push 8"


def test_main_passes_initial_gui_config(monkeypatch) -> None:
    configs = []

    class FakeGui:
        def __init__(self, config: gui.GuiConfig) -> None:
            configs.append(config)

        def run(self) -> None:
            return None

    monkeypatch.setattr(gui, "HumanVsAgentGui", FakeGui)

    assert gui.main(["--human", "yellow", "--agent", "minimax", "--depth", "2", "--seed", "9"]) == 0
    assert configs == [
        gui.GuiConfig(
            human=Player.YELLOW,
            agent_name="minimax",
            seed=9,
            depth=2,
            iterations=gui.DEFAULT_MCTS_ITERATIONS,
            exploration=DEFAULT_EXPLORATION,
            fpu=DEFAULT_FPU,
            power_mean_p=DEFAULT_POWER_MEAN_P,
            llm_temperature=1.0,
            llm_max_tokens=None,
            enable_undo=True,
        )
    ]


def test_make_agent_creates_online_mcts_player() -> None:
    game = object.__new__(gui.HumanVsAgentGui)
    game.config = gui.GuiConfig(agent_name="uct", iterations=100, seed=3)

    agent = game._make_agent()

    from connect4_mcts.players.mcts import MCTSPlayer

    assert isinstance(agent, MCTSPlayer)
    assert agent.simulation_mode == "search"
    assert agent.iterations == 100
    assert agent.tree_size == 0


def test_setup_hyperparams_for_mcts_variants() -> None:
    assert [spec.key for spec in gui.setup_hyperparams_for("uct")] == ["iterations", "exploration"]
    assert [spec.key for spec in gui.setup_hyperparams_for("fpu")] == ["iterations", "exploration", "fpu"]
    assert [spec.key for spec in gui.setup_hyperparams_for("llm")] == ["llm_temperature", "llm_max_tokens"]


def test_adjust_hyperparam_updates_llm_temperature() -> None:
    config = gui.GuiConfig(llm_temperature=1.0)
    spec = gui.setup_hyperparams_for("llm")[0]

    gui.adjust_hyperparam(config, spec, 1)

    assert config.llm_temperature == pytest.approx(1.1)


def test_make_agent_passes_mcts_hyperparameters() -> None:
    game = object.__new__(gui.HumanVsAgentGui)
    game.config = gui.GuiConfig(
        agent_name="fpu",
        iterations=120,
        exploration=1.5,
        fpu=0.8,
        seed=2,
    )

    agent = game._make_agent()

    from connect4_mcts.players.mcts import MCTSPlayer

    assert isinstance(agent, MCTSPlayer)
    assert agent.iterations == 120
    assert agent.exploration == pytest.approx(1.5)
    assert agent.fpu == pytest.approx(0.8)


def test_make_llm_agent_uses_temperature_and_max_tokens(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_create_llm_player(model, **kwargs):
        captured["model"] = model
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("connect4_mcts.players.llm.create_llm_player", fake_create_llm_player)

    game = object.__new__(gui.HumanVsAgentGui)
    game.config = gui.GuiConfig(
        agent_name="llm",
        llm_model="demo-model",
        llm_temperature=0.7,
        llm_max_tokens=512,
    )

    game._make_llm_agent()

    assert captured["temperature"] == pytest.approx(0.7)
    assert captured["max_tokens"] == 512


def test_main_accepts_mcts_agent_and_iterations(monkeypatch) -> None:
    configs = []

    class FakeGui:
        def __init__(self, config: gui.GuiConfig) -> None:
            configs.append(config)

        def run(self) -> None:
            return None

    monkeypatch.setattr(gui, "HumanVsAgentGui", FakeGui)

    assert gui.main(["--agent", "fpu", "--iterations", "250", "--seed", "1"]) == 0
    assert configs[0].agent_name == "fpu"
    assert configs[0].iterations == 250


def test_undo_in_multiplayer_reverts_one_move(monkeypatch) -> None:
    game = object.__new__(gui.HumanVsAgentGui)
    game.mode = "game"
    game.config = gui.GuiConfig(two_player=True, enable_undo=True)
    game.state = GameState.new()
    game._undo_stack = [game.state]
    game._agent_busy = False
    game._async_generation = 0
    game.message = ""
    game.llm_alert = None
    game.llm_thinking_panel = gui.ScrollableTextPanel()
    monkeypatch.setattr(game, "_invalidate_async_work", lambda: None)
    monkeypatch.setattr(game, "_sync_agent_after_undo", lambda: None)

    game._apply_human_move(Move(MoveType.DROP, 0))
    assert game.state.move_count == 1
    assert len(game._undo_stack) == 2

    game._apply_human_move(Move(MoveType.DROP, 1))
    assert game.state.move_count == 2

    game._undo()
    assert game.state.move_count == 1
    assert game.state.board[5][0] is Player.RED
    assert game.state.board[5][1] is None

    game._undo()
    assert game.state.move_count == 0
    assert not game._can_undo()


def test_undo_in_single_player_returns_before_last_human_move(monkeypatch) -> None:
    game = object.__new__(gui.HumanVsAgentGui)
    game.mode = "game"
    game.config = gui.GuiConfig(human=Player.RED, agent_name="random", enable_undo=True)
    game.state = GameState.new()
    game._undo_stack = [game.state]
    game._agent_busy = False
    game._async_generation = 0
    game.message = ""
    game.llm_alert = None
    game.llm_thinking_panel = gui.ScrollableTextPanel()
    game.agent = object()

    monkeypatch.setattr(game, "_schedule_agent_turn", lambda: None)
    monkeypatch.setattr(game, "_invalidate_async_work", lambda: None)
    monkeypatch.setattr(game, "_sync_agent_after_undo", lambda: None)

    game._apply_human_move(Move(MoveType.DROP, 0))
    assert game.state.move_count == 1
    assert game.state.current_player is Player.YELLOW
    assert game._can_undo()

    game._undo()
    assert game.state.move_count == 0
    assert game.state.current_player is Player.RED

    game.state = game.state.apply_move(Move(MoveType.DROP, 0))
    game.state = game.state.apply_move(Move(MoveType.DROP, 1))
    game._push_undo_point_if_needed()
    assert game.state.move_count == 2
    assert game.state.current_player is Player.RED

    game._apply_human_move(Move(MoveType.DROP, 2))
    assert game.state.move_count == 3
    assert game._can_undo()

    game._undo()
    assert game.state.move_count == 2
    assert game.state.current_player is Player.RED


def test_apply_human_move_skips_undo_stack_when_disabled() -> None:
    game = object.__new__(gui.HumanVsAgentGui)
    game.mode = "game"
    game.config = gui.GuiConfig(human=Player.RED, agent_name="random", two_player=True)
    game.state = GameState.new()
    game._undo_stack = None
    game.message = ""

    game._apply_human_move(Move(MoveType.DROP, 0))

    assert game._undo_stack is None
    assert game.state.move_count == 1


def test_human_move_schedules_agent_turn(monkeypatch) -> None:
    events = []
    game = object.__new__(gui.HumanVsAgentGui)
    game.config = gui.GuiConfig(human=Player.RED, agent_name="random")
    game.state = GameState.new()
    game.message = ""

    def fake_schedule_agent_turn() -> None:
        events.append(("agent", game.state.move_count))

    monkeypatch.setattr(game, "_schedule_agent_turn", fake_schedule_agent_turn)

    game._apply_human_move(Move(MoveType.DROP, 0))

    assert events == [("agent", 1)]


def test_wrap_text_preserve_newlines_keeps_paragraph_breaks() -> None:
    pygame.font.init()
    font = pygame.font.SysFont("Arial", 18)
    lines = gui._wrap_text_preserve_newlines("first line\n\nsecond line", font, 400)
    assert "" in lines
    assert any("first line" in line for line in lines)
    assert any("second line" in line for line in lines)


def test_scrollable_panel_follows_streaming_updates() -> None:
    pygame.font.init()
    font = pygame.font.SysFont("Arial", 18)
    panel = gui.ScrollableTextPanel()
    panel.rect = pygame.Rect(0, 0, 200, 80)
    long_text = "\n".join(f"line {index}" for index in range(20))
    panel.set_text(long_text)
    panel.handle_wheel(1, font)
    assert panel._follow_bottom is False
    panel.set_text(long_text + "\nline 21")
    assert "line 21" in panel.text


def test_thinking_panel_hidden_without_cot_enabled() -> None:
    game = object.__new__(gui.HumanVsAgentGui)
    game.mode = "game"
    game.config = gui.GuiConfig(agent_name="llm", llm_cot_enabled=False)
    assert game._thinking_panel_width() == 0

    game.config = gui.GuiConfig(agent_name="llm", llm_cot_enabled=True, llm_base_url="https://api.openai.com/v1")
    game.width = 1200
    assert game._thinking_panel_width() > 0

    game.config = gui.GuiConfig(agent_name="random")
    assert game._thinking_panel_width() == 0


def test_scrollbar_drag_moves_content() -> None:
    pygame.font.init()
    font = pygame.font.SysFont("Arial", 18)
    panel = gui.ScrollableTextPanel()
    panel.rect = pygame.Rect(0, 0, 220, 120)
    panel.set_text("\n".join(f"line {index}" for index in range(30)))

    layout = panel._scroll_layout(font)
    assert layout is not None
    assert layout.max_scroll > 0

    thumb_center = layout.thumb.center
    assert panel.handle_mouse_down(thumb_center, font)
    panel.handle_mouse_motion((thumb_center[0], layout.track.top + 4), font)
    assert panel.scroll_y == 0
    assert panel._dragging_scrollbar is True

    at_top = panel._scroll_layout(font)
    assert at_top is not None
    bottom_thumb_top = at_top.track.bottom - at_top.thumb.height
    panel.handle_mouse_motion(
        (thumb_center[0], bottom_thumb_top + panel._drag_grab_offset),
        font,
    )
    assert panel.scroll_y == layout.max_scroll
    assert panel._follow_bottom is True

    assert panel.handle_mouse_up()
    assert panel._dragging_scrollbar is False
