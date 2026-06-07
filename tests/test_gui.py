from connect4_mcts.game import GameResult, GameState, GameStatus, Move, MoveType, Player
import connect4_mcts.gui as gui
from connect4_mcts.gui import BoardLayout, cell_center, column_from_position, format_move, gui_status_message, result_text


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
    fair_turn = GameState(board=GameState.new().board, current_player=Player.YELLOW, status=GameStatus.FAIR_TURN)

    assert gui_status_message(red_turn, human=Player.RED) == "Your turn"
    assert gui_status_message(yellow_turn, human=Player.RED) == "Random turn"
    assert gui_status_message(fair_turn, human=Player.YELLOW) == "Your turn - fair turn"


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
    assert configs == [gui.GuiConfig(human=Player.YELLOW, agent_name="minimax", seed=9, depth=2)]


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
