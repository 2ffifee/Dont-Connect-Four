"""Utilities for running games between agents."""

from __future__ import annotations

from dataclasses import dataclass

from connect4_mcts.game import GameState, GameStatus, IllegalMoveError, Move, Player
from connect4_mcts.players import Agent


class GameRunnerError(RuntimeError):
    """Raised when an automated game cannot continue."""


@dataclass(frozen=True, slots=True)
class MoveRecord:
    player: Player
    move: Move
    move_number: int


@dataclass(frozen=True, slots=True)
class PlayedGame:
    final_state: GameState
    moves: tuple[MoveRecord, ...]

    @property
    def result(self):
        return self.final_state.result


def play_game(
    red: Agent,
    yellow: Agent,
    initial_state: GameState | None = None,
    max_moves: int | None = None,
) -> PlayedGame:
    prepare_agents_for_game(red, yellow)
    state = initial_state or GameState.new(first_player=Player.RED)
    agents = {
        Player.RED: red,
        Player.YELLOW: yellow,
    }
    records: list[MoveRecord] = []

    while state.status is not GameStatus.FINISHED:
        if max_moves is not None and len(records) >= max_moves:
            raise GameRunnerError(f"game exceeded max_moves={max_moves}")

        player = state.current_player
        move = agents[player].choose_move(state)
        if not state.is_legal_move(move):
            raise IllegalMoveError(f"{player.value} agent returned illegal move: {move}")

        records.append(MoveRecord(player=player, move=move, move_number=state.move_count + 1))
        state = state.apply_move(move)

    return PlayedGame(final_state=state, moves=tuple(records))


def prepare_agents_for_game(red: Agent, yellow: Agent) -> None:
    """Reset per-game agent state and run LLM rules briefings when needed."""
    _begin_new_game_for_agents(red, yellow)
    for agent, llm_player in ((red, Player.RED), (yellow, Player.YELLOW)):
        send_briefing = getattr(agent, "send_rules_briefing", None)
        if callable(send_briefing):
            send_briefing(llm_player)


def _begin_new_game_for_agents(*agents: Agent) -> None:
    """Notify reused agents that a game ended (LLM players reset per-game state)."""
    for agent in agents:
        begin_new_game = getattr(agent, "begin_new_game", None)
        if callable(begin_new_game):
            begin_new_game()
