"""Simple match simulation tools for comparing agents."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass

from connect4_mcts.game import GameState, GameStatus, IllegalMoveError, Move, Player
from connect4_mcts.players import Agent
from connect4_mcts.players.factory import AGENT_CHOICES, AgentName, create_agent, format_agent_name
from connect4_mcts.runner import GameRunnerError


@dataclass(frozen=True, slots=True)
class TimedMoveRecord:
    player: Player
    agent_name: AgentName
    move: Move
    move_number: int
    decision_time_seconds: float


@dataclass(frozen=True, slots=True)
class SimulatedGame:
    red_agent: AgentName
    yellow_agent: AgentName
    final_state: GameState
    moves: tuple[TimedMoveRecord, ...]
    seed: int | None = None

    @property
    def winner_agent(self) -> AgentName | None:
        if self.final_state.result is None or self.final_state.result.winner is None:
            return None
        if self.final_state.result.winner is Player.RED:
            return self.red_agent
        return self.yellow_agent


@dataclass(frozen=True, slots=True)
class MatchSummary:
    games: tuple[SimulatedGame, ...]
    agent_names: tuple[AgentName, ...]
    wins: dict[AgentName, int]
    draws: int
    move_counts: dict[AgentName, int]
    decision_time_seconds: dict[AgentName, float]

    @property
    def game_count(self) -> int:
        return len(self.games)

    @property
    def average_moves(self) -> float:
        if not self.games:
            return 0.0
        return sum(len(game.moves) for game in self.games) / len(self.games)

    def average_decision_time(self, agent_name: AgentName) -> float:
        move_count = self.move_counts.get(agent_name, 0)
        if move_count == 0:
            return 0.0
        return self.decision_time_seconds.get(agent_name, 0.0) / move_count


def play_timed_game(
    red_agent_name: AgentName,
    yellow_agent_name: AgentName,
    red: Agent,
    yellow: Agent,
    initial_state: GameState | None = None,
    max_moves: int | None = None,
    seed: int | None = None,
) -> SimulatedGame:
    state = initial_state or GameState.new(first_player=Player.RED)
    agents = {
        Player.RED: (red_agent_name, red),
        Player.YELLOW: (yellow_agent_name, yellow),
    }
    records: list[TimedMoveRecord] = []

    while state.status is not GameStatus.FINISHED:
        if max_moves is not None and len(records) >= max_moves:
            raise GameRunnerError(f"game exceeded max_moves={max_moves}")

        player = state.current_player
        agent_name, agent = agents[player]
        started_at = time.perf_counter()
        move = agent.choose_move(state)
        elapsed = time.perf_counter() - started_at
        if not state.is_legal_move(move):
            raise IllegalMoveError(f"{player.value} agent returned illegal move: {move}")

        records.append(
            TimedMoveRecord(
                player=player,
                agent_name=agent_name,
                move=move,
                move_number=state.move_count + 1,
                decision_time_seconds=elapsed,
            )
        )
        state = state.apply_move(move)

    return SimulatedGame(
        red_agent=red_agent_name,
        yellow_agent=yellow_agent_name,
        final_state=state,
        moves=tuple(records),
        seed=seed,
    )


def run_match(
    red_agent: AgentName,
    yellow_agent: AgentName,
    games: int,
    seed: int | None = None,
    depth: int = 3,
    swap_sides: bool = False,
    max_moves: int | None = None,
) -> MatchSummary:
    if games < 1:
        raise ValueError("games must be at least 1")

    played_games: list[SimulatedGame] = []
    for game_index in range(games):
        current_red, current_yellow = _agent_pair_for_game(red_agent, yellow_agent, game_index, swap_sides)
        game_seed = None if seed is None else seed + game_index * 2
        red_seed = game_seed
        yellow_seed = None if game_seed is None else game_seed + 1

        played_games.append(
            play_timed_game(
                red_agent_name=current_red,
                yellow_agent_name=current_yellow,
                red=create_agent(current_red, seed=red_seed, depth=depth),
                yellow=create_agent(current_yellow, seed=yellow_seed, depth=depth),
                max_moves=max_moves,
                seed=game_seed,
            )
        )

    return summarize_games(tuple(played_games), agent_names=_unique_names(red_agent, yellow_agent))


def summarize_games(games: tuple[SimulatedGame, ...], agent_names: tuple[AgentName, ...]) -> MatchSummary:
    wins: dict[AgentName, int] = {agent_name: 0 for agent_name in agent_names}
    move_counts: dict[AgentName, int] = {agent_name: 0 for agent_name in agent_names}
    decision_time_seconds: dict[AgentName, float] = {agent_name: 0.0 for agent_name in agent_names}
    draws = 0

    for game in games:
        if game.winner_agent is None:
            draws += 1
        else:
            wins[game.winner_agent] += 1

        for move in game.moves:
            move_counts[move.agent_name] += 1
            decision_time_seconds[move.agent_name] += move.decision_time_seconds

    return MatchSummary(
        games=games,
        agent_names=agent_names,
        wins=wins,
        draws=draws,
        move_counts=move_counts,
        decision_time_seconds=decision_time_seconds,
    )


def format_match_summary(summary: MatchSummary) -> str:
    lines = [f"Games: {summary.game_count}"]
    for agent_name in summary.agent_names:
        wins = summary.wins.get(agent_name, 0)
        win_rate = wins / summary.game_count if summary.game_count else 0.0
        avg_decision_ms = summary.average_decision_time(agent_name) * 1000
        lines.append(f"{format_agent_name(agent_name)} wins: {wins} ({win_rate:.1%})")
        lines.append(f"{format_agent_name(agent_name)} avg decision: {avg_decision_ms:.3f} ms")

    draw_rate = summary.draws / summary.game_count if summary.game_count else 0.0
    lines.append(f"Draws: {summary.draws} ({draw_rate:.1%})")
    lines.append(f"Average moves: {summary.average_moves:.2f}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run automated matches between agents.")
    parser.add_argument("--red", choices=AGENT_CHOICES, default="minimax", help="Initial red agent.")
    parser.add_argument("--yellow", choices=AGENT_CHOICES, default="random", help="Initial yellow agent.")
    parser.add_argument("--games", type=int, default=20, help="Number of games to simulate.")
    parser.add_argument("--seed", type=int, default=None, help="Base seed for seeded agents.")
    parser.add_argument("--depth", type=int, default=3, help="Search depth for minimax.")
    parser.add_argument("--swap-sides", action="store_true", help="Alternate red/yellow assignment between games.")
    args = parser.parse_args(argv)

    summary = run_match(
        red_agent=args.red,
        yellow_agent=args.yellow,
        games=args.games,
        seed=args.seed,
        depth=args.depth,
        swap_sides=args.swap_sides,
    )
    print(format_match_summary(summary))
    return 0


def _agent_pair_for_game(
    red_agent: AgentName,
    yellow_agent: AgentName,
    game_index: int,
    swap_sides: bool,
) -> tuple[AgentName, AgentName]:
    if swap_sides and game_index % 2 == 1:
        return yellow_agent, red_agent
    return red_agent, yellow_agent


def _unique_names(*agent_names: AgentName) -> tuple[AgentName, ...]:
    return tuple(dict.fromkeys(agent_names))


if __name__ == "__main__":
    raise SystemExit(main())
