"""Simple command line interface for playing against a random agent."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from connect4_mcts.game import COLUMNS, ROWS, GameResult, GameState, GameStatus, IllegalMoveError, Move, MoveType, Player
from connect4_mcts.players import RandomPlayer


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Play modified Connect4 in the terminal.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for the random player.")
    parser.add_argument("--human", choices=("red", "yellow"), default="red", help="Human player color.")
    parser.add_argument("--demo", action="store_true", help="Run a random-vs-random demo instead of interactive play.")
    args = parser.parse_args(argv)

    if args.demo:
        run_random_demo(seed=args.seed)
    else:
        run_human_vs_random(human=Player(args.human), seed=args.seed)
    return 0


def run_human_vs_random(human: Player = Player.RED, seed: int | None = None) -> GameState:
    state = GameState.new(first_player=Player.RED)
    random_player = RandomPlayer(seed=seed)

    print("Commands: d <column> for drop, p <column> for push, q to quit.")
    print("Columns are numbered from 1 to 8.")

    while state.status is not GameStatus.FINISHED:
        print()
        print(render_board(state))
        print(status_message(state))

        if state.current_player is human:
            move = prompt_for_move(state)
        else:
            move = random_player.choose_move(state)
            print(f"Random plays: {format_move(move)}")

        state = state.apply_move(move)

    print()
    print(render_board(state))
    print(result_message(state.result))
    return state


def run_random_demo(seed: int | None = None) -> GameState:
    state = GameState.new(first_player=Player.RED)
    red = RandomPlayer(seed=seed)
    yellow = RandomPlayer(seed=None if seed is None else seed + 1)

    while state.status is not GameStatus.FINISHED:
        player = red if state.current_player is Player.RED else yellow
        move = player.choose_move(state)
        print(f"{state.move_count + 1:02d}. {state.current_player.value}: {format_move(move)}")
        state = state.apply_move(move)

    print()
    print(render_board(state))
    print(result_message(state.result))
    return state


def prompt_for_move(state: GameState) -> Move:
    while True:
        raw_input = input("Your move: ").strip()
        if raw_input.lower() in {"q", "quit", "exit"}:
            raise SystemExit(0)

        try:
            move = parse_move(raw_input)
        except ValueError as error:
            print(error)
            continue

        if state.is_legal_move(move):
            return move

        print("Illegal move for the current board.")


def parse_move(raw_input: str) -> Move:
    parts = raw_input.lower().split()
    if len(parts) != 2:
        raise ValueError("Enter a move as: d <column> or p <column>.")

    move_type = _parse_move_type(parts[0])
    try:
        column = int(parts[1]) - 1
    except ValueError as error:
        raise ValueError("Column must be a number from 1 to 8.") from error

    try:
        return Move(move_type, column)
    except ValueError as error:
        raise ValueError("Column must be a number from 1 to 8.") from error


def render_board(state: GameState) -> str:
    rendered_rows = []
    for row in state.board:
        rendered_rows.append("|" + "|".join(render_cell(cell) for cell in row) + "|")

    columns = " " + " ".join(str(column) for column in range(1, COLUMNS + 1))
    return "\n".join(rendered_rows + [columns])


def render_cell(cell: Player | None) -> str:
    if cell is Player.RED:
        return "R"
    if cell is Player.YELLOW:
        return "Y"
    return "."


def status_message(state: GameState) -> str:
    fair_turn = " (fair turn)" if state.status is GameStatus.FAIR_TURN else ""
    return f"Turn: {state.current_player.value}{fair_turn}"


def result_message(result: GameResult | None) -> str:
    if result is None:
        raise IllegalMoveError("finished game is missing a result")

    if result.is_draw:
        outcome = "Draw"
    else:
        outcome = f"Winner: {result.winner.value}"

    return f"{outcome}. Lines: red={result.red_lines}, yellow={result.yellow_lines}"


def format_move(move: Move) -> str:
    prefix = "d" if move.move_type is MoveType.DROP else "p"
    return f"{prefix} {move.column + 1}"


def _parse_move_type(raw_move_type: str) -> MoveType:
    if raw_move_type in {"d", "drop"}:
        return MoveType.DROP
    if raw_move_type in {"p", "push"}:
        return MoveType.PUSH
    raise ValueError("Move type must be d/drop or p/push.")


if __name__ == "__main__":
    raise SystemExit(main())
