"""Position evaluation helpers for the suicide Connect4 variant."""

from __future__ import annotations

from connect4_mcts.game import COLUMNS, ROWS, GameState, GameStatus, Player


TERMINAL_SCORE = 1_000_000
WINDOW_WEIGHTS = {
    1: 1,
    2: 3,
    3: 9,
}


def evaluate_position(state: GameState, player: Player) -> int:
    """Return a heuristic score for ``state`` from ``player``'s perspective."""
    if not isinstance(player, Player):
        raise ValueError("player must be a Player")

    if state.result is not None:
        return _terminal_score(state, player)

    if state.status is GameStatus.FAIR_TURN:
        return _fair_turn_score(state, player)

    opponent = player.opponent
    return _potential_line_score(state, opponent) - _potential_line_score(state, player)


def _fair_turn_score(state: GameState, player: Player) -> int:
    legal_moves = state.legal_moves()
    if not legal_moves:
        return 0

    scores = [evaluate_position(state.apply_move(move), player) for move in legal_moves]
    if state.current_player is player:
        return max(scores)
    return min(scores)


def _terminal_score(state: GameState, player: Player) -> int:
    if state.result is None:
        raise ValueError("state must have a result")

    if state.result.winner is None:
        return 0
    if state.result.winner is player:
        return TERMINAL_SCORE
    return -TERMINAL_SCORE


def _potential_line_score(state: GameState, player: Player) -> int:
    score = 0
    opponent = player.opponent

    for window in _windows_of_four(state):
        player_cells = sum(1 for cell in window if cell is player)
        opponent_cells = sum(1 for cell in window if cell is opponent)
        if opponent_cells > 0:
            continue
        score += WINDOW_WEIGHTS.get(player_cells, 0)

    return score


def _windows_of_four(state: GameState):
    directions = ((0, 1), (1, 0), (1, 1), (1, -1))

    for row in range(ROWS):
        for column in range(COLUMNS):
            for row_step, column_step in directions:
                end_row = row + row_step * 3
                end_column = column + column_step * 3
                if not (0 <= end_row < ROWS and 0 <= end_column < COLUMNS):
                    continue

                yield tuple(
                    state.board[row + row_step * offset][column + column_step * offset]
                    for offset in range(4)
                )
