"""MCTS/UCT agent and its modifications (FPU, LGR, PMBp).

The agent implements the :class:`~connect4_mcts.players.base.Agent` protocol
(``choose_move``) so it can be plugged directly into the existing game runner,
CLI, GUI and experiment tooling.

The search tree is **persistent**: it is stored as a transposition table on the
player and kept across moves and games. A "trained" player is therefore simply
one whose tree has already been grown (typically by self-play). Statistics
gathered earlier are reused as a warm start, and the whole tree can be saved and
restored together with the player.

A single :class:`MCTSPlayer` covers every variant described in the project
report by toggling hyperparameters:

* base UCT - exploration constant ``C``;
* FPU (First Play Urgency) - fixed optimistic value for unvisited children;
* PMBp (Power-Mean Backpropagation) - power-mean aggregation parameter ``p``;
* LGR (Last Good Reply) - rollout policy backed by a learnable reply memory.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from connect4_mcts.game import GameResult, GameState, GameStatus, Move, Player
from connect4_mcts.players.base import MoveSelectionError


WIN_REWARD = 1.0
DRAW_REWARD = 0.5
LOSS_REWARD = 0.0

RolloutPolicy = str  # "random" | "lgr"
FinalMoveRule = str  # "robust" | "max_value"


@dataclass(slots=True)
class LGRMemory:
    """Learnable "Last Good Reply" table shared across moves and games.

    Maps ``(player, opponent_last_move)`` to the reply that previously appeared
    in a simulation won by ``player``. Together with the search tree this is the
    persistent, trainable state of an LGR agent.
    """

    replies: dict[tuple[Player, Move | None], Move] = field(default_factory=dict)

    def reply_for(self, player: Player, last_move: Move | None) -> Move | None:
        return self.replies.get((player, last_move))

    def update_from_game(
        self,
        history: list[tuple[Player, Move | None, Move]],
        winner: Player | None,
    ) -> None:
        if winner is None:
            return
        for player, last_move, move in history:
            if player is winner:
                self.replies[(player, last_move)] = move

    def __len__(self) -> int:
        return len(self.replies)


@dataclass(frozen=True, slots=True)
class SearchEvaluation:
    """Result of evaluating a position with the search tree.

    All values are win probabilities in ``[0, 1]`` from the perspective of
    ``player_to_move`` (the side that is about to move in the evaluated state).

    * ``root_value`` - value of the position under best play (the value of the
      best move); this is the "state value" used by the Blunder Rate metric.
    * ``move_values`` - value of each evaluated move, i.e. the win probability
      for ``player_to_move`` after playing that move.
    * ``move_visits`` - how many simulations backed each move (search support).
    """

    player_to_move: Player
    root_value: float
    best_move: Move
    move_values: dict[Move, float]
    move_visits: dict[Move, int]

    def value_of(self, move: Move) -> float | None:
        return self.move_values.get(move)

    def regret_of(self, move: Move) -> float | None:
        """How much worse ``move`` is than the best move (``0`` for the best)."""
        value = self.move_values.get(move)
        if value is None:
            return None
        return self.root_value - value

    def is_blunder(self, move: Move, threshold: float = 0.3) -> bool:
        regret = self.regret_of(move)
        return regret is not None and regret > threshold


@dataclass(slots=True)
class _Node:
    """Statistics for a single game state in the transposition table.

    ``children`` maps each legal move to the resulting game state, which is the
    key of the corresponding child node in the table.
    """

    visits: int = 0
    value_sum: float = 0.0
    power_sum: float = 0.0
    children: dict[Move, GameState] = field(default_factory=dict)
    expanded: bool = False


@dataclass(slots=True)
class MCTSPlayer:
    """Configurable MCTS/UCT agent with a persistent search tree.

    Hyperparameters
    ---------------
    iterations:
        Search budget: number of MCTS iterations performed per move. The tree
        these iterations grow is kept between calls.
    exploration:
        UCT exploration constant ``C``. Higher values explore more.
    fpu:
        First Play Urgency value assigned to unvisited children during
        selection. ``None`` keeps standard UCT (unvisited moves are tried
        first, i.e. treated as infinitely urgent).
    power_mean_p:
        Power-mean exponent ``p`` used to aggregate simulation rewards into a
        node value (PMBp). ``p == 1`` recovers the arithmetic mean. Must be
        positive.
    rollout_policy:
        ``"random"`` for uniform rollouts or ``"lgr"`` for the Last Good Reply
        policy backed by ``lgr_memory``.
    lgr_memory:
        Shared :class:`LGRMemory`. Created automatically when the LGR policy is
        selected without one.
    final_move:
        Rule for picking the move once the budget is spent: ``"robust"`` (most
        visited child) or ``"max_value"`` (highest average value).
    max_rollout_moves:
        Optional safety cap on rollout length. ``None`` plays rollouts to a
        terminal state (the game always terminates).
    seed:
        Seed for the internal RNG controlling rollouts and tie-breaking.
    tree:
        Persistent transposition table mapping game states to their statistics.
        This is the grown search tree and is preserved on save/load.
    """

    iterations: int = 200
    exploration: float = math.sqrt(2.0)
    fpu: float | None = None
    power_mean_p: float = 1.0
    rollout_policy: RolloutPolicy = "random"
    lgr_memory: LGRMemory | None = None
    final_move: FinalMoveRule = "robust"
    max_rollout_moves: int | None = None
    seed: int | None = None
    tree: dict[GameState, _Node] = field(default_factory=dict)
    _rng: random.Random = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.iterations < 1:
            raise ValueError("iterations must be at least 1")
        if self.exploration < 0:
            raise ValueError("exploration must be non-negative")
        if self.power_mean_p <= 0:
            raise ValueError("power_mean_p must be positive")
        if self.rollout_policy not in {"random", "lgr"}:
            raise ValueError("rollout_policy must be 'random' or 'lgr'")
        if self.final_move not in {"robust", "max_value"}:
            raise ValueError("final_move must be 'robust' or 'max_value'")
        if self.max_rollout_moves is not None and self.max_rollout_moves < 1:
            raise ValueError("max_rollout_moves must be at least 1 when set")
        if self.rollout_policy == "lgr" and self.lgr_memory is None:
            self.lgr_memory = LGRMemory()
        self._rng = random.Random(self.seed)

    @property
    def tree_size(self) -> int:
        """Number of states currently stored in the persistent tree."""
        return len(self.tree)

    def choose_move(self, state: GameState) -> Move:
        """Grow the tree from ``state`` and return the greedy best move."""
        legal_moves = state.legal_moves()
        if not legal_moves:
            raise MoveSelectionError("cannot choose a move when no legal moves are available")
        if len(legal_moves) == 1:
            return legal_moves[0]

        self.search(state)
        return self._best_move(state)

    def sample_move(self, state: GameState, temperature: float = 1.0) -> Move:
        """Grow the tree from ``state`` and sample a move (used in self-play).

        Moves are sampled proportionally to ``visits ** (1 / temperature)`` so
        that training games explore many lines instead of always following the
        single greedy variant. ``temperature <= 0`` falls back to the greedy
        move.
        """
        legal_moves = state.legal_moves()
        if not legal_moves:
            raise MoveSelectionError("cannot choose a move when no legal moves are available")
        if len(legal_moves) == 1:
            return legal_moves[0]

        self.search(state)
        if temperature <= 0:
            return self._best_move(state)
        return self._sample_move(state, temperature)

    def evaluate(self, root_state: GameState, run_search: bool = True) -> SearchEvaluation:
        """Search ``root_state`` and report the root value and per-move values.

        Intended for use as a reference/oracle engine (e.g. for the Blunder Rate
        metric): it returns the win probability of the position under best play
        plus the win probability of every candidate move, all from the
        perspective of the player about to move.

        With ``run_search=False`` no new iterations are run and only the
        statistics already cached in the tree are reported, which is useful for
        reusing a pre-built oracle tree.
        """
        legal_moves = root_state.legal_moves()
        if not legal_moves:
            raise MoveSelectionError("cannot evaluate a state with no legal moves")

        if run_search:
            self.search(root_state)
        root = self._node(root_state)

        move_values: dict[Move, float] = {}
        move_visits: dict[Move, int] = {}
        for move, child_state in root.children.items():
            child = self._node(child_state)
            if child.visits > 0:
                move_values[move] = child.value_sum / child.visits
                move_visits[move] = child.visits

        if not move_values:
            # No move was simulated (e.g. iterations exhausted on a single
            # forced branch); fall back to a neutral, uninformative estimate.
            fallback = legal_moves[0]
            return SearchEvaluation(
                player_to_move=root_state.current_player,
                root_value=DRAW_REWARD,
                best_move=fallback,
                move_values={move: DRAW_REWARD for move in legal_moves},
                move_visits={move: 0 for move in legal_moves},
            )

        best_move = max(move_values, key=lambda move: (move_values[move], move_visits[move]))
        return SearchEvaluation(
            player_to_move=root_state.current_player,
            root_value=move_values[best_move],
            best_move=best_move,
            move_values=move_values,
            move_visits=move_visits,
        )

    def search(self, root_state: GameState) -> None:
        """Run ``iterations`` MCTS iterations from ``root_state`` into the tree."""
        root = self._node(root_state)
        self._expand(root, root_state)
        for _ in range(self.iterations):
            self._run_iteration(root_state)

    def _run_iteration(self, root_state: GameState) -> None:
        path: list[GameState] = []
        state = root_state

        while True:
            node = self._node(state)
            path.append(state)
            if state.status is GameStatus.FINISHED or not node.expanded or not node.children:
                break
            move = self._select_move(node)
            state = node.children[move]

        node = self._node(state)
        if state.status is not GameStatus.FINISHED and node.visits > 0 and not node.expanded:
            self._expand(node, state)
            if node.children:
                move = self._select_move(node)
                state = node.children[move]
                path.append(state)

        result, history = self._simulate(state)

        for visited_state in path:
            self._update_node(self._node(visited_state), visited_state, result)

        if self.rollout_policy == "lgr" and self.lgr_memory is not None:
            self.lgr_memory.update_from_game(history, result.winner)

    def _node(self, state: GameState) -> _Node:
        node = self.tree.get(state)
        if node is None:
            node = _Node()
            self.tree[state] = node
        return node

    def _expand(self, node: _Node, state: GameState) -> None:
        if node.expanded or state.status is GameStatus.FINISHED:
            return
        for move in state.legal_moves():
            node.children[move] = state.apply_move(move)
        node.expanded = True

    def _select_move(self, node: _Node) -> Move:
        parent_visits = node.visits
        best_score = -math.inf
        best_moves: list[Move] = []

        for move, child_state in node.children.items():
            child = self._node(child_state)
            score = self._uct_score(child, parent_visits)
            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

        if len(best_moves) == 1:
            return best_moves[0]
        return self._rng.choice(best_moves)

    def _uct_score(self, child: _Node, parent_visits: int) -> float:
        if child.visits == 0:
            return self.fpu if self.fpu is not None else math.inf

        exploitation = (child.power_sum / child.visits) ** (1.0 / self.power_mean_p)
        if parent_visits <= 0 or self.exploration == 0:
            return exploitation
        exploration = self.exploration * math.sqrt(math.log(parent_visits) / child.visits)
        return exploitation + exploration

    def _simulate(self, state: GameState) -> tuple[GameResult, list[tuple[Player, Move | None, Move]]]:
        sim = state
        history: list[tuple[Player, Move | None, Move]] = []
        last_move: Move | None = None
        steps = 0

        while sim.status is not GameStatus.FINISHED:
            if self.max_rollout_moves is not None and steps >= self.max_rollout_moves:
                break
            legal_moves = sim.legal_moves()
            if not legal_moves:
                break
            chosen = self._rollout_move(sim, last_move, legal_moves)
            if self.rollout_policy == "lgr":
                history.append((sim.current_player, last_move, chosen))
            last_move = chosen
            sim = sim.apply_move(chosen)
            steps += 1

        result = sim.result if sim.result is not None else GameResult.from_board(sim.board)
        return result, history

    def _rollout_move(self, state: GameState, last_move: Move | None, legal_moves: tuple[Move, ...]) -> Move:
        if self.rollout_policy == "lgr" and self.lgr_memory is not None:
            reply = self.lgr_memory.reply_for(state.current_player, last_move)
            if reply is not None and reply in legal_moves:
                return reply
        return self._rng.choice(legal_moves)

    def _update_node(self, node: _Node, state: GameState, result: GameResult) -> None:
        reward = _reward_for(result, state.current_player.opponent)
        node.visits += 1
        node.value_sum += reward
        node.power_sum += reward**self.power_mean_p

    def _best_move(self, root_state: GameState) -> Move:
        root = self._node(root_state)
        if not root.children:
            return self._rng.choice(root_state.legal_moves())

        visited = [(move, self._node(s)) for move, s in root.children.items() if self._node(s).visits > 0]
        if not visited:
            return self._rng.choice(list(root.children.keys()))

        best_score = -math.inf
        best_moves: list[Move] = []
        for move, child in visited:
            score = self._final_score(child)
            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

        if len(best_moves) == 1:
            return best_moves[0]
        return self._rng.choice(best_moves)

    def _final_score(self, child: _Node) -> float:
        average_value = child.value_sum / child.visits
        if self.final_move == "max_value":
            return average_value
        return child.visits + average_value

    def _sample_move(self, root_state: GameState, temperature: float) -> Move:
        root = self._node(root_state)
        visited = [(move, self._node(s)) for move, s in root.children.items() if self._node(s).visits > 0]
        if not visited:
            return self._rng.choice(root_state.legal_moves())

        weights = [child.visits ** (1.0 / temperature) for _, child in visited]
        total = sum(weights)
        if total <= 0:
            return self._rng.choice([move for move, _ in visited])

        threshold = self._rng.random() * total
        cumulative = 0.0
        for (move, _), weight in zip(visited, weights):
            cumulative += weight
            if threshold <= cumulative:
                return move
        return visited[-1][0]


def _reward_for(result: GameResult, player: Player) -> float:
    if result.winner is None:
        return DRAW_REWARD
    return WIN_REWARD if result.winner is player else LOSS_REWARD
