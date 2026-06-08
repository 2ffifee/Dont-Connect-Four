import pytest

from connect4_mcts.game import GameState
from connect4_mcts.players.mcts import LGRMemory, MCTSPlayer
from connect4_mcts.training import (
    grow_player_to_memory_cap,
    load_player,
    max_nodes_from_memory_gb,
    save_player,
    selfplay_train,
    train_fpu,
    train_lgr,
    train_pmbp,
    train_uct,
)


def test_max_nodes_from_memory_gb_respects_zero_budget() -> None:
    assert max_nodes_from_memory_gb(0.0) == 0
    assert max_nodes_from_memory_gb(7.0) > max_nodes_from_memory_gb(2.0)


def test_grow_player_to_memory_cap_skips_when_cap_is_zero() -> None:
    player = train_uct(iterations=20, seed=1)
    games = grow_player_to_memory_cap(player, max_nodes=0)
    assert games == 0
    assert player.tree_size == 0


def test_train_uct_sets_hyperparameters() -> None:
    player = train_uct(iterations=250, exploration=1.0, seed=5)

    assert isinstance(player, MCTSPlayer)
    assert player.iterations == 250
    assert player.exploration == 1.0
    assert player.rollout_policy == "random"
    assert player.fpu is None


def test_train_fpu_sets_first_play_urgency() -> None:
    player = train_fpu(iterations=100, fpu=0.4, seed=5)

    assert player.fpu == 0.4
    assert player.rollout_policy == "random"


def test_train_pmbp_sets_power_mean_exponent() -> None:
    player = train_pmbp(iterations=100, power_mean_p=3.0, seed=5)

    assert player.power_mean_p == 3.0


def test_train_lgr_uses_lgr_rollout_policy() -> None:
    player = train_lgr(iterations=100, seed=5)

    assert player.rollout_policy == "lgr"
    assert isinstance(player.lgr_memory, LGRMemory)


def test_train_lgr_warmup_populates_shared_memory() -> None:
    memory = LGRMemory()

    player = train_lgr(
        iterations=30,
        selfplay_games=2,
        selfplay_iterations=20,
        memory=memory,
        seed=5,
    )

    assert player.lgr_memory is memory
    assert len(memory) > 0


def test_train_lgr_rejects_negative_selfplay_games() -> None:
    with pytest.raises(ValueError, match="games"):
        train_lgr(selfplay_games=-1)


def test_selfplay_training_grows_the_persistent_tree() -> None:
    player = train_uct(iterations=20, selfplay_games=3, selfplay_iterations=15, seed=1)

    assert player.tree_size > 0


def test_selfplay_train_extends_existing_tree() -> None:
    player = train_uct(iterations=20, seed=1)
    assert player.tree_size == 0

    selfplay_train(player, games=2, iterations=15)
    first_size = player.tree_size
    assert first_size > 0

    selfplay_train(player, games=2, iterations=15)
    assert player.tree_size >= first_size


def test_selfplay_train_restores_iteration_budget() -> None:
    player = train_uct(iterations=50, seed=1)

    selfplay_train(player, games=1, iterations=10)

    assert player.iterations == 50


def test_trained_players_choose_legal_moves() -> None:
    state = GameState.new()
    players = [
        train_uct(iterations=40, seed=1),
        train_fpu(iterations=40, fpu=1.0, seed=1),
        train_pmbp(iterations=40, power_mean_p=2.0, seed=1),
        train_lgr(iterations=40, seed=1),
    ]

    for player in players:
        assert player.choose_move(state) in state.legal_moves()


def test_save_and_load_round_trip_preserves_player_and_tree(tmp_path) -> None:
    player = train_pmbp(iterations=30, power_mean_p=2.5, selfplay_games=2, selfplay_iterations=20, seed=11)
    path = tmp_path / "player.pkl"

    save_player(player, path)
    loaded = load_player(path)

    assert isinstance(loaded, MCTSPlayer)
    assert loaded.iterations == player.iterations
    assert loaded.power_mean_p == player.power_mean_p
    assert loaded.tree_size == player.tree_size
    assert loaded.tree_size > 0
    assert loaded.choose_move(GameState.new()) in GameState.new().legal_moves()


def test_save_and_load_preserves_learned_lgr_memory(tmp_path) -> None:
    player = train_lgr(iterations=30, selfplay_games=2, selfplay_iterations=20, seed=11)
    path = tmp_path / "lgr.pkl"

    save_player(player, path)
    loaded = load_player(path)

    assert loaded.rollout_policy == "lgr"
    assert len(loaded.lgr_memory) == len(player.lgr_memory)
    assert loaded.lgr_memory.replies == player.lgr_memory.replies
