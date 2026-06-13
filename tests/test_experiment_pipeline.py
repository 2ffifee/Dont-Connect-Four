import json

import pytest

from connect4_mcts.experiment_pipeline import (
    mark_stage_complete,
    parse_skip_stages,
    pipeline_state_path,
    stages_to_skip,
)
from connect4_mcts.tournament_progress import (
    build_checkpoint_payload,
    can_resume_tournament,
    flush_tournament_progress,
    load_checkpoint,
    tournament_is_complete,
    validate_checkpoint,
)


def test_parse_skip_stages_accepts_aliases_and_commas() -> None:
    stages = parse_skip_stages(["tournament,blunders", "analyze"])
    assert stages == {"run-tournament", "score-blunders", "analyze-tournament"}


def test_parse_skip_stages_rejects_unknown_stage() -> None:
    with pytest.raises(ValueError, match="unknown pipeline stage"):
        parse_skip_stages(["not-a-stage"])


def test_stages_to_skip_merges_explicit_and_resume(tmp_path) -> None:
    mark_stage_complete(tmp_path, "run-tournament")
    state = json.loads(pipeline_state_path(tmp_path).read_text(encoding="utf-8"))
    from connect4_mcts.experiment_pipeline import PipelineState

    pipeline_state = PipelineState(
        version=state["version"],
        completed_stages=tuple(state["completed_stages"]),
        updated_at=state["updated_at"],
        metadata=state["metadata"],
    )
    skipped = stages_to_skip(
        explicit_skip={"analyze-tournament"},
        resume_pipeline=True,
        state=pipeline_state,
    )
    assert skipped == {"run-tournament", "analyze-tournament"}


def test_tournament_checkpoint_round_trip(tmp_path) -> None:
    checkpoint = build_checkpoint_payload(
        config_path="configs/experiments/smoke.toml",
        base_seed=3,
        games_per_pair=2,
        player_ids=["a", "b"],
        completed_pairs=1,
        total_pairs=1,
        pair_ids=["a_vs_b"],
        status="complete",
    )
    flush_tournament_progress(
        tmp_path,
        game_rows=[{"game_id": "a_vs_b_g0000", "winner_agent": "a"}],
        move_rows=[{"game_id": "a_vs_b_g0000", "move_number": 1}],
        pair_summaries=[{"pair_id": "a_vs_b"}],
        checkpoint=checkpoint,
    )

    loaded = load_checkpoint(tmp_path)
    assert loaded is not None
    validate_checkpoint(
        loaded,
        config_path="configs/experiments/smoke.toml",
        base_seed=3,
        games_per_pair=2,
        player_ids=["a", "b"],
    )
    assert tournament_is_complete(tmp_path)
    assert not can_resume_tournament(tmp_path)
