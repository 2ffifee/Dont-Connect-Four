"""Tests for tournament failure reporting helpers."""

from __future__ import annotations

from connect4_mcts.tournament_reporting import (
    TournamentFailureContext,
    describe_exit_code,
    format_failure_banner,
    write_failure_report,
)


def test_describe_exit_code_explains_sigkill() -> None:
    assert "OOM" in describe_exit_code(137) or "SIGKILL" in describe_exit_code(137)
    assert "signal" in describe_exit_code(-9)


def test_format_failure_banner_includes_matchup_and_game() -> None:
    context = TournamentFailureContext(
        pair_index=16,
        pair_total=28,
        pair_id="minimax-d3_vs_oracle",
        left_player="minimax-d3",
        right_player="oracle",
        game_index=4,
        games_per_pair=10,
        game_id="minimax-d3_vs_oracle_g0004",
        seed=12345,
        red_agent="oracle",
        yellow_agent="minimax-d3",
        move_number=17,
        current_player="yellow",
    )
    banner = format_failure_banner(context, RuntimeError("simulated crash"))
    assert "minimax-d3 vs oracle" in banner
    assert "minimax-d3_vs_oracle_g0004" in banner
    assert "#17" in banner


def test_write_failure_report_persists_json(tmp_path) -> None:
    context = TournamentFailureContext(
        pair_index=0,
        pair_total=1,
        pair_id="a_vs_b",
        left_player="a",
        right_player="b",
        game_index=0,
        games_per_pair=1,
        game_id="a_vs_b_g0000",
        seed=1,
        red_agent="a",
        yellow_agent="b",
    )
    path = write_failure_report(tmp_path, context=context, exc=ValueError("bad move"))
    assert path.is_file()
    assert "traceback" in path.read_text(encoding="utf-8")
