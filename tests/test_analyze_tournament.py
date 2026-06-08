import csv
import json

import scripts.analyze_tournament as analyze_tournament


def test_analyze_tournament_writes_report_summaries(tmp_path) -> None:
    input_dir = tmp_path / "tournament"
    output_dir = tmp_path / "analysis"
    input_dir.mkdir()
    _write_games_csv(input_dir / "games.csv")
    _write_moves_jsonl(input_dir / "moves.jsonl")

    result = analyze_tournament.main(["--input-dir", str(input_dir), "--output-dir", str(output_dir)])

    assert result == 0
    agent_summary = _read_csv(output_dir / "agent_summary.csv")
    matchup_summary = _read_csv(output_dir / "matchup_summary.csv")
    llm_summary = _read_csv(output_dir / "llm_summary.csv")

    by_agent = {row["agent"]: row for row in agent_summary}
    assert by_agent["uct-m0"]["games"] == "2"
    assert by_agent["uct-m0"]["wins"] == "1"
    assert by_agent["uct-m0"]["draws"] == "1"
    assert float(by_agent["uct-m0"]["win_rate"]) == 0.5
    assert by_agent["uct-m0"]["games_as_red"] == "1"
    assert by_agent["uct-m0"]["games_as_yellow"] == "1"
    assert by_agent["uct-m0"]["tree_growth"] == "20"
    assert float(by_agent["uct-m0"]["avg_legal_moves"]) == 14.0

    llm = by_agent["llm-test"]
    assert llm["llm_requests"] == "5"
    assert llm["llm_unparseable"] == "1"
    assert llm["llm_illegal"] == "1"
    assert llm["llm_fallbacks"] == "1"
    assert float(llm["llm_invalid_response_rate"]) == 0.4

    assert len(matchup_summary) == 1
    assert matchup_summary[0]["pair_id"] == "uct-m0_vs_llm-test"
    assert matchup_summary[0]["left_wins"] == "1"
    assert matchup_summary[0]["draws"] == "1"

    assert len(llm_summary) == 1
    assert llm_summary[0]["agent"] == "llm-test"
    assert llm_summary[0]["llm_invalid_responses"] == "2"


def _write_games_csv(path) -> None:
    rows = [
        {
            "game_id": "g1",
            "pair_id": "uct-m0_vs_llm-test",
            "pair_index": 0,
            "game_index": 0,
            "seed": 1,
            "red_agent": "uct-m0",
            "yellow_agent": "llm-test",
            "red_kind": "mcts",
            "yellow_kind": "llm",
            "winner_agent": "uct-m0",
            "winner_color": "red",
            "is_draw": False,
            "red_lines": 0,
            "yellow_lines": 1,
            "moves": 2,
            "red_decisions": 1,
            "yellow_decisions": 1,
            "red_total_decision_seconds": 0.1,
            "yellow_total_decision_seconds": 1.0,
            "red_avg_decision_seconds": 0.1,
            "yellow_avg_decision_seconds": 1.0,
            "red_tree_size_before": 10,
            "red_tree_size_after": 20,
            "yellow_tree_size_before": "",
            "yellow_tree_size_after": "",
            "red_llm_requests": 0,
            "red_llm_unparseable": 0,
            "red_llm_illegal": 0,
            "red_llm_fallbacks": 0,
            "red_llm_moves": 0,
            "red_llm_invalid_response_rate": 0.0,
            "red_llm_fallback_rate": 0.0,
            "yellow_llm_requests": 3,
            "yellow_llm_unparseable": 1,
            "yellow_llm_illegal": 1,
            "yellow_llm_fallbacks": 1,
            "yellow_llm_moves": 1,
            "yellow_llm_invalid_response_rate": 0.666,
            "yellow_llm_fallback_rate": 1.0,
        },
        {
            "game_id": "g2",
            "pair_id": "uct-m0_vs_llm-test",
            "pair_index": 0,
            "game_index": 1,
            "seed": 2,
            "red_agent": "llm-test",
            "yellow_agent": "uct-m0",
            "red_kind": "llm",
            "yellow_kind": "mcts",
            "winner_agent": "",
            "winner_color": "",
            "is_draw": True,
            "red_lines": 1,
            "yellow_lines": 1,
            "moves": 2,
            "red_decisions": 1,
            "yellow_decisions": 1,
            "red_total_decision_seconds": 2.0,
            "yellow_total_decision_seconds": 0.2,
            "red_avg_decision_seconds": 2.0,
            "yellow_avg_decision_seconds": 0.2,
            "red_tree_size_before": "",
            "red_tree_size_after": "",
            "yellow_tree_size_before": 20,
            "yellow_tree_size_after": 30,
            "red_llm_requests": 2,
            "red_llm_unparseable": 0,
            "red_llm_illegal": 0,
            "red_llm_fallbacks": 0,
            "red_llm_moves": 1,
            "red_llm_invalid_response_rate": 0.0,
            "red_llm_fallback_rate": 0.0,
            "yellow_llm_requests": 0,
            "yellow_llm_unparseable": 0,
            "yellow_llm_illegal": 0,
            "yellow_llm_fallbacks": 0,
            "yellow_llm_moves": 0,
            "yellow_llm_invalid_response_rate": 0.0,
            "yellow_llm_fallback_rate": 0.0,
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_moves_jsonl(path) -> None:
    rows = [
        {"game_id": "g1", "agent": "uct-m0", "legal_move_count": 16},
        {"game_id": "g1", "agent": "llm-test", "legal_move_count": 15},
        {"game_id": "g2", "agent": "llm-test", "legal_move_count": 14},
        {"game_id": "g2", "agent": "uct-m0", "legal_move_count": 12},
    ]
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))
