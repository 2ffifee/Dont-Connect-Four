import csv
import json

import scripts.run_tournament as run_tournament


def test_run_tournament_writes_metric_outputs(tmp_path) -> None:
    config_path = tmp_path / "tiny_tournament.toml"
    output_dir = tmp_path / "results"
    config_path.write_text(
        """
seed = 7
output_dir = "ignored"
games_per_pair = 1

[[players]]
id = "random-a"
type = "random"

[[players]]
id = "random-b"
type = "random"
""".strip(),
        encoding="utf-8",
    )

    result = run_tournament.main(
        [
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--max-moves",
            "200",
        ]
    )

    assert result == 0
    assert (output_dir / "games.csv").exists()
    assert (output_dir / "moves.jsonl").exists()
    assert (output_dir / "pair_summary.csv").exists()
    assert (output_dir / "standings.csv").exists()
    assert (output_dir / "run_metadata.json").exists()

    with (output_dir / "games.csv").open(newline="", encoding="utf-8") as file:
        games = list(csv.DictReader(file))

    assert len(games) == 1
    assert games[0]["game_id"] == "random-a_vs_random-b_g0000"
    assert games[0]["pair_id"] == "random-a_vs_random-b"
    assert games[0]["seed"] == "107"
    assert games[0]["red_agent"] == "random-a"
    assert games[0]["yellow_agent"] == "random-b"
    assert int(games[0]["moves"]) > 0

    with (output_dir / "moves.jsonl").open(encoding="utf-8") as file:
        first_move = json.loads(file.readline())

    assert first_move["game_id"] == "random-a_vs_random-b_g0000"
    assert first_move["move_number"] == 1
    assert first_move["agent"] == "random-a"
    assert first_move["state_before"]["current_player"] == "red"
    assert first_move["state_before"]["board"] == ["........"] * 6
    assert first_move["legal_move_count"] == 16

    with (output_dir / "pair_summary.csv").open(newline="", encoding="utf-8") as file:
        pairs = list(csv.DictReader(file))

    assert pairs[0]["left"] == "random-a"
    assert pairs[0]["right"] == "random-b"
    assert pairs[0]["games"] == "1"

    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["players"] == ["random-a", "random-b"]
    assert metadata["games_per_pair"] == 1

    checkpoint = json.loads((output_dir / "tournament_checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["status"] == "complete"
    assert checkpoint["completed_pairs"] == 1


def test_run_tournament_resume_continues_from_checkpoint(tmp_path) -> None:
    config_path = tmp_path / "tiny_tournament.toml"
    output_dir = tmp_path / "results"
    config_path.write_text(
        """
seed = 11
output_dir = "ignored"
games_per_pair = 1

[[players]]
id = "random-a"
type = "random"

[[players]]
id = "random-b"
type = "random"

[[players]]
id = "random-c"
type = "random"
""".strip(),
        encoding="utf-8",
    )

    result = run_tournament.main(
        [
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--max-moves",
            "200",
        ]
    )
    assert result == 0

    with (output_dir / "games.csv").open(newline="", encoding="utf-8") as file:
        full_games = list(csv.DictReader(file))
    assert len(full_games) == 3

    checkpoint = json.loads((output_dir / "tournament_checkpoint.json").read_text(encoding="utf-8"))
    checkpoint["status"] = "in_progress"
    checkpoint["completed_pairs"] = 1
    (output_dir / "tournament_checkpoint.json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")

    first_pair_id = full_games[0]["pair_id"]
    partial_games = [row for row in full_games if row["pair_id"] == first_pair_id]
    partial_pairs = [row for row in csv.DictReader((output_dir / "pair_summary.csv").open(encoding="utf-8")) if row["pair_id"] == first_pair_id]
    partial_moves = []
    with (output_dir / "moves.jsonl").open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            if row["pair_id"] == first_pair_id:
                partial_moves.append(row)

    with (output_dir / "games.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=partial_games[0].keys())
        writer.writeheader()
        writer.writerows(partial_games)
    with (output_dir / "pair_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=partial_pairs[0].keys())
        writer.writeheader()
        writer.writerows(partial_pairs)
    with (output_dir / "moves.jsonl").open("w", encoding="utf-8") as file:
        for row in partial_moves:
            file.write(json.dumps(row) + "\n")

    result = run_tournament.main(
        [
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--max-moves",
            "200",
            "--resume",
        ]
    )
    assert result == 0

    with (output_dir / "games.csv").open(newline="", encoding="utf-8") as file:
        resumed_games = list(csv.DictReader(file))
    assert len(resumed_games) == 3
    assert (output_dir / "tournament_checkpoint.json").read_text(encoding="utf-8").count('"status": "complete"') == 1
