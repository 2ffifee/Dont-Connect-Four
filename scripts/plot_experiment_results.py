"""Generate publication-ready figures from tournament experiment outputs.

Reads CSV summaries produced by the tournament pipeline
(``run_tournament.py``, ``analyze_tournament.py``, ``score_blunders.py``)
and writes PDF/PNG figures for the LaTeX report.

Usage::

    python scripts/plot_experiment_results.py
    python scripts/plot_experiment_results.py --input-dir results/main_final
    python scripts/plot_experiment_results.py --output-dir report/final/figures
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Consistent palette by agent family (report-friendly, color-blind aware).
CATEGORY_COLORS = {
    "mcts": "#2166ac",
    "minimax": "#b35806",
    "llm": "#d6604d",
    "random": "#878787",
    "other": "#4daf4a",
}

MCTS_AGENTS = ("uct-1000-1.41", "fpu-1000-1.0", "pmbp-1000-2.5", "lgr-1000")
MCTS_LABELS = {
    "uct-1000-1.41": "UCT",
    "fpu-1000-1.0": "FPU",
    "pmbp-1000-2.5": "PMBp",
    "lgr-1000": "LGR",
}

DISPLAY_NAMES = {
    "uct-1000-1.41": "UCT",
    "fpu-1000-1.0": "FPU",
    "pmbp-1000-2.5": "PMBp",
    "lgr-1000": "LGR",
    "minimax-d3": "Minimax d=3",
    "minimax-d4": "Minimax d=4",
    "random": "Losowy",
    "openchat": "OpenChat",
    "qwen2.5": "Qwen 2.5",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot tournament experiment results for the report.")
    parser.add_argument(
        "--input-dir",
        default="results/main_final",
        help="Directory with tournament CSV outputs (default: results/main_final).",
    )
    parser.add_argument(
        "--output-dir",
        default="report/final/figures",
        help="Directory for generated figures (default: report/final/figures).",
    )
    parser.add_argument("--dpi", type=int, default=200, help="PNG resolution (default: 200).")
    args = parser.parse_args(argv)

    input_dir = _resolve(args.input_dir)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    standings = _read_csv(input_dir / "standings.csv")
    agent_summary = _read_csv(input_dir / "agent_summary.csv")
    matchup_summary = _read_csv(input_dir / "matchup_summary.csv")
    blunder_summary = _read_csv(input_dir / "blunder_summary.csv")
    llm_summary = _read_csv(input_dir / "llm_summary.csv")

    if not standings:
        raise SystemExit(f"Brak pliku standings.csv w {input_dir}")

    _configure_style()

    generated: list[str] = []
    generated += _plot_standings_win_rate(standings, output_dir, args.dpi)
    generated += _plot_standings_outcomes(standings, output_dir, args.dpi)
    generated += _plot_draw_rate(standings, output_dir, args.dpi)
    generated += _plot_decision_time(agent_summary or standings, output_dir, args.dpi)
    generated += _plot_side_balance(agent_summary, output_dir, args.dpi)
    if matchup_summary:
        generated += _plot_matchup_heatmap(matchup_summary, standings, output_dir, args.dpi)
        generated += _plot_mcts_head_to_head(matchup_summary, output_dir, args.dpi)
    if blunder_summary:
        generated += _plot_blunder_rate(blunder_summary, output_dir, args.dpi)
        generated += _plot_blunder_regret(blunder_summary, output_dir, args.dpi)
        generated += _plot_skill_vs_blunders(standings, blunder_summary, output_dir, args.dpi)
        generated += _plot_mcts_variants(standings, blunder_summary, output_dir, args.dpi)
    if llm_summary:
        generated += _plot_llm_reliability(llm_summary, output_dir, args.dpi)
    human_rows = _read_human_games(input_dir / "humans.csv")
    if human_rows:
        generated += _plot_humans_vs_agents(human_rows, output_dir, args.dpi)
        generated += _plot_humans_per_participant(human_rows, output_dir, args.dpi)

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    for path in generated:
        print(f"  {path.name}")
    return 0


def _resolve(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def _display_name(agent_id: str) -> str:
    return DISPLAY_NAMES.get(agent_id, agent_id)


def _agent_category(agent_id: str, kind: str = "") -> str:
    lowered = agent_id.lower()
    if kind == "llm" or lowered in {"openchat", "qwen2.5"}:
        return "llm"
    if kind == "builtin" and "minimax" in lowered:
        return "minimax"
    if kind == "builtin" or lowered == "random":
        return "random"
    if kind == "mcts" or any(token in lowered for token in ("uct", "fpu", "pmbp", "lgr")):
        return "mcts"
    return "other"


def _color_for_agent(agent_id: str, kind: str = "") -> str:
    return CATEGORY_COLORS[_agent_category(agent_id, kind)]


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> list[Path]:
    paths: list[Path] = []
    for suffix in (".pdf", ".png"):
        path = output_dir / f"{stem}{suffix}"
        fig.savefig(path, dpi=dpi if suffix == ".png" else None)
        paths.append(path)
    plt.close(fig)
    return paths


def _sorted_by_win_rate(rows: list[dict[str, str]], agent_key: str = "player_id") -> list[dict[str, str]]:
    del agent_key  # kept for call-site clarity
    return sorted(rows, key=lambda row: float(row.get("win_rate", 0)), reverse=True)


def _plot_standings_win_rate(standings: list[dict[str, str]], output_dir: Path, dpi: int) -> list[Path]:
    rows = sorted(standings, key=lambda row: float(row["win_rate"]))
    labels = [_display_name(row["player_id"]) for row in rows]
    values = [float(row["win_rate"]) for row in rows]
    colors = [_color_for_agent(row["player_id"]) for row in rows]

    fig, ax = plt.subplots(figsize=(7.5, max(3.8, 0.42 * len(rows))))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Współczynnik wygranych")
    ax.set_title("Skuteczność agentów w turnieju round-robin")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    for bar, value in zip(bars, values, strict=True):
        ax.text(value + 0.01, bar.get_y() + bar.get_height() / 2, f"{value:.1%}", va="center", fontsize=8)
    _add_category_legend(ax)
    return _save_figure(fig, output_dir, "standings_win_rate", dpi)


def _plot_standings_outcomes(standings: list[dict[str, str]], output_dir: Path, dpi: int) -> list[Path]:
    rows = _sorted_by_win_rate(standings, "player_id")
    labels = [_display_name(row["player_id"]) for row in rows]
    wins = [float(row["win_rate"]) for row in rows]
    draws = [float(row["draw_rate"]) for row in rows]
    losses = [float(row["loss_rate"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8.5, max(3.8, 0.42 * len(rows))))
    y = np.arange(len(labels))
    ax.barh(y, wins, color="#4daf4a", label="Wygrane")
    ax.barh(y, draws, left=wins, color="#ffd92f", label="Remisy")
    ax.barh(y, losses, left=np.array(wins) + np.array(draws), color="#e41a1c", label="Porażki")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Udział wyników")
    ax.set_title("Rozkład wyników partii (wygrane / remisy / porażki)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(loc="lower right", framealpha=0.9)
    return _save_figure(fig, output_dir, "standings_outcomes", dpi)


def _plot_draw_rate(standings: list[dict[str, str]], output_dir: Path, dpi: int) -> list[Path]:
    rows = sorted(standings, key=lambda row: float(row["draw_rate"]), reverse=True)
    labels = [_display_name(row["player_id"]) for row in rows]
    values = [float(row["draw_rate"]) for row in rows]
    colors = [_color_for_agent(row["player_id"]) for row in rows]

    fig, ax = plt.subplots(figsize=(7.0, max(3.5, 0.38 * len(rows))))
    ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_xlim(0, max(values) * 1.15 if values else 1.0)
    ax.set_xlabel("Współczynnik remisów")
    ax.set_title("Częstość remisów — wskaźnik zbalansowania wariantu")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    return _save_figure(fig, output_dir, "draw_rate", dpi)


def _plot_decision_time(rows: list[dict[str, str]], output_dir: Path, dpi: int) -> list[Path]:
    agent_key = "agent" if rows and "agent" in rows[0] else "player_id"
    time_key = "avg_decision_seconds"
    sorted_rows = sorted(rows, key=lambda row: float(row[time_key]))
    labels = [_display_name(row[agent_key]) for row in sorted_rows]
    values = [float(row[time_key]) for row in sorted_rows]
    colors = [_color_for_agent(row[agent_key], row.get("kind", "")) for row in sorted_rows]

    fig, ax = plt.subplots(figsize=(7.5, max(3.8, 0.42 * len(sorted_rows))))
    ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_xlabel("Średni czas decyzji [s]")
    ax.set_title("Koszt obliczeniowy agentów")
    return _save_figure(fig, output_dir, "decision_time", dpi)


def _plot_side_balance(agent_summary: list[dict[str, str]], output_dir: Path, dpi: int) -> list[Path]:
    if not agent_summary:
        return []

    rows = sorted(agent_summary, key=lambda row: float(row["win_rate"]), reverse=True)
    labels = [_display_name(row["agent"]) for row in rows]
    red = [float(row["win_rate_as_red"]) for row in rows]
    yellow = [float(row["win_rate_as_yellow"]) for row in rows]

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(x - width / 2, red, width, label="Jako czerwony", color="#d73027")
    ax.bar(x + width / 2, yellow, width, label="Jako żółty", color="#f0c020", edgecolor="#666666", linewidth=0.4)
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Współczynnik wygranych")
    ax.set_title("Wpływ koloru startowego na wynik")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.0%}"))
    ax.legend()
    return _save_figure(fig, output_dir, "side_balance", dpi)


def _plot_matchup_heatmap(
    matchup_summary: list[dict[str, str]],
    standings: list[dict[str, str]],
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    players = [row["player_id"] for row in _sorted_by_win_rate(standings, "player_id")]
    index = {player: idx for idx, player in enumerate(players)}
    size = len(players)
    matrix = np.full((size, size), np.nan)
    np.fill_diagonal(matrix, 0.5)

    for row in matchup_summary:
        left = row["left"]
        right = row["right"]
        if left not in index or right not in index:
            continue
        left_rate = float(row["left_win_rate"])
        right_rate = float(row["right_win_rate"])
        matrix[index[left], index[right]] = left_rate
        matrix[index[right], index[left]] = right_rate

    labels = [_display_name(player) for player in players]
    cmap = LinearSegmentedColormap.from_list("wr", ["#b2182b", "#f7f7f7", "#2166ac"])

    fig, ax = plt.subplots(figsize=(9.5, 8.0))
    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(size), labels, rotation=45, ha="right")
    ax.set_yticks(range(size), labels)
    ax.set_title("Macierz wyników par (współczynnik wygranych wiersza vs kolumny)")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            value = matrix[i, j]
            if np.isnan(value):
                continue
            color = "white" if value < 0.35 or value > 0.65 else "black"
            ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=7, color=color)
    return _save_figure(fig, output_dir, "matchup_heatmap", dpi)


def _plot_mcts_head_to_head(matchup_summary: list[dict[str, str]], output_dir: Path, dpi: int) -> list[Path]:
    agents = list(MCTS_AGENTS)
    labels = [MCTS_LABELS[a] for a in agents]
    size = len(agents)
    matrix = np.full((size, size), np.nan)
    np.fill_diagonal(matrix, 0.5)

    for row in matchup_summary:
        left, right = row["left"], row["right"]
        if left in agents and right in agents:
            matrix[agents.index(left), agents.index(right)] = float(row["left_win_rate"])
            matrix[agents.index(right), agents.index(left)] = float(row["right_win_rate"])

    cmap = LinearSegmentedColormap.from_list("wr", ["#b2182b", "#f7f7f7", "#2166ac"])
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(size), labels)
    ax.set_yticks(range(size), labels)
    ax.set_title("Bezpośrednie pojedynki wariantów MCTS")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            value = matrix[i, j]
            if np.isnan(value):
                continue
            color = "white" if value < 0.35 or value > 0.65 else "black"
            ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=9, color=color)
    return _save_figure(fig, output_dir, "mcts_head_to_head", dpi)


def _plot_blunder_rate(blunder_summary: list[dict[str, str]], output_dir: Path, dpi: int) -> list[Path]:
    rows = sorted(blunder_summary, key=lambda row: float(row["blunder_rate"]))
    labels = [_display_name(row["agent"]) for row in rows]
    values = [float(row["blunder_rate"]) for row in rows]
    colors = [_color_for_agent(row["agent"]) for row in rows]

    fig, ax = plt.subplots(figsize=(7.5, max(3.8, 0.42 * len(rows))))
    ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_xlabel("Blunder Rate")
    ax.set_title("Odsetek rażących błędów (wyrocznia MCTS, próg regret = 0.3)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
    return _save_figure(fig, output_dir, "blunder_rate", dpi)


def _plot_blunder_regret(blunder_summary: list[dict[str, str]], output_dir: Path, dpi: int) -> list[Path]:
    rows = sorted(blunder_summary, key=lambda row: float(row["avg_regret"]), reverse=True)
    labels = [_display_name(row["agent"]) for row in rows]
    values = [float(row["avg_regret"]) for row in rows]
    colors = [_color_for_agent(row["agent"]) for row in rows]

    fig, ax = plt.subplots(figsize=(7.5, max(3.8, 0.42 * len(rows))))
    ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_xlabel("Średni regret")
    ax.set_title("Średnia strata jakości ruchu względem wyroczni")
    return _save_figure(fig, output_dir, "blunder_regret", dpi)


def _plot_skill_vs_blunders(
    standings: list[dict[str, str]],
    blunder_summary: list[dict[str, str]],
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    win_by_agent = {row["player_id"]: float(row["win_rate"]) for row in standings}
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for row in blunder_summary:
        agent = row["agent"]
        if agent not in win_by_agent:
            continue
        x = float(row["blunder_rate"])
        y = win_by_agent[agent]
        color = _color_for_agent(agent)
        ax.scatter(x, y, s=90, color=color, edgecolors="white", linewidths=0.8, zorder=3)
        ax.annotate(
            _display_name(agent),
            (x, y),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )
    ax.set_xlabel("Blunder Rate")
    ax.set_ylabel("Współczynnik wygranych")
    ax.set_title("Zależność skuteczności od częstości błędów")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.1%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.0%}"))
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.0)
    return _save_figure(fig, output_dir, "skill_vs_blunders", dpi)


def _plot_mcts_variants(
    standings: list[dict[str, str]],
    blunder_summary: list[dict[str, str]],
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    win_by_agent = {row["player_id"]: float(row["win_rate"]) for row in standings}
    blunder_by_agent = {row["agent"]: float(row["blunder_rate"]) for row in blunder_summary}
    labels = [MCTS_LABELS[a] for a in MCTS_AGENTS]
    win_rates = [win_by_agent.get(a, 0.0) for a in MCTS_AGENTS]
    blunder_rates = [blunder_by_agent.get(a, 0.0) for a in MCTS_AGENTS]

    x = np.arange(len(labels))
    width = 0.36
    fig, ax_win = plt.subplots(figsize=(7.8, 4.8))
    ax_blunder = ax_win.twinx()
    ax_blunder.spines["right"].set_visible(False)
    ax_blunder.spines["left"].set_position(("outward", -58))
    ax_blunder.spines["left"].set_visible(True)
    ax_blunder.yaxis.set_ticks_position("left")
    ax_blunder.yaxis.set_label_position("left")

    ax_win.bar(
        x + width / 2,
        win_rates,
        width,
        label="Współczynnik wygranych",
        color="#2166ac",
        zorder=1,
    )
    ax_blunder.bar(
        x - width / 2,
        blunder_rates,
        width,
        label="Blunder Rate",
        color="#d6604d",
        zorder=2,
    )
    ax_win.set_xticks(x, labels)
    ax_win.set_title("Porównanie wariantów MCTS (UCT, FPU, PMBp, LGR)")

    ax_win.set_ylim(0, max(win_rates) * 1.15)
    ax_win.set_ylabel("Współczynnik wygranych", color="#2166ac")
    ax_win.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.0%}"))
    ax_win.tick_params(axis="y", labelcolor="#2166ac")

    blunder_ymax = max(max(blunder_rates) * 1.3, 0.01)
    ax_blunder.set_ylim(0, blunder_ymax)
    ax_blunder.set_ylabel("Blunder Rate", color="#d6604d")
    ax_blunder.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.2%}"))
    ax_blunder.tick_params(axis="y", labelcolor="#d6604d")

    fig.subplots_adjust(left=0.16)
    handles = [
        mpatches.Patch(color="#2166ac", label="Współczynnik wygranych (oś lewa, wewnętrzna)"),
        mpatches.Patch(color="#d6604d", label="Blunder Rate (oś lewa, zewnętrzna)"),
    ]
    ax_win.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    return _save_figure(fig, output_dir, "mcts_variants", dpi)


def _plot_llm_reliability(llm_summary: list[dict[str, str]], output_dir: Path, dpi: int) -> list[Path]:
    rows = sorted(llm_summary, key=lambda row: row["agent"])
    labels = [_display_name(row["agent"]) for row in rows]
    invalid = [float(row["llm_invalid_response_rate"]) for row in rows]
    fallback = [float(row["llm_fallback_rate"]) for row in rows]

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(x - width / 2, invalid, width, label="Niepoprawna odpowiedź", color="#d6604d")
    ax.bar(x + width / 2, fallback, width, label="Fallback (losowy ruch)", color="#878787")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Udział ruchów")
    ax.set_title("Niezawodność modeli językowych w grze")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.1%}"))
    ax.legend()
    return _save_figure(fig, output_dir, "llm_reliability", dpi)


AGENT_ALIASES = {
    "minimaxd3": "minimax-d3",
    "minimaxd4": "minimax-d4",
}


def _normalize_agent_id(agent_id: str) -> str:
    return AGENT_ALIASES.get(agent_id.strip(), agent_id.strip())


def _is_human_id(player_id: str) -> bool:
    player_id = player_id.strip()
    return len(player_id) == 2 and player_id[0] == "H" and player_id[1].isdigit()


def _read_human_games(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.lower().startswith("playerr"):
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 3:
                continue
            winner = parts[2].strip().upper()
            if winner not in {"R", "Y"}:
                continue
            rows.append(
                {
                    "player_red": parts[0],
                    "player_yellow": parts[1],
                    "winner": winner,
                }
            )
    return rows


def _human_game_outcomes(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    outcomes: list[dict[str, str]] = []
    for row in rows:
        red = row["player_red"]
        yellow = row["player_yellow"]
        winner = row["winner"]
        if _is_human_id(red):
            outcomes.append(
                {
                    "human": red,
                    "agent": _normalize_agent_id(yellow),
                    "human_won": winner == "R",
                    "human_color": "red",
                }
            )
        elif _is_human_id(yellow):
            outcomes.append(
                {
                    "human": yellow,
                    "agent": _normalize_agent_id(red),
                    "human_won": winner == "Y",
                    "human_color": "yellow",
                }
            )
    return outcomes


def _plot_humans_vs_agents(rows: list[dict[str, str]], output_dir: Path, dpi: int) -> list[Path]:
    from collections import defaultdict

    outcomes = _human_game_outcomes(rows)
    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for outcome in outcomes:
        stats[outcome["agent"]][1] += 1
        stats[outcome["agent"]][0] += int(outcome["human_won"])

    agents = sorted(stats, key=lambda agent: stats[agent][0] / stats[agent][1])
    labels = [_display_name(agent) for agent in agents]
    human_rates = [stats[agent][0] / stats[agent][1] for agent in agents]
    agent_rates = [1.0 - rate for rate in human_rates]
    colors = [_color_for_agent(agent) for agent in agents]

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x - width / 2, human_rates, width, label="Człowiek", color="#4daf4a")
    ax.bar(x + width / 2, agent_rates, width, label="Agent", color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Współczynnik wygranych")
    ax.set_title("Wyniki testów człowiek--komputer (5 uczestników, 2 partie na parę)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.0%}"))
    ax.legend()
    return _save_figure(fig, output_dir, "humans_vs_agents", dpi)


def _plot_humans_per_participant(rows: list[dict[str, str]], output_dir: Path, dpi: int) -> list[Path]:
    from collections import defaultdict

    outcomes = _human_game_outcomes(rows)
    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for outcome in outcomes:
        stats[outcome["human"]][1] += 1
        stats[outcome["human"]][0] += int(outcome["human_won"])

    humans = sorted(stats)
    labels = humans
    rates = [stats[human][0] / stats[human][1] for human in humans]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.bar(labels, rates, color="#4daf4a", edgecolor="white", linewidth=0.6)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Uczestnik")
    ax.set_ylabel("Współczynnik wygranych")
    ax.set_title("Skuteczność poszczególnych graczy ludzkich")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.0%}"))
    for idx, rate in enumerate(rates):
        ax.text(idx, rate + 0.02, f"{rate:.0%}", ha="center", fontsize=9)
    return _save_figure(fig, output_dir, "humans_per_participant", dpi)


def _add_category_legend(ax: plt.Axes) -> None:
    handles = [
        mpatches.Patch(color=color, label=label)
        for label, color in (
            ("MCTS", CATEGORY_COLORS["mcts"]),
            ("Minimax", CATEGORY_COLORS["minimax"]),
            ("LLM", CATEGORY_COLORS["llm"]),
            ("Losowy", CATEGORY_COLORS["random"]),
        )
    ]
    ax.legend(handles=handles, loc="lower right", framealpha=0.9)


if __name__ == "__main__":
    raise SystemExit(main())
