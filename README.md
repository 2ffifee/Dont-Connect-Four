# Don't Connect Four

An academic project exploring a modified Connect Four game using Minimax, Monte Carlo Tree Search (MCTS/UCT), and several MCTS variants.

The repository contains the game engine, AI agents, CLI and GUI interfaces, automated tournaments, experiment configuration, and analysis tools.

## Quick start

```bash
git clone https://github.com/2ffifee/Dont-Connect-Four.git
cd Dont-Connect-Four

chmod +x scripts/setup.sh
./scripts/setup.sh
source .venv/bin/activate

python -m pytest

# Small experiment without external LLM dependencies
python scripts/run_full_experiment.py \
  --config configs/experiments/smoke.toml
```

On Windows PowerShell:

```powershell
.\scripts\setup.ps1
.\.venv\Scripts\Activate.ps1
```

---

## Game rules

The game is played on a `6x8` board. Red starts by default and players alternate turns.

Each non-full column allows two move types:

- `drop` — insert a piece from the top;
- `push` — insert a piece from the bottom, shifting the existing pieces in the column upward.

A full column allows neither move. `push` never removes the top piece.

Four-in-a-row segments are detected horizontally, vertically, and diagonally. Overlapping segments count separately.

### Losing condition

After every move, the engine compares the four-in-a-row segments on the new and previous boards.

- If new segments of one colour are created, the **owner of that colour immediately loses**, regardless of who made the move.
- If new segments of both colours are created in the same move, the game is a draw.
- If the board becomes full without creating a new segment, accumulated segment counts are compared; fewer segments wins and equal counts produce a draw.

Segment counters are cumulative: removing a line does not decrease the counter, and recreating the same segment later counts again.

In a normal game starting from an empty board, play usually ends when the first line is created.

---

## Implemented agents

| Agent | Description |
|---|---|
| `random` | random legal move |
| `minimax` | heuristic Minimax with alpha-beta pruning |
| `uct` | baseline MCTS using UCT |
| `fpu` | UCT with First Play Urgency |
| `lgr` | UCT with Last Good Reply |
| `pmbp` | UCT with Power-Mean Backpropagation |
| `llm` | optional language-model agent |

MCTS performs a new search budget before every move. The search tree is reused between moves of the same game and cleared before the next game.

---

## Running the game

### CLI

Human vs. Minimax:

```bash
python -m connect4_mcts.cli \
  --agent minimax \
  --depth 3
```

Agent-vs-agent demo:

```bash
python -m connect4_mcts.cli \
  --demo \
  --red minimax \
  --yellow random \
  --depth 3
```

CLI controls:

- `d 1` / `drop 1` — drop into column 1;
- `p 1` / `push 1` — push into column 1;
- `q` — quit.

### GUI

```bash
python -m connect4_mcts.gui
```

The GUI supports:

- human vs. AI;
- local two-player mode;
- Random, Minimax, MCTS variants, and optional LLM opponents;
- configurable search parameters.

Example:

```bash
python -m connect4_mcts.gui \
  --agent uct \
  --human yellow \
  --iterations 400 \
  --seed 1
```

---

## Experiments

Repeated agent-vs-agent games can be run directly:

```bash
python -m connect4_mcts.experiments \
  --red minimax \
  --yellow random \
  --games 100 \
  --depth 3 \
  --seed 1 \
  --swap-sides
```

The main experiment pipeline uses TOML configuration files.

Available presets include:

```text
configs/experiments/main_final.toml
configs/experiments/smoke.toml
configs/experiments/llm_small.toml
```

For a quick local run:

```bash
python scripts/run_full_experiment.py \
  --config configs/experiments/smoke.toml
```

The pipeline consists of:

1. round-robin tournament;
2. result aggregation;
3. optional Blunder Rate evaluation.

The full `main_final.toml` configuration also uses LLM-based agents and requires the corresponding optional dependencies and configured model endpoints.

---

## Experiment configuration

Example:

```toml
seed = 0
output_dir = "results/main_final"
games_per_pair = 10
blunder_threshold = 0.3

[[players]]
id = "random"
type = "random"

[[players]]
id = "uct"
type = "uct"
iterations = 1000
exploration = 1.414

[[players]]
id = "oracle"
type = "uct"
tags = ["ORACLE"]
iterations = 20000
exploration = 1.0
```

Tournament configuration primarily exposes:

- `iterations`;
- `exploration`;
- `fpu`;
- `power_mean_p`.

Additional MCTS parameters are available through the Python API.

---

## Blunder Rate

Game outcomes alone do not show how good individual decisions were, so the project also includes a Blunder Rate metric.

A stronger MCTS player tagged as `ORACLE` evaluates positions from completed games and estimates:

- the value of the current position;
- the best move found by the search;
- the estimated value of each explored move;
- the regret of the move that was actually played.

A move is classified as a blunder when its estimated regret exceeds a configurable threshold.

```bash
python scripts/score_blunders.py \
  --config configs/experiments/main_final.toml \
  --input-dir results/main_final
```

The MCTS values are finite-budget estimates rather than exact game-theoretic values.

Outcomes are scored as:

```text
win  = 1.0
draw = 0.5
loss = 0.0
```

so an estimated position value corresponds to:

```text
P(win) + 0.5 * P(draw)
```

from the perspective of the player to move.

---

## LLM opponent

An optional LLM agent can be used through an OpenAI-compatible API.

Install the optional dependency with:

```bash
pip install '.[llm]'
```

Example:

```bash
python -m connect4_mcts.gui \
  --llm \
  --llm-model gpt-4o-mini
```

A local OpenAI-compatible server can also be used:

```bash
python -m connect4_mcts.gui \
  --llm \
  --llm-model llama3 \
  --llm-base-url http://localhost:11434/v1
```

The model receives the game rules, current board, and legal moves and is asked to select a move.

Invalid or unparsable responses are retried and eventually fall back to a random legal move.

---

## Result analysis

Tournament results can be analysed in:

```text
notebooks/tournament_results_analysis.ipynb
```

The notebook's `RESULTS_DIR` should point to the output directory of the experiment being analysed.

Figures used by the report can be regenerated with:

```bash
python scripts/plot_experiment_results.py \
  --input-dir results/main_final \
  --output-dir report/figures
```

---

## Self-play and serialization

The current tournament pipeline creates MCTS agents online and does not require pre-trained `.pkl` files.

Separate utilities are available for self-play experiments and serialising MCTS players:

```python
from connect4_mcts import train_uct, save_player, load_player

player = train_uct(
    iterations=1000,
    selfplay_games=200,
    seed=1,
)

save_player(player, "uct_player.pkl")
restored = load_player("uct_player.pkl")
```

Starting a new game clears the MCTS search tree, so serialized trees should not be interpreted as permanent pre-training that is automatically reused across later games.

---

## Project structure

```text
src/connect4_mcts/   main package
configs/             experiment configurations
scripts/             experiment and analysis utilities
tests/               automated tests
notebooks/           result analysis
report/              project report and figures
```

---

## Tests

Run the complete test suite with:

```bash
python -m pytest
```

The game engine, agents, experiment utilities, and supporting functionality are covered by automated tests.
