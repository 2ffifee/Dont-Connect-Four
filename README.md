# Don't Connect 4

Projekt badawczy dla zmodyfikowanej gry Connect4 z agentami opartymi o MCTS/UCT.

## Zasady wariantu

Gramy na planszy `6x8`. Celem nie jest ulozenie czterech swoich zetonow w linii, tylko unikniecie tego. Po zakonczeniu gry liczone sa wszystkie segmenty czterech zetonow w rzedzie (poziomo, pionowo, po przekatnej) dla kazdego koloru; nakladajace sie segmenty licza sie osobno (np. szesc zetonow w rzedzie to trzy segmenty). Przegrywa gracz z wieksza liczba segmentow.

Dostepne sa dwa typy ruchow:

- `drop` - klasyczne wrzucenie zetonu do kolumny,
- `push` - wlozenie zetonu od spodu niezapelnionej kolumny; pozostale zetony w tej kolumnie przesuwaja sie o jedno pole w gore.

Obowiazuje **natychmiastowa porażka właściciela linii**: gdy w ruchu powstaje segment
czterech zetonow w rzędzie, **przegrywa gracz, do którego należy ten segment** — niezależnie
od tego, kto wykonał ruch. Jesli w jednym ruchu powstaja linie obu graczy, wynik to **remis**.
Jesli nie powstaje nowa linia, gra trwa dalej az do zapelnienia planszy; wtedy wygrywa gracz
z mniejsza **skumulowana** liczba segmentow (zniszczenie linii na planszy nie zmniejsza
wczesniej naliczonego wyniku).

Remis: rowna liczba segmentow po zapelnieniu planszy albo obie linie w jednym ruchu.

## Zakres implementacji

- rdzen gry: plansza 6x8, wariant suicide, ruchy drop i push, natychmiastowa porażka właściciela linii,
- gracze bazowi: random oraz heurystyczny,
- gracze MCTS/UCT i wybrane modyfikacje,
- proste GUI do rozgrywek czlowiek-komputer,
- narzedzia do automatycznych eksperymentow.

## Struktura

```text
src/connect4_mcts/   kod projektu
configs/             pliki TOML eksperymentow (turniej, wyrocznia ORACLE)
scripts/             skrypty turnieju i pipeline
tests/               testy automatyczne
notebooks/           analiza wynikow turnieju
```


## Przygotowanie srodowiska

Windows PowerShell:

```powershell
.\scripts\setup.ps1
```

Linux/macOS:

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

Skrypty tworza lokalne srodowisko `.venv`, aktualizuja `pip` i instaluja projekt z zaleznosciami developerskimi oraz analitycznymi do notebookow.

Aktywacja srodowiska na Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Aktywacja srodowiska na Linux/macOS:

```bash
source .venv/bin/activate
```

## Testy

Po przygotowaniu srodowiska:

```bash
python -m pytest
```

## Uruchamianie CLI

Dostepni gracze:

- `random` - wybiera losowy legalny ruch,
- `minimax` - uzywa heurystyki pozycyjnej oraz minimax z alpha-beta pruning,
- `uct` - bazowy MCTS/UCT,
- `fpu` - UCT z First Play Urgency,
- `lgr` - UCT z polityka Last Good Reply w rolloutach,
- `pmbp` - UCT z Power-Mean Backpropagation.

Parametr `--depth` steruje glebokoscia przeszukiwania minimaxa. Wieksza wartosc zwykle oznacza silniejsza gre, ale istotnie zwieksza czas decyzji. Na start praktyczne sa wartosci `2` lub `3`. Gracze MCTS uruchamiani z CLI/GUI/eksperymentow wykonuja **swieze symulacje przed kazdym ruchem** (domyslny budzet iteracji); pelna kontrole nad hiperparametrami daja pliki konfiguracyjne eksperymentow (ponizej) oraz modul treningowy (legacy).

Gra czlowiek kontra losowy agent:

```bash
python -m connect4_mcts.cli --seed 1
```

Gra czlowiek kontra minimax:

```bash
python -m connect4_mcts.cli --agent minimax --depth 3
```

Gra jako zolty, czyli agent zaczyna jako czerwony:

```bash
python -m connect4_mcts.cli --human yellow --agent minimax --depth 3 --seed 1
```

Demo random kontra random:

```bash
python -m connect4_mcts.cli --demo --seed 1
```

Demo minimax kontra random:

```bash
python -m connect4_mcts.cli --demo --red minimax --yellow random --depth 3 --seed 1
```

Sterowanie w CLI:

- `d 1` albo `drop 1` - drop do kolumny 1,
- `p 1` albo `push 1` - push do kolumny 1,
- kolumny sa numerowane od `1` do `8`,
- `q` konczy gre.

## Uruchamianie GUI

GUI uruchamia ekran wyboru ustawien. W aplikacji mozna wybrac:

- tryb gry: `Single` (czlowiek kontra agent) albo `2 Players` (dwoch ludzi lokalnie),
- kolor czlowieka: `Red` albo `Yellow`,
- przeciwnika: `Random`, `Minimax`, warianty MCTS (`UCT`, `FPU`, `LGR`, `PMBp`) albo `Play vs LLM...`,
- dla Minimax: glebokosc; dla MCTS: iteracje na ruch, exploration C oraz parametry wariantu (FPU / power-mean p); dla LLM: temperatura i opcjonalnie max tokens; dla Random: opcjonalny seed,

Okno jest skalowalne - plansza oraz menu sa wysrodkowane i dopasowuja sie do rozmiaru okna.

Start z domyslnymi ustawieniami:

```bash
python -m connect4_mcts.gui
```

Start z wybranym przeciwnikiem MCTS i kolorem:

```bash
python -m connect4_mcts.gui --agent uct --human yellow --iterations 400 --seed 1
```

Start od razu w trybie dwoch graczy:

```bash
python -m connect4_mcts.gui --two-player
```

Gracze MCTS w GUI uruchamiaja swieze symulacje przed kazdym ruchem (`--agent`
`uct`/`fpu`/`lgr`/`pmbp`, `--iterations`). Nie trzeba wczytywac pliku pickle.

### Gra przeciwko LLM w GUI

Na ekranie ustawien przycisk `Play vs LLM...` konfiguruje przeciwnika opartego na
modelu jezykowym. Kreator dziala tak:

1. **Base URL** - puste = OpenAI; wlasny adres kieruje na **lokalny serwer**
   zgodny z API OpenAI (np. `http://localhost:11434/v1` dla Ollamy albo
   `http://localhost:8000/v1` dla vLLM/LM Studio). Uzywane adresy sa zapisywane
   lokalnie w `~/.config/connect4-mcts/llm_endpoints.json` i podpowiadane przy
   kolejnym uruchomieniu (wpisz fragment, np. `11434`, aby przefiltrowac liste).
2. **API key** - puste = uzyj zmiennej `OPENAI_API_KEY` (lokalne serwery zwykle
   nie wymagaja klucza). Klucze **nie** sa zapisywane na dysku.
3. Aplikacja **laczy sie i pobiera liste modeli** z endpointu (to zarazem test
   polaczenia) i pokazuje **powiadomienie o sukcesie albo bledzie** (zly klucz,
   brak serwera itp.).
4. Po udanym polaczeniu wybierasz model z **listy rozwijanej** (mozna tez wpisac
   wlasny identyfikator). Nie trzeba znac nazwy modelu z gory.

Wymaga pakietu `openai` (`pip install '.[llm]'`). Mozna tez ustawic model wprost
z linii komend (z pominieciem listy):

```bash
# OpenAI (klucz z OPENAI_API_KEY)
python -m connect4_mcts.gui --llm --llm-model gpt-4o-mini

# Lokalny serwer zgodny z OpenAI (np. Ollama)
python -m connect4_mcts.gui --llm --llm-model llama3 --llm-base-url http://localhost:11434/v1
```

Bledy LLM (np. sieci/klucza) nie wieszaja gry: przeciwnik wykonuje wtedy losowy
legalny ruch i pokazuje komunikat.

Sterowanie w GUI:

- klikniecie kolumny wykonuje ruch,
- w trybie `2 Players` gracze `Red` i `Yellow` klikaja na zmiane,
- przyciski `Drop` i `Push` wybieraja typ ruchu,
- `Menu` wraca do ekranu ustawien,
- `Space` przelacza `drop/push`,
- `R` resetuje partie,
- `Esc` zamyka okno.

## Symulacje wielu gier

Modul eksperymentow pozwala uruchamiac serie gier agent-agent i szybko porownac wyniki:

```bash
python -m connect4_mcts.experiments --red minimax --yellow random --games 100 --depth 3 --seed 1 --swap-sides
```

Najwazniejsze opcje:

- `--red random|minimax|uct|fpu|lgr|pmbp` - agent grajacy jako czerwony w pierwszej grze,
- `--yellow random|minimax|uct|fpu|lgr|pmbp` - agent grajacy jako zolty w pierwszej grze,
- `--games N` - liczba partii,
- `--seed N` - bazowe ziarno losowosci,
- `--depth N` - glebokosc minimaxa,
- `--swap-sides` - zamienia strony co druga partie; przy parzystej liczbie gier kazdy agent gra tyle samo razy jako czerwony i zolty.

Wynik zawiera:

- liczbe zwyciestw i win rate kazdego agenta,
- liczbe remisow i draw rate,
- srednia liczbe ruchow na partie,
- sredni czas decyzji kazdego agenta.

## Trenowanie i zapisywanie graczy MCTS (legacy)

Modul `connect4_mcts.training` nadal udostepnia API do **self-play i zapisu drzewa**
w plikach `.pkl`. Turniej i pelny pipeline eksperymentow **nie wymagaja treningu** —
gracze MCTS sa budowani online z parametrow w pliku TOML (patrz ponizej).

Funkcje treningowe zwracaja gotowe obiekty graczy (implementujace `choose_move`):

```python
from connect4_mcts import (
    train_uct, train_fpu, train_lgr, train_pmbp, selfplay_train,
    save_player, load_player, play_game,
)

# Bazowy UCT wytrenowany 200 grami self-play (rozbudowuje drzewo).
uct = train_uct(iterations=1000, exploration=1.41, selfplay_games=200, seed=1)

# First Play Urgency: wartosc przypisywana nieodwiedzonym ruchom.
fpu = train_fpu(iterations=1000, fpu=1.0, selfplay_games=200, seed=1)

# Power-Mean Backpropagation: wykladnik sredniej potegowej p.
pmbp = train_pmbp(iterations=1000, power_mean_p=2.0, selfplay_games=200, seed=1)

# Last Good Reply: self-play buduje drzewo oraz pamiec odpowiedzi.
lgr = train_lgr(iterations=1000, selfplay_games=200, seed=1)
print("rozmiar drzewa:", uct.tree_size)

# Mozna dotrenowac istniejacego gracza, jeszcze bardziej rozbudowujac drzewo.
selfplay_train(uct, games=100, temperature=1.0)

# Gracze dzialaja od razu w mechanizmie gry (i nadal ucza sie podczas partii).
game = play_game(red=uct, yellow=pmbp)

# Zapis wytrenowanego gracza wraz z drzewem i pamiecia LGR oraz pozniejszy odczyt.
save_player(lgr, "lgr_player.pkl")
restored = load_player("lgr_player.pkl")
```

Gotowy skrypt trenuje i zapisuje przykladowego gracza (domyslnie UCT do
`models/uct_example.pkl`):

```bash
python scripts/train_example_player.py --algorithm uct --selfplay-games 40
```

Najwazniejsze opcje skryptu: `--algorithm {uct,fpu,lgr,pmbp}`, `--output`,
`--iterations`, `--selfplay-games`, `--selfplay-iterations`, `--max-rollout-moves`
oraz `--seed`.

Osobne skrypty legacy (nie sa czescia domyslnego pipeline eksperymentow):

```bash
python scripts/train_oracle.py --config configs/oracle.toml
python scripts/train_tournament_grid.py --config configs/tournament_grid.toml
```

## Konfiguracja eksperymentow

Eksperymenty turniejowe opisuje **jeden plik TOML**. Globalnie ustawiasz tylko:

- `seed` — bazowe ziarno losowosci,
- `output_dir` — katalog wynikow turnieju,
- `games_per_pair` — liczba partii na kazda pare graczy w round-robin,
- opcjonalnie `blunder_threshold`, `blunder_sample_every` — parametry metryki Blunder Rate.

Kazdy uczestnik to wpis `[[players]]` z polem `type`:

| `type` | Opis | Typowe hiperparametry |
|--------|------|------------------------|
| `random` | losowy legalny ruch | opcjonalny `seed` |
| `minimax` | minimax z heurystyka | `depth` |
| `uct`, `fpu`, `lgr`, `pmbp` | MCTS online | `iterations`, `exploration`, `fpu`, `power_mean_p` |
| `llm` | model jezykowy | `model`, `base_url`, opcjonalny `api_key`, `temperature`, `max_tokens`, `timeout` |

Gracz z tagiem **`ORACLE`** sluzy wylacznie do oceny blunderow (nie gra w turnieju).
Moze byc co najwyzej jeden taki gracz. Jesli go nie ma, etap Blunder Rate jest pomijany.

Przyklad (`configs/experiments/main_final.toml`):

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

[[players]]
id = "llm-local"
type = "llm"
model = "llama3"
base_url = "http://localhost:11434/v1"
temperature = 0.7
```

Gotowe presety:

- `configs/experiments/main_final.toml` — pelny turniej algorytmow,
- `configs/experiments/smoke.toml` — szybki test lokalny,
- `configs/experiments/llm_small.toml` — porownanie z LLM.

Ladowanie i budowa graczy z configu (modul `connect4_mcts.experiment_config`):

```python
from connect4_mcts.experiment_config import load_experiment_config, instantiate_player

config = load_experiment_config("configs/experiments/smoke.toml")
for spec in config.players:
    agent = instantiate_player(spec, game_seed=config.seed)
```

Stary format (`kind`, `builtin`, `algorithm`, `play_iterations`) jest nadal akceptowany
dla kompatybilnosci wstecznej.

### Pelny pipeline eksperymentu

Skrypt `scripts/run_full_experiment.py` uruchamia trzy etapy:

1. **turniej round-robin** (`run_tournament.py`) — kazda para gra `games_per_pair` partii,
2. **agregacja CSV** (`analyze_tournament.py`),
3. **Blunder Rate** (`score_blunders.py`) — tylko gdy w configu jest gracz z tagiem `ORACLE`.

Domyslny pelny run:

```bash
python scripts/run_full_experiment.py
python scripts/run_full_experiment.py --config configs/experiments/smoke.toml
```

Przed dlugim uruchomieniem mozna wypisac komendy bez wykonywania:

```bash
python scripts/run_full_experiment.py --dry-run
```

Pominiecie etapow (np. powtorzenie tylko analizy):

```bash
python scripts/run_full_experiment.py --skip-stage tournament,blunders
python scripts/run_full_experiment.py --resume-pipeline   # pomija juz ukonczone etapy
```

Najwazniejsze opcje:

- `--config PATH` — plik eksperymentu (domyslnie `configs/experiments/main_final.toml`),
- `--output-dir`, `--games-per-pair`, `--base-seed` — nadpisuja wartosci z configu,
- `--skip-blunders` — wymusza pominięcie Blunder Rate,
- `--blunder-max-positions N` — limit ocenianych ruchow przy probnym runie,
- `--verbose-games` — loguje kazda partie przed startem (diagnoza zawieszen).

Po zakonczeniu pelnego runu notebook `notebooks/tournament_results_analysis.ipynb`
powinien wskazywac na ten sam katalog:

```python
RESULTS_DIR = PROJECT_ROOT / "results" / "main_final"
```

### Turniej round-robin

Skrypt `scripts/run_tournament.py` czyta config eksperymentu, buduje graczy online
(bez plikow `.pkl`) i zapisuje m.in. `games.csv`, `moves.jsonl`, `standings.csv`:

```bash
python scripts/run_tournament.py --config configs/experiments/main_final.toml
python scripts/run_tournament.py --config configs/experiments/smoke.toml --games-per-pair 2
python scripts/run_tournament.py --config configs/experiments/main_final.toml --resume
```

Nadpisanie parametrow globalnych z CLI: `--output-dir`, `--games-per-pair`, `--base-seed`.
Plik `moves.jsonl` zawiera stan przed kazdym ruchem — sluzy do pozniejszej oceny blunderow.

### Wyrocznia i Blunder Rate

Wyrocznia to gracz MCTS (zwykle czysty `uct`) oznaczony tagiem `ORACLE` w tym samym
pliku configu co turniej. Skrypt `scripts/score_blunders.py` odczytuje `moves.jsonl`,
dla kazdej wybranej pozycji uruchamia `evaluate()` wyroczni w trybie online i liczy regret:

```bash
python scripts/score_blunders.py \
  --config configs/experiments/main_final.toml \
  --input-dir results/main_final
```

Prog blundera (`--threshold`) i probkowanie ruchow (`--sample-every`) domyslnie biora
wartosci z configu. Wyniki: `blunders.csv`, `blunder_summary.csv`.

Neutralna wyrocznia to UCT bez FPU/LGR (`power_mean_p = 1`), zeby nie faworyzowac
zadnej z badanych modyfikacji. Typowy budzet oceny to `iterations = 20000` w wpisie
`ORACLE` (silniejsza maszyna); uczestnicy turnieju moga miec nizszy budzet, np. `1000`.

### Ocena pozycji (wartosc korzenia i ruchow)

Metryka Blunder Rate potrzebuje wartosci stanu (szansy na wygrana przy
optymalnej grze) oraz wartosci pojedynczych ruchow. Sluzy do tego metoda
`MCTSPlayer.evaluate`, ktora przeszukuje pozycje i zwraca `SearchEvaluation`:

```python
ev = oracle.evaluate(state)              # przeszukuje budzetem `iterations`
ev.player_to_move                        # gracz na ruchu
ev.root_value                            # wartosc pozycji przy optymalnej grze (= wartosc najlepszego ruchu)
ev.best_move                             # najlepszy ruch wg wartosci
ev.move_values[move]                     # szansa na wygrana po danym ruchu (z perspektywy gracza na ruchu)
ev.move_visits[move]                     # liczba symulacji wspierajacych dany ruch
ev.regret_of(move)                       # o ile dany ruch jest gorszy od najlepszego
ev.is_blunder(move, threshold=0.3)       # czy ruch jest blunderem

# Reuzycie juz zbudowanego drzewa bez ponownego przeszukiwania (np. po self-play):
ev = oracle.evaluate(state, run_search=False)
```

Wszystkie wartosci sa prawdopodobienstwami wygranej w `[0, 1]` z perspektywy
gracza na ruchu.

### Wydajnosc rdzenia

`GameState.apply_move` jest na goracej sciezce MCTS i turnieju, dlatego zostalo
zoptymalizowane bez zmiany zachowania (zweryfikowane na 30000 losowych partiach):

- sprawdzanie legalnosci w O(1) zamiast budowania `legal_moves()`;
- przebudowa tylko zmienionych komorek planszy (wspoldzielenie niezmienionych,
  niemutowalnych wierszy);
- **inkrementalne** wykrywanie konczacej linii na podstawie komorek ruszonych
  przez ruch (stan `ONGOING` nie ma czworki, wiec nowa linia musi przechodzic
  przez te komorki) zamiast skanowania calej planszy po kazdym ruchu;
- pelne liczenie linii (kosztowne) tylko gdy faktycznie powstaje stan terminalny
  (do zbudowania `GameResult`);
- budowa stanu nastepnego bez ponownej walidacji `__post_init__`.

Efekt: przeszukiwanie UCT przyspieszylo z ~100 do ~4000 iteracji/s (rzedu 40x),
co skraca ocene pozycji w turnieju i przy ocenie Blunder Rate.

### Gracze oparci na LLM

`LLMPlayer` zaprzega model jezykowy jako zwykla *polityke* (zero-shot): w kazdej
turze pokazuje modelowi reguly, plansze i **liste legalnych ruchow**, a odpowiedz
parsuje do `Move` i waliduje. Implementuje protokol `Agent` (`choose_move`), wiec
dziala wszedzie tam, gdzie inni gracze (`play_game`, eksperymenty). Model jest
osiagany przez maly protokol `LLMClient`, dzieki czemu gracz jest niezalezny od
dostawcy:

- `OpenAIClient` - dowolny endpoint zgodny z API OpenAI (OpenAI, Google Gemini
  przez URL `generativelanguage.googleapis.com`, lub lokalny serwer; wybor przez
  `base_url`; typ providera wnioskuje sie z adresu); wymaga pakietu `openai`
  (`pip install '.[llm]'`).
- `MockLLMClient` - deterministyczny klient offline do testow/demonstracji.

```python
import os
from connect4_mcts import LLMPlayer, OpenAIClient, MCTSPlayer, play_game

llm = LLMPlayer(
    OpenAIClient(model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini")),
    model_label="gpt-4o-mini",
    max_attempts=3,       # ile razy odpytac model zanim nastapi fallback
    on_failure="random",  # po wyczerpaniu prob: losowy legalny ruch
)
oracle = MCTSPlayer(iterations=20000, seed=0)  # wyrocznia do Blunder Rate
game = play_game(red=llm, yellow=oracle)
print("nielegalne odpowiedzi:", f"{llm.illegal_move_rate:.0%}")
```

W eksperymentach turniejowych gracz LLM definiuje sie w pliku TOML (`type = "llm"`);
w kodzie programowym mozna uzyc `create_llm_player` / `create_agent("llm", ...)`.
Celowo **nie** ma typu `llm` w menu CLI (tryb offline), ale jest w GUI i configu turnieju.

Przy nielegalnej/niewyparsowalnej odpowiedzi gracz ponawia zapytanie z informacja
o bledzie, a po `max_attempts` stosuje fallback. Liczniki `requests`,
`unparseable`, `illegal`, `fallbacks`, `moves` oraz `illegal_move_rate` sluza do
metryki odsetka nielegalnych ruchow.

Uwaga badawcza: to **wariant samobojczy** - ulozenie czworki jest zle. LLM-y maja
silny prior "connect four = wygrana", dlatego prompt wielokrotnie podkresla
odwrocony cel; sprawdzenie, czy model potrafi przelamac ten prior, jest ciekawym
elementem porownania z wytrenowanymi agentami MCTS.

Wspolne hiperparametry: `iterations` (budzet przeszukiwania na ruch),
`exploration` (stala `C` w UCT), `final_move` (`"robust"` - najczesciej
odwiedzane dziecko, albo `"max_value"` - najwyzsza srednia wartosc),
`max_rollout_moves` (limit dlugosci rolloutu) oraz `seed`. Parametry treningu:
`selfplay_games` (liczba gier self-play), `selfplay_iterations` (budzet
przeszukiwania na ruch w trakcie treningu, jesli ma byc inny niz docelowy) oraz
`selfplay_temperature` (stopien eksploracji przy wyborze ruchow treningowych).
Modyfikacje dodaja wlasne parametry: `fpu` (FPU) oraz `power_mean_p` (PMBp).
