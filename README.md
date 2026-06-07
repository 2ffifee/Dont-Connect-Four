# Don't Connect 4

Projekt badawczy dla zmodyfikowanej gry Connect4 z agentami opartymi o MCTS/UCT.

## Zasady wariantu

Gramy na planszy `6x8`. Celem nie jest ulozenie czterech swoich zetonow w linii, tylko unikniecie tego. Po zakonczeniu gry liczone sa wszystkie segmenty czterech zetonow w rzedzie (poziomo, pionowo, po przekatnej) dla kazdego koloru; nakladajace sie segmenty licza sie osobno (np. szesc zetonow w rzedzie to trzy segmenty). Przegrywa gracz z wieksza liczba segmentow.

Dostepne sa dwa typy ruchow:

- `drop` - klasyczne wrzucenie zetonu do kolumny,
- `push` - wlozenie zetonu od spodu niezapelnionej kolumny; pozostale zetony w tej kolumnie przesuwaja sie o jedno pole w gore.

Obowiazuje sprawiedliwosc turowa: jesli segment czterech zetonow pojawi sie po ruchu gracza rozpoczynajacego, drugi gracz dostaje jeszcze jeden ruch. Jesli po tym ruchu obaj gracze maja **ta sama liczbe** segmentow, gra toczy sie dalej. W przeciwnym razie wygrywa gracz z mniejsza liczba segmentow.

Remis: gdy liczba segmentow jest rowna, gra toczy sie dalej (albo konczy sie remisem, gdy plansza jest pelna).

## Zakres implementacji

- rdzen gry: plansza 6x8, wariant suicide, ruchy drop i push, sprawiedliwosc turowa,
- gracze bazowi: random oraz heurystyczny,
- gracze MCTS/UCT i wybrane modyfikacje,
- proste GUI do rozgrywek czlowiek-komputer,
- narzedzia do automatycznych eksperymentow.

## Struktura

```text
src/connect4_mcts/   kod projektu
tests/               testy automatyczne
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

Skrypty tworza lokalne srodowisko `.venv`, aktualizuja `pip` i instaluja projekt z zaleznosciami developerskimi.

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

Parametr `--depth` steruje glebokoscia przeszukiwania minimaxa. Wieksza wartosc zwykle oznacza silniejsza gre, ale istotnie zwieksza czas decyzji. Na start praktyczne sa wartosci `2` lub `3`. Gracze MCTS uruchamiani z CLI/eksperymentow korzystaja z domyslnego budzetu iteracji; pelna kontrole nad hiperparametrami daja funkcje treningowe (ponizej).

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
- przeciwnika: `Random` albo `Minimax`,
- `Load player...` - wczytanie wytrenowanego gracza MCTS z pliku pickle (otwiera okno wyboru pliku),
- `Play vs LLM...` - gra przeciwko modelowi jezykowemu (zob. nizej),
- glebokosc minimaxa.

Okno jest skalowalne - plansza oraz menu sa wysrodkowane i dopasowuja sie do rozmiaru okna.

Start z domyslnymi ustawieniami:

```bash
python -m connect4_mcts.gui
```

Start z wybranym przeciwnikiem i kolorem:

```bash
python -m connect4_mcts.gui --agent minimax --human yellow --depth 3 --seed 1
```

Start od razu w trybie dwoch graczy:

```bash
python -m connect4_mcts.gui --two-player
```

Gra od razu przeciwko wczytanemu, wytrenowanemu graczowi:

```bash
python -m connect4_mcts.gui --load models/uct_example.pkl
```

W GUI mozna tez wczytac gracza w trakcie - na ekranie ustawien przyciskiem `Load player...` (otwiera systemowe okno wyboru pliku `.pkl`). Jezeli okno dialogowe nie jest dostepne, uzyj flagi `--load`.

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

## Trenowanie i zapisywanie graczy MCTS

Gracz MCTS (`MCTSPlayer`) utrzymuje **trwale drzewo przeszukiwania** (tablice
transpozycji) wspoldzielone miedzy ruchami i grami. Wytrenowany gracz to taki,
ktory ma juz w jakims stopniu zbudowane drzewo - jego ksztalt zalezy od
hiperparametrow danego algorytmu. Trening polega na **rozgrywaniu przez gracza
gier z samym soba**, co rozbudowuje wspolne drzewo (a dla LGR takze pamiec
odpowiedzi).

Funkcje treningowe z modulu `connect4_mcts.training` zwracaja gotowe obiekty
graczy (implementujace `choose_move`), ktore dzialaja bezposrednio z mechanizmem
gry (`play_game`, CLI, GUI, eksperymenty):

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
(limit dlugosci rolloutu, przyspiesza trening) oraz `--seed`. Tak zapisany plik
mozna wczytac w GUI (`Load player...` lub `--load`).

### Wyrocznia do metryki Blunder Rate

Osobny skrypt `scripts/train_oracle.py` buduje neutralny silnik referencyjny
(wyrocznie) uzywany do metryki Blunder Rate. Wyrocznia to **czysty UCT**
(`power_mean_p = 1`, bez FPU/LGR), aby nie faworyzowac zadnej z badanych
modyfikacji. Jej trwale drzewo transpozycji jest rozbudowywane przez self-play i
pelni potem role cache wartosci przy ocenianiu pozycji turniejowych.

Konfiguracja znajduje sie w `configs/oracle.toml` (wypelniona zaproponowanymi
wartosciami: limit pamieci `5 GB` ~= 1,8 mln wezlow, budzet budowania 1000
iteracji/ruch, budzet oceny 20000 iteracji/pozycje, prog blundera 0.3).

```bash
python scripts/train_oracle.py                       # uzywa configs/oracle.toml
python scripts/train_oracle.py --config configs/oracle.toml
python scripts/train_oracle.py --resume              # dorozbuduj istniejace drzewo
```

Trening zatrzymuje sie po osiagnieciu limitu pamieci (przeliczonego na liczbe
wezlow), liczby gier albo limitu czasu, zapisujac po drodze checkpointy. Po
optymalizacji rdzenia gry przeszukiwanie osiaga ~4000 iteracji/s (zob.
"Wydajnosc rdzenia"), wiec zapelnienie drzewa do 5 GB to rzad kilkunastu minut.

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

# Reuzycie gotowego drzewa wyroczni bez ponownego przeszukiwania:
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
co skraca zarowno ocene w turnieju, jak i budowanie drzewa wyroczni.

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

W eksperymentach mozna uzyc nazwy `"llm"` w `create_agent`/`run_match`
(model brany z `OPENAI_MODEL`, klucz z `OPENAI_API_KEY`, opcjonalnie
`OPENAI_BASE_URL`). Celowo **nie** ma jej w menu GUI/CLI, zeby te tryby
dzialaly offline.

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
