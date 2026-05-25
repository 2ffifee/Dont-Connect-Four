# Don't Connect 4

Projekt badawczy dla zmodyfikowanej gry Connect4 z agentami opartymi o MCTS/UCT.

## Zasady wariantu

Gramy na planszy `6x8`. Celem nie jest ulozenie czterech swoich zetonow w linii, tylko unikniecie tego. Po zakonczeniu gry liczone sa wszystkie linie dlugosci 4 w obu kolorach, a przegrywa gracz, ktory ma ich wiecej.

Dostepne sa dwa typy ruchow:

- `drop` - klasyczne wrzucenie zetonu do kolumny,
- `push` - wlozenie zetonu od spodu niezapelnionej kolumny; pozostale zetony w tej kolumnie przesuwaja sie o jedno pole w gore.

Obowiazuje sprawiedliwosc turowa: jesli linia pojawi sie po ruchu gracza rozpoczynajacego, drugi gracz dostaje jeszcze jeden ruch. Po tym ruchu gra konczy sie i porownywana jest liczba linii obu kolorow.

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
- `minimax` - uzywa heurystyki pozycyjnej oraz minimax z alpha-beta pruning.

Parametr `--depth` steruje glebokoscia przeszukiwania minimaxa. Wieksza wartosc zwykle oznacza silniejsza gre, ale istotnie zwieksza czas decyzji. Na start praktyczne sa wartosci `2` lub `3`.

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

- kolor czlowieka: `Red` albo `Yellow`,
- przeciwnika: `Random` albo `Minimax`,
- glebokosc minimaxa.

Start z domyslnymi ustawieniami:

```bash
python -m connect4_mcts.gui
```

Start z wybranym przeciwnikiem i kolorem:

```bash
python -m connect4_mcts.gui --agent minimax --human yellow --depth 3 --seed 1
```

Sterowanie w GUI:

- klikniecie kolumny wykonuje ruch,
- przyciski `Drop` i `Push` wybieraja typ ruchu,
- `Menu` wraca do wyboru przeciwnika i koloru,
- `Space` przelacza `drop/push`,
- `R` resetuje partie,
- `Esc` zamyka okno.

## Symulacje wielu gier

Modul eksperymentow pozwala uruchamiac serie gier agent-agent i szybko porownac wyniki:

```bash
python -m connect4_mcts.experiments --red minimax --yellow random --games 100 --depth 3 --seed 1 --swap-sides
```

Najwazniejsze opcje:

- `--red random|minimax` - agent grajacy jako czerwony w pierwszej grze,
- `--yellow random|minimax` - agent grajacy jako zolty w pierwszej grze,
- `--games N` - liczba partii,
- `--seed N` - bazowe ziarno losowosci,
- `--depth N` - glebokosc minimaxa,
- `--swap-sides` - zamienia strony co druga partie; przy parzystej liczbie gier kazdy agent gra tyle samo razy jako czerwony i zolty.

Wynik zawiera:

- liczbe zwyciestw i win rate kazdego agenta,
- liczbe remisow i draw rate,
- srednia liczbe ruchow na partie,
- sredni czas decyzji kazdego agenta.
