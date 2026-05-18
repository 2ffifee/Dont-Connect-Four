# MCTS Connect4

Projekt badawczy dla zmodyfikowanej gry Connect4 z agentami opartymi o MCTS/UCT.

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
