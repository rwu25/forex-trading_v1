# Forex Machine Learning and Technical Analysis Prototype – Version 2

Diese erweiterte Version des Prototyps dient der empirischen Untersuchung von **Potenzialen und Grenzen von Machine‑Learning‑Modellen im Vergleich zu klassischen technischen Analyseverfahren** im Forex‑Trading.  Neben der Berechnung technischer Indikatoren und dem Training mehrerer Klassifikationsmodelle ermöglicht Version 2:

* einen **zeitlich getrennten Trainings‑/Test‑Split** (z. B. Training 2018–2022, Test 2023–2024), um Datenlecks zu vermeiden,
* optionales **Hyperparameter‑Tuning** für Random Forest und MLP mittels `GridSearchCV`,
* die Berechnung von **finanzspezifischen Kennzahlen** (Sharpe Ratio, Max Drawdown, kumulativer Return) durch einfaches Backtesting der erzeugten Signale,
* die Erstellung von **Visualisierungen** wie Kapitalverläufen und Balkendiagrammen der Klassifikationsmetriken,
* (optional) einen Walk‑Forward‑Test (als Erweiterung möglich).

## Projektstruktur

```
forex-prototype-v2/
├── data/                    # (Optional) CSV‑Daten mit Forex‑Kursen
├── notebooks/               # Explorative Analysen und Beispielnotebooks
├── results/
│   ├── plots/               # Abgespeicherte Diagramme
│   └── reports/             # Vergleichstabellen als CSV/Markdown
├── src/
│   ├── data_loader.py       # Laden von Daten (API oder CSV)
│   ├── indicators.py        # Berechnung technischer Indikatoren
│   ├── utils.py             # Hilfsfunktionen (z. B. Feature Engineering)
│   ├── ml_models.py         # Training der ML‑Modelle mit Zeit‑Split und optionalem Tuning
│   ├── evaluator.py         # Generierung und Bewertung der SMA‑Strategie
│   ├── backtester.py        # Berechnung Sharpe Ratio, Max Drawdown, Return
│   ├── visualizer.py        # Plots (Equity Curves, Confusion Matrix, Balkendiagramme)
│   └── run_experiments.py    # CLI‑Skript zur Durchführung der Studie
├── requirements.txt         # Benötigte Python‑Pakete
└── README.md                # Diese Datei
```

## Abhängigkeiten

Die wichtigsten Bibliotheken sind in der Datei `requirements.txt` aufgeführt. Zur Installation empfiehlt sich ein virtuelles Environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Nutzung

### Schneller Einstieg

Die typische Nutzung erfolgt über das Kommandozeilen‑Skript.  Beispiel:

```bash
cd forex-prototype-v2
python -m src.run_experiments EURUSD=X \
    --train-start 2018-01-01 --train-end 2022-12-31 \
    --test-start 2023-01-01 --test-end 2024-12-31 \
    --short 50 --long 200 \
    --tune --plot
```

Dieses Kommando lädt Daten für `EURUSD=X` von 2018 bis 2024, berechnet
Indikatoren, trainiert die ML‑Modelle auf dem Zeitraum 2018–2022,
bewertet sie auf 2023–2024, führt eine GridSearch für Random Forest und
MLP durch (`--tune`) und speichert zusätzlich Plots (`--plot`) im
Ordner `results/plots/`.  Die Terminalausgabe enthält sowohl die
Klassifikationsmetriken als auch die finanzspezifischen Kennzahlen (Sharpe,
Max Drawdown, kumulativer Return).

### Programmierschnittstelle

Die Module können auch einzeln importiert werden:

```python
from src.data_loader import load_forex_data
from src.indicators import compute_indicators
from src.ml_models import train_classification_models
from src.evaluator import evaluate_classic_strategy
from src.backtester import backtest_signals
```

Weitere Beispiele und experimentelle Notebooks befinden sich im Ordner
`notebooks/`.

## Hinweis zur Datenbeschaffung

Der Prototyp verwendet standardmäßig [Yahoo Finance](https://finance.yahoo.com/) via das `yfinance`‑Paket, um historische Forex‑Kurse herunterzuladen【342124238352642†L59-L77】. Für eine eigene Datenquelle können CSV‑Dateien im Ordner `data/` abgelegt und mit `load_forex_data(csv_path=…)` geladen werden.  Falls eine andere API wie Alpha Vantage gewünscht ist, muss der entsprechende API‑Schlüssel in der Datei `src/data_loader.py` eingetragen werden.

## Ziel der Arbeit

Dieser Code stellt die technische Basis dar, um klassische technische Analyseverfahren (z. B. gleitende Durchschnitte, RSI, MACD, Bollinger‑Bänder) mit Machine‑Learning‑Methoden (Entscheidungsbäume, Random Forest, Support Vector Machine, MLP) zu vergleichen. Die Ergebnisse bilden die Grundlage für die Masterarbeit *„Potenziale und Grenzen von Machine Learning und klassischer technischer Analyse im Forex Trading – eine empirische Vergleichsstudie“*. Die vorliegende Implementierung ist modular aufgebaut, um eine einfache Erweiterung (z. B. weitere Indikatoren oder Modelle) zu ermöglichen.