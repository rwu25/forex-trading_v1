"""
Beispielszenario zum Vergleich von Machine‑Learning‑Modellen und
klassischer technischer Analyse auf historischen Forex‑Daten.

Dieses Skript kann direkt ausgeführt werden. Es lädt Devisendaten
über `yfinance`, berechnet typische technische Indikatoren,
erstellt eine binäre Zielvariable, trainiert mehrere ML‑Modelle und
bewertet zudem eine einfache Moving‑Average‑Crossover‑Strategie.
Die Ergebnisse werden als Tabelle ausgegeben.

Hinweis: Für eine vollständige Analyse sollten die Daten zunächst
lokal gecached oder in `data/` abgelegt werden, um API‑Limits zu
vermeiden.
"""

from __future__ import annotations

import argparse
import pandas as pd

from .data_loader import load_forex_data
from .indicators import compute_indicators
from .ml_models import train_classification_models
from .evaluator import evaluate_classic_strategy, compare_models
from .backtester import backtest_signals
from .visualizer import plot_equity_curves, plot_metric_bars


def main(
    symbol: str,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    short_window: int = 50,
    long_window: int = 200,
    tune: bool = False,
    plot: bool = False,
) -> None:
    """Führe den erweiterten Analyseprozess für ein Devisenpaar aus.

    Lädt die Daten für den gesamten Zeitraum (Trainings- und Testbereich),
    berechnet Indikatoren, trainiert ML‑Modelle mit einem zeitlichen
    Split, bewertet eine SMA‑Strategie, vergleicht die Klassifikationsmetriken
    und berechnet finanzielle Kennzahlen.  Optional werden Plots erzeugt.
    """
    overall_start = train_start
    overall_end = test_end
    print(f"Lade Daten für {symbol} von {overall_start} bis {overall_end}...")
    df = load_forex_data(symbol=symbol, start=overall_start, end=overall_end)

    print("Berechne technische Indikatoren...")
    df_ind = compute_indicators(df)

    print("Trainiere Machine‑Learning‑Modelle (zeitlicher Split)...")
    models, ml_metrics, predictions, probas = train_classification_models(
        df_ind,
        start_train=train_start,
        end_train=train_end,
        start_test=test_start,
        end_test=test_end,
        tune=tune,
    )

    print("Bewerte klassische SMA‑Strategie...")
    classic_res = evaluate_classic_strategy(
        df,
        short_window=short_window,
        long_window=long_window,
        return_signals=True,
        start_test=test_start,
        end_test=test_end,
    )
    classic_metrics = {k: v for k, v in classic_res.items() if k != "Signals"}
    classic_signals = classic_res.get("Signals", None)

    print("Vergleiche Klassifikationsmetriken...")
    results_classif = compare_models(ml_metrics, classic_metrics, save_path=None, plot=False)
    print(results_classif)

    print("Berechne finanzielle Kennzahlen...")
    # Close-Spalte ggf. von DataFrame auf Series reduzieren
    price_series = df["Close"]
    if hasattr(price_series, "ndim") and price_series.ndim > 1:
        price_series = price_series.iloc[:, 0]
    price_series = price_series.loc[pd.to_datetime(test_start): pd.to_datetime(test_end)]

    signals_dict = {name: pred for name, pred in predictions.items()}
    if classic_signals is not None:
        signals_dict[f"SMA{short_window}/{long_window}"] = classic_signals
    fin_metrics = backtest_signals(price_series, signals_dict)
    fin_df = pd.DataFrame(fin_metrics).T
    print(fin_df)

    if plot:
        print("Erstelle Plots...")
        output_dir = "results/plots"
        plot_equity_curves(price_series, signals_dict, output_dir)
        plot_metric_bars(results_classif, "F1", output_dir)
        print(f"Plots gespeichert in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forex ML vs. technische Analyse – erweiterte Version")
    parser.add_argument("symbol", help="Devisenpaar, z. B. 'EURUSD=X'")
    parser.add_argument("--train-start", dest="train_start", default="2018-01-01")
    parser.add_argument("--train-end", dest="train_end", default="2022-12-31")
    parser.add_argument("--test-start", dest="test_start", default="2023-01-01")
    parser.add_argument("--test-end", dest="test_end", default="2024-12-31")
    parser.add_argument("--short", type=int, default=50)
    parser.add_argument("--long", type=int, default=200)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    main(
        symbol=args.symbol,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        short_window=args.short,
        long_window=args.long,
        tune=args.tune,
        plot=args.plot,
    )
