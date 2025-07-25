"""
Vergleich von Handelsstrategien: klassische technische Analyse vs. Machine Learning.

Dieses Modul stellt Funktionen bereit, um einfache gleitende Durchschnitte
als Handelsstrategie zu evaluieren und die daraus resultierenden Metriken
mit den Ergebnissen der ML‑Modelle zu vergleichen.  Es können sowohl
klassische Klassifikationsmetriken als auch finanzspezifische Kennzahlen
wie die Sharpe Ratio berechnet werden.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .utils import create_target


def generate_sma_signal(
    df: pd.DataFrame,
    short_window: int = 50,
    long_window: int = 200,
) -> pd.Series:
    """Generiere ein binäres Signal basierend auf einer SMA‑Crossover‑Strategie.

    1 signalisiert einen Long‑Einstieg, 0 bedeutet keine Position.
    Das Signal entsteht bei einem Golden Cross (kurzer SMA kreuzt den langen SMA
    von unten nach oben).  Verkaufs‑ oder Short‑Signale werden nicht erzeugt.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame mit mindestens einer ``Close``‑Spalte.
    short_window : int
        Fensterlänge für den kurzfristigen Durchschnitt.
    long_window : int
        Fensterlänge für den langfristigen Durchschnitt.

    Returns
    -------
    pd.Series
        Binäre Signale (Index entspricht dem des Eingangsdatenframes).
    """
    data = df.copy()
    data[f"SMA_{short_window}"] = data["Close"].rolling(short_window).mean()
    data[f"SMA_{long_window}"] = data["Close"].rolling(long_window).mean()
    signal = pd.Series(0, index=data.index)
    cross_mask = (
        (data[f"SMA_{short_window}"].shift(1) < data[f"SMA_{long_window}"].shift(1))
        & (data[f"SMA_{short_window}"] >= data[f"SMA_{long_window}"])
    )
    signal.loc[cross_mask] = 1
    # Reduziere auf Series, falls DataFrame mit einer Spalte vorliegt
    if hasattr(signal, "ndim") and signal.ndim > 1:
        signal = signal.iloc[:, 0]
    return signal


def evaluate_classic_strategy(
    df: pd.DataFrame,
    short_window: int = 50,
    long_window: int = 200,
    return_signals: bool = False,
    start_test: Optional[str] = None,
    end_test: Optional[str] = None,
) -> Dict[str, float]:
    """Bewerte eine SMA‑Crossover‑Strategie anhand der Klassifikationsmetriken.

    Die Strategie generiert ein Signal (1 = Kauf, 0 = Flat) bei einem
    Golden Cross.  Optional kann ein Datumsbereich für den Test angegeben
    werden.  Zusätzlich kann die Signalserie selbst zurückgegeben werden.

    Parameters
    ----------
    df : pd.DataFrame
        Zeitreihe mit mindestens einer ``Close``‑Spalte.
    short_window : int
        Länge des kurzfristigen gleitenden Durchschnitts.
    long_window : int
        Länge des langfristigen gleitenden Durchschnitts.
    return_signals : bool, default False
        Wenn ``True``, wird unter dem Schlüssel ``"Signals"`` die
        Signalserie des Testzeitraums zurückgegeben.
    start_test, end_test : str, optional
        Begrenze die Bewertung auf diesen Datumsbereich.  Die
        ``start_test`` muss größer oder gleich dem Trainingsende sein.

    Returns
    -------
    dict
        Metriken ``Model``, ``Accuracy``, ``Precision``, ``Recall``, ``F1``.
        Optional enthält das Ergebnis die Schlüssel ``Signals``.
    """
    # Generiere Signale für das gesamte Datenframe
    signal_series = generate_sma_signal(df, short_window=short_window, long_window=long_window)
    # Zielvariable erzeugen
    target = create_target(df, horizon=1)
    # Optionalen Datumsbereich anwenden
    if start_test and end_test:
        start_dt = pd.to_datetime(start_test)
        end_dt = pd.to_datetime(end_test)
        mask_date = (signal_series.index >= start_dt) & (signal_series.index <= end_dt)
        signal_series = signal_series.loc[mask_date]
        target = target.loc[mask_date]
    # Stelle sicher, dass Target Serie ist
    if hasattr(target, "ndim") and target.ndim > 1:
        target = target.iloc[:, 0]
    # Entferne NaN‑Werte
    mask_valid = signal_series.notna() & target.notna()
    y_pred = signal_series.loc[mask_valid].astype(int)
    y_true = target.loc[mask_valid].astype(int)
    metrics = {
        "Model": f"SMA{short_window}/{long_window}",
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }
    if return_signals:
        metrics["Signals"] = signal_series
    return metrics


def compare_models(
    ml_metrics: pd.DataFrame,
    classic_metrics: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None,
    plot: bool = True,
) -> pd.DataFrame:
    """Vergleiche ML‑Modelle mit klassischen Strategien.

    Parameters
    ----------
    ml_metrics : pd.DataFrame
        Metriken der ML‑Modelle aus ``ml_models.train_classification_models``.
    classic_metrics : dict, optional
        Metriken der klassischen Strategie (z. B. aus ``evaluate_classic_strategy``).
    save_path : str, optional
        Wenn angegeben, wird die kombinierte Tabelle als CSV gespeichert.
    plot : bool
        Erstelle einen Balkendiagrammvergleich der Accuracy.

    Returns
    -------
    pd.DataFrame
        Kombinierte Tabelle aller Modelle.
    """
    all_metrics = ml_metrics.copy()
    if classic_metrics is not None:
        all_metrics = pd.concat([all_metrics, pd.DataFrame([classic_metrics])], ignore_index=True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        all_metrics.to_csv(save_path, index=False)

    if plot:
        plt.figure(figsize=(8, 4))
        plt.bar(all_metrics["Model"], all_metrics["Accuracy"], color="steelblue")
        plt.ylabel("Accuracy")
        plt.title("Modellvergleich: Accuracy")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plot_path = None
        if save_path:
            base, _ = os.path.splitext(save_path)
            plot_path = f"{base}_accuracy.png"
            plt.savefig(plot_path, dpi=150)
        else:
            plt.show()
        plt.close()
    return all_metrics