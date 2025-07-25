"""
Visualisierungsfunktionen für den Forex‑Prototyp.

Dieses Modul enthält Hilfsfunktionen, um die Resultate der Modelle
anschaulich darzustellen.  Plots werden standardmäßig im
Verzeichnis ``results/plots/`` gespeichert.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from .backtester import compute_strategy_returns, compute_equity_curve


def ensure_dir(path: str) -> None:
    """Erstelle das Verzeichnis, falls es nicht existiert."""
    os.makedirs(path, exist_ok=True)


def plot_equity_curves(
    price_series: pd.Series,
    signals_dict: Dict[str, pd.Series],
    output_dir: str,
    shift_signals: bool = True,
) -> None:
    """Erstelle einen Plot der Kapitalverläufe pro Modell.

    Parameters
    ----------
    price_series : pd.Series
        Schlusskurse.
    signals_dict : dict
        Mapping Modellname -> Signale.
    output_dir : str
        Zielverzeichnis für die Grafik.
    shift_signals : bool
        Ob die Signale um eine Periode verzögert werden sollen.
    """
    ensure_dir(output_dir)
    plt.figure(figsize=(10, 6))
    for name, sig in signals_dict.items():
        strat_returns = compute_strategy_returns(price_series, sig, shift_signals=shift_signals)
        equity = compute_equity_curve(strat_returns)
        plt.plot(equity.index, equity.values, label=name)
    plt.title("Kapitalverläufe der Strategien")
    plt.xlabel("Datum")
    plt.ylabel("Portfolio‑Wert (relativ)")
    plt.legend()
    plt.grid(True)
    outfile = os.path.join(output_dir, "equity_curves.png")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


def plot_confusion(y_true: pd.Series, y_pred: pd.Series, model_name: str, output_dir: str) -> None:
    """Erstelle eine Confusion Matrix für ein Modell.

    Die Grafik wird als PNG gespeichert.
    """
    ensure_dir(output_dir)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0", "1"])
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix – {model_name}")
    outfile = os.path.join(output_dir, f"confusion_{model_name}.png")
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metric_bars(metrics_df: pd.DataFrame, metric: str, output_dir: str) -> None:
    """Erstelle ein Balkendiagramm für einen gegebenen Metriknamen.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Tabelle mit Spalten ``Model`` und der ausgewählten Metrik.
    metric : str
        Name der Metrik (z. B. ``F1``).
    output_dir : str
        Zielverzeichnis für die Grafik.
    """
    ensure_dir(output_dir)
    plt.figure(figsize=(6, 4))
    sns.barplot(x="Model", y=metric, data=metrics_df)
    plt.title(f"{metric}‑Vergleich der Modelle")
    plt.ylabel(metric)
    plt.xlabel("")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    outfile = os.path.join(output_dir, f"bar_{metric}.png")
    plt.savefig(outfile, dpi=150)
    plt.close()