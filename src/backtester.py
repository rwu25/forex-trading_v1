"""
Backtesting‑Funktionen für Forex‑Strategien.

Dieses Modul stellt Funktionen bereit, um aus Signalreihen (z. B. aus
Klassifikatoren oder technischen Strategien) eine einfache
Portfolioentwicklung zu simulieren.
"""

from __future__ import annotations
from typing import Dict

import numpy as np
import pandas as pd


def compute_strategy_returns(
    price_series: pd.Series,
    signals: pd.Series,
    shift_signals: bool = True,
) -> pd.Series:
    """Berechne die täglichen Strategie‑Renditen anhand der Signale."""
    # price_series ggf. von DataFrame auf Series reduzieren
    if isinstance(price_series, pd.DataFrame) and price_series.shape[1] == 1:
        price_series = price_series.iloc[:, 0]

    returns = price_series.pct_change().fillna(0.0)

    # aus DataFrame eine Serie machen und auf Preisindex ausrichten
    sig = signals.squeeze() if hasattr(signals, "squeeze") else signals
    sig = pd.Series(sig, index=sig.index).reindex(price_series.index, fill_value=0.0).astype(float)
    if shift_signals:
        sig = sig.shift(1).fillna(0.0)

    return sig * returns


def compute_equity_curve(strategy_returns: pd.Series) -> pd.Series:
    return (1 + strategy_returns).cumprod()


def compute_financial_metrics(strategy_returns: pd.Series) -> Dict[str, float]:
    """Berechne Sharpe Ratio, Max Drawdown und Cumulative Return."""
    # Falls strategy_returns als DataFrame kommt, auf Series reduzieren
    if isinstance(strategy_returns, pd.DataFrame):
        if strategy_returns.shape[1] == 1:
            strategy_returns = strategy_returns.iloc[:, 0]
        else:
            strategy_returns = strategy_returns.mean(axis=1)

    if strategy_returns.empty:
        return {"Sharpe": np.nan, "MaxDrawdown": np.nan, "CumulativeReturn": np.nan}

    mean_daily = strategy_returns.mean()
    std_daily = strategy_returns.std()
    mean_val = float(mean_daily) if not isinstance(mean_daily, pd.Series) else float(mean_daily.iloc[0])
    std_val = float(std_daily) if not isinstance(std_daily, pd.Series) else float(std_daily.iloc[0])

    # Sharpe Ratio: 0 wenn keine Volatilität vorhanden ist
    sharpe = (mean_val / std_val) * np.sqrt(252) if std_val > 0 else 0.0

    equity = compute_equity_curve(strategy_returns)
    running_max = equity.cummax()
    drawdowns = equity / running_max - 1

    # Minimum Drawdown extrahieren
    max_drawdown = drawdowns.min()
    if isinstance(max_drawdown, pd.Series):
        max_drawdown = max_drawdown.min()
    max_drawdown_val = float(max_drawdown) if not np.isnan(max_drawdown) else 0.0

    # Kumulativer Return
    cumulative_return = equity.iloc[-1] - 1
    if isinstance(cumulative_return, pd.Series):
        cumulative_return = cumulative_return.iloc[-1]
    cumulative_return_val = float(cumulative_return) if not np.isnan(cumulative_return) else 0.0

    return {
        "Sharpe": sharpe,
        "MaxDrawdown": max_drawdown_val,
        "CumulativeReturn": cumulative_return_val,
    }


def backtest_signals(
    price_series: pd.Series,
    signals_dict: Dict[str, pd.Series],
    shift_signals: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Berechne finanzielle Kennzahlen für mehrere Strategien."""
    results: Dict[str, Dict[str, float]] = {}
    for name, sig in signals_dict.items():
        strat_returns = compute_strategy_returns(price_series, sig, shift_signals=shift_signals)
        results[name] = compute_financial_metrics(strat_returns)
    return results
