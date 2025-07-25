"""
Evaluation utilities for trading strategies and classification signals.

This module provides simple baseline strategy generation and
evaluation routines.  It currently implements a moving average
crossover strategy and computes standard classification metrics
to assess signal quality.  These functions are intended to
complement the machine‑learning models defined elsewhere in the
package.
"""

from __future__ import annotations

import pandas as pd
from typing import Dict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
)


def generate_sma_signal(
    df: pd.DataFrame,
    short_window: int = 10,
    long_window: int = 30,
) -> pd.Series:
    """Generate a simple moving average crossover signal.

    A signal of ``1`` indicates that the short moving average is
    above the long moving average, suggesting a bullish stance. A
    signal of ``0`` indicates a bearish or neutral stance.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a ``Close`` column.
    short_window : int, optional
        Window length for the short SMA (default 10).
    long_window : int, optional
        Window length for the long SMA (default 30).

    Returns
    -------
    pd.Series
        Binary signal series aligned with ``df``.
    """
    if "Close" not in df.columns:
        raise KeyError("Input DataFrame must contain a 'Close' column")
    sma_short = df["Close"].rolling(window=short_window).mean()
    sma_long = df["Close"].rolling(window=long_window).mean()
    signal = (sma_short > sma_long).astype(int)
    # Preserve the original index
    signal.name = "SMA_Signal"
    return signal


def evaluate_strategy(
    signals: pd.Series,
    target: pd.Series,
) -> Dict[str, float]:
    """Compute classification metrics for a trading signal.

    Both ``signals`` and ``target`` are aligned on their index.  NaN
    values are dropped prior to evaluation.

    Parameters
    ----------
    signals : pd.Series
        Series of predicted binary labels (e.g., from a strategy).
    target : pd.Series
        Series of true binary labels.

    Returns
    -------
    dict
        Dictionary with keys ``Accuracy``, ``Precision``, ``Recall``,
        ``F1`` and ``BalancedAccuracy``.
    """
    # Align and drop missing values
    df_eval = pd.concat([signals, target], axis=1, keys=["pred", "true"]).dropna()
    y_pred = df_eval["pred"].astype(int)
    y_true = df_eval["true"].astype(int)
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "BalancedAccuracy": balanced_accuracy_score(y_true, y_pred),
    }
    return metrics
