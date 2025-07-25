"""
Backtesting utilities for trading signals.

This module implements a straightforward backtesting engine that
applies binary trading signals to a price series, accounting for
transaction costs.  It computes the equity curve and several
common risk/return metrics such as the Sharpe ratio, maximum
drawdown and cumulative return.

The backtester assumes that positions are either fully long
(signal=1) or flat/short (signal=0 or negative).  Transaction
costs are applied whenever the position changes.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict


def backtest_signals(
    signals: pd.Series,
    prices: pd.Series,
    transaction_cost: float = 0.0,
) -> pd.DataFrame:
    """Backtest a series of trading signals against a price series.

    Parameters
    ----------
    signals : pd.Series
        Binary or integer signal values indicating position sizing. 1
        represents a long position; 0 represents no position.
    prices : pd.Series
        Series of closing prices aligned with ``signals``.
    transaction_cost : float, optional
        Proportional cost incurred on position changes (default 0.0).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``return`` and ``equity``, where
        ``return`` is the net daily return and ``equity`` is the
        cumulative equity curve starting from 1.0.
    """
    df = pd.concat([signals.rename("signal"), prices.rename("price")], axis=1).dropna()
    # Forward price changes
    price_pct_change = df["price"].pct_change().shift(-1)
    # Align signals with returns
    returns = price_pct_change * df["signal"]
    # Transaction cost applied on position change
    position_change = df["signal"].diff().abs()
    costs = position_change * transaction_cost
    net_returns = returns - costs
    # Replace NaNs from first rows with 0
    net_returns = net_returns.fillna(0.0)
    equity = (1.0 + net_returns).cumprod()
    result = pd.DataFrame({"return": net_returns, "equity": equity})
    return result


def compute_financial_metrics(
    returns: pd.Series,
    freq: int = 252,
) -> Dict[str, float]:
    """Calculate common financial performance metrics.

    Parameters
    ----------
    returns : pd.Series
        Series of net periodic returns.
    freq : int, optional
        Annualization factor (default 252 for daily data).

    Returns
    -------
    dict
        Dictionary containing Sharpe ratio, maximum drawdown and
        cumulative return.
    """
    # Drop NaNs to avoid affecting metrics
    r = returns.dropna()
    # Avoid division by zero
    if r.empty or r.std() == 0:
        sharpe = np.nan
    else:
        sharpe = (np.sqrt(freq) * r.mean()) / r.std()
    # Equity curve for drawdown and cumulative return
    equity = (1.0 + r).cumprod()
    # Maximum drawdown
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    # Cumulative return
    cumulative_return = equity.iloc[-1] - 1.0 if not equity.empty else np.nan
    return {
        "Sharpe": sharpe,
        "MaxDrawdown": max_drawdown,
        "CumulativeReturn": cumulative_return,
    }
