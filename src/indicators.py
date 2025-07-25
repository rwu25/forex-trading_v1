"""
Technical indicator computations for Forex data.

This module provides a flexible ``compute_indicators`` function that
calculates a variety of commonly used technical indicators.  In
addition to the basic suite of moving averages, RSI, MACD and
Bollinger bands, it also computes the Average True Range (ATR)
and the Stochastic Oscillator.  The function works with a
``pandas.DataFrame`` containing OHLCV data and returns a new
DataFrame with the added indicator columns.

The implementation does not rely on external libraries such as
``ta``; instead, it uses straightforward Pandas operations.  This
makes the code easy to audit and free from external dependencies.

Examples
--------
>>> from .data_loader import load_forex_data
>>> from .indicators import compute_indicators
>>> df = load_forex_data("EURUSD=X", start="2020-01-01", end="2021-01-01")
>>> df_ind = compute_indicators(df)
>>> df_ind.columns
Index([... 'SMA', 'EMA', 'RSI', 'MACD', 'MACD_Signal',
       'Bollinger_Middle', 'Bollinger_Upper', 'Bollinger_Lower',
       'ATR', 'Stoch_%K', 'Stoch_%D'],
      dtype='object')
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Tuple, Iterable


def compute_indicators(
    df: pd.DataFrame,
    sma_window: int = 14,
    ema_window: int = 14,
    rsi_window: int = 14,
    macd_params: Tuple[int, int, int] = (12, 26, 9),
    bollinger_window: int = 20,
    bollinger_std: float = 2.0,
    atr_window: int = 14,
    stoch_params: Tuple[int, int] = (14, 3),
) -> pd.DataFrame:
    """Compute a suite of technical indicators.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least ``Open``, ``High``, ``Low`` and ``Close`` columns.
    sma_window : int, optional
        Window size for the simple moving average (default 14).
    ema_window : int, optional
        Span for the exponential moving average (default 14).
    rsi_window : int, optional
        Window length for the Relative Strength Index (default 14).
    macd_params : tuple of int, optional
        Parameters ``(short_span, long_span, signal_span)`` for the MACD (default (12, 26, 9)).
    bollinger_window : int, optional
        Rolling window length for the Bollinger bands (default 20).
    bollinger_std : float, optional
        Number of standard deviations for Bollinger bands (default 2.0).
    atr_window : int, optional
        Window length for the Average True Range (default 14).
    stoch_params : tuple of int, optional
        Parameters ``(k_window, d_window)`` for the Stochastic Oscillator (default (14, 3)).

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with additional indicator columns.

    Notes
    -----
    The function will not modify the input DataFrame in place.
    Missing values will appear at the beginning of the series due
    to rolling computations.  Users should handle missing values
    before model training.
    """
    required_cols = {"Open", "High", "Low", "Close"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns for indicator computation: {missing}")
    data = df.copy()
    # Simple Moving Average
    data["SMA"] = data["Close"].rolling(window=sma_window).mean()
    # Exponential Moving Average
    data["EMA"] = data["Close"].ewm(span=ema_window, adjust=False).mean()
    # Relative Strength Index
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=rsi_window).mean()
    avg_loss = loss.rolling(window=rsi_window).mean()
    rs = avg_gain / avg_loss
    data["RSI"] = 100.0 - (100.0 / (1.0 + rs))
    # MACD and Signal
    short_span, long_span, signal_span = macd_params
    ema_short = data["Close"].ewm(span=short_span, adjust=False).mean()
    ema_long = data["Close"].ewm(span=long_span, adjust=False).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=signal_span, adjust=False).mean()
    data["MACD"] = macd
    data["MACD_Signal"] = signal
    # Bollinger Bands
    rolling_mean = data["Close"].rolling(window=bollinger_window).mean()
    rolling_std = data["Close"].rolling(window=bollinger_window).std()
    data["Bollinger_Middle"] = rolling_mean
    data["Bollinger_Upper"] = rolling_mean + bollinger_std * rolling_std
    data["Bollinger_Lower"] = rolling_mean - bollinger_std * rolling_std
    # Average True Range
    high = data["High"]
    low = data["Low"]
    close_prev = data["Close"].shift(1)
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    data["ATR"] = tr.rolling(window=atr_window).mean()
    # Stochastic Oscillator
    k_window, d_window = stoch_params
    low_min = data["Low"].rolling(window=k_window).min()
    high_max = data["High"].rolling(window=k_window).max()
    data["Stoch_%K"] = ((data["Close"] - low_min) / (high_max - low_min)) * 100.0
    data["Stoch_%D"] = data["Stoch_%K"].rolling(window=d_window).mean()
    return data
