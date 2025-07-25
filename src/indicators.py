"""
Berechnung technischer Indikatoren für Forex‑Zeitreihen.

Dieses Modul nutzt primär das Paket `ta` für die gängigen Indikatoren.
Falls `ta` nicht verfügbar ist, werden einfache Alternativen implementiert.
"""

from __future__ import annotations

import warnings
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    import ta  # type: ignore
except ImportError:  # pragma: no cover
    ta = None  # type: ignore


def compute_indicators(
    df: pd.DataFrame,
    sma_windows: Iterable[int] = (10, 20, 50),
    ema_windows: Iterable[int] = (10, 20),
    rsi_windows: Iterable[int] = (14,),
    bb_window: int = 20,
    macd_params: Tuple[int, int, int] = (12, 26, 9),
) -> pd.DataFrame:
    """Berechne eine Auswahl technischer Indikatoren.

    Zu jedem Indikator werden neue Spalten hinzugefügt.  Vorhandene Spalten
    bleiben unverändert.

    Parameters
    ----------
    df : pd.DataFrame
        Zeitreihe mit mindestens einer ``Close``‑Spalte.
    sma_windows : iterable of int
        Fensterlängen für einfache gleitende Durchschnitte.
    ema_windows : iterable of int
        Fensterlängen für exponentiell gleitende Durchschnitte.
    rsi_windows : iterable of int
        Fensterlängen für RSI.
    bb_window : int
        Fensterlänge für Bollinger‑Bänder.
    macd_params : tuple of int
        Parameter (fast, slow, signal) für den MACD.

    Returns
    -------
    pd.DataFrame
        Datenframe mit berechneten Indikatoren.
    """
    df = df.copy()
    close = df["Close"]
    # Stelle sicher, dass `close` eine eindimensionale Series ist
    if hasattr(close, "ndim") and close.ndim > 1:
        close = close.iloc[:, 0]

    # Simple Moving Averages (SMA)
    for window in sma_windows:
        df[f"SMA_{window}"] = close.rolling(window).mean()

    # Exponential Moving Averages (EMA)
    for window in ema_windows:
        df[f"EMA_{window}"] = close.ewm(span=window, adjust=False).mean()

    # Relative Strength Index (RSI)
    for window in rsi_windows:
        if ta:
            indicator = ta.momentum.RSIIndicator(close, window=window)
            df[f"RSI_{window}"] = indicator.rsi()
        else:
            # fallback einfache Berechnung
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window).mean()
            avg_loss = loss.rolling(window).mean()
            rs = avg_gain / avg_loss
            df[f"RSI_{window}"] = 100 - (100 / (1 + rs))

    # MACD
    fast, slow, signal = macd_params
    if ta:
        macd_indicator = ta.trend.MACD(close, window_slow=slow, window_fast=fast, window_sign=signal)
        df["MACD"] = macd_indicator.macd()
        df["MACD_Signal"] = macd_indicator.macd_signal()
        df["MACD_Hist"] = macd_indicator.macd_diff()
    else:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        df["MACD"] = macd_line
        df["MACD_Signal"] = signal_line
        df["MACD_Hist"] = macd_line - signal_line

    # Bollinger Bands
    if ta:
        bb_indicator = ta.volatility.BollingerBands(close, window=bb_window, window_dev=2)
        df["BB_High"] = bb_indicator.bollinger_hband()
        df["BB_Low"] = bb_indicator.bollinger_lband()
        df["BB_Mavg"] = bb_indicator.bollinger_mavg()
        df["BB_Percent"] = bb_indicator.bollinger_pband()
    else:
        ma = close.rolling(bb_window).mean()
        std = close.rolling(bb_window).std()
        df["BB_Mavg"] = ma
        df["BB_High"] = ma + 2 * std
        df["BB_Low"] = ma - 2 * std
        df["BB_Percent"] = (close - df["BB_Low"]) / (df["BB_High"] - df["BB_Low"])

    return df