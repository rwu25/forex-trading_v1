"""
Module zum Laden historischer Forex‑Daten.

Standardmäßig wird `yfinance` genutzt, um Devisenkurse wie EUR/USD
(`EURUSD=X`) von Yahoo Finance zu beziehen【342124238352642†L59-L77】.  Alternativ können bereits
heruntergeladene CSV‑Dateien eingelesen werden.  Die zurückgegebenen
DataFrames sind mit einem DatetimeIndex versehen und enthalten
mindestens die Spalten *Open*, *High*, *Low*, *Close* und *Volume*.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None  # type: ignore[assignment]


def load_forex_data(
    symbol: str,
    start: str,
    end: str,
    csv_path: Optional[str] = None,
    source: str = "yahoo",
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Lade Forex‑Zeitreihen.

    Parameters
    ----------
    symbol : str
        Kürzel des Devisenpaares, z. B. ``"EURUSD=X"``.
    start, end : str
        Datumsbereich im ISO‑Format ``"YYYY‑MM‑DD"``.
    csv_path : str, optional
        Pfad zu einer lokalen CSV‑Datei.  Falls angegeben, wird
        diese Datei bevorzugt geladen.
    source : {"yahoo", "csv"}
        Datenquelle.  ``"yahoo"`` nutzt `yfinance`, ``"csv"`` lädt
        aus `csv_path`.
    api_key : str, optional
        Für spätere Erweiterungen (z. B. Alpha Vantage).  Wird aktuell
        nicht verwendet.

    Returns
    -------
    pd.DataFrame
        Zeitreihe mit OHLCV‑Spalten.

    Raises
    ------
    ValueError
        Wenn keine gültige Datenquelle verfügbar ist.
    """
    if source == "csv":
        if not csv_path:
            raise ValueError("csv_path muss angegeben werden, wenn source='csv'")
        return load_csv_data(csv_path)

    if source == "yahoo":
        if yf is None:
            raise ImportError(
                "yfinance ist nicht installiert. Bitte mit pip install yfinance nachinstallieren."
            )
        # yfinance erwartet strings im ISO‑Format
        df = yf.download(symbol, start=start, end=end, progress=False)
        if df.empty:
            raise ValueError(f"Keine Daten für {symbol} im angegebenen Zeitraum gefunden.")
        # Normalisieren Spaltennamen (capitalise)
        df = df.rename(
            columns={
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Close": "Close",
                "Adj Close": "Adj Close",
                "Volume": "Volume",
            }
        )
        return df

    raise ValueError(f"Unbekannte Datenquelle: {source}")


def load_csv_data(file_path: str) -> pd.DataFrame:
    """Lese eine lokale CSV‑Datei mit Devisendaten ein.

    Die CSV sollte mindestens die Spalten *Date*, *Open*, *High*, *Low* und *Close*
    enthalten.  Weitere Spalten (z. B. *Volume*) werden übernommen.  Die
    Datumsspalte wird als Index verwendet.

    Parameters
    ----------
    file_path : str
        Pfad zur CSV‑Datei.

    Returns
    -------
    pd.DataFrame
        Zeitreihe mit DatetimeIndex.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    df = pd.read_csv(file_path)
    # Versuche, Datumsspalte zu identifizieren
    date_col = None
    for col in df.columns:
        if col.lower().startswith("date"):
            date_col = col
            break
    if date_col is None:
        raise ValueError("Keine Datumsspalte gefunden. Erwartet eine Spalte beginnend mit 'Date'")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    # Sortiere nach Datum
    df = df.sort_index()
    return df


def save_to_csv(df: pd.DataFrame, file_path: str) -> None:
    """Speichere die Zeitreihe als CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Zeitreihe mit Index.
    file_path : str
        Zielpfad.
    """
    df.to_csv(file_path)
