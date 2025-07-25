"""
Hilfsfunktionen für das Forex‑Projekt.

Dieses Modul enthält Funktionen zur Erstellung der Zielvariable und zur
Aufbereitung der Daten (z. B. Feature Engineering, Normalisierung,
Train/Test‑Aufteilung).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def create_target(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """Erzeuge eine binäre Zielvariable, die angibt, ob der Kurs in `horizon`
    Perioden steigt.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame mit mindestens einer ``Close``‑Spalte.
    horizon : int
        Vorhersagehorizont (Anzahl Perioden in die Zukunft).

    Returns
    -------
    pd.Series
        Binäre Zielvariable: 1 wenn ``Close[t + horizon] > Close[t]``, sonst 0.
    """
    # Zugriff auf die Schlusskurse. Manche Datenquellen (z. B. yfinance) liefern
    # diese Spalte als DataFrame mit einer Spalte statt als Series. In diesem Fall
    # muss sie auf eindimensional reduziert werden, sonst entstehen mehrdimensionale
    # Masken weiter unten im Code.
    close = df["Close"]
    # Falls 'close' zweidimensional ist, z. B. ein DataFrame mit einer Spalte,
    # extrahiere die erste Spalte, um eine Series zu erhalten.
    if hasattr(close, "ndim") and close.ndim > 1:
        close = close.iloc[:, 0]

    shifted = close.shift(-horizon)
    target = (shifted > close).astype(int)
    target.name = f"Target_{horizon}"
    return target


def train_test_split_scaled(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Teile die Daten in Train/Test‑Sets und skaliere die Features.

    Parameters
    ----------
    X : pd.DataFrame
        Feature‑Matrix.
    y : pd.Series
        Zielvariable.
    test_size : float
        Anteil der Daten im Testset.
    random_state : int
        Zufallssamen für Reproduzierbarkeit.

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler
        Geteilte und normalisierte Datensätze sowie der eingesetzte ``StandardScaler``.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Ravel die Zielarrays, um 1D‑Formate für scikit‑learn zu gewährleisten
    return (
        X_train_scaled,
        X_test_scaled,
        y_train.values.ravel(),
        y_test.values.ravel(),
        scaler,
    )