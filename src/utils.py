"""
Utility functions for feature engineering, target creation and data splitting.

The functions in this module provide common preprocessing tasks for
time‑series forecasting and classification problems.  Users can
create custom target variables with adjustable prediction horizons
and thresholds, and perform chronological train/test splits.

Examples
--------
>>> from .utils import create_target, chronological_split
>>> y = create_target(df, horizon=5, threshold=0.001)
>>> X_train, X_test, y_train, y_test = chronological_split(X, y, train_size=0.8)
"""

from __future__ import annotations

import pandas as pd
from typing import Tuple, Optional


def create_target(
    df: pd.DataFrame,
    horizon: int = 1,
    threshold: float = 0.0,
    binary: bool = True,
) -> pd.Series:
    """Construct a target variable for forecasting or classification.

    The target is derived from the future change in the closing price
    after ``horizon`` periods.  If ``binary`` is True, the target will
    be a binary indicator: 1 if the percentage change exceeds
    ``threshold``, 0 otherwise.  If ``binary`` is False, the target
    will be the raw percentage change.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a ``Close`` column.
    horizon : int, optional
        Number of periods ahead to look for the target (default 1).
    threshold : float, optional
        Minimum percentage change required to label a positive class
        when ``binary=True`` (default 0.0).  A value of 0 labels any
        increase as positive.
    binary : bool, optional
        Whether to produce a binary target (default True).  If False,
        the percentage change is returned instead.

    Returns
    -------
    pd.Series
        Target variable indexed like ``df``.
    """
    if "Close" not in df.columns:
        raise KeyError("Input DataFrame must contain a 'Close' column")
    # Percentage change between t and t+horizon
    future_close = df["Close"].shift(-horizon)
    pct_change = (future_close - df["Close"]) / df["Close"]
    if binary:
        target = (pct_change > threshold).astype(int)
    else:
        target = pct_change
    return target


def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: float = 0.7,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target chronologically into train and test sets.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix indexed by time (must be sorted).
    y : pd.Series
        Target variable aligned with ``X``.
    train_size : float, optional
        Fraction of the data to use for training (default 0.7).

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple of pd.DataFrame and pd.Series
        Chronologically ordered train and test splits.

    Notes
    -----
    No shuffling is performed.  Missing values are not handled and
    should be addressed before calling this function.
    """
    if not 0.0 < train_size < 1.0:
        raise ValueError("train_size must be between 0 and 1")
    split_idx = int(len(X) * train_size)
    X_train = X.iloc[:split_idx].copy()
    y_train = y.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_test = y.iloc[split_idx:].copy()
    return X_train, X_test, y_train, y_test
