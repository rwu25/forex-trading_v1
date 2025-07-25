"""
Training and evaluation of machine‑learning models for Forex data.

This module extends the original implementation by offering
optional hyperparameter tuning for Support Vector Machines and
Decision Trees, and by calculating additional evaluation metrics
such as balanced accuracy and ROC‑AUC.  The core logic remains
the same: a time‑based split separates training and testing
periods, features are scaled based on the training set, and
multiple models are trained on the same feature matrix.

Parameters
----------
df : pandas.DataFrame
    DataFrame with technical indicators and price columns.
start_train, end_train : str
    Start and end dates for the training period (inclusive).
start_test, end_test : str
    Start and end dates for the test period (inclusive).
tune_rf, tune_mlp, tune_svm, tune_dt : bool, optional
    Flags to indicate whether to perform grid search for the
    respective models.  RandomForest and MLP have sensible
    defaults; SVM and Decision Tree tuning is newly added.
random_state : int
    Random seed for reproducibility.

Returns
-------
models : dict
    Mapping from model name to trained estimator.
metrics_df : pandas.DataFrame
    Table of classification metrics (Accuracy, Precision,
    Recall, F1, BalancedAccuracy, ROC_AUC) for each model.
predictions : dict
    Mapping from model name to a Series of predicted labels for
    the test period.
probas : dict
    Mapping from model name to a Series of predicted class 1
    probabilities for the test period, or ``None`` if not
    available.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from .utils import create_target


def prepare_features(
    df: pd.DataFrame, dropna: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare the feature matrix and target variable from the DataFrame.

    All columns except ``Close``, ``Adj Close`` and ``Volume`` are used
    as features.  Missing values can be dropped if ``dropna`` is True.
    """
    df = df.copy()
    y = create_target(df, horizon=1)
    # Remove last row where the target is NaN
    df = df.iloc[:-1].copy()
    y = y.iloc[:-1]
    feature_cols = [col for col in df.columns if col not in {"Close", "Adj Close", "Volume"}]
    X = df[feature_cols]
    if dropna:
        mask = X.notna().all(axis=1)
        X = X[mask]
        y = y[mask]
    return X, y


def train_classification_models(
    df: pd.DataFrame,
    start_train: str,
    end_train: str,
    start_test: str,
    end_test: str,
    tune_rf: bool = False,
    tune_mlp: bool = False,
    tune_svm: bool = False,
    tune_dt: bool = False,
    tune_lr: bool = False,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, pd.Series], Dict[str, Optional[pd.Series]]]:
    """Train multiple classification models with a time‑based split.

    The data is split into training and testing periods according to
    the provided date ranges.  Optionally, hyperparameter tuning
    via ``GridSearchCV`` can be performed for RandomForest, MLP,
    SVM, DecisionTree and Logistic Regression models.  Standard scaling is applied
    separately on the training and test sets.
    """
    # Prepare features and target
    X, y = prepare_features(df)
    # Ensure the index is datetime
    if not isinstance(X.index, pd.DatetimeIndex):
        X = X.copy(); y = y.copy()
        X.index = pd.to_datetime(X.index)
        y.index = pd.to_datetime(y.index)
    # Apply date filters
    mask_train = (X.index >= pd.to_datetime(start_train)) & (X.index <= pd.to_datetime(end_train))
    mask_test = (X.index >= pd.to_datetime(start_test)) & (X.index <= pd.to_datetime(end_test))
    X_train, y_train = X.loc[mask_train], y.loc[mask_train]
    X_test, y_test = X.loc[mask_test], y.loc[mask_test]
    # Standard scaling
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Setup cross‑validation
    tscv = TimeSeriesSplit(n_splits=3)
    models: Dict[str, Any] = {}
    metrics_list: List[Dict[str, float]] = []
    predictions: Dict[str, pd.Series] = {}
    probas: Dict[str, Optional[pd.Series]] = {}

    # Decision Tree
    dt = DecisionTreeClassifier(random_state=random_state)
    if tune_dt:
        param_grid_dt = {
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5, 10],
        }
        grid_dt = GridSearchCV(dt, param_grid_dt, cv=tscv, scoring="f1", n_jobs=-1, verbose=0)
        grid_dt.fit(X_train_scaled, y_train)
        dt = grid_dt.best_estimator_
    else:
        dt.fit(X_train_scaled, y_train)
    models["DecisionTree"] = dt
    y_pred_dt = dt.predict(X_test_scaled)
    predictions["DecisionTree"] = pd.Series(y_pred_dt, index=y_test.index)
    proba_dt = None
    try:
        proba_arr = dt.predict_proba(X_test_scaled)[:, 1]
        proba_dt = pd.Series(proba_arr, index=y_test.index)
    except Exception:
        proba_dt = None
    probas["DecisionTree"] = proba_dt
    metrics_list.append(_evaluate_model(
        "DecisionTree", y_test, y_pred_dt, proba_dt
    ))

    # RandomForest
    rf = RandomForestClassifier(random_state=random_state)
    if tune_rf:
        param_grid_rf = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
        }
        grid_rf = GridSearchCV(rf, param_grid_rf, cv=tscv, scoring="f1", n_jobs=-1, verbose=0)
        grid_rf.fit(X_train_scaled, y_train)
        rf = grid_rf.best_estimator_
    else:
        rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
        rf.fit(X_train_scaled, y_train)
    models["RandomForest"] = rf
    y_pred_rf = rf.predict(X_test_scaled)
    predictions["RandomForest"] = pd.Series(y_pred_rf, index=y_test.index)
    proba_rf = None
    try:
        proba_arr = rf.predict_proba(X_test_scaled)[:, 1]
        proba_rf = pd.Series(proba_arr, index=y_test.index)
    except Exception:
        proba_rf = None
    probas["RandomForest"] = proba_rf
    metrics_list.append(_evaluate_model(
        "RandomForest", y_test, y_pred_rf, proba_rf
    ))

    # SVM
    svm = SVC(kernel="rbf", probability=True, random_state=random_state)
    if tune_svm:
        param_grid_svm = {
            "C": [0.1, 1, 10],
            "gamma": ["scale", "auto"],
        }
        grid_svm = GridSearchCV(svm, param_grid_svm, cv=tscv, scoring="f1", n_jobs=-1, verbose=0)
        grid_svm.fit(X_train_scaled, y_train)
        svm = grid_svm.best_estimator_
    else:
        svm.fit(X_train_scaled, y_train)
    models["SVM"] = svm
    y_pred_svm = svm.predict(X_test_scaled)
    predictions["SVM"] = pd.Series(y_pred_svm, index=y_test.index)
    proba_svm = None
    try:
        proba_arr = svm.predict_proba(X_test_scaled)[:, 1]
        proba_svm = pd.Series(proba_arr, index=y_test.index)
    except Exception:
        proba_svm = None
    probas["SVM"] = proba_svm
    metrics_list.append(_evaluate_model(
        "SVM", y_test, y_pred_svm, proba_svm
    ))

    # MLP Neural Network
    mlp = MLPClassifier(max_iter=200, random_state=random_state)
    if tune_mlp:
        param_grid_mlp = {
            "hidden_layer_sizes": [(50,), (100,), (100, 50)],
            "activation": ["relu", "tanh"],
        }
        grid_mlp = GridSearchCV(mlp, param_grid_mlp, cv=tscv, scoring="f1", n_jobs=-1, verbose=0)
        grid_mlp.fit(X_train_scaled, y_train)
        mlp = grid_mlp.best_estimator_
    else:
        mlp = MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
            max_iter=200, random_state=random_state
        )
        mlp.fit(X_train_scaled, y_train)
    models["MLP"] = mlp
    y_pred_mlp = mlp.predict(X_test_scaled)
    predictions["MLP"] = pd.Series(y_pred_mlp, index=y_test.index)
    proba_mlp = None
    try:
        proba_arr = mlp.predict_proba(X_test_scaled)[:, 1]
        proba_mlp = pd.Series(proba_arr, index=y_test.index)
    except Exception:
        proba_mlp = None
    probas["MLP"] = proba_mlp
    metrics_list.append(_evaluate_model(
        "MLP", y_test, y_pred_mlp, proba_mlp
    ))

    # Logistic Regression
    # A linear model baseline that can offer well‑calibrated probabilities.
    lr = LogisticRegression(max_iter=200, random_state=random_state, solver="lbfgs")
    if tune_lr:
        param_grid_lr = {
            "C": [0.1, 1, 10],
            "penalty": ["l2"],
        }
        grid_lr = GridSearchCV(lr, param_grid_lr, cv=tscv, scoring="f1", n_jobs=-1, verbose=0)
        grid_lr.fit(X_train_scaled, y_train)
        lr = grid_lr.best_estimator_
    else:
        lr.fit(X_train_scaled, y_train)
    models["LogisticRegression"] = lr
    y_pred_lr = lr.predict(X_test_scaled)
    predictions["LogisticRegression"] = pd.Series(y_pred_lr, index=y_test.index)
    proba_lr: Optional[pd.Series] = None
    try:
        proba_arr = lr.predict_proba(X_test_scaled)[:, 1]
        proba_lr = pd.Series(proba_arr, index=y_test.index)
    except Exception:
        proba_lr = None
    probas["LogisticRegression"] = proba_lr
    metrics_list.append(_evaluate_model(
        "LogisticRegression", y_test, y_pred_lr, proba_lr
    ))

    # Build DataFrame
    metrics_df = pd.DataFrame(metrics_list)
    return models, metrics_df, predictions, probas


def _evaluate_model(
    name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    proba: Optional[pd.Series],
) -> Dict[str, float]:
    """Compute classification metrics for a model.

    In addition to Accuracy, Precision, Recall and F1, this
    extended evaluation also reports balanced accuracy and
    ROC‑AUC (where probabilities are available).  If a probability
    vector is not provided or only a single class is present,
    ROC‑AUC will be set to ``np.nan``.
    """
    metrics: Dict[str, float] = {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "BalancedAccuracy": balanced_accuracy_score(y_true, y_pred),
        "ROC_AUC": np.nan,
    }
    if proba is not None:
        try:
            # ROC‑AUC only defined when both classes are present
            if len(np.unique(y_true)) > 1:
                metrics["ROC_AUC"] = roc_auc_score(y_true, proba)
        except Exception:
            metrics["ROC_AUC"] = np.nan
    return metrics