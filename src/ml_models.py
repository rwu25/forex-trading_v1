"""
Training und Evaluation von Machine‑Learning‑Modellen für Forex‑Daten.

Dieses Modul stellt Funktionen bereit, um Klassifikationsmodelle wie
Entscheidungsbäume, Random Forest, Support Vector Machine und MLP zu
trainieren.  Die Modelle verwenden technische Indikatoren als
Features und eine binäre Zielvariable (steigt der Kurs in den
kommenden Perioden?).
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from .utils import create_target, train_test_split_scaled


def prepare_features(df: pd.DataFrame, dropna: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """Bereite die Feature‑Matrix und Zielvariable aus dem DataFrame vor.

    Alle Spalten außer *Close* und *Volume* werden als Features genutzt.
    Fehlende Werte werden entfernt, wenn ``dropna`` wahr ist.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame mit Indikatoren.
    dropna : bool
        Falls ``True``, werden Zeilen mit fehlenden Werten verworfen.

    Returns
    -------
    X : pd.DataFrame
        Feature‑Matrix.
    y : pd.Series
        Binäre Zielvariable (Kurs steigt ja/nein).
    """
    df = df.copy()
    y = create_target(df, horizon=1)
    # Entferne letzte Zeile (Target ist NaN)
    df = df.iloc[:-1].copy()
    y = y.iloc[:-1]
    feature_cols = [col for col in df.columns if col not in {"Close", "Adj Close", "Volume"}]
    X = df[feature_cols]
    if dropna:
        # Entferne Zeilen mit fehlenden Werten
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
    tune: bool = False,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, pd.Series], Dict[str, Optional[pd.Series]]]:
    """Trainiere mehrere Klassifikationsmodelle auf einem zeitlich getrennten Split.

    Diese Funktion teilt die Daten anhand des DatetimeIndex in einen
    Trainings‑ und einen Testbereich auf.  Optional kann eine
    Hyperparameter‑Suche mit ``GridSearchCV`` aktiviert werden.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame mit technischen Indikatoren und Kursdaten.
    start_train, end_train : str
        Start‑ und Enddatum für den Trainingszeitraum (inkl.).
    start_test, end_test : str
        Start‑ und Enddatum für den Testzeitraum (inkl.).
    tune : bool, default False
        Wenn ``True``, wird für RandomForest und MLP eine GridSearchCV
        durchgeführt.
    random_state : int
        Zufallszustand für Reproduzierbarkeit.

    Returns
    -------
    models : dict
        Mapping Modellname -> trainierter Modellinstanz.
    metrics_df : pd.DataFrame
        Tabelle mit Klassifikationsmetriken je Modell.
    predictions : dict
        Mapping Modellname -> Vorhersageserie (0/1) für den Testzeitraum.
    probas : dict
        Mapping Modellname -> Wahrscheinlichkeit der Klasse 1 für den
        Testzeitraum (sofern verfügbar, sonst ``None``).
    """
    # Bereite Features und Ziel vor
    X, y = prepare_features(df)
    # Stelle sicher, dass der Index ein DatetimeIndex ist
    if not isinstance(X.index, pd.DatetimeIndex):
        X = X.copy()
        y = y.copy()
        X.index = pd.to_datetime(X.index)
        y.index = pd.to_datetime(y.index)
    # Filtere nach Datumsbereichen
    mask_train = (X.index >= pd.to_datetime(start_train)) & (X.index <= pd.to_datetime(end_train))
    mask_test = (X.index >= pd.to_datetime(start_test)) & (X.index <= pd.to_datetime(end_test))
    X_train, y_train = X.loc[mask_train], y.loc[mask_train]
    X_test, y_test = X.loc[mask_test], y.loc[mask_test]

    # Skaliere die Features basierend auf dem Training
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models: Dict[str, Any] = {}
    metrics_list: List[Dict[str, float]] = []
    predictions: Dict[str, pd.Series] = {}
    probas: Dict[str, Optional[pd.Series]] = {}

    # Setup Cross‑Validation für Tuning (Zeitreihe)
    tscv = TimeSeriesSplit(n_splits=3)

    # 1. Decision Tree (keine Hyperparameter‑Suche)
    dt = DecisionTreeClassifier(random_state=random_state)
    dt.fit(X_train_scaled, y_train)
    models["DecisionTree"] = dt
    y_pred_dt = dt.predict(X_test_scaled)
    predictions["DecisionTree"] = pd.Series(y_pred_dt, index=y_test.index)
    probas["DecisionTree"] = None
    metrics_list.append(_evaluate_model("DecisionTree", dt, y_test, y_pred_dt))

    # 2. RandomForest
    rf = RandomForestClassifier(random_state=random_state)
    if tune:
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
        }
        grid = GridSearchCV(
            rf,
            param_grid,
            cv=tscv,
            scoring="f1",
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train_scaled, y_train)
        rf = grid.best_estimator_
    else:
        rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
        rf.fit(X_train_scaled, y_train)
    models["RandomForest"] = rf
    y_pred_rf = rf.predict(X_test_scaled)
    predictions["RandomForest"] = pd.Series(y_pred_rf, index=y_test.index)
    try:
        proba_rf = rf.predict_proba(X_test_scaled)[:, 1]
        probas["RandomForest"] = pd.Series(proba_rf, index=y_test.index)
    except Exception:
        probas["RandomForest"] = None
    metrics_list.append(_evaluate_model("RandomForest", rf, y_test, y_pred_rf))

    # 3. Support Vector Machine
    svm = SVC(kernel="rbf", probability=True, random_state=random_state)
    svm.fit(X_train_scaled, y_train)
    models["SVM"] = svm
    y_pred_svm = svm.predict(X_test_scaled)
    predictions["SVM"] = pd.Series(y_pred_svm, index=y_test.index)
    try:
        proba_svm = svm.predict_proba(X_test_scaled)[:, 1]
        probas["SVM"] = pd.Series(proba_svm, index=y_test.index)
    except Exception:
        probas["SVM"] = None
    metrics_list.append(_evaluate_model("SVM", svm, y_test, y_pred_svm))

    # 4. MLP Neural Network
    mlp = MLPClassifier(max_iter=200, random_state=random_state)
    if tune:
        param_grid = {
            "hidden_layer_sizes": [(50,), (100,), (100, 50)],
            "activation": ["relu", "tanh"],
        }
        grid = GridSearchCV(
            mlp,
            param_grid,
            cv=tscv,
            scoring="f1",
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train_scaled, y_train)
        mlp = grid.best_estimator_
    else:
        mlp = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=200,
            random_state=random_state,
        )
        mlp.fit(X_train_scaled, y_train)
    models["MLP"] = mlp
    y_pred_mlp = mlp.predict(X_test_scaled)
    predictions["MLP"] = pd.Series(y_pred_mlp, index=y_test.index)
    try:
        proba_mlp = mlp.predict_proba(X_test_scaled)[:, 1]
        probas["MLP"] = pd.Series(proba_mlp, index=y_test.index)
    except Exception:
        probas["MLP"] = None
    metrics_list.append(_evaluate_model("MLP", mlp, y_test, y_pred_mlp))

    metrics_df = pd.DataFrame(metrics_list)
    return models, metrics_df, predictions, probas


def _evaluate_model(name: str, model: object, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Berechne Klassifikationsmetriken für ein Modell.

    Parameters
    ----------
    name : str
        Modellname.
    model : object
        Modellinstanz (wird hier nicht direkt genutzt, aber für mögliche
        Erweiterungen übergeben).
    y_true : pd.Series
        Tatsächliche Zielwerte.
    y_pred : np.ndarray
        Vorhergesagte Klassen.

    Returns
    -------
    dict
        Enthält Modellname, Accuracy, Precision, Recall und F1‑Score.
    """
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }