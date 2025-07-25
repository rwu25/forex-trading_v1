from __future__ import annotations

import os
import argparse
import pandas as pd
from typing import Dict, Tuple

from .data_loader import load_forex_data
from .indicators import compute_indicators
from .utils import create_target
from .ml_models import train_classification_models
from .evaluator import generate_sma_signal, evaluate_strategy
from .backtester import backtest_signals, compute_financial_metrics


def infer_train_test_dates(df: pd.DataFrame, train_ratio: float = 0.7) -> Tuple[str, str, str, str]:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex to infer dates")
    dates = df.index.sort_values()
    n = len(dates)
    split_idx = int(n * train_ratio)
    start_train = dates[0].strftime("%Y-%m-%d")
    end_train = dates[split_idx - 1].strftime("%Y-%m-%d")
    start_test = dates[split_idx].strftime("%Y-%m-%d") if split_idx < n else dates[-1].strftime("%Y-%m-%d")
    end_test = dates[-1].strftime("%Y-%m-%d")
    return start_train, end_train, start_test, end_test


def run_experiments(
    symbol: str,
    start_date: str,
    end_date: str,
    output_dir: str,
    train_start: str | None = None,
    train_end: str | None = None,
    test_start: str | None = None,
    test_end: str | None = None,
    transaction_cost: float = 0.0,
    horizon: int = 1,
    threshold: float = 0.0,
    tune_rf: bool = False,
    tune_mlp: bool = False,
    tune_svm: bool = False,
    tune_dt: bool = False,
    tune_lr: bool = False,
    cache_dir: str | None = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data for {symbol} from {start_date} to {end_date}...")
    df = load_forex_data(symbol, start=start_date, end=end_date, cache_dir=cache_dir)

    print("Computing technical indicators...")
    df_ind = compute_indicators(df)

    print("Creating target variable...")
    df_ind["Target"] = create_target(df_ind, horizon=horizon, threshold=threshold, binary=True)
    df_ind = df_ind.dropna()

    if all(x is not None for x in (train_start, train_end, test_start, test_end)):
        dates = (train_start, train_end, test_start, test_end)
    else:
        dates = infer_train_test_dates(df_ind, train_ratio=0.7)

    start_train, end_train, start_test, end_test = dates
    print(f"Training period: {start_train} to {end_train}")
    print(f"Testing period: {start_test} to {end_test}")

    print("Training machine-learning models...")
    models, metrics_df, predictions, probas = train_classification_models(
        df_ind,
        start_train=start_train,
        end_train=end_train,
        start_test=start_test,
        end_test=end_test,
        tune_rf=tune_rf,
        tune_mlp=tune_mlp,
        tune_svm=tune_svm,
        tune_dt=tune_dt,
        tune_lr=tune_lr,
    )

    print("Evaluating SMA crossover strategy...")
    baseline_signal = generate_sma_signal(df_ind)
    baseline_metrics = evaluate_strategy(baseline_signal, df_ind["Target"])
    baseline_backtest = backtest_signals(baseline_signal, df_ind["Close"], transaction_cost)
    baseline_financial = compute_financial_metrics(baseline_backtest["return"], freq=252)

    financial_results: Dict[str, Dict[str, float]] = {}
    for model_name, preds in predictions.items():
        signal = preds.reindex(df_ind.index).fillna(0).astype(int)
        backtest = backtest_signals(signal, df_ind["Close"], transaction_cost)
        fin_metrics = compute_financial_metrics(backtest["return"], freq=252)
        financial_results[model_name] = fin_metrics

    baseline_row = {"Model": "SMA_Crossover", **baseline_metrics}
    metrics_df = pd.concat([
        metrics_df,
        pd.DataFrame([baseline_row])
    ], ignore_index=True, sort=False).fillna(0)

    class_metrics_path = os.path.join(output_dir, "classification_metrics.csv")
    metrics_df.to_csv(class_metrics_path, index=False)
    print(f"Classification metrics saved to {class_metrics_path}")

    fin_metrics_df = pd.DataFrame.from_dict(financial_results, orient="index")
    fin_metrics_df.loc["SMA_Crossover"] = baseline_financial
    fin_metrics_path = os.path.join(output_dir, "financial_metrics.csv")
    fin_metrics_df.to_csv(fin_metrics_path)
    print(f"Financial metrics saved to {fin_metrics_path}")

    print("\nClassification Metrics:\n", metrics_df)
    print("\nFinancial Metrics:\n", fin_metrics_df)


def main():
    parser = argparse.ArgumentParser(description="Run Forex trading experiments")
    parser.add_argument("--symbol", type=str, required=True, help="Forex symbol (e.g., EURUSD=X)")
    parser.add_argument("--start-date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for results")
    parser.add_argument("--train-start", type=str, help="Training start date (YYYY-MM-DD)")
    parser.add_argument("--train-end", type=str, help="Training end date (YYYY-MM-DD)")
    parser.add_argument("--test-start", type=str, help="Testing start date (YYYY-MM-DD)")
    parser.add_argument("--test-end", type=str, help="Testing end date (YYYY-MM-DD)")
    parser.add_argument("--transaction-cost", type=float, default=0.0, help="Transaction cost per trade")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon for target variable")
    parser.add_argument("--threshold", type=float, default=0.0, help="Threshold for binary target")
    parser.add_argument("--tune-rf", action="store_true", help="Tune RandomForest hyperparameters")
    parser.add_argument("--tune-mlp", action="store_true", help="Tune MLP hyperparameters")
    parser.add_argument("--tune-svm", action="store_true", help="Tune SVM hyperparameters")
    parser.add_argument("--tune-dt", action="store_true", help="Tune Decision Tree hyperparameters")
    parser.add_argument("--tune-lr", action="store_true", help="Tune Logistic Regression hyperparameters")
    parser.add_argument("--cache-dir", type=str, default=None, help="Optional directory to cache downloaded data")

    args = parser.parse_args()

    run_experiments(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        transaction_cost=args.transaction_cost,
        horizon=args.horizon,
        threshold=args.threshold,
        tune_rf=args.tune_rf,
        tune_mlp=args.tune_mlp,
        tune_svm=args.tune_svm,
        tune_dt=args.tune_dt,
        tune_lr=args.tune_lr,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
