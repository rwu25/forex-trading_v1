
import os
import hashlib
import pandas as pd
import yfinance as yf


def load_forex_data(symbol: str, start: str, end: str, cache_dir: str | None = None) -> pd.DataFrame:
    """
    Load historical Forex data for a given symbol and date range.

    Parameters
    ----------
    symbol : str
        Yahoo Finance symbol (e.g. 'EURUSD=X').
    start : str
        Start date in 'YYYY-MM-DD' format.
    end : str
        End date in 'YYYY-MM-DD' format.
    cache_dir : str, optional
        Directory for caching downloaded data. If None, no caching is used.

    Returns
    -------
    pd.DataFrame
        DataFrame with OHLCV data indexed by date.
    """
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        key = f"{symbol}_{start}_{end}"
        hash_key = hashlib.md5(key.encode()).hexdigest()
        cache_path = os.path.join(cache_dir, f"{hash_key}.csv")
        if os.path.exists(cache_path):
            print(f"Loaded cached data from {cache_path}")
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            return df

    df = yf.download(symbol, start=start, end=end, progress=False)

    if df.empty:
        raise ValueError(f"No data found for {symbol} between {start} and {end}")

    df = df[["Open", "High", "Low", "Close", "Volume"]]  # dropped 'Adj Close'
    df = df.dropna()

    if cache_dir:
        df.to_csv(cache_path)
        print(f"Data cached to {cache_path}")

    return df
