"""
data_loader.py
==============
Data loading utilities for PSX daily OHLCV stock price history, fetched
live via the `psxdata` library (https://psxdata.mintlify.app/) instead
of local CSV files.

This module refactors the data-loading cells previously duplicated
across the project's notebooks:
    - Codes/Portfolio- Univariate LSTM.ipynb
    - Codes/Portfolio- Multivariate LSTM.ipynb
    - Codes/MV Optimization (Univariate Returns).ipynb
    - Codes/MV Optimization (Multivariate Returns).ipynb

into a single, reusable set of functions, and adds:
    - Open/High/Low/Close/Volume (OHLCV) loading via `psxdata.stocks`
    - a strict chronological walk-forward train/validation/test split
    - MinMaxScaler fitting restricted to the training split only, to
      prevent forward-looking data leakage into validation/test data.
"""

from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import psxdata
from sklearn.preprocessing import MinMaxScaler


def fetch_stock(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch and clean a single stock's daily OHLCV history from PSX.

    Wraps `psxdata.stocks`, which returns a dataframe with lowercase
    `date, open, high, low, close, volume` columns, and normalizes it
    to the same shape previously produced by `load_stock_csv` when
    reading from the local `Data/` CSV files: a `Date`-indexed
    dataframe with `Open/High/Low/Close[/Volume]` columns, sorted in
    ascending date order.

    Parameters
    ----------
    ticker : str
        PSX ticker symbol, e.g. "ENGRO", "LUCK", "HUBC".
    start : str
        Start date, e.g. "2016-07-04".
    end : str
        End date, e.g. "2026-07-04".

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe indexed by Date, with Open/High/Low/Close
        (and Volume, if available) columns as floats, sorted in
        ascending date order.
    """
    data = psxdata.stocks(ticker, start=start, end=end)
    data = data.rename(columns={c: c.capitalize() for c in data.columns})

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date", "Open", "High", "Low", "Close"])
    data = data.sort_values("Date").reset_index(drop=True)
    data = data.set_index("Date")

    ohlcv_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in data.columns]
    return data[ohlcv_cols]


def compute_returns(data: pd.DataFrame, column: str = "Close") -> pd.DataFrame:
    """Compute simple percentage returns for a single price column.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing at least the given price `column`.
    column : str
        Name of the price column to compute returns from (default "Close").

    Returns
    -------
    pd.DataFrame
        Single-column dataframe of returns named "{column}_Return".
    """
    returns = data[column].pct_change().dropna().to_frame(name=f"{column}_Return")
    return returns


def load_portfolio_data(
    tickers: Iterable[str], start: str, end: str
) -> pd.DataFrame:
    """Fetch and combine multiple stocks' closing prices into one dataframe.

    Refactored from the data-loading cell in the MV Optimization
    notebooks, where all stock CSVs in a directory were loaded and
    combined into a single wide dataframe of closing prices (one
    column per ticker). Now fetches each ticker's history directly
    from PSX via `psxdata.stocks` instead of reading local CSV files.

    Parameters
    ----------
    tickers : Iterable[str]
        PSX ticker symbols to fetch, e.g. ["ENGRO", "SYS", "LUCK"].
    start : str
        Start date, e.g. "2016-07-04".
    end : str
        End date, e.g. "2026-07-04".

    Returns
    -------
    pd.DataFrame
        Wide dataframe indexed by Date, one column per ticker, sorted
        by date with rows containing any missing values dropped.
    """
    price_data: Dict[str, pd.Series] = {}
    for ticker in tickers:
        ohlcv = fetch_stock(ticker, start=start, end=end)
        price_data[ticker] = ohlcv["Close"]

    prices = pd.DataFrame(price_data)
    prices = prices.sort_index()
    prices = prices.dropna()

    return prices


def chronological_train_val_test_split(
    data: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time-ordered dataframe into train/validation/test sets
    using a strict chronological walk-forward split (no shuffling).

    The data is assumed to already be sorted in ascending chronological
    order (as returned by `fetch_stock` / `load_portfolio_data`).
    The first `train_frac` of rows become the training set, the next
    `val_frac` become validation, and the remaining rows become the
    test set - preserving time order throughout, unlike scikit-learn's
    randomized `train_test_split`.

    Parameters
    ----------
    data : pd.DataFrame
        Chronologically sorted OHLCV (or returns) dataframe.
    train_frac : float
        Fraction of rows assigned to the training set (default 0.70).
    val_frac : float
        Fraction of rows assigned to the validation set (default 0.15).
    test_frac : float
        Fraction of rows assigned to the test set (default 0.15).

    Returns
    -------
    (pd.DataFrame, pd.DataFrame, pd.DataFrame)
        (train, validation, test) dataframes, in chronological order.
    """
    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-8:
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    n = len(data)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train = data.iloc[:train_end]
    validation = data.iloc[train_end:val_end]
    test = data.iloc[val_end:]

    return train, validation, test


def fit_minmax_scaler_on_train(
    train_data, feature_range: Tuple[int, int] = (0, 1)
) -> MinMaxScaler:
    """Fit a MinMaxScaler using only the training split.

    Parameters
    ----------
    train_data : array-like or pd.DataFrame
        Training data only. The scaler must never see validation or
        test data during fitting, to prevent forward-looking data
        leakage.
    feature_range : (int, int)
        Desired range of the scaled data (default (0, 1)).

    Returns
    -------
    MinMaxScaler
        A scaler fitted exclusively on `train_data`.
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    scaler.fit(train_data)
    return scaler


def scale_train_val_test(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_range: Tuple[int, int] = (0, 1),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """Scale train/validation/test splits with a MinMaxScaler fitted
    only on the training split, then applied unchanged to validation
    and test data.

    This prevents forward data leakage: validation and test data never
    influence the scaler's learned min/max values.

    Parameters
    ----------
    train, validation, test : pd.DataFrame
        Chronologically ordered splits, as produced by
        `chronological_train_val_test_split`.
    feature_range : (int, int)
        Desired range of the scaled data (default (0, 1)).

    Returns
    -------
    (np.ndarray, np.ndarray, np.ndarray, MinMaxScaler)
        Scaled train, validation, and test arrays, plus the fitted
        scaler (for later inverse-transforming predictions).
    """
    scaler = fit_minmax_scaler_on_train(train, feature_range=feature_range)

    train_scaled = scaler.transform(train)
    val_scaled = scaler.transform(validation)
    test_scaled = scaler.transform(test)

    return train_scaled, val_scaled, test_scaled, scaler


def create_windowed_dataset(dataset: np.ndarray, step: int):
    """Create sliding-window (X, y) sequences for LSTM input.

    Refactored from the `create_ds` helper in
    Codes/Portfolio- Univariate LSTM.ipynb.

    Parameters
    ----------
    dataset : np.ndarray
        1-D (or single-feature 2-D) array of scaled values.
    step : int
        Number of look-back timesteps per sequence.

    Returns
    -------
    (np.ndarray, np.ndarray)
        X of shape (n_samples, step), y of shape (n_samples,).
    """
    x, y = [], []
    for i in range(len(dataset) - step - 1):
        x.append(dataset[i:(i + step), 0])
        y.append(dataset[i + step, 0])
    return np.array(x), np.array(y)
