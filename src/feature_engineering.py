"""
feature_engineering.py
=======================
Feature generation pipeline for daily OHLCV data.

Implements:
    Trend & Momentum:   SMA, EMA, RSI, MACD, Stochastic Oscillator
    Volatility & Volume: ATR, Bollinger Bands, OBV, VWAP
    Lag features:        standard lagged OHLCV columns

Every indicator group and the lag-feature step can be independently
enabled or disabled via the `feature_engineering` section of
`config.yaml`, to support future ablation experiments.

This module only generates features - it does not perform feature
selection (see `src/feature_selection.py`) or model training.
"""

import copy
from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_FEATURE_CONFIG = {
    "enabled": True,
    "indicators": {
        "sma": True,
        "ema": True,
        "rsi": True,
        "macd": True,
        "stochastic_oscillator": True,
        "atr": True,
        "bollinger_bands": True,
        "obv": True,
        "vwap": True,
    },
    "params": {
        "sma_windows": [10, 20, 50],
        "ema_windows": [10, 20, 50],
        "rsi_length": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "stochastic_k_period": 14,
        "stochastic_d_period": 3,
        "atr_length": 14,
        "bollinger_length": 20,
        "bollinger_num_std": 2,
        "vwap_window": 14,
    },
    "lag_features": {
        "enabled": True,
        "columns": ["Open", "High", "Low", "Close", "Volume"],
        "lags": [1, 2, 3, 5, 10],
    },
}


def _merge_config(default: dict, override: Optional[dict]) -> dict:
    """Recursively merge a user-supplied config dict over the defaults."""
    merged = copy.deepcopy(default)
    if not override:
        return merged

    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


# ---------------------------------------------------------------------------
# Trend & Momentum indicators
# ---------------------------------------------------------------------------
def sma(series: pd.Series, length: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=length, min_periods=1).mean()


def ema(series: pd.Series, length: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=length, min_periods=length).mean()
    avg_loss = loss.rolling(window=length, min_periods=length).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """Moving Average Convergence Divergence.

    Returns
    -------
    pd.DataFrame
        Columns: "MACD", "MACD_Signal", "MACD_Histogram".
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line

    return pd.DataFrame(
        {"MACD": macd_line, "MACD_Signal": signal_line, "MACD_Histogram": histogram}
    )


def stochastic_oscillator(
    data: pd.DataFrame, k_period: int = 14, d_period: int = 3
) -> pd.DataFrame:
    """Stochastic Oscillator (%K, %D). Requires High/Low/Close columns."""
    low_min = data["Low"].rolling(window=k_period, min_periods=k_period).min()
    high_max = data["High"].rolling(window=k_period, min_periods=k_period).max()

    percent_k = 100 * (data["Close"] - low_min) / (high_max - low_min)
    percent_d = percent_k.rolling(window=d_period, min_periods=d_period).mean()

    return pd.DataFrame({"Stoch_%K": percent_k, "Stoch_%D": percent_d})


# ---------------------------------------------------------------------------
# Volatility & Volume indicators
# ---------------------------------------------------------------------------
def atr(data: pd.DataFrame, length: int = 14) -> pd.Series:
    """Average True Range. Requires High/Low/Close columns."""
    high_low = data["High"] - data["Low"]
    high_prev_close = (data["High"] - data["Close"].shift()).abs()
    low_prev_close = (data["Low"] - data["Close"].shift()).abs()

    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    return true_range.rolling(window=length, min_periods=1).mean()


def bollinger_bands(series: pd.Series, length: int = 20, num_std: int = 2) -> pd.DataFrame:
    """Bollinger Bands (Upper, Middle, Lower)."""
    middle = series.rolling(window=length, min_periods=length).mean()
    std = series.rolling(window=length, min_periods=length).std()

    upper = middle + num_std * std
    lower = middle - num_std * std

    return pd.DataFrame({"BB_Upper": upper, "BB_Middle": middle, "BB_Lower": lower})


def obv(data: pd.DataFrame) -> pd.Series:
    """On-Balance Volume. Requires Close/Volume columns."""
    direction = np.sign(data["Close"].diff()).fillna(0)
    return (direction * data["Volume"]).cumsum()


def vwap(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """Rolling Volume Weighted Average Price.

    Computed over a rolling `window` (default 14 days) rather than a
    single cumulative-since-start value, since this project's daily
    OHLCV history spans multiple years. Requires High/Low/Close/Volume
    columns.
    """
    typical_price = (data["High"] + data["Low"] + data["Close"]) / 3
    price_volume = typical_price * data["Volume"]

    rolling_pv = price_volume.rolling(window=window, min_periods=window).sum()
    rolling_volume = data["Volume"].rolling(window=window, min_periods=window).sum()

    return rolling_pv / rolling_volume


# ---------------------------------------------------------------------------
# Lag features
# ---------------------------------------------------------------------------
def generate_lag_features(data: pd.DataFrame, columns, lags) -> pd.DataFrame:
    """Generate standard lagged versions of the given OHLCV columns.

    Parameters
    ----------
    data : pd.DataFrame
        Chronologically ordered OHLCV dataframe.
    columns : list[str]
        Column names to lag (e.g. ["Open", "High", "Low", "Close", "Volume"]).
    lags : list[int]
        Number of periods to shift back for each lag feature.

    Returns
    -------
    pd.DataFrame
        One column per (column, lag) pair, named "{column}_lag{lag}".
    """
    lagged = {}
    for column in columns:
        if column not in data.columns:
            continue
        for lag in lags:
            lagged[f"{column}_lag{lag}"] = data[column].shift(lag)

    return pd.DataFrame(lagged, index=data.index)


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------
def generate_features(data: pd.DataFrame, config: Optional[dict] = None) -> pd.DataFrame:
    """Generate the full feature set for daily OHLCV data.

    Reads the `feature_engineering` section of `config.yaml` (or an
    equivalent dict) to determine which indicator groups and lag
    features to compute, so individual groups can be toggled on/off
    for ablation experiments without editing code.

    Parameters
    ----------
    data : pd.DataFrame
        Chronologically ordered OHLCV dataframe (as returned by
        `src.data_loader.load_stock_csv`), with Open/High/Low/Close
        (and ideally Volume) columns.
    config : dict, optional
        Either the full parsed config.yaml dict (containing a
        top-level `feature_engineering` key) or the
        `feature_engineering` section itself. Defaults to
        `DEFAULT_FEATURE_CONFIG` (all indicators enabled) if omitted.

    Returns
    -------
    pd.DataFrame
        A copy of `data` with enabled feature columns appended. If
        `feature_engineering.enabled` is False, returns `data`
        unchanged (as a copy).
    """
    if config and "feature_engineering" in config:
        fe_config = _merge_config(DEFAULT_FEATURE_CONFIG, config["feature_engineering"])
    else:
        fe_config = _merge_config(DEFAULT_FEATURE_CONFIG, config)

    result = data.copy()

    if not fe_config["enabled"]:
        return result

    indicators = fe_config["indicators"]
    params = fe_config["params"]
    close = result["Close"]

    if indicators.get("sma"):
        for window in params["sma_windows"]:
            result[f"SMA_{window}"] = sma(close, window)

    if indicators.get("ema"):
        for window in params["ema_windows"]:
            result[f"EMA_{window}"] = ema(close, window)

    if indicators.get("rsi"):
        result["RSI"] = rsi(close, length=params["rsi_length"])

    if indicators.get("macd"):
        macd_df = macd(
            close,
            fast=params["macd_fast"],
            slow=params["macd_slow"],
            signal=params["macd_signal"],
        )
        result = pd.concat([result, macd_df], axis=1)

    if indicators.get("stochastic_oscillator") and {"High", "Low", "Close"}.issubset(
        result.columns
    ):
        stoch_df = stochastic_oscillator(
            result,
            k_period=params["stochastic_k_period"],
            d_period=params["stochastic_d_period"],
        )
        result = pd.concat([result, stoch_df], axis=1)

    if indicators.get("atr") and {"High", "Low", "Close"}.issubset(result.columns):
        result["ATR"] = atr(result, length=params["atr_length"])

    if indicators.get("bollinger_bands"):
        bb_df = bollinger_bands(
            close, length=params["bollinger_length"], num_std=params["bollinger_num_std"]
        )
        result = pd.concat([result, bb_df], axis=1)

    if indicators.get("obv") and "Volume" in result.columns:
        result["OBV"] = obv(result)

    if indicators.get("vwap") and {"High", "Low", "Close", "Volume"}.issubset(result.columns):
        result["VWAP"] = vwap(result, window=params["vwap_window"])

    lag_config = fe_config["lag_features"]
    if lag_config.get("enabled"):
        lag_df = generate_lag_features(
            data, columns=lag_config["columns"], lags=lag_config["lags"]
        )
        result = pd.concat([result, lag_df], axis=1)

    return result
