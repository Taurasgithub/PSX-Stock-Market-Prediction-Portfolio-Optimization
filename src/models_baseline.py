"""
models_baseline.py
===================
Classical baseline forecasting models: ARIMA and XGBoost.

These serve as the non-deep-learning baselines that future deep
learning models (see `src/models_deep_learning.py`) will be compared
against. Both baselines are evaluated with the same metrics - RMSE and
MAE - so results are directly comparable across model types.

No deep learning models are defined in this module.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor


# ---------------------------------------------------------------------------
# ARIMA baseline
# ---------------------------------------------------------------------------
def build_arima_model(train_series, order: Tuple[int, int, int] = (5, 1, 0)) -> ARIMA:
    """Construct an ARIMA model for a univariate training series.

    Parameters
    ----------
    train_series : array-like
        1-D chronologically ordered training series (e.g. Close prices
        or returns from the training split only).
    order : (int, int, int)
        The (p, d, q) ARIMA order. Defaults to (5, 1, 0), a common
        baseline configuration for daily financial time series.

    Returns
    -------
    ARIMA
        An unfitted statsmodels ARIMA model instance.
    """
    return ARIMA(train_series, order=order)


def fit_arima_model(train_series, order: Tuple[int, int, int] = (5, 1, 0)):
    """Build and fit an ARIMA model on the training series.

    Parameters
    ----------
    train_series : array-like
        1-D chronologically ordered training series.
    order : (int, int, int)
        The (p, d, q) ARIMA order (default (5, 1, 0)).

    Returns
    -------
    statsmodels ARIMAResults
        The fitted ARIMA results object.
    """
    model = build_arima_model(train_series, order=order)
    return model.fit()


def forecast_arima(fitted_model, steps: int) -> np.ndarray:
    """Produce an out-of-sample forecast from a fitted ARIMA model.

    Parameters
    ----------
    fitted_model : statsmodels ARIMAResults
        The result of `fit_arima_model`.
    steps : int
        Number of steps ahead to forecast (typically len(test_series)).

    Returns
    -------
    np.ndarray
        Forecasted values, length `steps`.
    """
    return np.asarray(fitted_model.forecast(steps=steps))


# ---------------------------------------------------------------------------
# XGBoost baseline
# ---------------------------------------------------------------------------
def build_xgboost_model(
    n_estimators: int = 200,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    seed: Optional[int] = None,
    **kwargs,
) -> XGBRegressor:
    """Construct an XGBoost regressor for baseline return/price forecasting.

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds (default 200).
    max_depth : int
        Maximum tree depth (default 5).
    learning_rate : float
        Boosting learning rate (default 0.05).
    seed : int, optional
        Random seed, passed through as `random_state` for reproducibility.
    **kwargs
        Additional keyword arguments forwarded to `XGBRegressor`.

    Returns
    -------
    XGBRegressor
        An unfitted XGBoost regressor.
    """
    return XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=seed,
        objective="reg:squarederror",
        **kwargs,
    )


def fit_xgboost_model(model: XGBRegressor, x_train, y_train) -> XGBRegressor:
    """Fit an XGBoost regressor on training features/targets.

    Parameters
    ----------
    model : XGBRegressor
        An unfitted model from `build_xgboost_model`.
    x_train : array-like
        Training feature matrix.
    y_train : array-like
        Training targets.

    Returns
    -------
    XGBRegressor
        The fitted model (same object, fitted in place).
    """
    model.fit(x_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Shared evaluation utilities
# ---------------------------------------------------------------------------
def evaluate_predictions(y_true, y_pred) -> dict:
    """Compute RMSE and MAE for a set of predictions.

    Parameters
    ----------
    y_true : array-like
        Ground-truth values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    dict
        {"rmse": float, "mae": float}
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    return {"rmse": rmse, "mae": mae}


def compare_baseline_models(results: dict) -> pd.DataFrame:
    """Assemble per-model RMSE/MAE results into a single comparison table.

    Intended to hold the ARIMA and XGBoost baseline results side by
    side, in a format that future deep learning model results (e.g.
    from `src/models_deep_learning.py`) can be appended to for
    comparison.

    Parameters
    ----------
    results : dict
        Mapping of model name -> {"rmse": float, "mae": float}, e.g.
        {"ARIMA": {"rmse": 0.01, "mae": 0.008},
         "XGBoost": {"rmse": 0.009, "mae": 0.007}}

    Returns
    -------
    pd.DataFrame
        Indexed by model name, with "rmse" and "mae" columns.
    """
    return pd.DataFrame.from_dict(results, orient="index")[["rmse", "mae"]]
