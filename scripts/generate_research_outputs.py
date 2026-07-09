"""
generate_research_outputs.py
=============================
Generates the project's final publication tables and figures. This
script is orchestration only: it calls the existing functions in
`src/` (data_loader, feature_engineering, feature_selection,
models_baseline, models_deep_learning, models_dl,
portfolio_optimization, statistical_tests, explainability) and does
not implement, modify, or extend any forecasting/portfolio algorithm.

Publication tables (saved to ../reports/tables/)
-------------------------------------------------
1. forecasting_metrics.csv / .md
   RMSE and MAPE for every forecasting model (ARIMA, XGBoost,
   Univariate LSTM, Multivariate LSTM, GRU, BiLSTM), plus
   Diebold-Mariano test p-values for the deep learning models against
   both baselines (src/statistical_tests.py).
2. portfolio_metrics.csv / .md
   Sharpe Ratio and Maximum Drawdown for every portfolio strategy
   (Equal Weight, Mean-Variance, Max Sharpe;
   src/portfolio_optimization.py).

Publication figures (saved to ../reports/figures/)
----------------------------------------------------
1. efficient_frontier.png
2. cumulative_returns.png
3. shap_summary_<model_name>.png
4. training_loss_curves.png

Two execution tiers
--------------------
- PORTFOLIO TIER (`run_portfolio_tier`): only needs numpy / pandas /
  scipy / matplotlib / scikit-learn plus this project's CSV data.
  Produces table 2 and figures 1-2 for real, using
  `src/portfolio_optimization.py` unchanged.
- MODELING TIER (`run_modeling_tier`): additionally needs
  tensorflow, xgboost, statsmodels and shap (see requirements.txt) to
  actually fit ARIMA / XGBoost / LSTM / GRU / BiLSTM models. Produces
  table 1 and figures 3-4. If these packages are not importable, this
  tier raises `ModelingDependenciesMissing` with a clear message
  instead of fabricating numbers.

Because the model-predicted returns needed by the portfolio strategies
(Mean-Variance / Max Sharpe) are themselves produced by the modeling
tier, `run_portfolio_tier` falls back to a simple, clearly-labeled
naive expected-return estimate - a trailing rolling mean of realized
returns - whenever trained model predictions are not supplied. This is
NOT a new forecasting algorithm; it is only a placeholder input to the
already-existing `rolling_backtest` function, used so the portfolio
tables/figures can still be produced end-to-end. Once modeling-tier
predictions are available, pass them in via `predicted_returns` to use
the project's actual GRU/BiLSTM forecasts instead.
"""

import os
import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
TABLES_DIR = os.path.join(REPORTS_DIR, "tables")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

import sys

sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import (  # noqa: E402
    chronological_train_val_test_split,
    load_portfolio_data,
    load_stock_csv,
)
from src.portfolio_optimization import (  # noqa: E402
    compute_performance_metrics,
    equal_weight_weights,
    max_sharpe_weights,
    mean_variance_weights,
    rolling_backtest,
)


class ModelingDependenciesMissing(RuntimeError):
    """Raised when tensorflow/xgboost/statsmodels/shap are not
    importable, so the modeling tier cannot fit any model."""


def _ensure_dirs() -> None:
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


def _save_table(df: pd.DataFrame, name: str) -> None:
    _ensure_dirs()
    csv_path = os.path.join(TABLES_DIR, f"{name}.csv")
    md_path = os.path.join(TABLES_DIR, f"{name}.md")
    df.to_csv(csv_path)
    with open(md_path, "w") as f:
        f.write(df.to_markdown())
    print(f"[saved] {csv_path}\n[saved] {md_path}")


def _apply_publication_style(ax, title: str, xlabel: str, ylabel: str) -> None:
    """Shared, minimal publication styling (labels/titles/legend/grid
    only) applied to every figure. Purely cosmetic - no algorithmic
    content."""
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ===========================================================================
# Table 2 + Figures 1-2: portfolio tier (real data, no ML dependencies)
# ===========================================================================


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error, skipping zero-valued targets to
    avoid division by zero. A simple reporting metric, not a model."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def load_real_returns(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Load the project's real multi-stock closing-price history and
    convert it to daily simple returns, via `src.data_loader`."""
    prices = load_portfolio_data(data_dir)
    returns = prices.pct_change().dropna(how="all").dropna()
    return returns


def naive_expected_returns(realized_returns: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """A trailing rolling-mean return estimate, used ONLY as a
    placeholder for `predicted_returns` when trained
    LSTM/GRU/BiLSTM forecasts are not supplied (see module docstring).
    Shifted by one day so no future information leaks into the
    'prediction' for day t.
    """
    return realized_returns.rolling(window=window, min_periods=window).mean().shift(1).dropna()


def run_portfolio_tier(
    predicted_returns: Optional[pd.DataFrame] = None,
    data_dir: str = DATA_DIR,
    rebalance_frequency: int = 21,
    cov_lookback: int = 60,
    trading_days: int = 252,
    n_frontier_portfolios: int = 5000,
    seed: int = 42,
) -> Dict:
    """Produce Table 2 (Sharpe / Max Drawdown per strategy) and
    Figures 1-2 (Efficient Frontier, Cumulative Return comparison)
    using only `src/portfolio_optimization.py`, unmodified.

    Parameters
    ----------
    predicted_returns : pd.DataFrame, optional
        Model-predicted returns (e.g. the project's trained
        GRU/BiLSTM output), indexed by date, one column per asset. If
        omitted, a naive trailing rolling-mean estimate is used
        instead (see module docstring) and every output is labeled
        accordingly.
    data_dir, rebalance_frequency, cov_lookback, trading_days :
        Forwarded to `src.data_loader.load_portfolio_data` /
        `src.portfolio_optimization.rolling_backtest`.
    n_frontier_portfolios : int
        Number of randomly-weighted portfolios sampled for the
        Efficient Frontier scatter plot.
    seed : int
        Random seed for the frontier's random weight sampling.

    Returns
    -------
    dict
        {"portfolio_metrics_table": pd.DataFrame,
         "backtests": dict, "used_naive_returns": bool}
    """
    _ensure_dirs()

    realized_returns = load_real_returns(data_dir)
    _, _, test_returns = chronological_train_val_test_split(realized_returns)

    used_naive_returns = predicted_returns is None
    if predicted_returns is None:
        warnings.warn(
            "No trained model predictions supplied; using a naive trailing "
            "rolling-mean return estimate in place of the project's "
            "GRU/BiLSTM forecasts for the Mean-Variance / Max Sharpe "
            "strategies. Pass `predicted_returns` once modeling-tier "
            "forecasts are available.",
            stacklevel=2,
        )
        predicted_returns = naive_expected_returns(realized_returns)

    common_index = test_returns.index.intersection(predicted_returns.index)
    test_returns = test_returns.loc[common_index]
    predicted_returns = predicted_returns.loc[common_index, test_returns.columns]

    strategies = ["equal_weight", "mean_variance", "max_sharpe"]
    backtests = {}
    metrics_rows = []
    for strategy in strategies:
        returns, weights = rolling_backtest(
            predicted_returns,
            test_returns,
            strategy=strategy,
            rebalance_frequency=rebalance_frequency,
            cov_lookback=cov_lookback,
        )
        metrics = compute_performance_metrics(returns, trading_days=trading_days)
        backtests[strategy] = {"returns": returns, "weights": weights, "metrics": metrics}
        metrics_rows.append(
            {
                "strategy": strategy,
                "sharpe_ratio": metrics["sharpe_ratio"],
                "max_drawdown": metrics["max_drawdown"],
                "annual_return": metrics["annual_return"],
                "volatility": metrics["volatility"],
            }
        )

    portfolio_metrics_table = pd.DataFrame(metrics_rows).set_index("strategy")
    note = (
        "expected returns: naive trailing rolling-mean (placeholder; see "
        "module docstring)" if used_naive_returns else "expected returns: trained model forecasts"
    )
    portfolio_metrics_table.attrs["note"] = note
    _save_table(portfolio_metrics_table, "portfolio_metrics")

    plot_efficient_frontier(
        realized_returns.loc[common_index],
        trading_days=trading_days,
        n_portfolios=n_frontier_portfolios,
        seed=seed,
        used_naive_returns=used_naive_returns,
    )
    plot_cumulative_returns(backtests, used_naive_returns=used_naive_returns)

    return {
        "portfolio_metrics_table": portfolio_metrics_table,
        "backtests": backtests,
        "used_naive_returns": used_naive_returns,
    }


def plot_efficient_frontier(
    returns: pd.DataFrame,
    trading_days: int = 252,
    n_portfolios: int = 5000,
    seed: int = 42,
    used_naive_returns: bool = False,
) -> str:
    """Publication figure 1: Efficient Frontier.

    Randomly-weighted long-only portfolios are sampled (Dirichlet
    weights, a standard visualization device - not a competing
    optimizer) and plotted as annualized volatility vs. annualized
    return, colored by Sharpe ratio. The Equal-Weight, Mean-Variance,
    and Max-Sharpe portfolios (`src.portfolio_optimization`'s actual
    optimizer functions) are overlaid as reference points.
    """
    import matplotlib.pyplot as plt

    mean_daily = returns.mean().to_numpy()
    cov_daily = returns.cov().to_numpy()
    n_assets = len(mean_daily)

    rng = np.random.RandomState(seed)
    weights = rng.dirichlet(np.ones(n_assets), size=n_portfolios)

    port_return = weights @ mean_daily * trading_days
    port_vol = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov_daily, weights)) * np.sqrt(
        trading_days
    )
    port_sharpe = np.divide(
        port_return, port_vol, out=np.full_like(port_return, np.nan), where=port_vol > 0
    )

    ew = equal_weight_weights(n_assets)
    mv = mean_variance_weights(mean_daily * trading_days, cov_daily * trading_days)
    ms = max_sharpe_weights(mean_daily * trading_days, cov_daily * trading_days)

    def _ret_vol(w):
        r = w @ mean_daily * trading_days
        v = float(np.sqrt(w @ (cov_daily * trading_days) @ w))
        return r, v

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        port_vol, port_return, c=port_sharpe, cmap="viridis", s=8, alpha=0.6, linewidths=0
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Sharpe Ratio", fontsize=10)

    for label, w, marker, color in [
        ("Equal Weight", ew, "o", "red"),
        ("Mean-Variance", mv, "^", "black"),
        ("Max Sharpe", ms, "*", "gold"),
    ]:
        r, v = _ret_vol(w)
        ax.scatter(v, r, marker=marker, color=color, s=180, edgecolors="black", label=label, zorder=5)

    title = "Efficient Frontier (Real KSE-100 Daily Returns)"
    if used_naive_returns:
        title += "\n(reference portfolios; annualized from historical mean/covariance)"
    _apply_publication_style(
        ax, title, "Annualized Volatility", "Annualized Expected Return"
    )
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "efficient_frontier.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_path}")
    return out_path


def plot_cumulative_returns(backtests: Dict, used_naive_returns: bool = False) -> str:
    """Publication figure 2: Cumulative Return comparison across the
    three portfolio strategies, from `rolling_backtest` output."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for strategy, result in backtests.items():
        cumulative = (1.0 + result["returns"]).cumprod()
        ax.plot(cumulative.index, cumulative.values, label=strategy.replace("_", " ").title(), linewidth=1.8)

    title = "Cumulative Return Comparison"
    if used_naive_returns:
        title += "\n(Mean-Variance/Max Sharpe use a naive rolling-mean return estimate)"
    _apply_publication_style(ax, title, "Date", "Cumulative Growth of $1")
    ax.legend(loc="best", frameon=True)
    fig.autofmt_xdate()
    fig.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "cumulative_returns.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_path}")
    return out_path


# ===========================================================================
# Table 1 + Figures 3-4: modeling tier (needs tensorflow / xgboost /
# statsmodels / shap - see requirements.txt)
# ===========================================================================


def _check_modeling_dependencies() -> None:
    missing = []
    for module_name in ("tensorflow", "xgboost", "statsmodels", "shap", "keras_tuner"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        raise ModelingDependenciesMissing(
            "The modeling tier (forecasting metrics table, SHAP summary "
            "plot, training loss curves) requires the following packages, "
            f"which are not importable in this environment: {missing}. "
            "Install everything in requirements.txt and re-run "
            "`run_modeling_tier(...)` to generate these outputs for real - "
            "no results are fabricated in their absence."
        )


def run_modeling_tier(
    data_dir: str = DATA_DIR,
    representative_ticker_csv: str = "Lucky Cement Stock Price History.csv",
    window: int = 20,
    epochs: int = 100,
    n_selected_features: int = 25,
    seed: Optional[int] = None,
) -> Dict:
    """Fit ARIMA, XGBoost, the Univariate/Multivariate LSTM
    (`src/models_deep_learning.py`) and the GRU/BiLSTM
    (`src/models_dl.py`) on one representative stock's daily Close
    series, evaluate RMSE/MAPE on the held-out test split, run the
    Diebold-Mariano test (`src/statistical_tests.py`) comparing each
    deep learning model against both baselines, and produce:
        - Table 1 (`reports/tables/forecasting_metrics.*`)
        - Figure 3, SHAP summary plot for the GRU model
          (`src/explainability.py`)
        - Figure 4, training loss curves for every trained model

    Raises
    ------
    ModelingDependenciesMissing
        If tensorflow / xgboost / statsmodels / shap / keras_tuner are
        not importable. No numbers or plots are fabricated in that
        case - install `requirements.txt` and re-run.
    """
    _check_modeling_dependencies()

    # Imported lazily so this module can still be imported (and the
    # portfolio tier used) in environments without these packages.
    from sklearn.preprocessing import MinMaxScaler

    from src.data_loader import create_windowed_dataset, scale_train_val_test
    from src.explainability import explain_model
    from src.feature_engineering import generate_features
    from src.feature_selection import select_features_rfe
    from src.models_baseline import (
        build_xgboost_model,
        evaluate_predictions,
        fit_arima_model,
        fit_xgboost_model,
        forecast_arima,
    )
    from src.models_deep_learning import build_multivariate_lstm, build_univariate_lstm, train_model
    from src.models_dl import build_bilstm_model, build_gru_model, default_callbacks
    from src.seed_utils import load_config, set_global_seed
    from src.statistical_tests import compare_dl_against_baselines

    if seed is None:
        seed = load_config()["random_seed"]
    set_global_seed(seed)

    ohlcv = load_stock_csv(os.path.join(data_dir, representative_ticker_csv))
    features = generate_features(ohlcv)
    features["target"] = features["Close"].shift(-1)
    features = features.replace([np.inf, -np.inf], np.nan).dropna()

    train_df, val_df, test_df = chronological_train_val_test_split(features)
    candidate_columns = [c for c in features.columns if c != "target"]

    _, selected_columns = select_features_rfe(
        train_df[candidate_columns], train_df["target"], n_features=n_selected_features, seed=seed
    )

    # --- ARIMA (univariate, on the Close series) ---
    arima_fit = fit_arima_model(train_df["Close"])
    arima_forecast = forecast_arima(arima_fit, steps=len(test_df))
    arima_metrics = evaluate_predictions(test_df["target"], arima_forecast)
    arima_mape = compute_mape(test_df["target"].to_numpy(), arima_forecast)

    # --- XGBoost (multivariate, on the RFE-selected features) ---
    xgb_model = fit_xgboost_model(
        build_xgboost_model(seed=seed), train_df[selected_columns], train_df["target"]
    )
    xgb_pred = xgb_model.predict(test_df[selected_columns])
    xgb_metrics = evaluate_predictions(test_df["target"], xgb_pred)
    xgb_mape = compute_mape(test_df["target"].to_numpy(), xgb_pred)

    # --- Univariate LSTM ---
    uni_scaler = MinMaxScaler()
    uni_train = uni_scaler.fit_transform(train_df[["Close"]])
    uni_val = uni_scaler.transform(val_df[["Close"]])
    uni_test = uni_scaler.transform(test_df[["Close"]])
    uni_full = np.concatenate([uni_train, uni_val, uni_test])
    x_uni, y_uni = create_windowed_dataset(uni_full, step=window)
    x_uni = x_uni.reshape(x_uni.shape[0], x_uni.shape[1], 1)
    n_test = len(test_df) - window - 1
    x_uni_train, y_uni_train = x_uni[:-n_test], y_uni[:-n_test]
    x_uni_test, y_uni_test = x_uni[-n_test:], y_uni[-n_test:]

    uni_model = build_univariate_lstm((x_uni.shape[1], 1))
    uni_history = train_model(uni_model, x_uni_train, y_uni_train, epochs=epochs)
    uni_pred_scaled = uni_model.predict(x_uni_test, verbose=0).ravel()
    uni_pred = uni_scaler.inverse_transform(uni_pred_scaled.reshape(-1, 1)).ravel()
    uni_true = uni_scaler.inverse_transform(y_uni_test.reshape(-1, 1)).ravel()
    uni_metrics = evaluate_predictions(uni_true, uni_pred)
    uni_mape = compute_mape(uni_true, uni_pred)

    # --- Multivariate LSTM / GRU / BiLSTM (shared windowed feature set) ---
    x_train_s, x_val_s, x_test_s, x_scaler = scale_train_val_test(
        train_df[selected_columns], val_df[selected_columns], test_df[selected_columns]
    )
    y_scaler = MinMaxScaler()
    y_train_s = y_scaler.fit_transform(train_df[["target"]])
    y_val_s = y_scaler.transform(val_df[["target"]])
    y_test_s = y_scaler.transform(test_df[["target"]])

    def _windows(x, y):
        xs, ys = [], []
        for i in range(len(x) - window):
            xs.append(x[i : i + window])
            ys.append(y[i + window, 0])
        return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)

    x_tr, y_tr = _windows(x_train_s, y_train_s)
    x_va, y_va = _windows(x_val_s, y_val_s)
    x_te, y_te = _windows(x_test_s, y_test_s)
    input_shape = (x_tr.shape[1], x_tr.shape[2])

    def _fit_and_eval(model, name, checkpoint_name):
        callbacks = default_callbacks(
            checkpoint_path=os.path.join(REPORTS_DIR, "checkpoints", f"{checkpoint_name}.weights.h5")
        )
        os.makedirs(os.path.dirname(callbacks[1].filepath), exist_ok=True)
        history = model.fit(
            x_tr, y_tr, validation_data=(x_va, y_va), epochs=epochs, batch_size=32,
            callbacks=callbacks, verbose=0,
        )
        pred_scaled = np.asarray(model.predict(x_te, verbose=0)).ravel()
        pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        true = y_scaler.inverse_transform(y_te.reshape(-1, 1)).ravel()
        metrics = evaluate_predictions(true, pred)
        mape = compute_mape(true, pred)
        return {"name": name, "history": history, "pred": pred, "true": true,
                "metrics": metrics, "mape": mape}

    multi_lstm_model = build_multivariate_lstm(input_shape)
    multi_lstm_result = _fit_and_eval(multi_lstm_model, "Multivariate LSTM", "multivariate_lstm")

    gru_model = build_gru_model(input_shape, seed=seed)
    gru_result = _fit_and_eval(gru_model, "GRU", "gru")

    bilstm_model = build_bilstm_model(input_shape, seed=seed)
    bilstm_result = _fit_and_eval(bilstm_model, "BiLSTM", "bilstm")

    # --- Table 1: RMSE / MAPE + DM test p-values ---
    rows = [
        {"model": "ARIMA", "rmse": arima_metrics["rmse"], "mape": arima_mape},
        {"model": "XGBoost", "rmse": xgb_metrics["rmse"], "mape": xgb_mape},
        {"model": "Univariate LSTM", "rmse": uni_metrics["rmse"], "mape": uni_mape},
        {"model": "Multivariate LSTM", "rmse": multi_lstm_result["metrics"]["rmse"],
         "mape": multi_lstm_result["mape"]},
        {"model": "GRU", "rmse": gru_result["metrics"]["rmse"], "mape": gru_result["mape"]},
        {"model": "BiLSTM", "rmse": bilstm_result["metrics"]["rmse"], "mape": bilstm_result["mape"]},
    ]
    metrics_table = pd.DataFrame(rows).set_index("model")

    dl_dm_results = {}
    for result in (multi_lstm_result, gru_result, bilstm_result):
        n = min(len(result["true"]), len(xgb_pred), len(arima_forecast))
        dm_table = compare_dl_against_baselines(
            result["true"][-n:], result["pred"][-n:], np.asarray(arima_forecast)[-n:], np.asarray(xgb_pred)[-n:],
            dl_model_name=result["name"],
        )
        dl_dm_results[result["name"]] = dm_table
    dm_pvalues = pd.concat(dl_dm_results.values())["p_value"]
    metrics_table = metrics_table.join(dm_pvalues.rename("dm_test_p_value"), how="left")

    _save_table(metrics_table, "forecasting_metrics")

    # --- Figure 3: SHAP summary plot (GRU) ---
    explain_model(
        gru_model,
        background_data=x_tr,
        x_explain=x_te,
        feature_names=selected_columns,
        model_name="gru",
        output_dir=FIGURES_DIR,
    )
    print(f"[saved] {os.path.join(FIGURES_DIR, 'gru_shap_summary.png')}")

    # --- Figure 4: training loss curves ---
    plot_training_loss_curves(
        {
            "Multivariate LSTM": multi_lstm_result["history"],
            "GRU": gru_result["history"],
            "BiLSTM": bilstm_result["history"],
        }
    )

    return {
        "forecasting_metrics_table": metrics_table,
        "dm_test_tables": dl_dm_results,
        "histories": {
            "Multivariate LSTM": multi_lstm_result["history"],
            "GRU": gru_result["history"],
            "BiLSTM": bilstm_result["history"],
        },
    }


def plot_training_loss_curves(histories: Dict) -> str:
    """Publication figure 4: training vs. validation loss curves for
    every trained deep learning model, one subplot per model, from
    each model's Keras `History` object.
    """
    import matplotlib.pyplot as plt

    n = len(histories)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), squeeze=False)
    axes = axes[0]

    for ax, (name, history) in zip(axes, histories.items()):
        h = history.history
        ax.plot(h["loss"], label="Train Loss", linewidth=1.8)
        if "val_loss" in h:
            ax.plot(h["val_loss"], label="Validation Loss", linewidth=1.8, linestyle="--")
        _apply_publication_style(ax, name, "Epoch", "Loss (MSE)")
        ax.legend(loc="best", frameon=True)

    fig.suptitle("Training Loss Curves", fontsize=14, fontweight="bold")
    fig.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "training_loss_curves.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_path}")
    return out_path


if __name__ == "__main__":
    print("=== Portfolio tier (Table 2, Figures 1-2) ===")
    run_portfolio_tier()

    print("\n=== Modeling tier (Table 1, Figures 3-4) ===")
    try:
        run_modeling_tier()
    except ModelingDependenciesMissing as exc:
        print(f"[skipped] {exc}")
