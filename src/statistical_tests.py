"""
statistical_tests.py
=====================
Statistical significance testing for the PSX Stock Market Prediction
project.

Implements exactly four things:

    1. Diebold-Mariano (DM) test, comparing a deep learning model's
       forecast errors against the classical ARIMA and XGBoost
       baselines built in `src/models_baseline.py`
       (`diebold_mariano_test`, `compare_dl_against_baselines`).

    2. A paired significance test on two return series - the paired
       t-test (`scipy.stats.ttest_rel`) or the Wilcoxon Signed-Rank
       test (`scipy.stats.wilcoxon`) - intended for comparing two
       portfolio return series such as those produced by
       `src/portfolio_optimization.py`'s `rolling_backtest`
       (`paired_return_test`, `compare_returns`).

    3. An ablation-study runner that retrains the project's deep
       learning pipeline (`src/feature_engineering.py`,
       `src/feature_selection.py`, `src/models_dl.py`) under three
       ablated configurations:
           - without technical indicators
           - without feature selection
           - without hyperparameter tuning
       (`run_ablation_study`, `run_single_variant`).

    4. A comparison step that statistically compares every ablation
       case against the complete ("full") model, using both the DM
       test (on squared forecast errors) and the paired t-test /
       Wilcoxon test (on the two models' absolute errors)
       (`compare_ablation_to_full`, built into `run_ablation_study`).

Scope
-----
This module only implements statistical hypothesis testing and the
ablation-study orchestration needed to run/compare those tests. It
does not generate publication figures or plots of any kind - results
are returned as plain dicts / pandas DataFrames only (see
`src/explainability.py` for SHAP-based visualizations).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.data_loader import chronological_train_val_test_split
from src.feature_engineering import generate_features
from src.feature_selection import select_features_rfe
from src.models_dl import build_gru_model, default_callbacks, get_tuner
from src.seed_utils import load_config, set_global_seed

# ===========================================================================
# 1. Diebold-Mariano (DM) test
# ===========================================================================


def _loss_differential(
    y_true: np.ndarray, pred1: np.ndarray, pred2: np.ndarray, loss: str = "MSE"
) -> np.ndarray:
    """Per-observation loss differential d_t = g(e1_t) - g(e2_t) used by
    the Diebold-Mariano test, where e1/e2 are the two models' forecast
    errors and g is the chosen loss function.

    Parameters
    ----------
    y_true : array-like
        Ground-truth values.
    pred1, pred2 : array-like
        The two models' predictions, aligned with `y_true`.
    loss : {"MSE", "MAE"}
        Loss function applied to each model's forecast error before
        differencing.

    Returns
    -------
    np.ndarray
        The loss differential series, length len(y_true).
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    pred1 = np.asarray(pred1, dtype=float).ravel()
    pred2 = np.asarray(pred2, dtype=float).ravel()

    if not (len(y_true) == len(pred1) == len(pred2)):
        raise ValueError("y_true, pred1 and pred2 must all be the same length.")

    e1 = y_true - pred1
    e2 = y_true - pred2

    if loss == "MSE":
        g1, g2 = e1**2, e2**2
    elif loss == "MAE":
        g1, g2 = np.abs(e1), np.abs(e2)
    else:
        raise ValueError(f"Unknown loss '{loss}'; expected 'MSE' or 'MAE'.")

    return g1 - g2


def diebold_mariano_test(
    y_true: np.ndarray,
    pred1: np.ndarray,
    pred2: np.ndarray,
    h: int = 1,
    loss: str = "MSE",
    harvey_correction: bool = True,
) -> Dict[str, float]:
    """Diebold-Mariano test of equal predictive accuracy between two
    forecasts, `pred1` and `pred2`, of the same series `y_true`.

    The null hypothesis is that the two forecasts have equal expected
    loss (`E[d_t] = 0`, where `d_t` is the loss differential). A
    negative DM statistic (with a small p-value) indicates `pred1` has
    lower loss than `pred2` (i.e. `pred1` is significantly more
    accurate); a positive statistic indicates the reverse.

    The long-run variance of the loss differential is estimated using
    the Newey-West-style autocovariance sum over lags `0..h-1` (the DM
    test's standard heteroskedasticity/autocorrelation-consistent
    variance estimator for an `h`-step-ahead forecast), and the
    small-sample correction of Harvey, Leybourne and Newbold (1997) is
    applied by default, comparing the corrected statistic to a
    Student's t distribution with `n - 1` degrees of freedom instead of
    the standard normal.

    Parameters
    ----------
    y_true : array-like
        Ground-truth values.
    pred1, pred2 : array-like
        The two models' predictions, aligned with `y_true` and with
        each other (same dates/order).
    h : int
        Forecast horizon (number of steps ahead). Used to determine
        how many autocovariance lags are included in the long-run
        variance estimate, and in the Harvey small-sample correction.
        Use `h=1` for one-step-ahead forecasts (the typical case for
        this project's next-day return/price predictions).
    loss : {"MSE", "MAE"}
        Loss function used to score each model's forecast errors.
    harvey_correction : bool
        If True (default), apply the Harvey, Leybourne & Newbold
        (1997) small-sample correction and use a Student's t reference
        distribution; if False, use the asymptotic standard normal
        distribution.

    Returns
    -------
    dict
        {
            "dm_statistic": float,
            "p_value": float,
            "mean_loss_differential": float,  # mean(g(e1) - g(e2))
            "h": int,
            "loss": str,
            "n_obs": int,
        }
        A `mean_loss_differential` < 0 means `pred1` had lower average
        loss than `pred2` over the sample.
    """
    d = _loss_differential(y_true, pred1, pred2, loss=loss)
    n = d.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 observations to run the DM test.")

    d_mean = float(np.mean(d))

    # Long-run variance: gamma_0 + 2 * sum_{k=1}^{h-1} gamma_k
    d_centered = d - d_mean
    gamma0 = float(np.dot(d_centered, d_centered) / n)
    long_run_var = gamma0
    for lag in range(1, h):
        if lag >= n:
            break
        gamma_k = float(np.dot(d_centered[:-lag], d_centered[lag:]) / n)
        long_run_var += 2.0 * gamma_k

    var_d_mean = long_run_var / n

    if var_d_mean <= 0 or np.isnan(var_d_mean):
        return {
            "dm_statistic": float("nan"),
            "p_value": float("nan"),
            "mean_loss_differential": d_mean,
            "h": h,
            "loss": loss,
            "n_obs": n,
        }

    dm_stat = d_mean / np.sqrt(var_d_mean)

    if harvey_correction:
        correction = np.sqrt(
            max((n + 1 - 2 * h + h * (h - 1) / n) / n, 0.0)
        )
        dm_stat = dm_stat * correction
        p_value = 2.0 * (1.0 - stats.t.cdf(np.abs(dm_stat), df=n - 1))
    else:
        p_value = 2.0 * (1.0 - stats.norm.cdf(np.abs(dm_stat)))

    return {
        "dm_statistic": float(dm_stat),
        "p_value": float(p_value),
        "mean_loss_differential": d_mean,
        "h": h,
        "loss": loss,
        "n_obs": n,
    }


def compare_dl_against_baselines(
    y_true: np.ndarray,
    dl_pred: np.ndarray,
    arima_pred: np.ndarray,
    xgb_pred: np.ndarray,
    dl_model_name: str = "DeepLearningModel",
    h: int = 1,
    loss: str = "MSE",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run the Diebold-Mariano test comparing a deep learning model's
    forecasts against both the ARIMA and XGBoost baselines from
    `src/models_baseline.py`, on the same held-out (test) targets.

    Parameters
    ----------
    y_true : array-like
        Ground-truth test-set values.
    dl_pred : array-like
        Deep learning model's predictions (e.g. from
        `src/models_dl.py` or `src/models_deep_learning.py`), aligned
        with `y_true`.
    arima_pred : array-like
        ARIMA baseline predictions (e.g. from
        `src.models_baseline.forecast_arima`), aligned with `y_true`.
    xgb_pred : array-like
        XGBoost baseline predictions (e.g. from a fitted
        `src.models_baseline.build_xgboost_model`), aligned with
        `y_true`.
    dl_model_name : str
        Label used for the deep learning model in the returned table.
    h, loss :
        Forwarded to `diebold_mariano_test`.
    alpha : float
        Significance level used to flag results as significant in the
        returned table.

    Returns
    -------
    pd.DataFrame
        Indexed by comparison (e.g. "DeepLearningModel vs ARIMA"),
        with columns "dm_statistic", "p_value",
        "mean_loss_differential", "significant", "more_accurate_model".
    """
    comparisons = {
        f"{dl_model_name} vs ARIMA": diebold_mariano_test(
            y_true, dl_pred, arima_pred, h=h, loss=loss
        ),
        f"{dl_model_name} vs XGBoost": diebold_mariano_test(
            y_true, dl_pred, xgb_pred, h=h, loss=loss
        ),
    }

    rows = []
    for name, result in comparisons.items():
        significant = bool(result["p_value"] < alpha) if not np.isnan(result["p_value"]) else False
        if not significant or np.isnan(result["mean_loss_differential"]):
            winner = "no significant difference"
        elif result["mean_loss_differential"] < 0:
            winner = dl_model_name
        else:
            winner = name.split(" vs ")[1]

        rows.append(
            {
                "comparison": name,
                "dm_statistic": result["dm_statistic"],
                "p_value": result["p_value"],
                "mean_loss_differential": result["mean_loss_differential"],
                "significant": significant,
                "more_accurate_model": winner,
            }
        )

    return pd.DataFrame(rows).set_index("comparison")


# ===========================================================================
# 2. Paired t-test / Wilcoxon Signed-Rank test on portfolio returns
# ===========================================================================


def paired_return_test(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    test: str = "ttest",
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """Paired significance test between two aligned portfolio return
    series (e.g. two strategies' daily returns from
    `src.portfolio_optimization.rolling_backtest`, over the same
    dates).

    Parameters
    ----------
    returns_a, returns_b : array-like
        Two return series of equal length, paired observation-by-
        observation (e.g. the same trading days for two different
        models/strategies).
    test : {"ttest", "wilcoxon"}
        "ttest" runs a paired Student's t-test (`scipy.stats.ttest_rel`),
        which assumes the return differences are approximately
        normally distributed. "wilcoxon" runs the non-parametric
        Wilcoxon Signed-Rank test (`scipy.stats.wilcoxon`), which
        makes no normality assumption and is more appropriate when
        return differences are skewed or heavy-tailed.
    alternative : {"two-sided", "less", "greater"}
        Alternative hypothesis, forwarded to the underlying scipy
        test. "greater" tests whether `returns_a` has systematically
        higher paired values than `returns_b`.

    Returns
    -------
    dict
        {
            "test": str,
            "statistic": float,
            "p_value": float,
            "mean_diff": float,   # mean(returns_a - returns_b)
            "n_obs": int,
        }
    """
    a = np.asarray(returns_a, dtype=float).ravel()
    b = np.asarray(returns_b, dtype=float).ravel()

    if len(a) != len(b):
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]

    diff = a - b

    if test == "ttest":
        result = stats.ttest_rel(a, b, alternative=alternative)
        statistic, p_value = float(result.statistic), float(result.pvalue)
    elif test == "wilcoxon":
        if np.allclose(diff, 0.0):
            statistic, p_value = 0.0, 1.0
        else:
            result = stats.wilcoxon(a, b, alternative=alternative, zero_method="wilcox")
            statistic, p_value = float(result.statistic), float(result.pvalue)
    else:
        raise ValueError(f"Unknown test '{test}'; expected 'ttest' or 'wilcoxon'.")

    return {
        "test": test,
        "statistic": statistic,
        "p_value": p_value,
        "mean_diff": float(np.mean(diff)),
        "n_obs": int(len(diff)),
    }


def compare_returns(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    label_a: str = "Model A",
    label_b: str = "Model B",
    alpha: float = 0.05,
    normality_alpha: float = 0.05,
) -> Dict:
    """Compare two paired portfolio return series, automatically
    choosing between the paired t-test and the Wilcoxon Signed-Rank
    test based on a Shapiro-Wilk normality check of the paired
    differences.

    The Shapiro-Wilk test is run on `returns_a - returns_b`. If it
    fails to reject normality (p >= `normality_alpha`), the paired
    t-test is used; otherwise, the non-parametric Wilcoxon
    Signed-Rank test is used instead, since the t-test's normality
    assumption would be violated.

    Parameters
    ----------
    returns_a, returns_b : array-like
        Two paired return series of equal length.
    label_a, label_b : str
        Labels for the two series, used in the returned summary.
    alpha : float
        Significance level for the chosen paired test.
    normality_alpha : float
        Significance level for the Shapiro-Wilk normality pre-test.

    Returns
    -------
    dict
        {
            "normality_test": {"statistic": float, "p_value": float},
            "test_used": "ttest" | "wilcoxon",
            "result": dict,          # output of `paired_return_test`
            "significant": bool,
            "better_performing": str,
        }
    """
    a = np.asarray(returns_a, dtype=float).ravel()
    b = np.asarray(returns_b, dtype=float).ravel()
    n = min(len(a), len(b))
    diff = a[:n] - b[:n]

    if n < 3:
        normality_stat, normality_p = float("nan"), float("nan")
        test_used = "wilcoxon"
    else:
        shapiro = stats.shapiro(diff)
        normality_stat, normality_p = float(shapiro.statistic), float(shapiro.pvalue)
        test_used = "ttest" if normality_p >= normality_alpha else "wilcoxon"

    result = paired_return_test(a, b, test=test_used)
    significant = bool(result["p_value"] < alpha) if not np.isnan(result["p_value"]) else False

    if not significant:
        better = "no significant difference"
    else:
        better = label_a if result["mean_diff"] > 0 else label_b

    return {
        "normality_test": {"statistic": normality_stat, "p_value": normality_p},
        "test_used": test_used,
        "result": result,
        "significant": significant,
        "better_performing": better,
    }


# ===========================================================================
# 3 & 4. Ablation studies, each compared against the complete model
# ===========================================================================

ABLATION_VARIANTS = (
    "full",
    "no_technical_indicators",
    "no_feature_selection",
    "no_hyperparameter_tuning",
)


@dataclass
class AblationResult:
    """Container for a single ablation variant's trained-model output."""

    variant: str
    y_true: np.ndarray
    y_pred: np.ndarray
    metrics: Dict[str, float]
    n_candidate_features: int
    n_selected_features: int
    used_feature_selection: bool
    used_hyperparameter_tuning: bool
    used_technical_indicators: bool


def _make_sequences(x: np.ndarray, y: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build sliding-window (X, y) sequences for a multivariate
    recurrent model out of a 2-D feature matrix `x` and 1-D target
    `y`: row `i + window` of `y` is predicted from the `window` rows
    of `x` immediately preceding (and including up to) it.
    """
    xs, ys = [], []
    for i in range(len(x) - window):
        xs.append(x[i : i + window])
        ys.append(y[i + window])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def _prepare_ablation_features(
    ohlcv: pd.DataFrame,
    use_technical_indicators: bool,
    feature_config: Optional[dict] = None,
) -> pd.DataFrame:
    """Run `src.feature_engineering.generate_features` and append the
    next-day Close return prediction target, optionally with every
    technical indicator group disabled (lag features are always kept,
    since they are not "technical indicators").
    """
    config = {"feature_engineering": {} if feature_config is None else dict(feature_config)}
    fe_section = config["feature_engineering"]
    fe_section.setdefault("indicators", {})

    if not use_technical_indicators:
        fe_section["indicators"] = {
            key: False
            for key in [
                "sma",
                "ema",
                "rsi",
                "macd",
                "stochastic_oscillator",
                "atr",
                "bollinger_bands",
                "obv",
                "vwap",
            ]
        }

    features = generate_features(ohlcv, config=config)
    features["target"] = features["Close"].pct_change().shift(-1)
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    return features


def run_single_variant(
    ohlcv: pd.DataFrame,
    variant: str,
    n_selected_features: int = 25,
    window: int = 20,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    epochs: int = 50,
    batch_size: int = 32,
    tuner_max_trials: int = 10,
    seed: Optional[int] = None,
) -> AblationResult:
    """Train and evaluate one variant of the deep learning pipeline
    (a stacked GRU regressor, `src.models_dl.build_gru_model` /
    `get_tuner`), predicting next-day Close returns.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        Cleaned OHLCV dataframe for a single stock (e.g. from
        `src.data_loader.fetch_stock`).
    variant : str
        One of `ABLATION_VARIANTS`:
            - "full": technical indicators enabled, RFE feature
              selection applied, hyperparameter tuning run.
            - "no_technical_indicators": technical indicator groups
              disabled (lag features kept); RFE + tuning still run.
            - "no_feature_selection": RFE is skipped and every
              candidate engineered feature is used as model input;
              indicators + tuning still run.
            - "no_hyperparameter_tuning": the hyperparameter search is
              skipped and the model is built with
              `build_gru_model`'s fixed default architecture instead;
              indicators + RFE still run.
    n_selected_features : int
        Number of features RFE keeps, when feature selection is used.
    window : int
        Number of look-back timesteps per input sequence.
    train_frac, val_frac, test_frac : float
        Chronological train/validation/test split fractions.
    epochs, batch_size : int
        Training configuration for the non-tuned variants.
    tuner_max_trials : int
        `max_trials` passed to `src.models_dl.get_tuner` for the
        tuned variants.
    seed : int, optional
        Random seed. Defaults to the project's `config.yaml`
        `random_seed` via `src.seed_utils.load_config`.

    Returns
    -------
    AblationResult
    """
    if variant not in ABLATION_VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'; expected one of {ABLATION_VARIANTS}.")

    if seed is None:
        seed = load_config()["random_seed"]
    set_global_seed(seed)

    use_indicators = variant != "no_technical_indicators"
    use_feature_selection = variant != "no_feature_selection"
    use_tuning = variant != "no_hyperparameter_tuning"

    features = _prepare_ablation_features(ohlcv, use_technical_indicators=use_indicators)
    candidate_columns = [c for c in features.columns if c != "target"]

    train_df, val_df, test_df = chronological_train_val_test_split(
        features, train_frac=train_frac, val_frac=val_frac, test_frac=test_frac
    )

    if use_feature_selection:
        _, selected_columns = select_features_rfe(
            train_df[candidate_columns],
            train_df["target"],
            n_features=min(n_selected_features, len(candidate_columns)),
            seed=seed,
        )
    else:
        selected_columns = candidate_columns

    from sklearn.preprocessing import MinMaxScaler

    x_scaler = MinMaxScaler(feature_range=(0, 1))
    x_scaler.fit(train_df[selected_columns])

    y_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler.fit(train_df[["target"]])

    def _to_sequences(split_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        x_scaled = x_scaler.transform(split_df[selected_columns])
        y_scaled = y_scaler.transform(split_df[["target"]]).ravel()
        return _make_sequences(x_scaled, y_scaled, window)

    x_train, y_train = _to_sequences(train_df)
    x_val, y_val = _to_sequences(val_df)
    x_test, y_test = _to_sequences(test_df)

    input_shape = (x_train.shape[1], x_train.shape[2])

    if use_tuning:
        tuner = get_tuner(
            "gru",
            input_shape,
            max_trials=tuner_max_trials,
            directory=f"tuner_results_ablation/{variant}",
            project_name=variant,
            seed=seed,
        )
        tuner.search(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            verbose=0,
        )
        model = tuner.get_best_models(num_models=1)[0]
    else:
        model = build_gru_model(input_shape, seed=seed)
        callbacks = default_callbacks(
            checkpoint_path=f"/tmp/ablation_{variant}_best.weights.h5", patience=10
        )
        import os

        os.makedirs(os.path.dirname(callbacks[1].filepath), exist_ok=True)
        model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            shuffle=True,
            verbose=0,
        )

    y_pred_scaled = np.asarray(model.predict(x_test, verbose=0)).ravel()
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_true = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    return AblationResult(
        variant=variant,
        y_true=y_true,
        y_pred=y_pred,
        metrics={"rmse": rmse, "mae": mae},
        n_candidate_features=len(candidate_columns),
        n_selected_features=len(selected_columns),
        used_feature_selection=use_feature_selection,
        used_hyperparameter_tuning=use_tuning,
        used_technical_indicators=use_indicators,
    )


def compare_ablation_to_full(
    full_result: AblationResult,
    ablated_result: AblationResult,
    h: int = 1,
    loss: str = "MSE",
    alpha: float = 0.05,
) -> Dict:
    """Statistically compare one ablated variant's predictions against
    the complete ("full") model's predictions, on the same test-set
    targets, using both:
        - the Diebold-Mariano test on the two models' squared forecast
          errors, and
        - the paired t-test / Wilcoxon Signed-Rank test (auto-selected
          via `compare_returns`) on the two models' absolute forecast
          errors.

    Parameters
    ----------
    full_result : AblationResult
        Result of `run_single_variant(..., variant="full")`.
    ablated_result : AblationResult
        Result of `run_single_variant(..., variant=<ablation>)`.
    h, loss :
        Forwarded to `diebold_mariano_test`.
    alpha : float
        Significance level used by both tests.

    Returns
    -------
    dict
        {
            "variant": str,
            "dm_test": dict,             # diebold_mariano_test output
            "paired_test": dict,         # compare_returns output
            "full_metrics": dict,
            "ablated_metrics": dict,
        }
    """
    n = min(len(full_result.y_true), len(ablated_result.y_true))
    y_true = full_result.y_true[-n:]
    full_pred = full_result.y_pred[-n:]
    ablated_pred = ablated_result.y_pred[-n:]

    dm_result = diebold_mariano_test(y_true, full_pred, ablated_pred, h=h, loss=loss)

    full_abs_err = np.abs(y_true - full_pred)
    ablated_abs_err = np.abs(y_true - ablated_pred)
    paired_result = compare_returns(
        full_abs_err,
        ablated_abs_err,
        label_a="full (lower error)",
        label_b=ablated_result.variant,
        alpha=alpha,
    )

    return {
        "variant": ablated_result.variant,
        "dm_test": dm_result,
        "paired_test": paired_result,
        "full_metrics": full_result.metrics,
        "ablated_metrics": ablated_result.metrics,
    }


def run_ablation_study(
    ohlcv: pd.DataFrame,
    variants: Sequence[str] = (
        "no_technical_indicators",
        "no_feature_selection",
        "no_hyperparameter_tuning",
    ),
    n_selected_features: int = 25,
    window: int = 20,
    epochs: int = 50,
    batch_size: int = 32,
    tuner_max_trials: int = 10,
    seed: Optional[int] = None,
) -> Dict:
    """Run the complete ("full") deep learning pipeline plus each
    requested ablation variant, then statistically compare every
    ablation case against the complete model.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        Cleaned OHLCV dataframe for a single stock (e.g. from
        `src.data_loader.fetch_stock`).
    variants : sequence of str
        Which ablations to run, from `ABLATION_VARIANTS` (excluding
        "full", which is always run as the baseline for comparison).
    n_selected_features, window, epochs, batch_size, tuner_max_trials, seed :
        Forwarded to `run_single_variant`.

    Returns
    -------
    dict
        {
            "results": {variant: AblationResult, ...},   # includes "full"
            "comparisons": {variant: dict, ...},          # compare_ablation_to_full output, per ablation
            "summary_table": pd.DataFrame,                # RMSE/MAE + DM/paired-test p-values, one row per ablation
        }
    """
    common_kwargs = dict(
        n_selected_features=n_selected_features,
        window=window,
        epochs=epochs,
        batch_size=batch_size,
        tuner_max_trials=tuner_max_trials,
        seed=seed,
    )

    results: Dict[str, AblationResult] = {
        "full": run_single_variant(ohlcv, "full", **common_kwargs)
    }
    comparisons: Dict[str, Dict] = {}
    summary_rows: List[Dict] = []

    for variant in variants:
        if variant == "full":
            continue
        results[variant] = run_single_variant(ohlcv, variant, **common_kwargs)
        comparison = compare_ablation_to_full(results["full"], results[variant])
        comparisons[variant] = comparison

        summary_rows.append(
            {
                "variant": variant,
                "full_rmse": comparison["full_metrics"]["rmse"],
                "ablated_rmse": comparison["ablated_metrics"]["rmse"],
                "full_mae": comparison["full_metrics"]["mae"],
                "ablated_mae": comparison["ablated_metrics"]["mae"],
                "dm_statistic": comparison["dm_test"]["dm_statistic"],
                "dm_p_value": comparison["dm_test"]["p_value"],
                "paired_test_used": comparison["paired_test"]["test_used"],
                "paired_p_value": comparison["paired_test"]["result"]["p_value"],
                "significantly_worse_than_full": (
                    comparison["dm_test"]["p_value"] < 0.05
                    and comparison["dm_test"]["mean_loss_differential"] < 0
                )
                if not np.isnan(comparison["dm_test"]["p_value"])
                else False,
            }
        )

    summary_table = pd.DataFrame(summary_rows).set_index("variant")

    return {
        "results": results,
        "comparisons": comparisons,
        "summary_table": summary_table,
    }
