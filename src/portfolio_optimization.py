"""
portfolio_optimization.py
==========================
Portfolio construction strategies, a rolling backtesting pipeline with
periodic rebalancing, and performance-evaluation metrics for the PSX
Stock Market Prediction project.

Implements exactly three portfolio strategies:
    1. Mean-Variance (Markowitz) optimization, using model-predicted
       returns as the expected-return input (`mean_variance_weights`).
    2. Equal Weight allocation (`equal_weight_weights`).
    3. Maximum Sharpe Ratio optimization, using model-predicted
       returns and a covariance matrix estimated from the test set's
       realized returns (`max_sharpe_weights`).

`rolling_backtest` runs any of the three strategies over a test
period, rebalancing every `rebalance_frequency` trading days: at each
rebalance date, the covariance matrix is re-estimated from a trailing
window of realized (test-set) returns already observed as of that
date (no look-ahead), expected returns come from the model's
predictions for that date, and the resulting weights are held fixed
until the next rebalance.

`compute_performance_metrics` calculates, for a return series:
    Annual Return, Volatility, Sharpe Ratio, Sortino Ratio,
    Calmar Ratio, Maximum Drawdown (MDD), and Value at Risk (VaR).

Scope
-----
This module covers portfolio construction, backtesting, and
descriptive performance metrics only. It does not implement
statistical significance testing (e.g. hypothesis tests comparing
strategies, bootstrapped confidence intervals, Diebold-Mariano tests,
etc.).
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Portfolio strategies
# ---------------------------------------------------------------------------


def equal_weight_weights(n_assets: int) -> np.ndarray:
    """Equal Weight strategy: allocate 1/N to each of the N assets.

    Parameters
    ----------
    n_assets : int

    Returns
    -------
    np.ndarray
        Length-`n_assets` array of weights, each equal to `1/n_assets`.
    """
    return np.full(n_assets, 1.0 / n_assets)


def mean_variance_weights(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_aversion: float = 1.0,
    allow_short: bool = False,
) -> np.ndarray:
    """Mean-Variance (Markowitz) strategy: choose weights that
    maximize `w'*expected_returns - (risk_aversion / 2) * w'*cov*w`,
    subject to weights summing to 1.

    Parameters
    ----------
    expected_returns : np.ndarray
        Length-N array of predicted (model) expected returns, one per
        asset.
    cov_matrix : np.ndarray
        N x N covariance matrix of asset returns.
    risk_aversion : float
        Higher values penalize variance more heavily, producing more
        conservative (lower-risk) portfolios.
    allow_short : bool
        If False (default), weights are constrained to [0, 1]
        (long-only). If True, weights may be negative.

    Returns
    -------
    np.ndarray
        Length-N array of optimized portfolio weights, summing to 1.
    """
    n = len(expected_returns)

    def objective(w: np.ndarray) -> float:
        return 0.5 * risk_aversion * w @ cov_matrix @ w - w @ expected_returns

    weights = _solve_weights(objective, n, allow_short)
    return weights


def max_sharpe_weights(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.0,
    allow_short: bool = False,
) -> np.ndarray:
    """Maximum Sharpe Ratio strategy: choose weights that maximize
    `(w'*expected_returns - risk_free_rate) / sqrt(w'*cov*w)`, subject
    to weights summing to 1.

    Parameters
    ----------
    expected_returns : np.ndarray
        Length-N array of predicted (model) expected returns, one per
        asset.
    cov_matrix : np.ndarray
        N x N covariance matrix of asset returns, estimated from the
        test set (see `rolling_backtest`).
    risk_free_rate : float
        Risk-free rate, in the same units (e.g. annualized) as
        `expected_returns`.
    allow_short : bool
        If False (default), weights are constrained to [0, 1]
        (long-only). If True, weights may be negative.

    Returns
    -------
    np.ndarray
        Length-N array of optimized portfolio weights, summing to 1.
    """
    n = len(expected_returns)

    def objective(w: np.ndarray) -> float:
        port_return = w @ expected_returns
        port_vol = np.sqrt(w @ cov_matrix @ w)
        if port_vol <= 1e-12:
            return 1e6
        return -(port_return - risk_free_rate) / port_vol

    weights = _solve_weights(objective, n, allow_short)
    return weights


def _solve_weights(objective, n_assets: int, allow_short: bool) -> np.ndarray:
    """Shared SLSQP solver used by `mean_variance_weights` and
    `max_sharpe_weights`: minimizes `objective(w)` subject to
    `sum(w) == 1`, starting from an equal-weight guess.

    Falls back to equal weights (with no error raised) if the
    optimizer fails to converge, since a rolling backtest should not
    halt on a single failed rebalance.
    """
    w0 = equal_weight_weights(n_assets)
    bounds = None if allow_short else [(0.0, 1.0)] * n_assets
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    result = minimize(
        objective, w0, method="SLSQP", bounds=bounds, constraints=constraints
    )

    if not result.success:
        return w0
    return result.x


_STRATEGIES = {"equal_weight", "mean_variance", "max_sharpe"}


def get_strategy_weights(
    strategy: str,
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.0,
    risk_aversion: float = 1.0,
    allow_short: bool = False,
) -> np.ndarray:
    """Dispatch to the requested strategy's weight-computation
    function.

    Parameters
    ----------
    strategy : {"equal_weight", "mean_variance", "max_sharpe"}
    expected_returns : np.ndarray
        Ignored by "equal_weight".
    cov_matrix : np.ndarray
        Ignored by "equal_weight".
    risk_free_rate, risk_aversion, allow_short :
        Forwarded to the underlying strategy function.

    Returns
    -------
    np.ndarray
    """
    if strategy == "equal_weight":
        return equal_weight_weights(len(expected_returns))
    if strategy == "mean_variance":
        return mean_variance_weights(
            expected_returns, cov_matrix, risk_aversion=risk_aversion, allow_short=allow_short
        )
    if strategy == "max_sharpe":
        return max_sharpe_weights(
            expected_returns, cov_matrix, risk_free_rate=risk_free_rate, allow_short=allow_short
        )
    raise ValueError(f"Unknown strategy '{strategy}'; expected one of {sorted(_STRATEGIES)}.")


# ---------------------------------------------------------------------------
# Rolling backtest with periodic rebalancing
# ---------------------------------------------------------------------------


def rolling_backtest(
    predicted_returns: pd.DataFrame,
    actual_returns: pd.DataFrame,
    strategy: str,
    rebalance_frequency: int = 21,
    cov_lookback: int = 60,
    risk_free_rate: float = 0.0,
    risk_aversion: float = 1.0,
    allow_short: bool = False,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Run a rolling backtest of `strategy` over the test period,
    rebalancing every `rebalance_frequency` trading days.

    At each rebalance date `t` (spaced `rebalance_frequency` trading
    days apart, starting once `cov_lookback` days of test-set history
    are available):
        - expected returns are read from `predicted_returns.loc[t]`
          (the model's predicted returns for each asset, as of `t`);
        - the covariance matrix is estimated from the `cov_lookback`
          trading days of `actual_returns` immediately *before* `t`
          (i.e. only test-set data already observed by `t` - no
          look-ahead);
        - weights are computed via the chosen strategy and held fixed
          (no intra-period drift-adjustment) until the next rebalance
          date.

    Parameters
    ----------
    predicted_returns : pd.DataFrame
        Model-predicted returns, indexed by date, one column per
        asset, with the same columns as `actual_returns`.
    actual_returns : pd.DataFrame
        Realized (test-set) returns, indexed by date (chronologically
        sorted, no gaps assumed), aligned to `predicted_returns`'s
        columns.
    strategy : {"equal_weight", "mean_variance", "max_sharpe"}
    rebalance_frequency : int
        Number of trading days between rebalances.
    cov_lookback : int
        Number of trailing trading days of `actual_returns` used to
        estimate the covariance matrix at each rebalance. Also sets
        how many initial days are skipped as a warm-up period.
    risk_free_rate : float
        Risk-free rate used by the max-Sharpe objective (same units
        as the returns being passed in, e.g. daily if returns are
        daily).
    risk_aversion : float
        Risk-aversion coefficient for the mean-variance objective.
    allow_short : bool
        If False (default), weights are constrained long-only.

    Returns
    -------
    (pd.Series, pd.DataFrame)
        - `returns`: daily portfolio returns over the backtest period.
        - `weights`: rebalance-date weight history (index = rebalance
          date, columns = assets).
    """
    if strategy not in _STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'; expected one of {sorted(_STRATEGIES)}.")

    assets = list(actual_returns.columns)
    dates = actual_returns.index

    rebalance_locs = list(range(cov_lookback, len(dates), rebalance_frequency))
    if not rebalance_locs:
        raise ValueError(
            "Not enough test-set history: need at least `cov_lookback` days "
            "before the first rebalance."
        )

    portfolio_returns = pd.Series(np.nan, index=dates, dtype=float)
    weight_history: Dict = {}

    for i, loc in enumerate(rebalance_locs):
        reb_date = dates[loc]

        if reb_date not in predicted_returns.index:
            # No prediction available for this rebalance date; skip it
            # and let the previous weights continue to hold (if any),
            # or leave this period unallocated if it's the first one.
            continue

        cov_window = actual_returns.iloc[loc - cov_lookback : loc]
        cov_matrix = cov_window.cov().values

        expected_returns = predicted_returns.loc[reb_date, assets].to_numpy(dtype=float)

        weights = get_strategy_weights(
            strategy,
            expected_returns,
            cov_matrix,
            risk_free_rate=risk_free_rate,
            risk_aversion=risk_aversion,
            allow_short=allow_short,
        )
        weight_history[reb_date] = weights

        next_loc = rebalance_locs[i + 1] if i + 1 < len(rebalance_locs) else len(dates)
        period_returns = actual_returns.iloc[loc:next_loc][assets].to_numpy(dtype=float)
        portfolio_returns.iloc[loc:next_loc] = period_returns @ weights

    portfolio_returns = portfolio_returns.dropna()
    weights_df = pd.DataFrame.from_dict(weight_history, orient="index", columns=assets)
    weights_df.index.name = actual_returns.index.name

    return portfolio_returns, weights_df


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------


def annual_return(returns: pd.Series, trading_days: int = 252) -> float:
    """Compound Annual Growth Rate (CAGR) implied by a periodic return
    series.

    Parameters
    ----------
    returns : array-like
        Periodic (e.g. daily) simple returns.
    trading_days : int
        Number of trading periods per year, used to annualize.

    Returns
    -------
    float
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)
    if n == 0:
        return float("nan")
    compounded = np.prod(1.0 + r)
    return float(compounded ** (trading_days / n) - 1.0)


def volatility(returns: pd.Series, trading_days: int = 252) -> float:
    """Annualized standard deviation of a periodic return series.

    Parameters
    ----------
    returns : array-like
    trading_days : int

    Returns
    -------
    float
    """
    r = np.asarray(returns, dtype=float)
    if len(r) < 2:
        return float("nan")
    return float(np.std(r, ddof=1) * np.sqrt(trading_days))


def sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, trading_days: int = 252
) -> float:
    """Annualized Sharpe ratio: mean excess return over its standard
    deviation, annualized by `sqrt(trading_days)`.

    Parameters
    ----------
    returns : array-like
        Periodic simple returns.
    risk_free_rate : float
        Annual risk-free rate.
    trading_days : int

    Returns
    -------
    float
    """
    r = np.asarray(returns, dtype=float)
    if len(r) < 2:
        return float("nan")
    daily_rf = risk_free_rate / trading_days
    excess = r - daily_rf
    std = np.std(excess, ddof=1)
    if std == 0:
        return float("nan")
    return float((np.mean(excess) / std) * np.sqrt(trading_days))


def sortino_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, trading_days: int = 252
) -> float:
    """Annualized Sortino ratio: mean excess return over the standard
    deviation of only the negative excess returns (downside
    deviation), annualized by `sqrt(trading_days)`.

    Parameters
    ----------
    returns : array-like
    risk_free_rate : float
        Annual risk-free rate.
    trading_days : int

    Returns
    -------
    float
    """
    r = np.asarray(returns, dtype=float)
    if len(r) < 2:
        return float("nan")
    daily_rf = risk_free_rate / trading_days
    excess = r - daily_rf
    downside = excess[excess < 0]
    if len(downside) < 2:
        return float("nan")
    downside_std = np.std(downside, ddof=1)
    if downside_std == 0:
        return float("nan")
    return float((np.mean(excess) / downside_std) * np.sqrt(trading_days))


def max_drawdown(returns: pd.Series) -> float:
    """Maximum Drawdown (MDD): the largest peak-to-trough decline in
    the cumulative wealth curve implied by `returns`.

    Parameters
    ----------
    returns : array-like
        Periodic simple returns.

    Returns
    -------
    float
        A value in [-1, 0]; e.g. -0.23 means a 23% peak-to-trough
        decline.
    """
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return float("nan")
    wealth = np.cumprod(1.0 + r)
    running_max = np.maximum.accumulate(wealth)
    drawdown = wealth / running_max - 1.0
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series, trading_days: int = 252) -> float:
    """Calmar ratio: annualized return divided by the absolute value
    of the maximum drawdown.

    Parameters
    ----------
    returns : array-like
    trading_days : int

    Returns
    -------
    float
    """
    mdd = max_drawdown(returns)
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    return float(annual_return(returns, trading_days) / abs(mdd))


def value_at_risk(
    returns: pd.Series, confidence_level: float = 0.95, method: str = "historical"
) -> float:
    """Value at Risk (VaR) of the periodic return series, at the given
    confidence level.

    Parameters
    ----------
    returns : array-like
        Periodic simple returns (same period as the reported VaR,
        e.g. daily returns in -> 1-day VaR out; not annualized).
    confidence_level : float
        E.g. 0.95 for a 95% VaR.
    method : {"historical", "parametric"}
        "historical" uses the empirical return-distribution
        percentile; "parametric" assumes normally-distributed returns
        and uses the sample mean/standard deviation.

    Returns
    -------
    float
        A non-negative number representing the magnitude of the
        potential loss (as a positive fraction of portfolio value) not
        expected to be exceeded at the given confidence level.
    """
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return float("nan")

    if method == "historical":
        var = -np.percentile(r, (1.0 - confidence_level) * 100.0)
    elif method == "parametric":
        mean = np.mean(r)
        std = np.std(r, ddof=1)
        z = norm.ppf(1.0 - confidence_level)
        var = -(mean + z * std)
    else:
        raise ValueError(f"Unknown method '{method}'; expected 'historical' or 'parametric'.")

    return float(max(var, 0.0))


def compute_performance_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    trading_days: int = 252,
    var_confidence: float = 0.95,
    var_method: str = "historical",
) -> Dict[str, float]:
    """Compute all seven performance metrics for a periodic return
    series: Annual Return, Volatility, Sharpe Ratio, Sortino Ratio,
    Calmar Ratio, Maximum Drawdown, and Value at Risk.

    Parameters
    ----------
    returns : array-like
        Periodic (e.g. daily) portfolio returns, such as those
        returned by `rolling_backtest`.
    risk_free_rate : float
        Annual risk-free rate, used by the Sharpe/Sortino ratios.
    trading_days : int
        Number of trading periods per year, used for annualization.
    var_confidence : float
        Confidence level for Value at Risk (e.g. 0.95).
    var_method : {"historical", "parametric"}
        VaR estimation method.

    Returns
    -------
    dict
        {"annual_return", "volatility", "sharpe_ratio",
         "sortino_ratio", "calmar_ratio", "max_drawdown",
         "value_at_risk"}
    """
    return {
        "annual_return": annual_return(returns, trading_days),
        "volatility": volatility(returns, trading_days),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate, trading_days),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate, trading_days),
        "calmar_ratio": calmar_ratio(returns, trading_days),
        "max_drawdown": max_drawdown(returns),
        "value_at_risk": value_at_risk(returns, var_confidence, var_method),
    }


# ---------------------------------------------------------------------------
# End-to-end convenience wrapper
# ---------------------------------------------------------------------------


def run_backtest_comparison(
    predicted_returns: pd.DataFrame,
    actual_returns: pd.DataFrame,
    strategies: Sequence[str] = ("equal_weight", "mean_variance", "max_sharpe"),
    rebalance_frequency: int = 21,
    cov_lookback: int = 60,
    risk_free_rate: float = 0.0,
    risk_aversion: float = 1.0,
    allow_short: bool = False,
    trading_days: int = 252,
    var_confidence: float = 0.95,
    var_method: str = "historical",
) -> Dict[str, Dict]:
    """Run `rolling_backtest` for each of `strategies` and summarize
    each one's resulting return series with
    `compute_performance_metrics`.

    Parameters
    ----------
    predicted_returns, actual_returns :
        See `rolling_backtest`.
    strategies : sequence of {"equal_weight", "mean_variance", "max_sharpe"}
        Which strategies to run and compare.
    rebalance_frequency, cov_lookback, risk_free_rate, risk_aversion,
    allow_short :
        Forwarded to `rolling_backtest`.
    trading_days, var_confidence, var_method :
        Forwarded to `compute_performance_metrics`.

    Returns
    -------
    dict
        {strategy_name: {"returns": pd.Series, "weights": pd.DataFrame,
                          "metrics": dict}}
    """
    results: Dict[str, Dict] = {}
    for strategy in strategies:
        returns, weights = rolling_backtest(
            predicted_returns,
            actual_returns,
            strategy,
            rebalance_frequency=rebalance_frequency,
            cov_lookback=cov_lookback,
            risk_free_rate=risk_free_rate,
            risk_aversion=risk_aversion,
            allow_short=allow_short,
        )
        metrics = compute_performance_metrics(
            returns,
            risk_free_rate=risk_free_rate,
            trading_days=trading_days,
            var_confidence=var_confidence,
            var_method=var_method,
        )
        results[strategy] = {"returns": returns, "weights": weights, "metrics": metrics}

    return results
