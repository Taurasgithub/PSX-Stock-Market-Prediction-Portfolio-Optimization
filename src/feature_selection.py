"""
feature_selection.py
=====================
Feature-selection utilities for the candidate feature set produced by
`src/feature_engineering.py`.

Implements exactly two selection strategies:
    1. Recursive Feature Elimination (RFE) with a Random Forest
       regressor as the underlying estimator.
    2. Random Forest feature importance ranking (mean decrease in
       impurity, `feature_importances_`).

Every call to either selector appends a record of the run (timestamp,
method, parameters, and the resulting feature list) to a persistent
log file, so the exact feature set used for any downstream experiment
can always be traced back after the fact.

This module is scoped to feature *selection* only. It does not train,
evaluate, or persist any predictive model - that is the responsibility
of `src/models_baseline.py` / `src/models_deep_learning.py`. The
Random Forest estimators fitted here exist solely as a mechanism to
rank/eliminate candidate features and are discarded after use.
"""

import json
import os
from datetime import datetime, timezone
from typing import List, Optional, Sequence, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE

# Default location for the feature-selection execution log, resolved
# relative to the project root (one level up from `src/`).
DEFAULT_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs",
    "feature_selection.log",
)


def _log_selection_run(
    method: str,
    selected_features: Sequence[str],
    log_path: str = DEFAULT_LOG_PATH,
    **params,
) -> None:
    """Append a single execution record to the feature-selection log.

    Parameters
    ----------
    method : str
        Name of the selection method that produced the result, e.g.
        "RFE" or "RandomForestImportance".
    selected_features : Sequence[str]
        The feature names selected in this run.
    log_path : str
        Path of the log file to append to. The parent directory is
        created automatically if it does not exist.
    **params
        Any additional run parameters/metadata to record alongside the
        selected feature list (e.g. n_features, seed, n_estimators).
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": method,
        "n_selected": len(selected_features),
        "selected_features": list(selected_features),
        **params,
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def select_features_rfe(
    x: pd.DataFrame,
    y: pd.Series,
    n_features: int = 25,
    seed: Optional[int] = None,
    n_estimators: int = 500,
    log_path: str = DEFAULT_LOG_PATH,
) -> Tuple[RFE, List[str]]:
    """Select the top `n_features` using Recursive Feature Elimination
    with a Random Forest regressor as the base estimator.

    Parameters
    ----------
    x : pd.DataFrame
        Candidate feature matrix.
    y : pd.Series
        Regression target.
    n_features : int
        Number of features to keep (default 25).
    seed : int, optional
        Random seed passed to the underlying RandomForestRegressor and
        RFE for reproducibility.
    n_estimators : int
        Number of trees in the Random Forest estimator used by RFE at
        each elimination step.
    log_path : str
        Path of the text/log file that this run's selected feature
        list is appended to.

    Returns
    -------
    (RFE, list[str])
        The fitted RFE selector and the list of selected column names.
    """
    rfe = RFE(
        RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1),
        n_features_to_select=n_features,
        verbose=0,
    )
    rfe.fit(x, y)

    selected_features = list(x.columns[rfe.support_])

    _log_selection_run(
        method="RFE",
        selected_features=selected_features,
        log_path=log_path,
        n_features_requested=n_features,
        n_candidate_features=x.shape[1],
        n_estimators=n_estimators,
        seed=seed,
    )

    return rfe, selected_features


def select_features_rf_importance(
    x: pd.DataFrame,
    y: pd.Series,
    n_features: int = 25,
    seed: Optional[int] = None,
    n_estimators: int = 500,
    log_path: str = DEFAULT_LOG_PATH,
) -> Tuple[RandomForestRegressor, List[str], pd.Series]:
    """Select the top `n_features` ranked by Random Forest feature
    importance (mean decrease in impurity).

    Parameters
    ----------
    x : pd.DataFrame
        Candidate feature matrix.
    y : pd.Series
        Regression target.
    n_features : int
        Number of top-ranked features to keep (default 25).
    seed : int, optional
        Random seed passed to the underlying RandomForestRegressor for
        reproducibility.
    n_estimators : int
        Number of trees in the Random Forest estimator.
    log_path : str
        Path of the text/log file that this run's selected feature
        list is appended to.

    Returns
    -------
    (RandomForestRegressor, list[str], pd.Series)
        The fitted Random Forest estimator, the list of selected
        column names (highest importance first), and a Series of all
        candidate features' importance scores, sorted descending.
    """
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
    rf.fit(x, y)

    importances = pd.Series(rf.feature_importances_, index=x.columns).sort_values(
        ascending=False
    )
    selected_features = list(importances.head(n_features).index)

    _log_selection_run(
        method="RandomForestImportance",
        selected_features=selected_features,
        log_path=log_path,
        n_features_requested=n_features,
        n_candidate_features=x.shape[1],
        n_estimators=n_estimators,
        seed=seed,
        importances=importances.head(n_features).round(6).to_dict(),
    )

    return rf, selected_features, importances
