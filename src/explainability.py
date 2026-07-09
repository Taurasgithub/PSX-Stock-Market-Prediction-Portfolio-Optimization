"""
explainability.py
==================
SHAP integration for the trained GRU / BiLSTM models built in
`src/models_dl.py`: summary and single-day waterfall plots, computed
in a memory-bounded way regardless of dataset size.

Implements:
    - `build_shap_explainer` : wraps a trained GRU/BiLSTM Keras model
      in a `shap.GradientExplainer`, using a small, randomly sampled
      background dataset (rather than the full training set) to keep
      memory usage bounded.
    - `compute_shap_values`  : computes SHAP values batch-by-batch
      (with a capped `nsamples` per gradient estimate and explicit
      garbage collection between batches), then sums each feature's
      per-timestep contributions into a single value per feature.
    - `plot_shap_summary`    : SHAP summary (beeswarm) plot across all
      explained days, saved to a file.
    - `plot_shap_waterfall`  : SHAP waterfall plot explaining a single
      prediction day, saved to a file.
    - `explain_model`        : end-to-end convenience wrapper tying the
      above together.

This module only computes/visualizes SHAP attributions for
already-trained models - it does not build or train models (see
`src/models_dl.py`). Plots use SHAP's/matplotlib's default styling;
no publication formatting (custom themes, fonts, sizing, export
formats, etc.) is implemented here.
"""

import gc
import os
from typing import Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from tensorflow import keras

from src.seed_utils import load_config, set_global_seed

# ---------------------------------------------------------------------------
# Explainer construction (memory-bounded background)
# ---------------------------------------------------------------------------


def _subsample_background(
    background_data: np.ndarray,
    background_size: int = 50,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Randomly subsample the background dataset used to estimate the
    SHAP baseline distribution.

    `GradientExplainer`'s memory/compute cost scales with the size of
    the background set, so a small random subsample is used instead of
    the full training set whenever it's larger than `background_size`.

    Parameters
    ----------
    background_data : np.ndarray
        (n_samples, timesteps, n_features) candidate background
        windows.
    background_size : int
        Maximum number of background windows to keep.
    seed : int, optional
        Random seed for the subsample.

    Returns
    -------
    np.ndarray
        (min(n_samples, background_size), timesteps, n_features),
        cast to float32.
    """
    background_data = np.asarray(background_data, dtype=np.float32)
    n = background_data.shape[0]
    if n <= background_size:
        return background_data

    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=background_size, replace=False)
    return background_data[idx]


def build_shap_explainer(
    model: keras.Model,
    background_data: np.ndarray,
    background_size: int = 50,
    seed: Optional[int] = None,
) -> Tuple[shap.GradientExplainer, np.ndarray]:
    """Build a `shap.GradientExplainer` for a trained GRU or BiLSTM
    Keras model.

    `GradientExplainer` is used because it works directly with any
    differentiable TensorFlow/Keras model - including the stacked
    GRU / Bidirectional-LSTM regressors built in `src/models_dl.py` -
    without a model-specific wrapper, and handles their 3-D
    (samples, timesteps, features) input shape natively.

    Parameters
    ----------
    model : keras.Model
        A trained GRU or BiLSTM model (e.g. from
        `src/models_dl.build_gru_model` / `build_bilstm_model`).
    background_data : np.ndarray
        (n_samples, timesteps, n_features) windows used to estimate
        the SHAP baseline/expected value. Only a small random
        subsample (`background_size`) is actually used, to bound
        memory usage.
    background_size : int
        Maximum number of background windows to sample.
    seed : int, optional
        Random seed for the background subsample. Defaults to the
        project's global seed (`config.yaml`) via `src/seed_utils`.

    Returns
    -------
    (shap.GradientExplainer, np.ndarray)
        The explainer, and the exact background subsample it was
        built with (needed to compute the waterfall plot's base
        value - see `_background_base_value`).
    """
    if seed is None:
        seed = load_config()["random_seed"]
    set_global_seed(seed)

    background = _subsample_background(background_data, background_size, seed)
    explainer = shap.GradientExplainer(model, background)
    return explainer, background


def _background_base_value(model: keras.Model, background: np.ndarray) -> float:
    """Compute the SHAP base/expected value used by the waterfall
    plot: the model's mean predicted output over the background
    dataset.

    `GradientExplainer` (expected gradients) does not expose an
    `expected_value` attribute the way `DeepExplainer`/`TreeExplainer`
    do; its SHAP values are defined to sum to
    `prediction - E[model(background)]`, so that mean background
    prediction is what a waterfall plot needs as its base value.
    """
    preds = np.asarray(model.predict(background, verbose=0))
    return float(preds.reshape(-1).mean())


# ---------------------------------------------------------------------------
# SHAP value computation (batched, to avoid memory errors)
# ---------------------------------------------------------------------------


def compute_shap_values(
    explainer: shap.GradientExplainer,
    x: np.ndarray,
    batch_size: int = 16,
    nsamples: int = 200,
) -> np.ndarray:
    """Compute per-feature SHAP values for `x`, processed batch by
    batch to keep peak memory usage bounded regardless of how many
    rows are being explained.

    Each batch is explained with a capped `nsamples` (the number of
    gradient samples `GradientExplainer` averages per prediction) and
    intermediate per-batch arrays are freed (`gc.collect()`) before
    the next batch starts. Per-timestep SHAP values are then summed
    into a single value per feature: since SHAP values are additive,
    a feature's total contribution to a prediction equals the sum of
    its contributions across every input timestep.

    Parameters
    ----------
    explainer : shap.GradientExplainer
        Explainer built by `build_shap_explainer`.
    x : np.ndarray
        (n_samples, timesteps, n_features) windows to explain.
    batch_size : int
        Number of windows explained per batch. Lower this if memory
        is still constrained.
    nsamples : int
        Number of gradient samples per `GradientExplainer` call.
        Lower values use less memory/time at the cost of noisier
        SHAP estimates.

    Returns
    -------
    np.ndarray
        (n_samples, n_features) SHAP values, aggregated across
        timesteps.
    """
    x = np.asarray(x, dtype=np.float32)
    n_samples, _, n_features = x.shape

    aggregated = np.empty((n_samples, n_features), dtype=np.float32)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = x[start:end]

        batch_shap = explainer.shap_values(batch, nsamples=nsamples)
        # GradientExplainer returns a list of arrays, one per model
        # output unit; these regressors have a single output.
        if isinstance(batch_shap, list):
            batch_shap = batch_shap[0]
        batch_shap = np.asarray(batch_shap)
        if batch_shap.ndim == 4:
            # Some SHAP versions add a trailing size-1 output axis.
            batch_shap = batch_shap[..., 0]

        aggregated[start:end] = batch_shap.sum(axis=1)  # sum over timesteps

        del batch, batch_shap
        gc.collect()

    return aggregated


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_shap_summary(
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    feature_names: Sequence[str],
    output_path: str,
    max_display: int = 20,
) -> str:
    """Generate and save a SHAP summary (beeswarm) plot.

    Parameters
    ----------
    shap_values : np.ndarray
        (n_samples, n_features) SHAP values, e.g. from
        `compute_shap_values`.
    feature_values : np.ndarray
        (n_samples, n_features) raw feature values corresponding to
        `shap_values` (used for the plot's colour scale).
    feature_names : Sequence[str]
        Column names, in the same order as the feature axis.
    output_path : str
        File path the plot image is saved to (parent directories are
        created automatically).
    max_display : int
        Maximum number of features shown on the plot.

    Returns
    -------
    str
        `output_path`, for convenience/chaining.
    """
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.figure()
    shap.summary_plot(
        shap_values,
        feature_values,
        feature_names=list(feature_names),
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_shap_waterfall(
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    feature_names: Sequence[str],
    day_index: int,
    base_value: float,
    output_path: str,
    max_display: int = 15,
) -> str:
    """Generate and save a SHAP waterfall plot explaining a single
    prediction day.

    Parameters
    ----------
    shap_values : np.ndarray
        (n_samples, n_features) SHAP values, e.g. from
        `compute_shap_values`.
    feature_values : np.ndarray
        (n_samples, n_features) raw feature values corresponding to
        `shap_values`.
    feature_names : Sequence[str]
        Column names, in the same order as the feature axis.
    day_index : int
        Row index (into `shap_values` / `feature_values`) of the
        single prediction day to explain.
    base_value : float
        The model's expected/base output, e.g. from
        `build_shap_explainer(...).expected_value`.
    output_path : str
        File path the plot image is saved to (parent directories are
        created automatically).
    max_display : int
        Maximum number of features shown on the plot.

    Returns
    -------
    str
        `output_path`, for convenience/chaining.
    """
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    explanation = shap.Explanation(
        values=np.asarray(shap_values)[day_index],
        base_values=base_value,
        data=np.asarray(feature_values)[day_index],
        feature_names=list(feature_names),
    )

    plt.figure()
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


# ---------------------------------------------------------------------------
# End-to-end convenience wrapper
# ---------------------------------------------------------------------------


def explain_model(
    model: keras.Model,
    background_data: np.ndarray,
    x_explain: np.ndarray,
    feature_names: Sequence[str],
    day_index: int = -1,
    output_dir: str = "reports/shap",
    model_name: str = "model",
    background_size: int = 50,
    batch_size: int = 16,
    nsamples: int = 200,
    seed: Optional[int] = None,
) -> dict:
    """End-to-end SHAP explanation for a trained GRU/BiLSTM model:
    builds the explainer, computes SHAP values in a memory-bounded
    way, and saves both a summary plot and a single-day waterfall
    plot.

    Parameters
    ----------
    model : keras.Model
        A trained model from `src/models_dl.py` (GRU or BiLSTM).
    background_data : np.ndarray
        (n_samples, timesteps, n_features) windows used as the SHAP
        baseline distribution (e.g. a slice of the training set).
    x_explain : np.ndarray
        (n_samples, timesteps, n_features) windows to explain (e.g.
        the validation/test set).
    feature_names : Sequence[str]
        Column names, in the same order as the feature axis of
        `background_data` / `x_explain`.
    day_index : int
        Row index into `x_explain` used for the waterfall plot
        (default: the last/most recent day).
    output_dir : str
        Directory the summary/waterfall plot images are saved to.
    model_name : str
        Filename prefix, e.g. "gru" -> "gru_shap_summary.png".
    background_size, batch_size, nsamples, seed :
        Forwarded to `build_shap_explainer` / `compute_shap_values`.

    Returns
    -------
    dict
        {
            "shap_values": np.ndarray,           # (n_samples, n_features)
            "summary_plot_path": str,
            "waterfall_plot_path": str,
        }
    """
    explainer, background = build_shap_explainer(model, background_data, background_size, seed)
    shap_values = compute_shap_values(explainer, x_explain, batch_size, nsamples)
    base_value = _background_base_value(model, background)

    # Represent each window by its most recent timestep's raw feature
    # values, for plot colouring/axis display.
    display_values = np.asarray(x_explain, dtype=np.float32)[:, -1, :]

    summary_path = plot_shap_summary(
        shap_values,
        display_values,
        feature_names,
        os.path.join(output_dir, f"{model_name}_shap_summary.png"),
    )
    waterfall_path = plot_shap_waterfall(
        shap_values,
        display_values,
        feature_names,
        day_index,
        base_value,
        os.path.join(output_dir, f"{model_name}_shap_waterfall.png"),
    )

    return {
        "shap_values": shap_values,
        "summary_plot_path": summary_path,
        "waterfall_plot_path": waterfall_path,
    }
