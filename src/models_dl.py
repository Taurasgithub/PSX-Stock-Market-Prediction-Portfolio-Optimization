"""
models_dl.py
============
GRU and Bidirectional LSTM (BiLSTM) model architectures, standard
training callbacks, and Keras Tuner-based hyperparameter search for
the PSX Stock Market Prediction project.

Implements:
    - GRU model architecture                  (`build_gru_model`)
    - Bidirectional LSTM model architecture    (`build_bilstm_model`)
    - EarlyStopping on validation loss, ModelCheckpoint (best weights
      only), and ReduceLROnPlateau              (`default_callbacks`)
    - Keras Tuner HyperModels (`GRUHyperModel`, `BiLSTMHyperModel`)
      that search over stacked layer sizes, dropout, and batch size
    - `get_tuner`, which wires a HyperModel up to a Keras Tuner search
      strategy (RandomSearch or Hyperband), seeded from the project's
      single global `random_seed` (see `src/seed_utils.py`) so that
      hyperparameter search trials are reproducible.

Scope
-----
This module only builds architectures and configures/searches
hyperparameters. It does not implement:
    - explainability (e.g. SHAP-based feature attribution), or
    - portfolio optimization (see `src/portfolio_optimization.py`).
The project's original fixed LSTM architectures remain in
`src/models_deep_learning.py`; this module is additive and scoped to
GRU / BiLSTM + tuning only.
"""

from typing import Optional, Tuple

import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import GRU, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.models import Sequential

from src.seed_utils import load_config, set_global_seed

# ---------------------------------------------------------------------------
# Fixed-architecture builders
# ---------------------------------------------------------------------------


def build_gru_model(
    input_shape: Tuple[int, int],
    units: Tuple[int, int, int] = (120, 80, 50),
    dropout: float = 0.1,
    learning_rate: float = 0.001,
    seed: Optional[int] = None,
) -> keras.Model:
    """Build a 3-layer stacked GRU regression model.

    Parameters
    ----------
    input_shape : (int, int)
        (timesteps, n_features) shape expected by the first GRU layer.
    units : (int, int, int)
        Number of units in each of the three stacked GRU layers.
    dropout : float
        Dropout rate applied after every GRU layer.
    learning_rate : float
        Adam optimizer learning rate.
    seed : int, optional
        Random seed used to initialize every layer's weights, for
        reproducibility (typically the project's global seed from
        `src/seed_utils.py`).

    Returns
    -------
    keras.Model
        A compiled Sequential model, ready to call `.fit()` on.
    """
    u1, u2, u3 = units
    initializer = keras.initializers.GlorotUniform(seed=seed)

    model = Sequential()
    model.add(
        GRU(
            units=u1,
            return_sequences=True,
            input_shape=input_shape,
            kernel_initializer=initializer,
        )
    )
    model.add(Dropout(dropout))
    model.add(GRU(units=u2, return_sequences=True, kernel_initializer=initializer))
    model.add(Dropout(dropout))
    model.add(GRU(units=u3, kernel_initializer=initializer))
    model.add(Dropout(dropout))
    model.add(Dense(units=1, activation="relu", kernel_initializer=initializer))

    model.compile(keras.optimizers.Adam(learning_rate=learning_rate), loss="mean_squared_error")
    return model


def build_bilstm_model(
    input_shape: Tuple[int, int],
    units: Tuple[int, int, int] = (120, 80, 50),
    dropout: float = 0.1,
    learning_rate: float = 0.001,
    seed: Optional[int] = None,
) -> keras.Model:
    """Build a 3-layer stacked Bidirectional LSTM (BiLSTM) regression
    model.

    Parameters
    ----------
    input_shape : (int, int)
        (timesteps, n_features) shape expected by the first
        Bidirectional LSTM layer.
    units : (int, int, int)
        Number of units in each of the three stacked (per-direction)
        LSTM layers wrapped by `Bidirectional`.
    dropout : float
        Dropout rate applied after every Bidirectional LSTM layer.
    learning_rate : float
        Adam optimizer learning rate.
    seed : int, optional
        Random seed used to initialize every layer's weights, for
        reproducibility (typically the project's global seed from
        `src/seed_utils.py`).

    Returns
    -------
    keras.Model
        A compiled Sequential model, ready to call `.fit()` on.
    """
    u1, u2, u3 = units
    initializer = keras.initializers.GlorotUniform(seed=seed)

    model = Sequential()
    model.add(
        Bidirectional(
            LSTM(units=u1, return_sequences=True, kernel_initializer=initializer),
            input_shape=input_shape,
        )
    )
    model.add(Dropout(dropout))
    model.add(
        Bidirectional(LSTM(units=u2, return_sequences=True, kernel_initializer=initializer))
    )
    model.add(Dropout(dropout))
    model.add(Bidirectional(LSTM(units=u3, kernel_initializer=initializer)))
    model.add(Dropout(dropout))
    model.add(Dense(units=1, activation="relu", kernel_initializer=initializer))

    model.compile(keras.optimizers.Adam(learning_rate=learning_rate), loss="mean_squared_error")
    return model


# ---------------------------------------------------------------------------
# Callbacks: EarlyStopping + ModelCheckpoint + ReduceLROnPlateau
# ---------------------------------------------------------------------------


def default_callbacks(
    checkpoint_path: str,
    patience: int = 10,
    reduce_lr_factor: float = 0.5,
    reduce_lr_patience: Optional[int] = None,
):
    """Build the standard training callback stack shared by the GRU and
    BiLSTM models: EarlyStopping, ModelCheckpoint (best weights only),
    and ReduceLROnPlateau - all monitoring validation loss.

    Parameters
    ----------
    checkpoint_path : str
        File path the best-epoch weights are saved to (e.g.
        "checkpoints/gru_best.weights.h5"). The parent directory is
        NOT created automatically; the caller is responsible for it
        existing before training starts.
    patience : int
        Number of epochs with no `val_loss` improvement before
        EarlyStopping halts training.
    reduce_lr_factor : float
        Factor the learning rate is multiplied by when `val_loss`
        plateaus.
    reduce_lr_patience : int, optional
        Number of epochs with no `val_loss` improvement before the
        learning rate is reduced. Defaults to `max(1, patience // 2)`
        so the learning rate backs off before EarlyStopping fires.

    Returns
    -------
    list
        [EarlyStopping, ModelCheckpoint, ReduceLROnPlateau], all
        monitoring `val_loss`.
    """
    if reduce_lr_patience is None:
        reduce_lr_patience = max(1, patience // 2)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-10,
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
        verbose=1,
    )
    return [early_stopping, checkpoint, reduce_lr]


# ---------------------------------------------------------------------------
# Keras Tuner hypermodels: search over layer sizes, dropout, batch size
# ---------------------------------------------------------------------------


class _BaseSequenceHyperModel(kt.HyperModel):
    """Shared Keras Tuner search space for the GRU/BiLSTM HyperModels:
    three stacked recurrent layer sizes, a shared dropout rate, the
    Adam learning rate, and the training batch size.

    Subclasses only need to implement `_recurrent_layer` to plug in
    the concrete recurrent layer type (GRU vs. Bidirectional LSTM).
    """

    def __init__(self, input_shape: Tuple[int, int]):
        super().__init__()
        self.input_shape = input_shape

    def _recurrent_layer(
        self,
        units: int,
        return_sequences: bool,
        input_shape: Optional[Tuple[int, int]] = None,
    ):
        raise NotImplementedError

    def build(self, hp: kt.HyperParameters) -> keras.Model:
        """Sample layer sizes and dropout, and build the model graph.

        Search space
        ------------
        units_1, units_2, units_3 : int
            Sizes of the three stacked recurrent layers.
        dropout : float
            Dropout rate applied after every recurrent layer.
        learning_rate : float
            Adam optimizer learning rate.
        """
        units_1 = hp.Int("units_1", min_value=32, max_value=160, step=32)
        units_2 = hp.Int("units_2", min_value=32, max_value=160, step=32)
        units_3 = hp.Int("units_3", min_value=16, max_value=96, step=16)
        dropout = hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.05)
        learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 5e-4, 1e-4])

        model = Sequential()
        model.add(
            self._recurrent_layer(units_1, return_sequences=True, input_shape=self.input_shape)
        )
        model.add(Dropout(dropout))
        model.add(self._recurrent_layer(units_2, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(self._recurrent_layer(units_3, return_sequences=False))
        model.add(Dropout(dropout))
        model.add(Dense(units=1, activation="relu"))

        model.compile(
            keras.optimizers.Adam(learning_rate=learning_rate), loss="mean_squared_error"
        )
        return model

    def fit(self, hp: kt.HyperParameters, model: keras.Model, *args, **kwargs):
        """Sample the training batch size and delegate to `model.fit`.

        Batch size is a *training*, not architecture, hyperparameter,
        so it is tuned here (per Keras Tuner's recommended pattern)
        rather than in `build`.
        """
        batch_size = hp.Choice("batch_size", values=[16, 32, 64, 128])
        return model.fit(*args, batch_size=batch_size, **kwargs)


class GRUHyperModel(_BaseSequenceHyperModel):
    """Keras Tuner search space for the stacked GRU architecture."""

    def _recurrent_layer(
        self,
        units: int,
        return_sequences: bool,
        input_shape: Optional[Tuple[int, int]] = None,
    ):
        kwargs = dict(units=units, return_sequences=return_sequences)
        if input_shape is not None:
            kwargs["input_shape"] = input_shape
        return GRU(**kwargs)


class BiLSTMHyperModel(_BaseSequenceHyperModel):
    """Keras Tuner search space for the stacked Bidirectional LSTM
    architecture."""

    def _recurrent_layer(
        self,
        units: int,
        return_sequences: bool,
        input_shape: Optional[Tuple[int, int]] = None,
    ):
        lstm = LSTM(units=units, return_sequences=return_sequences)
        if input_shape is not None:
            return Bidirectional(lstm, input_shape=input_shape)
        return Bidirectional(lstm)


def get_tuner(
    model_type: str,
    input_shape: Tuple[int, int],
    max_trials: int = 20,
    executions_per_trial: int = 1,
    directory: str = "tuner_results",
    project_name: Optional[str] = None,
    seed: Optional[int] = None,
    tuner_type: str = "random_search",
) -> kt.Tuner:
    """Build a Keras Tuner search over the GRU or BiLSTM hyperparameter
    space (stacked layer sizes, dropout, batch size), seeded with the
    project's global `random_seed` so tuning runs are reproducible.

    Parameters
    ----------
    model_type : {"gru", "bilstm"}
        Which architecture's hyperparameter space to search.
    input_shape : (int, int)
        (timesteps, n_features) shape passed through to the HyperModel.
    max_trials : int
        For `tuner_type="random_search"`: number of hyperparameter
        combinations to try. For `tuner_type="hyperband"`: the
        `max_epochs` budget for the largest bracket.
    executions_per_trial : int
        Number of models trained per hyperparameter combination, to
        average out training-run noise.
    directory : str
        Directory Keras Tuner writes trial results/checkpoints to.
    project_name : str, optional
        Sub-directory name for this search. Defaults to
        "gru_tuning" / "bilstm_tuning" based on `model_type`.
    seed : int, optional
        Random seed applied to the tuner's hyperparameter sampling.
        If not provided, it is loaded from `config.yaml` via
        `src/seed_utils.load_config`, and `src/seed_utils.set_global_seed`
        is applied before the tuner is constructed - this ensures
        hyperparameter search uses the same single project-wide seed
        as every other stage of the pipeline (per requirement).
    tuner_type : {"random_search", "hyperband"}
        Which Keras Tuner search strategy to use.

    Returns
    -------
    keras_tuner.Tuner
        A configured (not yet run) tuner. Call `.search(x, y, callbacks=...,
        validation_data=...)` on it to run the hyperparameter search;
        pass the callbacks from `default_callbacks(...)` so EarlyStopping /
        ModelCheckpoint / ReduceLROnPlateau apply to every trial.
    """
    if seed is None:
        seed = load_config()["random_seed"]
    set_global_seed(seed)

    if model_type == "gru":
        hypermodel = GRUHyperModel(input_shape)
        default_project_name = "gru_tuning"
    elif model_type == "bilstm":
        hypermodel = BiLSTMHyperModel(input_shape)
        default_project_name = "bilstm_tuning"
    else:
        raise ValueError(f"Unknown model_type '{model_type}'; expected 'gru' or 'bilstm'.")

    project_name = project_name or default_project_name

    if tuner_type == "random_search":
        tuner = kt.RandomSearch(
            hypermodel,
            objective="val_loss",
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            seed=seed,
            directory=directory,
            project_name=project_name,
            overwrite=True,
        )
    elif tuner_type == "hyperband":
        tuner = kt.Hyperband(
            hypermodel,
            objective="val_loss",
            max_epochs=max_trials,
            seed=seed,
            directory=directory,
            project_name=project_name,
            overwrite=True,
        )
    else:
        raise ValueError(
            f"Unknown tuner_type '{tuner_type}'; expected 'random_search' or 'hyperband'."
        )

    return tuner
