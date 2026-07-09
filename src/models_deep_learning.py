"""
models_deep_learning.py
========================
Deep learning (LSTM) model definitions, refactored from the
model-construction cells previously duplicated across:
    - Codes/Portfolio- Univariate LSTM.ipynb
    - Codes/Portfolio- Multivariate LSTM.ipynb

These functions build (but do not fit) the same fixed architectures
used in the original notebooks. Training is left to the caller so
that the same model-building code can be reused for both fitting and
later reloading/inspection.

Note: these are the project's deep learning models, evaluated against
the classical baselines in `src/models_baseline.py` (ARIMA, XGBoost).
"""

from typing import Optional, Tuple

from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential


def build_univariate_lstm(input_shape: Tuple[int, int]) -> keras.Model:
    """Build the fixed univariate LSTM architecture.

    Refactored from the model-construction cell in
    Codes/Portfolio- Univariate LSTM.ipynb.

    Parameters
    ----------
    input_shape : (int, int)
        (timesteps, n_features) shape expected by the first LSTM layer.

    Returns
    -------
    keras.Model
        A compiled (uncompiled-optimizer-state) Sequential model, ready
        to call `.fit()` on.
    """
    model = Sequential()
    model.add(LSTM(units=150, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.05))
    model.add(LSTM(units=70, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.05))
    model.add(LSTM(units=60, input_shape=input_shape))
    model.add(Dropout(0.05))
    model.add(Dense(units=1, activation="relu"))

    model.compile(keras.optimizers.Adam(learning_rate=0.001), loss="mean_squared_error")
    return model


def build_multivariate_lstm(input_shape: Tuple[int, int]) -> keras.Model:
    """Build the fixed multivariate LSTM architecture.

    Refactored from the model-construction cell in
    Codes/Portfolio- Multivariate LSTM.ipynb.

    Parameters
    ----------
    input_shape : (int, int)
        (timesteps, n_features) shape expected by the first LSTM layer.

    Returns
    -------
    keras.Model
        A compiled Sequential model, ready to call `.fit()` on.
    """
    model = Sequential()
    model.add(LSTM(units=120, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.1))
    model.add(LSTM(units=170, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.1))
    model.add(LSTM(units=50, input_shape=input_shape))
    model.add(Dropout(0.1))
    model.add(Dense(units=1, activation="relu"))

    model.compile(keras.optimizers.Adam(learning_rate=0.001), loss="mean_squared_error")
    return model


def default_callbacks(patience: int = 10):
    """Return the EarlyStopping/ReduceLROnPlateau callbacks used for
    training both baseline models in the original notebooks.
    """
    es = EarlyStopping(monitor="loss", min_delta=1e-10, patience=patience, verbose=1)
    rlr = ReduceLROnPlateau(monitor="loss", factor=0.5, patience=patience, verbose=1)
    return [es, rlr]


def train_model(
    model: keras.Model,
    x_train,
    y_train,
    epochs: int = 100,
    batch_size: int = 32,
    validation_split: float = 0.2,
    validation_data: Optional[Tuple] = None,
    shuffle: bool = True,
):
    """Fit a baseline model using the same training configuration
    (epochs, batch size, callbacks) used in the original notebooks.

    Returns
    -------
    keras.callbacks.History
        The training history object.
    """
    callbacks = default_callbacks()

    fit_kwargs = dict(
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=shuffle,
    )
    if validation_data is not None:
        fit_kwargs["validation_data"] = validation_data
    else:
        fit_kwargs["validation_split"] = validation_split

    return model.fit(x_train, y_train, **fit_kwargs)
