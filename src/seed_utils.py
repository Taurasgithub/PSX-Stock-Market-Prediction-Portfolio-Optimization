"""
seed_utils.py
=============
Centralized configuration loading and reproducibility utilities.

This module reads the single global `random_seed` value from
`config.yaml` (project root) and applies it consistently across every
library used in this project:

    - Python's built-in `random` module
    - NumPy (`numpy.random`)
    - TensorFlow / Keras (`tensorflow.random`)
    - scikit-learn estimators (via the returned seed, passed as
      `random_state=` wherever an estimator is constructed)

Usage
-----
    from src.seed_utils import load_config, set_global_seed

    config = load_config()
    SEED = set_global_seed(config["random_seed"])

    # SEED can then be passed to any scikit-learn estimator, e.g.:
    # RandomForestRegressor(random_state=SEED)
"""

import os
import random

import numpy as np
import yaml

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"
)


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Load the project's YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to the config.yaml file. Defaults to the config.yaml
        located at the project root.

    Returns
    -------
    dict
        Parsed configuration dictionary (must contain `random_seed`).
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if "random_seed" not in config:
        raise KeyError("config.yaml must define a top-level `random_seed` key.")

    return config


def set_global_seed(seed: int) -> int:
    """Apply a single global random seed across all ML/data libraries.

    This seeds, in order:
        1. Python's built-in `random` module
        2. NumPy's global random state
        3. TensorFlow's global random state (if TensorFlow is installed)

    scikit-learn does not have a single global seed; instead, the same
    `seed` value returned here should be passed explicitly as the
    `random_state` argument to individual scikit-learn estimators and
    functions (e.g. `train_test_split`, `RandomForestRegressor`, `RFE`).

    Parameters
    ----------
    seed : int
        The global random seed to apply.

    Returns
    -------
    int
        The same seed, for convenience (e.g. to pass into
        scikit-learn's `random_state` arguments).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        # TensorFlow is an optional dependency for modules that don't
        # need it (e.g. pure data loading/portfolio optimization code).
        pass

    return seed
