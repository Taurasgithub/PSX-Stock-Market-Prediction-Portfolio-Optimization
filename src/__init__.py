"""
src
===
Modular Python package for the PSX Stock Market Prediction & Portfolio
Optimization project (v2-daily-data-framework).

Modules
-------
seed_utils            : config.yaml loading + global random seed control
data_loader            : CSV loading/cleaning, walk-forward split, scaling
feature_engineering    : config-driven technical indicator + lag feature generation
feature_selection      : RFE-based feature selection utilities
models_baseline        : classical baselines (ARIMA, XGBoost) + RMSE/MAE evaluation
models_deep_learning   : deep learning models (univariate/multivariate LSTM)
portfolio_optimization : Mean-Variance / Equal-Weight / Max-Sharpe strategies + rolling backtest
"""
