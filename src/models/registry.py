"""Model registry.

Every forecaster predicts the *forward return* for one horizon and exposes the
same two methods. Price forecasts are then `base_price * (1 + return)`, which
keeps models scale-free across the expanding training window (raw price levels
drift far outside the training range in a trending market and degrade badly).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.logging_config import get_logger

log = get_logger(__name__)


class NaiveForecaster:
    """Last observed price carried forward: zero predicted return.

    This is the benchmark every sentiment-driven model has to beat.
    """

    name = "naive"

    def fit(self, features: pd.DataFrame, returns: pd.Series) -> "NaiveForecaster":
        self.n_train_ = len(returns)
        return self

    def predict(self, features: pd.DataFrame, base_price: pd.Series | np.ndarray) -> np.ndarray:
        return np.zeros(len(features), dtype=float)


class SklearnForecaster:
    """Adapter so scikit-learn estimators match the forecaster interface."""

    def __init__(self, name: str, estimator) -> None:
        self.name = name
        self.estimator = estimator
        self.n_train_ = 0

    def fit(self, features: pd.DataFrame, returns: pd.Series) -> "SklearnForecaster":
        self.estimator.fit(features.to_numpy(dtype=float), returns.to_numpy(dtype=float))
        self.n_train_ = len(returns)
        return self

    def predict(self, features: pd.DataFrame, base_price: pd.Series | np.ndarray) -> np.ndarray:
        return np.asarray(self.estimator.predict(features.to_numpy(dtype=float)), dtype=float)


def _random_forest(seed: int) -> SklearnForecaster:
    from sklearn.ensemble import RandomForestRegressor

    estimator = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=3,
        max_features=0.5,
        random_state=seed,
        n_jobs=-1,
    )
    return SklearnForecaster("random_forest", estimator)


def _hist_gbm(seed: int) -> SklearnForecaster:
    from sklearn.ensemble import HistGradientBoostingRegressor

    estimator = HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=10,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=seed,
    )
    return SklearnForecaster("hist_gbm", estimator)


def _xgboost(seed: int) -> SklearnForecaster:
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ModelNotAvailableError(
            "xgboost is not installed. Install it (pip install xgboost) or remove it from MODELS."
        ) from exc
    estimator = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )
    return SklearnForecaster("xgboost", estimator)


BUILDERS = {
    "naive": lambda seed: NaiveForecaster(),
    "random_forest": _random_forest,
    "hist_gbm": _hist_gbm,
    "xgboost": _xgboost,
}


class ModelNotAvailableError(RuntimeError):
    """Raised when a configured model cannot be constructed."""


def build_model(name: str, seed: int = 42):
    if name not in BUILDERS:
        raise ModelNotAvailableError(
            f"unknown model '{name}'. Available: {', '.join(sorted(BUILDERS))}."
        )
    return BUILDERS[name](seed)


def resolve_models(names: list[str] | tuple[str, ...], seed: int = 42) -> dict[str, object]:
    """Construct the requested models, skipping unavailable ones with a warning."""
    models: dict[str, object] = {}
    for name in names:
        try:
            models[name] = build_model(name, seed)
        except ModelNotAvailableError as exc:
            log.warning("skipping model '%s': %s", name, exc)
    if "naive" not in models:
        models["naive"] = NaiveForecaster()
        log.info("naive baseline added automatically -- every run is compared against it")
    return models


def feature_importance(model, feature_names: list[str], top_n: int = 10) -> pd.Series:
    """Permutation-free importance where the estimator exposes one, else empty."""
    estimator = getattr(model, "estimator", None)
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return pd.Series(dtype=float)
    series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    return series.head(top_n)
