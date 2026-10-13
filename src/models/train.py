"""Fitting forecasters on the full history and producing forward forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.logging_config import get_logger
from src.ingestion.timeutils import interval_timedelta
from src.models.registry import build_model, feature_importance, resolve_models

log = get_logger(__name__)


def fit_on_history(
    features: pd.DataFrame,
    target_returns: pd.DataFrame,
    model_names: list[str] | tuple[str, ...],
    seed: int = 42,
) -> dict[tuple[str, int], object]:
    """Fit every (model, horizon) pair on all rows whose target is realised.

    Used for the forward-looking n+1..n+5 forecast table. Walk-forward
    evaluation refits its own models and never touches these objects.
    """
    models = resolve_models(model_names, seed)
    fitted: dict[tuple[str, int], object] = {}
    horizons = sorted(int(column.split("_")[1]) for column in target_returns.columns)

    for horizon in horizons:
        column = f"ret_{horizon}"
        valid = target_returns[column].notna()
        usable = features[valid]
        if usable.empty:
            log.warning("no rows with a realised target for n+%d; skipping", horizon)
            continue
        for name in models:
            # A fresh estimator per horizon: reusing one object would leave every
            # horizon holding the state of the last fit.
            trained = build_model(name, seed).fit(usable, target_returns.loc[usable.index, column])
            fitted[(name, horizon)] = trained
            log.info(
                "fit %s for n+%d on %d row(s)", name, horizon, getattr(trained, "n_train_", len(usable))
            )
    return fitted


def forecast_ahead(
    fitted: dict[tuple[str, int], object],
    latest_features: pd.DataFrame,
    base_price: float,
    horizons: list[int] | tuple[int, ...],
    origin_ts: pd.Timestamp,
    interval: str,
) -> pd.DataFrame:
    """Project the latest observed bar forward across every horizon.

    n+0 is emitted as the observed price with model 'observed' so the table
    reads n+0 ... n+5 without pretending the current price was predicted.
    """
    rows = [
        {
            "model": "observed",
            "horizon": 0,
            "origin_ts": origin_ts,
            "target_ts": origin_ts,
            "base_price": base_price,
            "predicted_return": 0.0,
            "predicted_price": base_price,
            "actual_return": np.nan,
            "actual_price": np.nan,
        }
    ]
    span = interval_timedelta(interval)
    for horizon in horizons:
        target_ts = origin_ts + span * horizon
        for (name, fitted_horizon), model in sorted(fitted.items()):
            if fitted_horizon != horizon:
                continue
            predicted_return = float(model.predict(latest_features, pd.Series([base_price]))[0])
            rows.append(
                {
                    "model": name,
                    "horizon": horizon,
                    "origin_ts": origin_ts,
                    "target_ts": target_ts,
                    "base_price": base_price,
                    "predicted_return": predicted_return,
                    "predicted_price": base_price * (1.0 + predicted_return),
                    "actual_return": np.nan,
                    "actual_price": np.nan,
                }
            )
    return pd.DataFrame(rows)


def importance_table(
    fitted: dict[tuple[str, int], object], feature_names: list[str], top_n: int = 10
) -> pd.DataFrame:
    frames = []
    for (name, horizon), model in sorted(fitted.items()):
        series = feature_importance(model, feature_names, top_n)
        if series.empty:
            continue
        for rank, (feature, value) in enumerate(series.items(), start=1):
            frames.append(
                {
                    "model": name,
                    "horizon": horizon,
                    "rank": rank,
                    "feature": feature,
                    "importance": float(value),
                }
            )
    return pd.DataFrame(frames)
