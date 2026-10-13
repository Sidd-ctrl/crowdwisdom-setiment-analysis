"""Expanding-window walk-forward validation.

For each test origin i (the close of bar i) and each horizon k:

1. Training rows are j <= i - k, so every training target ``close[j+k]`` was
   already realised at origin i. Rows whose target would land at or after the
   origin are excluded -- that is the leakage the naive implementation misses.
2. Models are refit on that expanding window (every `refit_every` steps to keep
   the runtime sane; the window still only grows, never shifts).
3. The forecast for origin i is stored with the realised value once the target
   bar closes.

No scaler, encoder or imputer is fit on data beyond the training window: the
feature frame itself contains only causal columns, and each refit sees strictly
past observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config.logging_config import get_logger
from src.models.registry import resolve_models

log = get_logger(__name__)


@dataclass
class WalkForwardReport:
    predictions: pd.DataFrame
    steps: int = 0
    fits: dict = field(default_factory=dict)
    skipped: dict = field(default_factory=dict)
    first_origin: pd.Timestamp | None = None
    last_origin: pd.Timestamp | None = None

    @property
    def empty(self) -> bool:
        return self.predictions.empty


def run_walk_forward(
    features: pd.DataFrame,
    target_returns: pd.DataFrame,
    base_prices: pd.Series,
    model_names: list[str] | tuple[str, ...],
    horizons: list[int] | tuple[int, ...],
    min_train_samples: int = 60,
    refit_every: int = 5,
    seed: int = 42,
) -> WalkForwardReport:
    features = features.sort_index()
    target_returns = target_returns.loc[features.index]
    base_prices = base_prices.loc[features.index]
    n = len(features)
    max_horizon = max(horizons)
    start = min_train_samples + max_horizon

    if n <= start:
        log.error(
            "not enough history for walk-forward validation: %d usable bar(s), need more than %d "
            "(MIN_TRAIN_SAMPLES=%d plus the largest horizon n+%d). Increase PERIOD_DAYS or lower "
            "MIN_TRAIN_SAMPLES.",
            n,
            start,
            min_train_samples,
            max_horizon,
        )
        return WalkForwardReport(predictions=pd.DataFrame(columns=_columns()))

    records: list[dict] = []
    fits: dict[str, int] = {}
    skipped: dict[str, int] = {}
    origins_seen: set[pd.Timestamp] = set()

    for horizon in horizons:
        column = f"ret_{horizon}"
        if column not in target_returns.columns:
            skipped[f"missing_target_{column}"] = 1
            continue

        # Fresh estimator objects per horizon so a refit can never leak state
        # from another horizon's fit.
        models = resolve_models(model_names, seed)
        cache: dict[str, object] = {}
        last_fit: dict[str, int] = {}

        for position in range(start, n):
            train_end = position - horizon  # last row whose target is realised at origin
            if train_end - 0 + 1 < min_train_samples:
                skipped[f"n+{horizon}_insufficient_training"] = (
                    skipped.get(f"n+{horizon}_insufficient_training", 0) + 1
                )
                continue

            actual_return = target_returns[column].iloc[position]
            if pd.isna(actual_return):
                skipped[f"n+{horizon}_no_actual"] = skipped.get(f"n+{horizon}_no_actual", 0) + 1
                continue

            train_features = features.iloc[: train_end + 1]
            train_targets = target_returns[column].iloc[: train_end + 1]
            usable_mask = train_targets.notna() & train_features.notna().all(axis=1)
            train_features = train_features[usable_mask.to_numpy()]
            train_targets = train_targets[usable_mask]

            if len(train_features) < min_train_samples:
                skipped[f"n+{horizon}_insufficient_training"] = (
                    skipped.get(f"n+{horizon}_insufficient_training", 0) + 1
                )
                continue

            test_features = features.iloc[position : position + 1]
            base_price = float(base_prices.iloc[position])
            origin = features.index[position]
            origins_seen.add(origin)

            for name, model in models.items():
                needs_fit = name not in cache or (position - last_fit.get(name, -10**9)) >= refit_every
                if needs_fit:
                    if not _has_variance(train_targets):
                        skipped[f"n+{horizon}_{name}_degenerate_training"] = (
                            skipped.get(f"n+{horizon}_{name}_degenerate_training", 0) + 1
                        )
                        continue
                    try:
                        cache[name] = model.fit(train_features, train_targets)
                    except Exception as exc:  # surfaced, never silently ignored
                        log.error("fit failed for %s at n+%d origin %s: %s", name, horizon, origin, exc)
                        skipped[f"n+{horizon}_{name}_fit_error"] = (
                            skipped.get(f"n+{horizon}_{name}_fit_error", 0) + 1
                        )
                        continue
                    last_fit[name] = position
                    fits[f"{name}_n+{horizon}"] = fits.get(f"{name}_n+{horizon}", 0) + 1
                if name not in cache:
                    continue

                try:
                    predicted_return = float(
                        cache[name].predict(test_features, pd.Series([base_price]))[0]
                    )
                except Exception as exc:
                    log.error("predict failed for %s at n+%d origin %s: %s", name, horizon, origin, exc)
                    skipped[f"n+{horizon}_{name}_predict_error"] = (
                        skipped.get(f"n+{horizon}_{name}_predict_error", 0) + 1
                    )
                    continue

                records.append(
                    {
                        "origin_ts": origin,
                        "model": name,
                        "horizon": int(horizon),
                        "base_price": base_price,
                        "predicted_return": predicted_return,
                        "predicted_price": base_price * (1.0 + predicted_return),
                        "actual_return": float(actual_return),
                        "actual_price": base_price * (1.0 + float(actual_return)),
                        "target_ts": _target_ts(features.index, position, horizon),
                        "train_rows": len(train_features),
                    }
                )

    predictions = pd.DataFrame(records, columns=_columns())
    if not predictions.empty:
        predictions = predictions.sort_values(["model", "horizon", "origin_ts"]).reset_index(drop=True)

    report = WalkForwardReport(
        predictions=predictions,
        steps=len(origins_seen),
        fits=fits,
        skipped=skipped,
        first_origin=predictions["origin_ts"].min() if not predictions.empty else None,
        last_origin=predictions["origin_ts"].max() if not predictions.empty else None,
    )
    log.info(
        "walk-forward complete: %d origin(s), %d forecast(s), %d refit(s)",
        report.steps,
        len(predictions),
        sum(fits.values()),
    )
    if skipped:
        log.info("walk-forward skipped steps: %s", skipped)
    return report


def _columns() -> list[str]:
    return [
        "origin_ts",
        "model",
        "horizon",
        "base_price",
        "predicted_return",
        "predicted_price",
        "actual_return",
        "actual_price",
        "target_ts",
        "train_rows",
    ]


def _target_ts(index: pd.DatetimeIndex, position: int, horizon: int) -> pd.Timestamp | None:
    target = position + horizon
    return index[target] if target < len(index) else None


def _has_variance(targets: pd.Series) -> bool:
    values = targets.to_numpy(dtype=float)
    return bool(len(values) > 1 and np.nanstd(values) > 0)
