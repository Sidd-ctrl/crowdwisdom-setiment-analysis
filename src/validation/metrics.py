"""Quantitative evaluation metrics.

Definitions used throughout the project:

* MAE / RMSE are computed on **price levels** (and, for reference, on returns).
* Directional accuracy is the share of forecasts whose predicted sign matches
  the realised sign of the forward return. Exactly-zero realised returns and
  exactly-zero forecasts are excluded rather than counted as misses, so a
  baseline that makes no directional claim reports "undefined" instead of 0.
* The trading signal for horizon k is ``sign(predicted return)``: long when the
  model expects a rise, short when it expects a fall, flat when it predicts
  exactly zero. Trades are entered at the origin bar close and exited k bars
  later; sampled origins are spaced k bars apart so blocks do not overlap.
* Trade return = ``position * realised k-bar return - 2 * cost`` where cost is
  the one-way transaction cost (entry plus exit). Cost is configurable via
  TRANSACTION_COST_BPS; set it to 0 to see the cost-free numbers.
* Sharpe and Sortino annualise with ``sqrt(periods_per_year / k)`` and use a
  zero risk-free rate. Sortino uses the downside deviation of negative returns
  only. Both are reported alongside the sample size, because a Sharpe computed
  on a few dozen overlapping blocks is not statistically meaningful.
* Maximum drawdown is the largest peak-to-trough fall of the compounded equity
  curve of the sampled trade returns, reported as a negative fraction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12

ERROR_METRICS = ("mae", "rmse", "mae_ret", "rmse_ret")
STRATEGY_METRICS = ("sharpe", "sortino", "max_drawdown", "mean_trade_return", "win_rate", "n_trades")


def mean_absolute_error(actual, predicted) -> float:
    actual, predicted = _aligned(actual, predicted)
    if len(actual) == 0:
        return float("nan")
    return float(np.mean(np.abs(actual - predicted)))


def root_mean_squared_error(actual, predicted) -> float:
    actual, predicted = _aligned(actual, predicted)
    if len(actual) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def directional_accuracy(actual_returns, predicted_returns) -> float:
    """Share of forecasts whose predicted sign matches the realised sign.

    Realised returns of exactly zero and forecasts of exactly zero (the naive
    baseline makes no directional claim) are excluded. If nothing directional
    remains the metric is undefined rather than 0, which would read as "always
    wrong".
    """
    actual, predicted = _aligned(actual_returns, predicted_returns)
    mask = (np.abs(actual) > EPS) & (np.abs(predicted) > EPS)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.sign(actual[mask]) == np.sign(predicted[mask])))


def equity_curve(returns) -> pd.Series:
    series = _as_series(returns)
    return (1.0 + series).cumprod()


def max_drawdown(returns) -> float:
    """Largest peak-to-trough decline of the compounded equity curve (<= 0)."""
    series = _as_series(returns)
    if series.empty:
        return float("nan")
    equity = equity_curve(series)
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def sharpe_ratio(returns, periods_per_year: int) -> float:
    series = _drop_nan(returns)
    if len(series) < 2:
        return float("nan")
    std = float(series.std(ddof=1))
    if std < EPS:
        return float("nan")
    return float(series.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(returns, periods_per_year: int) -> float:
    series = _drop_nan(returns)
    if len(series) < 2:
        return float("nan")
    downside = series[series < 0]
    if downside.empty:
        return float("nan")
    downside_std = float(np.sqrt(np.mean(downside**2)))
    if downside_std < EPS:
        return float("nan")
    return float(series.mean() / downside_std * np.sqrt(periods_per_year))


def sample_non_overlapping(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Keep every k-th origin so k-bar holding periods never overlap."""
    ordered = frame.sort_values("origin_ts")
    stride = max(1, int(horizon))
    return ordered.iloc[::stride].copy()


def signal_positions(frame: pd.DataFrame, horizon: int) -> pd.Series:
    """Long/short/flat position implied by the sign of each forecast."""
    return _positions_from(sample_non_overlapping(frame, horizon))


def _positions_from(sampled: pd.DataFrame) -> pd.Series:
    """Positions for an already-sampled frame (no further striding)."""
    if sampled.empty:
        return pd.Series(dtype=float)
    positions = np.sign(sampled["predicted_return"].to_numpy(dtype=float))
    return pd.Series(positions, index=pd.DatetimeIndex(sampled["origin_ts"]))


def strategy_returns(frame: pd.DataFrame, horizon: int, cost_bps: float) -> pd.Series:
    """Returns of the sign-of-forecast strategy on non-overlapping blocks."""
    sampled = sample_non_overlapping(frame, horizon)
    if sampled.empty:
        return pd.Series(dtype=float)
    position = _positions_from(sampled).to_numpy(dtype=float)
    realised = sampled["actual_return"].to_numpy(dtype=float)
    cost = float(cost_bps) / 10_000.0 * 2.0
    returns = np.where(position == 0, 0.0, position * realised - cost)
    return pd.Series(returns, index=pd.DatetimeIndex(sampled["origin_ts"]))


def buy_hold_returns(frame: pd.DataFrame, horizon: int, cost_bps: float) -> pd.Series:
    """Always-long benchmark over the same sampled origins."""
    sampled = sample_non_overlapping(frame, horizon)
    if sampled.empty:
        return pd.Series(dtype=float)
    realised = sampled["actual_return"].to_numpy(dtype=float)
    cost = float(cost_bps) / 10_000.0 * 2.0
    return pd.Series(realised - cost, index=pd.DatetimeIndex(sampled["origin_ts"]))


def evaluate_predictions(
    predictions: pd.DataFrame,
    periods_per_year: int,
    cost_bps: float,
) -> pd.DataFrame:
    """Long-form metrics table: one row per (model, horizon, metric)."""
    if predictions.empty:
        return pd.DataFrame(
            columns=["model", "horizon", "metric_name", "metric_value", "n_samples", "notes"]
        )

    rows: list[dict] = []
    horizons = sorted(predictions["horizon"].unique())

    for horizon in horizons:
        block = predictions[predictions["horizon"] == horizon]
        annual = periods_per_year / max(1, horizon)

        for model, group in block.groupby("model"):
            valid = group.dropna(subset=["actual_return", "actual_price"])
            if valid.empty:
                continue
            trades = strategy_returns(valid, horizon, cost_bps)
            positions = signal_positions(valid, horizon)
            n_trades = int((positions != 0).sum())
            flat_note = f"{cost_bps:.1f} bps one-way, rf=0, annualised /{max(1, horizon)} bars"
            if n_trades == 0:
                flat_note += "; no trades taken (signal never leaves zero)"
            win_rate = float(np.mean(trades[positions != 0] > 0)) if n_trades else float("nan")
            rows.extend(
                [
                    _row(model, horizon, "mae", mean_absolute_error(valid["actual_price"], valid["predicted_price"]), len(valid), "price level"),
                    _row(model, horizon, "rmse", root_mean_squared_error(valid["actual_price"], valid["predicted_price"]), len(valid), "price level"),
                    _row(model, horizon, "mae_ret", mean_absolute_error(valid["actual_return"], valid["predicted_return"]), len(valid), "return units"),
                    _row(model, horizon, "rmse_ret", root_mean_squared_error(valid["actual_return"], valid["predicted_return"]), len(valid), "return units"),
                    _row(model, horizon, "direction_accuracy", directional_accuracy(valid["actual_return"], valid["predicted_return"]), len(valid), "sign match, zero ties excluded"),
                    _row(model, horizon, "sharpe", sharpe_ratio(trades, annual), len(trades), flat_note),
                    _row(model, horizon, "sortino", sortino_ratio(trades, annual), len(trades), flat_note),
                    _row(model, horizon, "max_drawdown", max_drawdown(trades), len(trades), "compounded sampled trade returns"),
                    _row(model, horizon, "mean_trade_return", float(trades.mean()) if len(trades) else float("nan"), len(trades), "per sampled block"),
                    _row(model, horizon, "win_rate", win_rate, n_trades, "share of profitable trades"),
                    _row(model, horizon, "n_trades", n_trades, len(trades), "blocks with a non-zero position"),
                ]
            )

        benchmark = buy_hold_returns(block.dropna(subset=["actual_return"]), horizon, cost_bps)
        if not benchmark.empty:
            rows.extend(
                [
                    _row("buy_hold", horizon, "sharpe", sharpe_ratio(benchmark, annual), len(benchmark), "always long, same sampled origins"),
                    _row("buy_hold", horizon, "sortino", sortino_ratio(benchmark, annual), len(benchmark), "always long, same sampled origins"),
                    _row("buy_hold", horizon, "max_drawdown", max_drawdown(benchmark), len(benchmark), "always long, same sampled origins"),
                    _row("buy_hold", horizon, "mean_trade_return", float(benchmark.mean()), len(benchmark), "per sampled block"),
                    _row(
                        "buy_hold",
                        horizon,
                        "direction_accuracy",
                        _share_positive(block.dropna(subset=["actual_return"]), horizon),
                        len(benchmark),
                        "share of up blocks",
                    ),
                ]
            )

    return pd.DataFrame(rows)


def metrics_matrix(metrics: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long-form metrics into model x metric tables per horizon."""
    if metrics.empty:
        return pd.DataFrame()
    return metrics.pivot_table(
        index=["horizon", "model"], columns="metric_name", values="metric_value", aggfunc="first"
    )


def _share_positive(frame: pd.DataFrame, horizon: int) -> float:
    sampled = sample_non_overlapping(frame, horizon)
    if sampled.empty:
        return float("nan")
    return float(np.mean(sampled["actual_return"].to_numpy(dtype=float) > 0))


def _row(model: str, horizon: int, name: str, value, n_samples: int, notes: str) -> dict:
    number = float(value) if value is not None else float("nan")
    return {
        "model": model,
        "horizon": int(horizon),
        "metric_name": name,
        "metric_value": number,
        "n_samples": int(n_samples),
        "notes": notes,
    }


def _aligned(actual, predicted) -> tuple[np.ndarray, np.ndarray]:
    a = pd.Series(_as_series(actual)).astype(float)
    p = pd.Series(_as_series(predicted)).astype(float)
    frame = pd.concat([a.rename("a"), p.rename("p")], axis=1).dropna()
    return frame["a"].to_numpy(), frame["p"].to_numpy()


def _as_series(values) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.astype(float)
    return pd.Series(np.asarray(values, dtype=float))


def _drop_nan(returns) -> pd.Series:
    return _as_series(returns).dropna()
