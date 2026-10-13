"""Evaluation metrics against hand-computed values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.validation.metrics import (
    directional_accuracy,
    equity_curve,
    evaluate_predictions,
    max_drawdown,
    mean_absolute_error,
    metrics_matrix,
    root_mean_squared_error,
    sample_non_overlapping,
    sharpe_ratio,
    sortino_ratio,
    strategy_returns,
)


def _predictions(rows):
    frame = pd.DataFrame(rows)
    frame["base_price"] = 100.0
    frame["predicted_price"] = 100.0 * (1 + frame["predicted_return"])
    frame["actual_price"] = 100.0 * (1 + frame["actual_return"])
    return frame


def test_mae_and_rmse_are_hand_checkable():
    actual = [100.0, 102.0, 99.0]
    predicted = [101.0, 100.0, 99.0]
    assert mean_absolute_error(actual, predicted) == pytest.approx((1 + 2 + 0) / 3)
    assert root_mean_squared_error(actual, predicted) == pytest.approx(np.sqrt((1 + 4 + 0) / 3))


def test_metrics_ignore_mismatched_nan_rows():
    assert mean_absolute_error([1.0, np.nan, 3.0], [1.0, 2.0, np.nan]) == 0.0
    assert np.isnan(mean_absolute_error([], []))


def test_directional_accuracy_counts_sign_matches():
    assert directional_accuracy([0.01, -0.02, 0.03], [0.05, -0.01, 0.02]) == pytest.approx(1.0)
    assert directional_accuracy([0.01, -0.02, 0.03], [-0.05, 0.01, 0.02]) == pytest.approx(1 / 3)


def test_directional_accuracy_is_undefined_without_directional_forecasts():
    assert np.isnan(directional_accuracy([0.01, -0.02], [0.0, 0.0]))
    assert np.isnan(directional_accuracy([0.0, 0.0], [0.01, 0.02]))


def test_sharpe_and_sortino_annualisation():
    returns = pd.Series([0.01, -0.005, 0.01, -0.005])
    expected_sharpe = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    assert sharpe_ratio(returns, 252) == pytest.approx(expected_sharpe)
    downside = np.sqrt(np.mean(returns[returns < 0] ** 2))
    assert sortino_ratio(returns, 252) == pytest.approx(returns.mean() / downside * np.sqrt(252))


def test_sharpe_is_undefined_for_constant_returns():
    assert np.isnan(sharpe_ratio(pd.Series([0.0, 0.0, 0.0]), 252))
    assert np.isnan(sharpe_ratio(pd.Series([0.01]), 252))


def test_sortino_is_undefined_without_downside():
    assert np.isnan(sortino_ratio(pd.Series([0.01, 0.02, 0.03]), 252))


def test_max_drawdown_of_a_known_path():
    returns = pd.Series([0.10, -0.50, 0.10])
    equity = equity_curve(returns)
    assert equity.iloc[0] == pytest.approx(1.10)
    assert max_drawdown(returns) == pytest.approx(0.55 / 1.10 - 1.0)


def test_max_drawdown_is_zero_for_a_monotonic_path():
    assert max_drawdown(pd.Series([0.01, 0.02, 0.01])) == pytest.approx(0.0)


def test_non_overlapping_sampling_uses_horizon_stride():
    frame = _predictions(
        {
            "origin_ts": pd.date_range("2024-01-01", periods=10, freq="D"),
            "model": "m",
            "horizon": 3,
            "predicted_return": 0.01,
            "actual_return": 0.02,
        }
    )
    sampled = sample_non_overlapping(frame, 3)
    assert len(sampled) == 4
    assert (sampled["origin_ts"].diff().dropna() == pd.Timedelta(days=3)).all()


def test_strategy_returns_apply_the_round_trip_cost():
    frame = _predictions(
        {
            "origin_ts": pd.date_range("2024-01-01", periods=4, freq="D"),
            "model": "m",
            "horizon": 1,
            "predicted_return": [0.01, -0.01, 0.02, 0.0],
            "actual_return": [0.02, 0.01, 0.03, 0.05],
        }
    )
    returns = strategy_returns(frame, 1, cost_bps=10.0)
    cost = 2 * 10.0 / 10_000.0
    assert returns.iloc[0] == pytest.approx(0.02 - cost)
    assert returns.iloc[1] == pytest.approx(-0.01 - cost)  # short into an up move
    assert returns.iloc[3] == 0.0  # flat signal takes no trade


def test_evaluate_predictions_covers_every_required_metric():
    frame = _predictions(
        {
            "origin_ts": pd.date_range("2024-01-01", periods=20, freq="D"),
            "model": ["m"] * 20,
            "horizon": [1] * 20,
            "predicted_return": np.linspace(-0.01, 0.01, 20),
            "actual_return": np.linspace(-0.02, 0.02, 20),
        }
    )
    metrics = evaluate_predictions(frame, periods_per_year=252, cost_bps=10.0)
    names = set(metrics["metric_name"])
    assert {
        "mae",
        "rmse",
        "direction_accuracy",
        "sharpe",
        "sortino",
        "max_drawdown",
        "n_trades",
    } <= names
    assert (metrics["n_samples"] > 0).all()
    assert not metrics_matrix(metrics).empty


def test_evaluate_predictions_adds_a_buy_hold_benchmark():
    frame = _predictions(
        {
            "origin_ts": pd.date_range("2024-01-01", periods=12, freq="D"),
            "model": ["m"] * 12,
            "horizon": [2] * 12,
            "predicted_return": 0.01,
            "actual_return": np.linspace(-0.01, 0.03, 12),
        }
    )
    metrics = evaluate_predictions(frame, periods_per_year=365, cost_bps=0.0)
    assert "buy_hold" in set(metrics["model"])
    horizon_rows = metrics[(metrics["model"] == "buy_hold") & (metrics["horizon"] == 2)]
    assert set(horizon_rows["metric_name"]) >= {"sharpe", "sortino", "max_drawdown"}


def test_evaluate_predictions_handles_empty_input():
    metrics = evaluate_predictions(pd.DataFrame(), 252, 10.0)
    assert metrics.empty
    assert "metric_name" in metrics.columns


def test_annualisation_factor_scales_sharpe_with_sqrt_periods():
    rows = {
        "origin_ts": pd.date_range("2024-01-01", periods=20, freq="D"),
        "model": ["m"] * 20,
        "horizon": [1] * 20,
        "predicted_return": np.linspace(-0.01, 0.01, 20),
        "actual_return": np.linspace(-0.02, 0.02, 20),
    }
    frame = _predictions(rows)
    daily = evaluate_predictions(frame, periods_per_year=252, cost_bps=0.0)
    quarterly = evaluate_predictions(frame, periods_per_year=63, cost_bps=0.0)
    sharpe_daily = float(daily[daily.metric_name == "sharpe"]["metric_value"].iloc[0])
    sharpe_quarterly = float(quarterly[quarterly.metric_name == "sharpe"]["metric_value"].iloc[0])
    assert sharpe_daily == pytest.approx(sharpe_quarterly * 2.0)


def test_horizon_k_annualises_by_the_holding_period():
    """A k-bar strategy is annualised with sqrt(periods_per_year / k)."""
    rows = {
        "origin_ts": pd.date_range("2024-01-01", periods=20, freq="D"),
        "model": ["m"] * 20,
        "horizon": [2] * 20,
        "predicted_return": np.linspace(-0.01, 0.01, 20),
        "actual_return": np.linspace(-0.02, 0.02, 20),
    }
    metrics = evaluate_predictions(_predictions(rows), periods_per_year=252, cost_bps=0.0)
    row = metrics[metrics.metric_name == "sharpe"].iloc[0]
    assert "annualised /2 bars" in row["notes"]
    assert int(row["n_samples"]) == 10  # 20 origins sampled every 2 bars
