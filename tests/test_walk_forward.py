"""Walk-forward validation: training order, leakage and horizon handling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import registry
from src.models.train import fit_on_history, forecast_ahead
from src.validation.walk_forward import run_walk_forward

HORIZONS = (1, 2, 3)


class MeanModel:
    """Deterministic stand-in: predicts the mean of the training targets."""

    name = "mean_model"

    def fit(self, features, returns):
        self.mean_ = float(returns.mean())
        self.n_train_ = len(returns)
        self.last_train_index_ = features.index[-1]
        return self

    def predict(self, features, base_price=None):
        return np.full(len(features), self.mean_)


class RecordingModel(MeanModel):
    """Records every (last training bar, origin bar) pair it is asked about."""

    name = "recorder"

    def __init__(self):
        self.pairs = []

    def fit(self, features, returns):
        super().fit(features, returns)
        self._last_train = features.index[-1]
        return self

    def predict(self, features, base_price=None):
        self.pairs.append((self._last_train, features.index[-1]))
        return super().predict(features, base_price)


@pytest.fixture(autouse=True)
def register_test_models(monkeypatch):
    """Expose the deterministic stand-ins to the model registry."""
    monkeypatch.setitem(registry.BUILDERS, "mean_model_probe", lambda seed: MeanModel())


@pytest.fixture()
def recorder_model(monkeypatch):
    model = RecordingModel()
    monkeypatch.setitem(registry.BUILDERS, "recorder", lambda seed: model)
    return model


@pytest.fixture()
def dataset(prices, scored_posts):
    from src.features.builder import build_features
    from src.sentiment.aggregator import aggregate_bars

    sentiment = aggregate_bars(scored_posts, prices.index, "1d")
    features, returns, _, base = build_features(prices, sentiment, HORIZONS)
    return features, returns, base


def _run(dataset, recorder_model, **overrides):
    features, returns, base = dataset
    kwargs = dict(
        model_names=("recorder",),
        horizons=HORIZONS,
        min_train_samples=10,
        refit_every=2,
        seed=42,
    )
    kwargs.update(overrides)
    return run_walk_forward(features, returns, base, **kwargs)


def test_training_rows_are_always_realised_before_the_origin(dataset, recorder_model, prices):
    report = _run(dataset, recorder_model)
    assert not report.empty
    assert recorder_model.pairs, "the recording model was never called"
    index = prices.index
    for last_train, origin in recorder_model.pairs:
        train_position = index.get_loc(last_train)
        origin_position = index.get_loc(origin)
        # every training target close[j+k] must be known at origin i
        assert origin_position - train_position >= min(HORIZONS)
        assert train_position < origin_position


def test_predictions_have_one_row_per_model_horizon_origin(dataset, recorder_model):
    report = _run(dataset, recorder_model)
    frame = report.predictions
    assert not frame.duplicated(["model", "horizon", "origin_ts"]).any()
    assert set(frame["horizon"]) <= set(HORIZONS)
    assert (frame["actual_price"] == frame["base_price"] * (1 + frame["actual_return"])).all()
    assert (frame["predicted_price"] == frame["base_price"] * (1 + frame["predicted_return"])).all()


def test_truncating_the_future_does_not_change_earlier_forecasts(dataset, recorder_model):
    features, returns, base = dataset
    full = run_walk_forward(
        features, returns, base, model_names=("mean_model_probe",), horizons=HORIZONS,
        min_train_samples=10, refit_every=2, seed=42,
    )
    cut = len(features) - 4
    truncated = run_walk_forward(
        features.iloc[:cut], returns.iloc[:cut], base.iloc[:cut],
        model_names=("mean_model_probe",), horizons=HORIZONS,
        min_train_samples=10, refit_every=2, seed=42,
    )
    if full.empty or truncated.empty:
        pytest.skip("not enough rows for both runs")
    common = truncated.predictions[truncated.predictions["horizon"] == min(HORIZONS)].set_index("origin_ts")
    reference = full.predictions[full.predictions["horizon"] == min(HORIZONS)].set_index("origin_ts")
    overlap = common.index.intersection(reference.index)
    assert len(overlap) > 0
    pd.testing.assert_series_equal(
        common.loc[overlap, "predicted_return"], reference.loc[overlap, "predicted_return"], check_names=False
    )


def test_expanding_window_never_shrinks(dataset, recorder_model):
    report = _run(dataset, recorder_model)
    sizes = (
        report.predictions[report.predictions["horizon"] == min(HORIZONS)]
        .sort_values("origin_ts")["train_rows"]
        .to_numpy()
    )
    assert (np.diff(sizes) >= 0).all()


def test_insufficient_history_returns_empty_report(dataset):
    features, returns, base = dataset
    report = run_walk_forward(
        features.iloc[:8], returns.iloc[:8], base.iloc[:8],
        model_names=("naive",), horizons=HORIZONS, min_train_samples=50, refit_every=1, seed=42,
    )
    assert report.empty
    assert report.steps == 0


def test_naive_baseline_predicts_zero_return(dataset):
    features, returns, base = dataset
    report = run_walk_forward(
        features, returns, base, model_names=("naive",), horizons=HORIZONS,
        min_train_samples=10, refit_every=5, seed=42,
    )
    naive = report.predictions[report.predictions["model"] == "naive"]
    assert not naive.empty
    assert (naive["predicted_return"] == 0).all()
    assert (naive["predicted_price"] == naive["base_price"]).all()


def test_fit_on_history_covers_every_horizon(dataset):
    features, returns, _ = dataset
    fitted = fit_on_history(features, returns, ("naive",), seed=42)
    assert {(name, horizon) for name, horizon in fitted} == {("naive", horizon) for horizon in HORIZONS}


def test_forward_forecast_starts_at_the_observed_price(dataset, prices):
    features, returns, base = dataset
    fitted = fit_on_history(features, returns, ("naive", "mean_model_probe"), seed=42)
    forecast = forecast_ahead(
        fitted, features.iloc[[-1]], float(base.iloc[-1]), HORIZONS, features.index[-1], "1d"
    )
    observed = forecast[forecast["horizon"] == 0]
    assert len(observed) == 1
    assert observed["model"].iloc[0] == "observed"
    assert observed["predicted_price"].iloc[0] == pytest.approx(float(base.iloc[-1]))
    assert set(forecast["horizon"]) == {0, *HORIZONS}
    assert forecast["target_ts"].notna().all()


def test_each_horizon_gets_its_own_fitted_model(dataset):
    """Regression: a shared estimator would leave every horizon on the last fit."""
    features, returns, base = dataset
    fitted = fit_on_history(features, returns, ("mean_model_probe",), seed=42)
    objects = [fitted[("mean_model_probe", horizon)] for horizon in HORIZONS]
    assert len({id(model) for model in objects}) == len(HORIZONS)
    means = [model.mean_ for model in objects]
    assert len(set(round(value, 10) for value in means)) > 1


def test_forward_projections_differ_across_horizons(dataset):
    features, returns, base = dataset
    fitted = fit_on_history(features, returns, ("mean_model_probe",), seed=42)
    forecast = forecast_ahead(
        fitted, features.iloc[[-1]], float(base.iloc[-1]), HORIZONS, features.index[-1], "1d"
    )
    prices = forecast[forecast["model"] == "mean_model_probe"].sort_values("horizon")
    assert prices["predicted_price"].round(10).nunique() > 1
