"""CLI surface: flag placement and exit codes."""

from __future__ import annotations

import app
import pytest


def test_flags_are_accepted_before_and_after_the_subcommand():
    before = app.build_parser().parse_args(["--symbol", "NVDA", "demo"])
    after = app.build_parser().parse_args(["demo", "--symbol", "NVDA"])
    assert before.symbol == after.symbol == "NVDA"
    assert before.command == after.command == "demo"


def test_every_stage_is_exposed():
    parser = app.build_parser()
    for stage in ("check", "status", "run-all", "run", "demo", "ingest",
                  "sentiment", "features", "model", "charts", "report"):
        args = parser.parse_args([stage])
        assert args.command == stage


def test_check_returns_nonzero_without_credentials(monkeypatch, capsys):
    for key in ("APIFY_TOKEN", "APIFY_ACTOR_ID", "APIFY_DATASET_ID", "TAVILY_API_KEY", "EXA_API_KEY", "DEMO_MODE"):
        monkeypatch.delenv(key, raising=False)
    assert app.main(["check"]) == 2


def test_check_succeeds_in_demo_mode(monkeypatch, capsys):
    for key in ("APIFY_TOKEN", "APIFY_ACTOR_ID", "APIFY_DATASET_ID", "TAVILY_API_KEY", "EXA_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    assert app.main(["--demo", "check"]) == 0


def test_invalid_configuration_is_rejected(monkeypatch):
    for key in ("APIFY_TOKEN", "TAVILY_API_KEY", "EXA_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("BAR_INTERVAL", "fortnight")
    assert app.main(["check"]) == 2
    monkeypatch.setenv("BAR_INTERVAL", "1d")
    monkeypatch.setenv("MODELS", "not_a_model")
    assert app.main(["check"]) == 2
