#!/usr/bin/env python3
"""CrowdWisdomTrading X sentiment & price forecasting pipeline.

Single entry point:

    python app.py demo                     # full run on labelled synthetic data
    python app.py run-all                  # full live run (needs credentials)
    python app.py run-all --symbol NVDA    # any yfinance symbol
    python app.py check                    # validate configuration only
    python app.py status                   # show what is stored in SQLite
    python app.py ingest|sentiment|features|model|charts|report
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.logging_config import get_logger, setup_logging  # noqa: E402
from config.settings import load_settings, validate_settings  # noqa: E402
from src.pipeline import (  # noqa: E402
    run_pipeline,
    stage_charts,
    stage_features,
    stage_ingest,
    stage_model,
    stage_report,
    stage_sentiment,
)
from src.storage.database import Database  # noqa: E402

log = get_logger("app")

STAGE_FUNCTIONS = {
    "ingest": stage_ingest,
    "sentiment": stage_sentiment,
    "features": stage_features,
    "charts": stage_charts,
}


def build_parser() -> argparse.ArgumentParser:
    # SUPPRESS keeps a flag given *before* the subcommand from being reset to a
    # default when the subparser runs, so both placements work.
    sup = argparse.SUPPRESS
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--env-file", default=sup, help="path to a .env file (default: ./.env)")
    common.add_argument("--symbol", default=sup, help="asset symbol, e.g. BTC-USD, TSLA, NVDA")
    common.add_argument("--interval", default=sup, help="bar interval: 1d, 1h, 60m, 15m, ...")
    common.add_argument("--days", type=int, default=sup, help="lookback window in days")
    common.add_argument("--db", default=sup, help="SQLite database path")
    common.add_argument("--models", default=sup, help="comma separated: naive,random_forest,hist_gbm,xgboost")
    common.add_argument("--horizons", default=sup, help="comma separated horizons, e.g. 1,2,3,4,5")
    common.add_argument("--seed", type=int, default=sup, help="random seed")
    common.add_argument("--log-level", default=sup, help="DEBUG, INFO, WARNING, ERROR")
    common.add_argument("--demo", action="store_true", default=sup, help="force demo mode (synthetic data)")
    common.add_argument("--live", action="store_true", default=sup, help="force live mode even if DEMO_MODE=true in .env")

    parser = argparse.ArgumentParser(
        prog="app.py",
        parents=[common],
        description="X sentiment and price forecasting pipeline (CrowdWisdomTrading assessment).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("check", parents=[common], help="validate configuration and exit")
    subparsers.add_parser("status", parents=[common], help="print stored row counts and exit")
    subparsers.add_parser("run-all", parents=[common], help="run every stage in order")
    subparsers.add_parser("run", parents=[common], help="alias for run-all")
    subparsers.add_parser("demo", parents=[common], help="run every stage on labelled synthetic data")
    for stage in ("ingest", "sentiment", "features", "model", "charts", "report"):
        subparsers.add_parser(stage, parents=[common], help=f"run only the {stage} stage")
    return parser


def apply_overrides(args: argparse.Namespace) -> None:
    """CLI flags win over the environment, so nothing has to be edited by hand."""
    mapping = {
        "symbol": "ASSET_SYMBOL",
        "interval": "BAR_INTERVAL",
        "days": "PERIOD_DAYS",
        "db": "DB_PATH",
        "models": "MODELS",
        "horizons": "FORECAST_HORIZONS",
        "seed": "RANDOM_SEED",
        "log_level": "LOG_LEVEL",
    }
    for attribute, variable in mapping.items():
        value = getattr(args, attribute, None)
        if value is not None:
            os.environ[variable] = str(value)
    if getattr(args, "demo", False):
        os.environ["DEMO_MODE"] = "true"
    if getattr(args, "live", False):
        os.environ["DEMO_MODE"] = "false"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    apply_overrides(args)

    try:
        settings = load_settings(getattr(args, "env_file", None))
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    setup_logging(settings.log_level)
    command = args.command or "run-all"
    if command == "demo" or (command in ("run-all", "run") and getattr(args, "demo", False)):
        settings = replace(settings, demo_mode=True)

    report = validate_settings(settings)
    log.info("\n%s", report.as_text(settings))

    if command == "check":
        return 0 if report.ok else 2

    if not report.ok:
        for error in report.errors:
            log.error("configuration: %s", error)
        log.error("run 'python app.py check' for details, or use 'python app.py demo'")
        return 2

    settings.ensure_directories()
    db = Database(settings.db_path)

    if command == "status":
        for table, count in db.table_counts().items():
            log.info("%-18s %d row(s)", table, count)
        return 0

    if command in ("run-all", "run", "demo"):
        outcome = run_pipeline(settings, db)
    elif command in STAGE_FUNCTIONS:
        result = STAGE_FUNCTIONS[command](settings, db)
        outcome = {"run_id": None, "results": [result], "failed": [] if result.ok else [result.name], "counts": db.table_counts()}
    elif command == "model":
        result = stage_model(settings, db)
        outcome = {"run_id": result.summary.get("run_id"), "results": [result], "failed": [] if result.ok else ["model"], "counts": db.table_counts()}
    elif command == "report":
        result = stage_report(settings, db)
        outcome = {"run_id": None, "results": [result], "failed": [] if result.ok else ["report"], "counts": db.table_counts()}
    else:  # pragma: no cover - argparse restricts the choices
        log.error("unknown command '%s'", command)
        return 2

    print()
    print("Pipeline summary")
    print(f"  mode      : {'DEMO (synthetic)' if settings.demo_mode else 'LIVE'}")
    print(f"  run id    : {outcome.get('run_id') or 'n/a'}")
    print(f"  database  : {db.path}")
    for table, count in outcome["counts"].items():
        print(f"  {table:<16}: {count}")
    for result in outcome["results"]:
        print(f"  {result.describe()}")
    for artefact_dir, label in ((settings.chart_dir, "charts"), (settings.report_dir, "reports"), (settings.forecast_dir, "forecasts")):
        files = sorted(artefact_dir.glob("*")) if artefact_dir.exists() else []
        if files:
            print(f"  {label:<10}: {len(files)} file(s) in {artefact_dir}")
    if settings.demo_mode:
        print("\n  DEMO MODE: all data is synthetic and clearly labelled. Not a market result.")

    return 1 if outcome["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
