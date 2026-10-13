"""Evaluation report: shared block structure rendered to Markdown and PDF.

Metrics in this report are read from the evaluation table produced by the
walk-forward run -- nothing is estimated, rounded for appearance or filled in.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.logging_config import get_logger
from config.settings import AppSettings
from src.features.builder import FEATURE_DEFINITIONS, FEATURE_GROUPS
from src.validation.metrics import metrics_matrix

log = get_logger(__name__)

Block = tuple[str, Any]


# --------------------------------------------------------------------- builders
def h1(text: str) -> Block:
    return ("h1", text)


def h2(text: str) -> Block:
    return ("h2", text)


def para(text: str) -> Block:
    return ("para", text)


def bullets(items: list[str]) -> Block:
    return ("bullets", items)


def table(headers: list[str], rows: list[list[str]], caption: str | None = None) -> Block:
    return ("table", {"headers": headers, "rows": rows, "caption": caption})


def image(path: Path | str, caption: str | None = None) -> Block:
    return ("image", {"path": str(path), "caption": caption})


def fmt(value: float | None, digits: int = 3, suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:,.{digits}f}{suffix}"


def build_report_blocks(
    settings: AppSettings,
    coverage: dict,
    walk_forward_summary: dict,
    metrics: pd.DataFrame,
    forecast: pd.DataFrame,
    importance: pd.DataFrame,
    chart_paths: list[Path],
    run_id: str,
) -> list[Block]:
    symbol = settings.asset_symbol
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    mode = "DEMO — SYNTHETIC DATA" if settings.demo_mode else "LIVE"

    blocks: list[Block] = [
        h1(f"{symbol} — X Sentiment & Price Forecast Evaluation"),
        para(
            f"Generated {generated} · Run `{run_id}` · Mode: **{mode}** · "
            f"Bars: {settings.bar_interval} · Window: {settings.period_days} days"
        ),
    ]
    if settings.demo_mode:
        blocks.append(
            para(
                "**This report was produced in DEMO MODE.** Prices and posts are seeded synthetic "
                "data from `src/ingestion/synthetic.py`. The pipeline mechanics, metrics and charts "
                "are real; the market content is not. Do not quote these numbers as market results."
            )
        )

    blocks += _methodology(settings)
    blocks += _data_coverage(settings, coverage)
    blocks += _feature_section()
    blocks += _model_section(settings)
    blocks += _validation_section(settings, walk_forward_summary)
    blocks += _accuracy_section(metrics)
    blocks += _baseline_section(metrics)
    blocks += _risk_section(settings, metrics)
    blocks += _forecast_section(forecast)
    blocks += _importance_section(importance)
    blocks += _charts_section(chart_paths, settings)
    blocks += _limitations(settings, metrics, walk_forward_summary)
    return blocks


def _methodology(settings: AppSettings) -> list[Block]:
    return [
        h2("1. Methodology"),
        para(
            "The pipeline ingests X (Twitter) posts and historical OHLCV prices for one asset, scores "
            "every post for sentiment, aggregates posts onto the price-bar grid, engineers a causal "
            "feature set, and forecasts the close at n+1 … n+5. n+0 is reported as the latest observed "
            "close rather than treated as a forecast."
        ),
        bullets(
            [
                "Ingestion: Apify actor for X posts, yfinance for prices, Tavily/Exa for news context.",
                "Storage: SQLite tables for prices, posts, bar sentiment, news, forecasts and metrics; "
                "re-runs upsert on natural keys so nothing is duplicated.",
                "Modelling: forward returns are the regression target; price forecasts are "
                "`base_price × (1 + predicted return)`.",
                "Evaluation: expanding-window walk-forward with a strict target-realisation rule, then "
                "error, directional and risk metrics against a naive baseline.",
                "News is deliberately excluded from the feature set — it is research context only.",
            ]
        ),
        para(
            f"Cost assumption: {settings.transaction_cost_bps:.1f} bps one-way, applied on entry and exit "
            f"of every simulated trade. Annualisation factor: {settings.resolved_annualization} periods/year, "
            f"with horizon-k strategies scaled by sqrt({settings.resolved_annualization}/k). Risk-free rate: 0."
        ),
    ]


def _data_coverage(settings: AppSettings, coverage: dict) -> list[Block]:
    rows = [
        [
            "Prices",
            "yfinance" if not settings.demo_mode else "synthetic (demo)",
            f"{coverage.get('price_bars', 0):,}",
            coverage.get("price_range", "n/a"),
            "OHLCV",
        ],
        [
            "X posts",
            "Apify actor" if not settings.demo_mode else "synthetic (demo)",
            f"{coverage.get('post_count', 0):,}",
            coverage.get("post_range", "n/a"),
            f"{coverage.get('bars_with_posts', 0):,} / {coverage.get('price_bars', 0):,} bars covered",
        ],
        [
            "News context",
            "Tavily + Exa" if not settings.demo_mode else "not queried in demo mode",
            f"{coverage.get('news_count', 0):,}",
            coverage.get("news_range", "n/a"),
            "context only, never a feature",
        ],
    ]
    notes = [
        f"Posts dropped during normalisation: {coverage.get('posts_dropped', 0):,}",
        f"Bars with no posts: {coverage.get('bars_without_posts', 0):,} "
        "(their sentiment is carried forward from the last bar that had posts)",
        f"Sentiment backend: {settings.sentiment_backend}",
        f"Ingestion problems recorded: {coverage.get('ingestion_notes', 'none')}",
    ]
    blocks = [h2("2. Data sources and coverage"), table(["Dataset", "Source", "Records", "Time range", "Notes"], rows)]
    if int(coverage.get("post_count") or 0) == 0:
        blocks.append(
            para(
                "**No X posts were available for this window.** The sentiment features are pinned "
                "to a neutral value, so every number in this report describes a price-only model. "
                "Do not read the tables below as evidence about sentiment. Re-run once X ingestion "
                "succeeds (check APIFY_ACTOR_ID / APIFY_DATASET_ID) before drawing conclusions."
            )
        )
    blocks.append(bullets(notes))
    return blocks


def _feature_section() -> list[Block]:
    rows = []
    for group, names in FEATURE_GROUPS.items():
        for name in names:
            rows.append([name, group, FEATURE_DEFINITIONS.get(name, "")])
    return [
        h2("3. Feature engineering"),
        para(
            f"{len(rows)} features, all computed from information available at the close of the "
            "prediction bar. Forward-looking columns exist only in the target frame."
        ),
        table(["Feature", "Group", "Definition"], rows),
    ]


def _model_section(settings: AppSettings) -> list[Block]:
    return [
        h2("4. Model selection"),
        para(
            "Models: " + ", ".join(f"`{name}`" for name in settings.models) + ". "
            "`naive` carries the last observed price forward (zero predicted return) and is the "
            "benchmark; the tree ensembles fit the same causal feature matrix per horizon."
        ),
        bullets(
            [
                "Random Forest: robust to the small, noisy sample and gives a readable importance ranking.",
                "HistGradientBoosting: handles the sparse sentiment bars natively and usually has the "
                "lowest variance at this sample size.",
                "XGBoost (optional): included only when installed; otherwise skipped with a warning.",
                "Each horizon is a separate regression target, so n+1 … n+5 are independent fits rather "
                "than a recursively chained forecast.",
            ]
        ),
    ]


def _validation_section(settings: AppSettings, summary: dict) -> list[Block]:
    rows = [
        ["Origins evaluated", f"{summary.get('steps', 0):,}"],
        ["First / last origin", f"{summary.get('first_origin', 'n/a')} → {summary.get('last_origin', 'n/a')}"],
        ["Out-of-sample forecasts", f"{summary.get('total_forecasts', 0):,}"],
        ["Minimum training rows", f"{settings.min_train_samples:,}"],
        ["Refit cadence", f"every {settings.refit_every} origin(s), expanding window only"],
        ["Skipped steps", summary.get("skipped_text", "none")],
    ]
    return [
        h2("5. Walk-forward validation"),
        para(
            "Expanding window. At origin i the model may only train on rows j ≤ i − k, so every training "
            "target close[j+k] was already realised when the forecast was made. Rows whose target would "
            "land at or after the origin are excluded — the leakage that a naive train/test split on "
            "time-indexed rows silently introduces."
        ),
        table(["Setting", "Value"], rows),
        para(
            "No scaler or imputer is fit across the split: the feature frame contains only causal "
            "columns and each refit sees strictly past observations."
        ),
    ]


def _accuracy_section(metrics: pd.DataFrame) -> list[Block]:
    if metrics.empty:
        return [h2("6. Accuracy metrics"), para("_No walk-forward predictions were produced._")]
    matrix = metrics_matrix(metrics)
    rows = []
    for (horizon, model), row in matrix.iterrows():
        if model in ("buy_hold", "observed"):
            continue
        rows.append(
            [
                f"n+{int(horizon)}",
                model,
                fmt(row.get("mae"), 2),
                fmt(row.get("rmse"), 2),
                fmt(row.get("mae_ret"), 5),
                fmt(row.get("direction_accuracy"), 3),
                f"{int(_n_samples(metrics, model, int(horizon)))}",
            ]
        )
    sample_note = ""
    if not metrics.empty:
        counts = metrics[metrics["metric_name"] == "mae"]["n_samples"]
        sample_note = f"Sample size per horizon: {int(counts.min())}–{int(counts.max())} origins."
    return [
        h2("6. Accuracy metrics (out-of-sample)"),
        table(
            ["Horizon", "Model", "MAE (price)", "RMSE (price)", "MAE (return)", "Directional acc.", "n"],
            rows,
            caption=sample_note,
        ),
        para(
            "MAE and RMSE are price-level errors; the return-scale columns are included because price "
            "levels drift and can mask the size of the relative error."
        ),
    ]


def _baseline_section(metrics: pd.DataFrame) -> list[Block]:
    if metrics.empty:
        return [h2("7. Baseline comparison"), para("_No metrics available._")]
    naive = {
        int(row["horizon"]): float(row["metric_value"])
        for _, row in metrics[(metrics["model"] == "naive") & (metrics["metric_name"] == "mae")].iterrows()
    }
    naive_dir = {
        int(row["horizon"]): float(row["metric_value"])
        for _, row in metrics[(metrics["model"] == "naive") & (metrics["metric_name"] == "direction_accuracy")].iterrows()
    }
    rows = []
    verdicts: dict[str, list[int]] = {}
    for _, row in metrics[(metrics["metric_name"] == "mae") & (metrics["model"] != "naive")].iterrows():
        horizon = int(row["horizon"])
        base = naive.get(horizon, float("nan"))
        ratio = float(row["metric_value"]) / base if base and not np.isnan(base) else float("nan")
        improvement = (1.0 - ratio) * 100.0 if not np.isnan(ratio) else float("nan")
        direction = _lookup(metrics, row["model"], horizon, "direction_accuracy")
        improvement_text = "n/a" if np.isnan(improvement) else f"{improvement:+.1f}%"
        rows.append(
            [
                f"n+{horizon}",
                row["model"],
                fmt(float(row["metric_value"]), 2),
                fmt(base, 2),
                fmt(ratio, 3),
                improvement_text,
                fmt(direction, 3),
                fmt(naive_dir.get(horizon, float("nan")), 3),
            ]
        )
        verdicts.setdefault(row["model"], []).append(-1 if not np.isnan(ratio) and ratio < 1.0 else 1)

    summary_lines = []
    for model, flags in verdicts.items():
        wins = sum(1 for flag in flags if flag < 0)
        summary_lines.append(
            f"`{model}` beats the naive baseline on MAE in {wins} of {len(flags)} horizons."
        )
    summary_lines.append(
        "A small MAE improvement over the naive last-price forecast is expected for short horizons in a "
        "low-volatility window and is not, on its own, evidence that sentiment carries predictive value. "
        "Read the directional accuracy and the strategy metrics next to it before drawing that conclusion."
    )
    return [
        h2("7. Baseline comparison"),
        table(
            ["Horizon", "Model", "MAE model", "MAE naive", "MAE ratio", "Improvement", "Dir. acc. model", "Dir. acc. naive"],
            rows,
            caption="Ratio < 1 means the model is closer to the realised price than the last-price baseline.",
        ),
        bullets(summary_lines),
    ]


def _risk_section(settings: AppSettings, metrics: pd.DataFrame) -> list[Block]:
    if metrics.empty:
        return [h2("8. Strategy risk metrics"), para("_No metrics available._")]
    rows = []
    for _, row in metrics[metrics["metric_name"] == "sharpe"].iterrows():
        horizon = int(row["horizon"])
        rows.append(
            [
                f"n+{horizon}",
                row["model"],
                fmt(float(row["metric_value"]), 2),
                fmt(_lookup(metrics, row["model"], horizon, "sortino"), 2),
                fmt(_lookup(metrics, row["model"], horizon, "max_drawdown"), 3),
                fmt(_lookup(metrics, row["model"], horizon, "win_rate"), 3),
                f"{int(row['n_samples'])}",
            ]
        )
    hold_rows = []
    for _, row in metrics[(metrics["model"] == "buy_hold") & (metrics["metric_name"] == "sharpe")].iterrows():
        horizon = int(row["horizon"])
        hold_rows.append(
            [
                f"n+{horizon}",
                fmt(float(row["metric_value"]), 2),
                fmt(_lookup(metrics, "buy_hold", horizon, "sortino"), 2),
                fmt(_lookup(metrics, "buy_hold", horizon, "max_drawdown"), 3),
                fmt(_lookup(metrics, "buy_hold", horizon, "mean_trade_return"), 5),
            ]
        )
    return [
        h2("8. Strategy risk metrics"),
        para(
            f"Signal: sign of the predicted return at the origin (long / short / flat). Entry at the origin "
            f"close, exit k bars later, origins sampled k bars apart so holding periods never overlap. "
            f"Trade return = position × realised k-bar return − 2 × {settings.transaction_cost_bps:.1f} bps. "
            f"Sharpe and Sortino annualise with sqrt({settings.resolved_annualization}/k) at a zero risk-free rate."
        ),
        table(["Horizon", "Model", "Sharpe", "Sortino", "MaxDD", "Win rate", "n blocks"], rows),
        table(["Horizon", "Sharpe", "Sortino", "MaxDD", "Mean block return"], hold_rows, caption="Always-long benchmark over the same sampled origins."),
        para(
            "These figures describe the behaviour of a mechanical rule applied to model output on a small "
            "out-of-sample sample. They are not a profitability claim: with few blocks the confidence "
            "interval around a Sharpe ratio is wider than the estimate itself."
        ),
    ]


def _forecast_section(forecast: pd.DataFrame) -> list[Block]:
    if forecast.empty:
        return [h2("9. Latest forecast"), para("_No forward forecast was produced._")]
    rows = []
    for _, row in forecast.sort_values(["horizon", "model"]).iterrows():
        rows.append(
            [
                f"n+{int(row['horizon'])}",
                row["model"],
                f"{row['base_price']:,.2f}",
                f"{row['predicted_price']:,.2f}",
                f"{row['predicted_return'] * 100:+.2f}%",
                str(row["target_ts"]),
            ]
        )
    return [
        h2("9. Latest forward forecast"),
        table(["Horizon", "Model", "Base (n+0)", "Predicted", "Implied return", "Target bar"], rows),
    ]


def _importance_section(importance: pd.DataFrame) -> list[Block]:
    if importance is None or importance.empty:
        return [h2("10. Feature importance"), para("_No importance ranking available (estimator does not expose one)._")]
    rows = []
    for _, row in importance.head(20).iterrows():
        rows.append([row["model"], f"n+{int(row['horizon'])}", int(row["rank"]), row["feature"], fmt(float(row["importance"]), 4)])
    return [
        h2("10. Feature importance"),
        table(["Model", "Horizon", "Rank", "Feature", "Importance"], rows),
        para(
            "Tree-split importance describes how the fitted model uses features in-sample; it is not a "
            "causal statement and it is not a feature-selection result."
        ),
    ]


def _charts_section(chart_paths: list[Path], settings: AppSettings) -> list[Block]:
    blocks = [h2("11. Charts")]
    for path in chart_paths:
        blocks.append(image(path, Path(path).stem.replace("_", " ")))
    return blocks


def _limitations(settings: AppSettings, metrics: pd.DataFrame, summary: dict) -> list[Block]:
    sample = 0
    if not metrics.empty:
        counts = metrics[metrics["metric_name"] == "mae"]["n_samples"]
        sample = int(counts.min()) if len(counts) else 0
    items = [
        f"**Sample size.** {summary.get('steps', 0)} walk-forward origins and about {sample} scored "
        "forecasts per horizon. Metrics on this sample are noisy and should be re-run on a longer "
        "history before any conclusion is drawn.",
        "**Social-media sampling bias.** X posts over-represent vocal, active accounts. Bots, "
        "coordinated campaigns and paid promotion are not filtered, and coverage is uneven across bars.",
        "**Sentiment model limits.** The scorer is lexicon/model based; sarcasm, memes and multi-asset "
        "threads are routinely mis-scored, which attenuates any real signal.",
        "**Correlation is not causation.** Sentiment moves with price. A model can pick up sentiment "
        "that merely restates the last return.",
        "**Execution.** Fills are assumed at bar closes with a fixed cost. Slippage, funding, borrow "
        "costs and partial fills are ignored.",
        "**Survivorship of the chosen asset.** One ticker over one window says nothing about the "
        "stability of the approach.",
    ]
    if settings.demo_mode:
        items.insert(
            0,
            "**DEMO MODE.** All data is synthetic and seeded; results demonstrate the pipeline, not a "
            "market edge.",
        )
    return [
        h2("12. Limitations and conclusions"),
        bullets(items),
        para(
            "Conclusion: the pipeline produces reproducible, out-of-sample forecasts and honest risk "
            "metrics for sentiment-augmented price prediction. Whether sentiment adds predictive value "
            "for this asset is answered by the baseline comparison and directional accuracy above — "
            "read them together, not in isolation."
        ),
    ]


# ---------------------------------------------------------------------- renderers
def to_markdown(blocks: list[Block], base_dir: Path | None = None) -> str:
    lines: list[str] = []
    for kind, payload in blocks:
        if kind == "h1":
            lines += [f"# {payload}", ""]
        elif kind == "h2":
            lines += [f"## {payload}", ""]
        elif kind == "para":
            lines += [payload, ""]
        elif kind == "bullets":
            lines += [f"- {item}" for item in payload] + [""]
        elif kind == "table":
            headers = payload["headers"]
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for row in payload["rows"]:
                lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
            if payload.get("caption"):
                lines += ["", f"_{payload['caption']}_"]
            lines.append("")
        elif kind == "image":
            path = Path(payload["path"])
            relative = path.name if base_dir is None else _relative(path, base_dir)
            caption = payload.get("caption") or path.stem
            lines += [f"![{caption}]({relative})", ""]
    return "\n".join(lines).rstrip() + "\n"


def write_markdown(blocks: list[Block], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_markdown(blocks, base_dir=path.parent), encoding="utf-8")
    log.info("evaluation report written to %s", path)
    return path


def write_pdf(blocks: list[Block], path: Path) -> Path | None:
    """Render the same blocks to PDF with reportlab (skipped if not installed)."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            Image as RLImage,
            LongTable,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            TableStyle,
        )
    except ImportError:
        log.warning("reportlab is not installed; skipping the PDF report (Markdown is still written)")
        return None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=8.6, leading=11.5, spaceAfter=5)
    heading1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=15, spaceAfter=8, textColor=colors.HexColor("#1B2126"))
    heading2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=11.5, spaceBefore=9, spaceAfter=5, textColor=colors.HexColor("#25706A"))
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=9, bulletIndent=2)
    caption = ParagraphStyle("caption", parent=body, fontSize=7.2, textColor=colors.HexColor("#5C6B70"))

    story: list = []
    for kind, payload in blocks:
        if kind == "h1":
            story.append(Paragraph(_escape(payload), heading1))
        elif kind == "h2":
            story.append(Paragraph(_escape(payload), heading2))
        elif kind == "para":
            story.append(Paragraph(_md_inline(payload), body))
        elif kind == "bullets":
            for item in payload:
                story.append(Paragraph(_md_inline(item), bullet, bulletText="•"))
        elif kind == "table":
            data = [[Paragraph(f"<b>{_escape(str(cell))}</b>", body) for cell in payload["headers"]]]
            data += [[Paragraph(_escape(str(cell)), body) for cell in row] for row in payload["rows"]]
            grid = LongTable(data, repeatRows=1, hAlign="LEFT")
            grid.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2A3238")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#B8C0C2")),
                        ("FONTSIZE", (0, 0), (-1, -1), 7.2),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F4F4")]),
                    ]
                )
            )
            story.append(grid)
            if payload.get("caption"):
                story.append(Spacer(1, 2 * mm))
                story.append(Paragraph(_escape(payload["caption"]), caption))
            story.append(Spacer(1, 3 * mm))
        elif kind == "image":
            file_path = Path(payload["path"])
            if file_path.exists():
                chart = RLImage(str(file_path))
                chart.drawWidth = 170 * mm
                chart.drawHeight = chart.drawHeight * (170 * mm / chart.drawWidth) if chart.drawWidth else 90 * mm
                story.append(chart)
                if payload.get("caption"):
                    story.append(Paragraph(_escape(payload["caption"]), caption))
                story.append(Spacer(1, 3 * mm))

    document = SimpleDocTemplate(
        str(path), pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=16 * mm, bottomMargin=16 * mm
    )
    document.build(story)
    log.info("evaluation PDF written to %s", path)
    return path


def _escape(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _md_inline(text: str) -> str:
    """Minimal markdown emphasis handling for the PDF renderer."""
    import re

    escaped = _escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"`(.+?)`", r'<font face="Courier">\1</font>', escaped)
    return escaped


def _relative(path: Path, base_dir: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(base_dir).resolve()))
    except ValueError:
        import os

        return os.path.relpath(Path(path).resolve(), Path(base_dir).resolve()).replace("\\", "/")


def _n_samples(metrics: pd.DataFrame, model: str, horizon: int) -> int:
    match = metrics[(metrics["model"] == model) & (metrics["horizon"] == horizon)]
    return int(match["n_samples"].max()) if not match.empty else 0


def _lookup(metrics: pd.DataFrame, model: str, horizon: int, name: str) -> float:
    match = metrics[
        (metrics["model"] == model) & (metrics["horizon"] == horizon) & (metrics["metric_name"] == name)
    ]
    return float("nan") if match.empty else float(match["metric_value"].iloc[0])
