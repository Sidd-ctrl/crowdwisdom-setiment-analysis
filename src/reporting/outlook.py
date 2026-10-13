"""One-page sentiment outlook.

The report keeps measured numbers and interpretation in separate sections, and
every interpretive label is tied to the threshold that produced it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config.logging_config import get_logger
from config.settings import AppSettings

log = get_logger(__name__)

TONE_STRONG = 0.20
TONE_MILD = 0.08
SCORE_POSITIVE = 0.05
SCORE_NEGATIVE = -0.05


def _pct(value: float | None) -> str:
    return "n/a" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value * 100:+.2f}%"


def _num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:.{digits}f}"


def describe_tone(net_ratio: float, mean_score: float) -> tuple[str, str]:
    """Label the crowd tone and return the rule that produced the label."""
    rule = (
        f"net tone = pos_ratio − neg_ratio over the window; strong if |net| > {TONE_STRONG}, "
        f"mild if |net| > {TONE_MILD}; mean sentiment positive above {SCORE_POSITIVE}, "
        f"negative below {SCORE_NEGATIVE}"
    )
    if np.isnan(net_ratio) or np.isnan(mean_score):
        return "insufficient data", rule
    if net_ratio > TONE_STRONG and mean_score > SCORE_POSITIVE:
        return "clearly positive", rule
    if net_ratio < -TONE_STRONG and mean_score < SCORE_NEGATIVE:
        return "clearly negative", rule
    if net_ratio > TONE_MILD:
        return "mildly positive", rule
    if net_ratio < -TONE_MILD:
        return "mildly negative", rule
    return "mixed / neutral", rule


def build_outlook(
    settings: AppSettings,
    prices: pd.DataFrame,
    sentiment: pd.DataFrame,
    posts: pd.DataFrame,
    news: pd.DataFrame,
    forecast: pd.DataFrame,
    metrics: pd.DataFrame,
    run_id: str,
) -> str:
    symbol = settings.asset_symbol
    close = prices["close"]
    latest_ts = prices.index[-1]
    latest_price = float(close.iloc[-1])
    mode = "DEMO — SYNTHETIC DATA (no live API calls were made)" if settings.demo_mode else "LIVE"

    window = min(7, len(sentiment))
    recent = sentiment.tail(window)
    prior = sentiment.iloc[-2 * window : -window] if len(sentiment) >= 2 * window else sentiment.iloc[:0]

    mean_score = float(recent["mean_score"].mean()) if recent["mean_score"].notna().any() else float("nan")
    net_ratio = float((recent["pos_ratio"] - recent["neg_ratio"]).mean())
    tone, tone_rule = describe_tone(net_ratio, mean_score)

    posts_recent = float(recent["post_count"].sum())
    posts_prior = float(prior["post_count"].sum()) if not prior.empty else float("nan")
    volume_change = (
        (posts_recent / posts_prior - 1.0) if posts_prior and not np.isnan(posts_prior) and posts_prior > 0 else float("nan")
    )
    engagement_recent = float(
        (recent["likes"].fillna(0) + 2 * recent["reposts"].fillna(0) + recent["views"].fillna(0) / 100).sum()
    )
    engagement_prior = (
        float((prior["likes"].fillna(0) + 2 * prior["reposts"].fillna(0) + prior["views"].fillna(0) / 100).sum())
        if not prior.empty
        else float("nan")
    )
    engagement_change = (
        engagement_recent / engagement_prior - 1.0
        if engagement_prior and not np.isnan(engagement_prior) and engagement_prior > 0
        else float("nan")
    )
    bars_with_posts = int((sentiment["post_count"] > 0).sum())

    lines = [
        f"# {symbol} — X Sentiment Outlook",
        "",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · "
        f"**Data mode:** {mode} · **Run:** `{run_id}`",
        "",
        "> n+0 is the latest observed close, not a forecast. n+1 … n+5 are model forecasts.",
        "",
        "## 1. Measured snapshot",
        "",
        "| Measure | Value |",
        "| --- | --- |",
        f"| Latest observed price (n+0) | {latest_price:,.2f} |",
        f"| Change vs previous bar | {_pct(_safe_pct(close, 1))} |",
        f"| Change over last 5 bars | {_pct(_safe_pct(close, 5))} |",
        f"| Change over last 20 bars | {_pct(_safe_pct(close, 20))} |",
        f"| Realised volatility (10-bar, annualised reference {settings.resolved_annualization}/yr) | "
        f"{_num(float(close.pct_change().tail(10).std()))} per bar |",
        f"| Mean sentiment, last {window} bar(s) | {_num(mean_score)} |",
        f"| Positive share, last {window} bar(s) | {_num(float(recent['pos_ratio'].mean()), 3)} |",
        f"| Negative share, last {window} bar(s) | {_num(float(recent['neg_ratio'].mean()), 3)} |",
        f"| Net tone (pos − neg) | {_num(net_ratio, 3)} |",
        f"| Posts, last {window} bar(s) | {posts_recent:,.0f} |",
        f"| Post volume change vs prior window | {_pct(volume_change)} |",
        f"| Weighted engagement change | {_pct(engagement_change)} |",
        f"| Bars with ≥1 post (coverage) | {bars_with_posts} / {len(sentiment)} |",
        f"| Total posts analysed | {len(posts):,} |",
        "",
        "## 2. Interpretation",
        "",
        f"- **Crowd tone: {tone}.** Rule applied: {tone_rule}.",
        f"- Evidence: mean sentiment {_num(mean_score)} with net tone {_num(net_ratio, 3)} over "
        f"{window} bar(s) covering {posts_recent:,.0f} posts.",
    ]

    if not np.isnan(volume_change):
        direction = "up" if volume_change > 0 else "down"
        lines.append(
            f"- Attention is {direction} {abs(volume_change) * 100:.1f}% versus the prior window "
            f"({posts_recent:,.0f} vs {posts_prior:,.0f} posts). Rising attention amplifies both "
            "positive and negative readings; it is not directional evidence by itself."
        )
    else:
        lines.append("- Attention change could not be computed: the prior window has no posts.")

    if bars_with_posts / max(1, len(sentiment)) < 0.6:
        lines.append(
            f"- **Coverage warning:** only {bars_with_posts} of {len(sentiment)} bars contain posts, so "
            "bar-level sentiment is sparse and largely carried forward from earlier bars."
        )

    lines += [
        "- Sentiment is a *description* of crowd commentary, not a causal driver. Nothing here should be "
        "read as a trading recommendation.",
        "",
        "## 3. Model forecast summary",
        "",
    ]
    lines += _forecast_table(forecast, metrics)
    lines += [
        "",
        "## 4. Recent news context",
        "",
    ]
    lines += _news_section(news, settings)
    lines += [
        "",
        "## 5. Risks and data limitations",
        "",
        f"- **Sample size:** the walk-forward evaluation covers a limited number of origins "
        f"(see the evaluation report). Sharpe/Sortino/MaxDD on that many overlapping blocks are "
        "indicative only.",
        "- **Social-media sampling bias:** X posts over-represent active, opinionated accounts; "
        "retail chatter is not a representative sample of market participants, and bot or "
        "coordinated activity is not filtered out.",
        "- **Sentiment model limits:** VADER is a lexicon-based general-purpose scorer. Sarcasm, "
        "memes, cashtag irony and multi-asset threads are frequently mis-scored.",
        "- **Costs and execution:** the strategy metrics assume fills at bar closes with the "
        f"configured {settings.transaction_cost_bps:.1f} bps one-way cost. Slippage, funding and "
        "partial fills are not modelled.",
        "- **No profitability claim:** out-of-sample error and risk metrics are reported so the "
        "forecast quality can be judged, not as evidence of a tradable edge.",
    ]
    if settings.demo_mode:
        lines.append(
            "- **DEMO MODE:** every number above comes from seeded synthetic data generated by "
            "`src/ingestion/synthetic.py`. It demonstrates pipeline mechanics only and must not be "
            "quoted as a market result."
        )
    lines.append("")
    return "\n".join(lines)


def _forecast_table(forecast: pd.DataFrame, metrics: pd.DataFrame) -> list[str]:
    if forecast.empty:
        return ["_No forecasts available for this run._"]
    lines = [
        "| Horizon | Model | Predicted price | Implied return | Walk-forward MAE | Directional accuracy |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in forecast.sort_values(["horizon", "model"]).iterrows():
        mae = _metric(metrics, row["model"], int(row["horizon"]), "mae")
        direction = _metric(metrics, row["model"], int(row["horizon"]), "direction_accuracy")
        mae_text = "n/a" if np.isnan(mae) else f"{mae:,.2f}"
        direction_text = "n/a" if np.isnan(direction) else f"{direction * 100:.1f}%"
        lines.append(
            f"| n+{int(row['horizon'])} | {row['model']} | {row['predicted_price']:,.2f} | "
            f"{row['predicted_return'] * 100:+.2f}% | {mae_text} | {direction_text} |"
        )
    return lines


def _news_section(news: pd.DataFrame, settings: AppSettings) -> list[str]:
    if news is None or news.empty:
        reason = (
            "Demo mode does not call news providers."
            if settings.demo_mode
            else "No news provider returned results (check TAVILY_API_KEY / EXA_API_KEY)."
        )
        return [f"_No news context available. {reason}_", "", "News is research context only and is never used as a model feature."]
    lines = []
    for _, row in news.head(8).iterrows():
        published = row.get("published_at") or "unknown date"
        title = row.get("title") or "(untitled)"
        url = row.get("url")
        lines.append(f"- **{title}** — {row['source']}, {published}")
        if row.get("summary"):
            lines.append(f"  - {str(row['summary'])[:280]}")
        if url:
            lines.append(f"  - {url}")
    lines.append("")
    lines.append("_News is research context only and is never used as a model feature._")
    return lines


def _metric(metrics: pd.DataFrame, model: str, horizon: int, name: str) -> float:
    if metrics is None or metrics.empty:
        return float("nan")
    match = metrics[
        (metrics["model"] == model) & (metrics["horizon"] == horizon) & (metrics["metric_name"] == name)
    ]
    return float("nan") if match.empty else float(match["metric_value"].iloc[0])


def _safe_pct(close: pd.Series, periods: int) -> float | None:
    if len(close) <= periods:
        return None
    return float(close.iloc[-1] / close.iloc[-1 - periods] - 1.0)


def write_outlook(markdown: str, settings: AppSettings, run_id: str) -> Path:
    settings.report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = settings.report_dir / f"{settings.asset_symbol}_sentiment_outlook_{stamp}_{run_id[:8]}.md"
    path.write_text(markdown, encoding="utf-8")
    log.info("sentiment outlook written to %s", path)
    return path
