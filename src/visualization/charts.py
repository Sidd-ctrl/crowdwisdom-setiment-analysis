"""Chart generation.

Static PNGs are always written; the two most useful views are additionally
exported as self-contained interactive HTML (plotly.js is inlined so the file
opens offline). Every chart states its data mode in the caption.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.logging_config import get_logger
from src.visualization.theme import (
    DIVERGING,
    PALETTE,
    footer,
    new_figure,
    new_subplots,
    save_png,
    style_ax,
)

log = get_logger(__name__)


def _mode_tag(demo_mode: bool) -> str:
    return "DEMO / SYNTHETIC DATA" if demo_mode else "LIVE DATA"


# -------------------------------------------------------------------------- 1
def price_vs_sentiment(
    prices: pd.DataFrame,
    sentiment: pd.DataFrame,
    symbol: str,
    out_dir: Path,
    demo_mode: bool = False,
    interactive: bool = True,
) -> list[Path]:
    """Close price against bar-level mean sentiment on a shared time axis."""
    out_dir = Path(out_dir)
    fig, ax = new_figure(11.0, 4.8)
    ax.plot(prices.index, prices["close"], color=PALETTE["white"], linewidth=1.4, label=f"{symbol} close")
    ax.fill_between(prices.index, prices["close"], prices["close"].min(), color=PALETTE["teal"], alpha=0.10)
    style_ax(ax, f"{symbol} — price vs X sentiment", ylabel="price")
    ax.legend(loc="upper left", frameon=True)

    twin = ax.twinx()
    twin.plot(
        sentiment.index,
        sentiment["mean_score"],
        color=PALETTE["amber"],
        linewidth=1.1,
        alpha=0.95,
        label="mean sentiment (VADER compound)",
    )
    twin.axhline(0.0, color=PALETTE["muted"], linewidth=0.7, linestyle="--")
    twin.set_ylabel("mean sentiment", color=PALETTE["amber"])
    twin.tick_params(axis="y", colors=PALETTE["amber"], length=2)
    twin.set_ylim(-1, 1)
    twin.spines["top"].set_visible(False)
    twin.grid(False)
    handles, labels = ax.get_legend_handles_labels()
    twin_handles, twin_labels = twin.get_legend_handles_labels()
    ax.legend(handles + twin_handles, labels + twin_labels, loc="upper left", frameon=True)
    footer(fig, _mode_tag(demo_mode) + " · sentiment aggregated per bar from posts published inside that bar")

    paths = [save_png(fig, out_dir / f"{symbol}_price_vs_sentiment.png")]
    if interactive:
        paths.append(_interactive_price_sentiment(prices, sentiment, symbol, out_dir, demo_mode))
    return paths


def _interactive_price_sentiment(
    prices: pd.DataFrame, sentiment: pd.DataFrame, symbol: str, out_dir: Path, demo_mode: bool
) -> Path:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=prices.index, y=prices["close"], name="close", line=dict(color=PALETTE["white"], width=1.6)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=sentiment.index,
            y=sentiment["mean_score"],
            name="mean sentiment",
            line=dict(color=PALETTE["amber"], width=1.2),
        ),
        secondary_y=True,
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PALETTE["canvas"],
        plot_bgcolor=PALETTE["panel"],
        title=f"{symbol} — price vs X sentiment ({_mode_tag(demo_mode)})",
        height=520,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08),
    )
    fig.update_yaxes(title_text="price", secondary_y=False)
    fig.update_yaxes(title_text="mean sentiment", range=[-1, 1], secondary_y=True)
    path = Path(out_dir) / f"{symbol}_price_vs_sentiment.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="inline", full_html=True)
    return path


# -------------------------------------------------------------------------- 2
def sentiment_and_volume(
    prices: pd.DataFrame,
    sentiment: pd.DataFrame,
    symbol: str,
    out_dir: Path,
    demo_mode: bool = False,
) -> list[Path]:
    """Sentiment intensity by sign, post volume, and engagement on three rows."""
    out_dir = Path(out_dir)
    fig, axes = new_subplots(3, 1, width=11.0, height=2.0, sharex=True)

    scores = sentiment["mean_score"].fillna(0.0)
    colours = np.where(scores >= 0, PALETTE["teal"], PALETTE["red"])
    axes[0].bar(sentiment.index, scores, color=colours, width=0.8, alpha=0.9)
    axes[0].axhline(0.0, color=PALETTE["muted"], linewidth=0.7)
    axes[0].set_ylim(-1, 1)
    style_ax(axes[0], f"{symbol} — sentiment intensity per bar", ylabel="mean score")

    axes[1].bar(sentiment.index, sentiment["post_count"], color=PALETTE["slate"], width=0.8, alpha=0.85)
    style_ax(axes[1], "post volume", ylabel="posts")

    engagement = np.log1p(
        sentiment["likes"].fillna(0) + 2 * sentiment["reposts"].fillna(0) + sentiment["views"].fillna(0) / 100
    )
    axes[2].plot(prices.index, prices["close"], color=PALETTE["white"], linewidth=1.2, label="close")
    axes[2].bar(sentiment.index, engagement, color=PALETTE["amber"], width=0.8, alpha=0.55, label="log engagement")
    style_ax(axes[2], "engagement vs price", ylabel="log1p engagement")
    axes[2].legend(loc="upper left", frameon=True)
    footer(fig, _mode_tag(demo_mode))
    return [save_png(fig, out_dir / f"{symbol}_sentiment_volume.png")]


# -------------------------------------------------------------------------- 3
def actual_vs_predicted(
    predictions: pd.DataFrame,
    symbol: str,
    out_dir: Path,
    demo_mode: bool = False,
    model: str | None = None,
) -> list[Path]:
    """One row per horizon: realised price path against out-of-sample forecasts."""
    if predictions.empty:
        log.warning("no walk-forward predictions to plot")
        return []
    out_dir = Path(out_dir)
    horizons = sorted(predictions["horizon"].unique())
    model = model or _best_model(predictions)
    block = predictions[predictions["model"] == model]

    fig, axes = new_subplots(len(horizons), 1, width=11.0, height=1.9, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, horizon in zip(axes, horizons):
        rows = block[block["horizon"] == horizon].sort_values("origin_ts")
        if rows.empty:
            ax.set_visible(False)
            continue
        ax.plot(rows["origin_ts"], rows["actual_price"], color=PALETTE["white"], linewidth=1.1, label="actual")
        ax.plot(
            rows["origin_ts"],
            rows["predicted_price"],
            color=PALETTE["teal"],
            linewidth=1.1,
            linestyle="--",
            label=f"predicted n+{horizon}",
        )
        ax.fill_between(rows["origin_ts"], rows["actual_price"], rows["predicted_price"], color=PALETTE["amber"], alpha=0.12)
        style_ax(ax, f"n+{horizon} · {model} · MAE {_mae(rows):,.2f}", ylabel="price")
        if ax is axes[0]:
            ax.legend(loc="upper left", frameon=True)
    footer(
        fig,
        f"{_mode_tag(demo_mode)} · expanding-window walk-forward · every point is an out-of-sample forecast",
    )
    return [save_png(fig, out_dir / f"{symbol}_actual_vs_predicted_{model}.png")]


# -------------------------------------------------------------------------- 4
def horizon_forecast(
    forecast: pd.DataFrame, symbol: str, out_dir: Path, demo_mode: bool = False
) -> list[Path]:
    """Forward projection n+0 (observed) through n+5 for each model."""
    if forecast.empty:
        log.warning("no forward forecasts to plot")
        return []
    out_dir = Path(out_dir)
    fig, ax = new_figure(9.5, 4.6)
    colours = {"naive": PALETTE["muted"], "random_forest": PALETTE["teal"], "hist_gbm": PALETTE["amber"], "xgboost": PALETTE["slate"]}

    base = forecast[forecast["model"] == "observed"]
    if not base.empty:
        origin = base["origin_ts"].iloc[0]
        price = float(base["predicted_price"].iloc[0])
        ax.scatter([origin], [price], color=PALETTE["white"], zorder=5, s=28, label="n+0 observed")

    for model, rows in forecast[forecast["model"] != "observed"].groupby("model"):
        rows = rows.sort_values("horizon")
        ax.plot(
            rows["horizon"],
            rows["predicted_price"],
            marker="o",
            markersize=4,
            linewidth=1.3,
            color=colours.get(model, PALETTE["slate"]),
            label=model,
        )
    ax.set_xticks(sorted(forecast["horizon"].unique()))
    ax.set_xticklabels([f"n+{int(h)}" for h in sorted(forecast["horizon"].unique())])
    style_ax(ax, f"{symbol} — forward price projection", xlabel="horizon", ylabel="price")
    ax.legend(loc="upper left", frameon=True)
    footer(fig, f"{_mode_tag(demo_mode)} · n+0 is the last observed close, not a forecast")
    return [save_png(fig, out_dir / f"{symbol}_forward_projection.png")]


# -------------------------------------------------------------------------- 5
def sentiment_price_matrix(
    prices: pd.DataFrame,
    sentiment: pd.DataFrame,
    symbol: str,
    out_dir: Path,
    demo_mode: bool = False,
    lookback: int = 40,
    predictions: pd.DataFrame | None = None,
) -> list[Path]:
    """Time-aligned matrix: sentiment shifts and price moves on the same rows."""
    out_dir = Path(out_dir)
    frame = pd.DataFrame(index=prices.index[-lookback:])
    frame["price chg %"] = (prices["close"].pct_change() * 100).reindex(frame.index)
    frame["sent mean"] = sentiment["mean_score"].reindex(frame.index)
    frame["Δ sent"] = sentiment["mean_score"].diff().reindex(frame.index)
    frame["posts"] = sentiment["post_count"].reindex(frame.index)
    frame["pos−neg"] = (sentiment["pos_ratio"] - sentiment["neg_ratio"]).reindex(frame.index)
    engagement = np.log1p(
        sentiment["likes"].fillna(0) + 2 * sentiment["reposts"].fillna(0) + sentiment["views"].fillna(0) / 100
    )
    frame["engage"] = engagement.reindex(frame.index)

    if predictions is not None and not predictions.empty:
        latest = (
            predictions[predictions["horizon"] == 1]
            .sort_values("origin_ts")
            .drop_duplicates("origin_ts", keep="last")
        )
        frame["fcst n+1 %"] = (latest.set_index("origin_ts")["predicted_return"] * 100).reindex(frame.index)

    display = frame.copy()
    normalised = display.apply(_symmetric_norm)
    height = max(3.6, 0.19 * len(display) + 1.6)
    fig, ax = new_figure(10.0, height)
    image = ax.imshow(
        normalised.to_numpy(dtype=float),
        cmap=DIVERGING,
        vmin=-1,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xticks(range(len(display.columns)))
    ax.set_xticklabels(display.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(display.index)))
    ax.set_yticklabels([stamp.strftime("%Y-%m-%d %H:%M") for stamp in display.index], fontsize=6.4)
    ax.grid(False)

    for row in range(len(display)):
        for col, name in enumerate(display.columns):
            value = display.iloc[row, col]
            text = "·" if pd.isna(value) else f"{value:,.2f}" if abs(value) < 100 else f"{value:,.0f}"
            ax.text(
                col,
                row,
                text,
                ha="center",
                va="center",
                fontsize=6.2,
                color=PALETTE["text"],
            )
    colourbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colourbar.set_label("column-normalised intensity (−1 … +1)", fontsize=7.5)
    colourbar.ax.tick_params(labelsize=7)
    style_ax(ax, f"{symbol} — aligned sentiment / price matrix (last {len(display)} bars)")
    footer(fig, _mode_tag(demo_mode) + " · each column scaled by its own maximum absolute value")
    return [save_png(fig, out_dir / f"{symbol}_sentiment_price_matrix.png")]


# -------------------------------------------------------------------------- 6
def sentiment_return_density(
    prices: pd.DataFrame, sentiment: pd.DataFrame, symbol: str, out_dir: Path, demo_mode: bool = False
) -> list[Path]:
    """Where do next-bar returns land for each change in sentiment?"""
    out_dir = Path(out_dir)
    frame = pd.DataFrame(
        {
            "delta_sent": sentiment["mean_score"].diff(),
            "next_return": prices["close"].pct_change().shift(-1),
        }
    ).dropna()
    if frame.empty:
        log.warning("not enough overlapping sentiment/return pairs for the density chart")
        return []

    fig, ax = new_figure(8.6, 4.8)
    counts, xedges, yedges, image = ax.hist2d(
        frame["delta_sent"],
        frame["next_return"] * 100,
        bins=[18, 18],
        cmap=DIVERGING,
    )
    ax.axhline(0, color=PALETTE["text"], linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axvline(0, color=PALETTE["text"], linewidth=0.8, linestyle="--", alpha=0.6)
    colourbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    colourbar.set_label("observations", fontsize=7.5)
    style_ax(
        ax,
        f"{symbol} — Δ sentiment vs next-bar return",
        xlabel="change in mean sentiment",
        ylabel="next-bar return (%)",
    )
    footer(fig, _mode_tag(demo_mode) + " · descriptive only; density is not evidence of predictability")
    return [save_png(fig, out_dir / f"{symbol}_sentiment_return_density.png")]


# -------------------------------------------------------------------------- 7
def evaluation_summary(
    metrics: pd.DataFrame, symbol: str, out_dir: Path, demo_mode: bool = False
) -> list[Path]:
    """MAE / RMSE by horizon and model, plus the headline risk metrics table."""
    if metrics.empty:
        log.warning("no metrics to plot")
        return []
    out_dir = Path(out_dir)
    fig, axes = new_subplots(1, 3, width=13.0, height=3.6)
    models = [m for m in metrics["model"].unique() if m != "buy_hold"]
    horizons = sorted(metrics["horizon"].unique())
    colours = {"naive": PALETTE["muted"], "random_forest": PALETTE["teal"], "hist_gbm": PALETTE["amber"], "xgboost": PALETTE["slate"]}

    for ax, metric, title in (
        (axes[0], "mae", "MAE by horizon"),
        (axes[1], "rmse", "RMSE by horizon"),
    ):
        width = 0.8 / max(1, len(models))
        for offset, model in enumerate(models):
            values = [
                _metric_value(metrics, model, horizon, metric) for horizon in horizons
            ]
            positions = np.arange(len(horizons)) + offset * width
            ax.bar(positions, values, width=width, color=colours.get(model, PALETTE["slate"]), label=model, alpha=0.92)
        ax.set_xticks(np.arange(len(horizons)) + width * (len(models) - 1) / 2)
        ax.set_xticklabels([f"n+{h}" for h in horizons])
        style_ax(ax, title, ylabel=metric.upper())
        ax.legend(loc="upper left", frameon=True)

    axes[2].axis("off")
    table_rows = []
    for model in models:
        for horizon in horizons:
            table_rows.append(
                [
                    model,
                    f"n+{horizon}",
                    f"{_metric_value(metrics, model, horizon, 'sharpe'):.2f}",
                    f"{_metric_value(metrics, model, horizon, 'sortino'):.2f}",
                    f"{_metric_value(metrics, model, horizon, 'max_drawdown') * 100:.1f}%",
                ]
            )
    table = axes[2].table(
        cellText=table_rows[:22],
        colLabels=["model", "horizon", "Sharpe", "Sortino", "MaxDD"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.12)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(PALETTE["grid"])
        cell.set_facecolor(PALETTE["panel"] if row else PALETTE["grid"])
        cell.set_text_props(color=PALETTE["text"])
    axes[2].set_title("strategy risk metrics (per sampled block)", loc="left", pad=8)
    footer(
        fig,
        _mode_tag(demo_mode)
        + " · strategy = sign of predicted return, held k bars, non-overlapping blocks, costs applied",
    )
    return [save_png(fig, out_dir / f"{symbol}_evaluation_summary.png")]


# --------------------------------------------------------------------- helpers
def _symmetric_norm(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    scale = float(np.nanmax(np.abs(values))) if values.notna().any() else 0.0
    return values / scale if scale > 1e-12 else values * 0.0


def _mae(rows: pd.DataFrame) -> float:
    return float(np.mean(np.abs(rows["actual_price"] - rows["predicted_price"])))


def _metric_value(metrics: pd.DataFrame, model: str, horizon: int, name: str) -> float:
    match = metrics[
        (metrics["model"] == model) & (metrics["horizon"] == horizon) & (metrics["metric_name"] == name)
    ]
    if match.empty:
        return float("nan")
    return float(match["metric_value"].iloc[0])


def _best_model(predictions: pd.DataFrame) -> str:
    """Lowest average MAE across horizons, excluding the naive baseline."""
    candidates = predictions[predictions["model"] != "naive"]
    if candidates.empty:
        return str(predictions["model"].iloc[0])
    errors = (candidates["actual_price"] - candidates["predicted_price"]).abs()
    return str(errors.groupby(candidates["model"]).mean().idxmin())
