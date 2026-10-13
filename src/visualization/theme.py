"""Financial-research chart theme: charcoal canvas, muted teal, amber accents."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

PALETTE = {
    "canvas": "#14181B",
    "panel": "#1B2126",
    "grid": "#2A3238",
    "text": "#E8E6E1",
    "muted": "#8E9AA0",
    "teal": "#3FA796",
    "teal_dark": "#25706A",
    "amber": "#D9A441",
    "amber_dark": "#8A6520",
    "red": "#C0625A",
    "slate": "#6E8494",
    "white": "#F5F3EE",
}

DIVERGING = LinearSegmentedColormap.from_list(
    "research_diverging", [PALETTE["red"], "#5C4A47", PALETTE["panel"], "#31605C", PALETTE["teal"]]
)


def apply_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PALETTE["canvas"],
            "axes.facecolor": PALETTE["panel"],
            "savefig.facecolor": PALETTE["canvas"],
            "axes.edgecolor": PALETTE["grid"],
            "axes.labelcolor": PALETTE["text"],
            "axes.titlecolor": PALETTE["text"],
            "axes.grid": True,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "text.color": PALETTE["text"],
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.titlesize": 10.5,
            "axes.labelsize": 8.5,
            "axes.titleweight": "600",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.facecolor": PALETTE["panel"],
            "legend.edgecolor": PALETTE["grid"],
            "legend.fontsize": 8,
            "legend.framealpha": 0.9,
            "font.size": 9,
            "figure.dpi": 130,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def new_figure(width: float = 10.0, height: float = 4.6):
    apply_theme()
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    return fig, ax


def new_subplots(rows: int, cols: int, width: float = 11.0, height: float = 3.2, sharex: bool = False):
    apply_theme()
    fig, axes = plt.subplots(rows, cols, figsize=(width, height * rows), constrained_layout=True, sharex=sharex)
    return fig, axes


def style_ax(ax, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None) -> None:
    if title:
        ax.set_title(title, loc="left", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.tick_params(axis="both", length=2)


def save_png(fig, path: str | Path, caption: str | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if caption:
        fig.text(0.01, -0.01, caption, color=PALETTE["muted"], fontsize=7.2, ha="left")
    fig.savefig(path)
    plt.close(fig)
    return path


def footer(fig, text: str) -> None:
    fig.text(0.005, 0.005, text, color=PALETTE["muted"], fontsize=7.0, ha="left")
