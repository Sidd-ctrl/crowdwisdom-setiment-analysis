"""Package the project into crowdwisdom_sentiment_pipeline.zip.

Ships source, configuration templates, tests, documentation and a small set of
regenerable example outputs (PNG charts, Markdown reports, the forward forecast
CSV). Large generated artefacts (SQLite DB, processed CSVs, interactive HTML,
PDF, the walk-forward CSV) are left out; `python app.py demo` recreates them.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT.parent / "crowdwisdom_sentiment_pipeline.zip"

TEXT_SUFFIXES = {".py", ".md", ".txt", ".ini", ".sh", ".ps1", ".example", ".gitignore", ".gitkeep"}
CHART_SUFFIXES = {".png"}

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    ".arena",
    ".cache",
}
EXCLUDED_FILES = {".env", "crowdwisdom_sentiment_pipeline.zip"}


def iter_files():
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        if path.name in EXCLUDED_FILES:
            continue
        rel = path.relative_to(ROOT)

        if path.name in {".gitignore", ".gitkeep"} or path.suffix in TEXT_SUFFIXES:
            yield rel
        elif path.suffix in CHART_SUFFIXES and rel.parts[0] == "outputs":
            yield rel
        elif (
            rel.parts[0] == "outputs"
            and rel.parts[1] == "reports"
            and path.suffix == ".md"
        ):
            yield rel
        elif (
            rel.parts[0] == "outputs"
            and rel.parts[1] == "forecasts"
            and path.name.endswith(".csv")
            and "walkforward" not in path.name
        ):
            yield rel
        # everything else (sqlite, html, pdf, processed csvs) is regenerated


def main() -> None:
    entries = list(iter_files())
    with zipfile.ZipFile(TARGET, "w", zipfile.ZIP_DEFLATED) as bundle:
        for rel in entries:
            bundle.write(ROOT / rel, f"crowdwisdom_sentiment_pipeline/{rel}")

    sizes = sorted(
        (info.filename, info.file_size) for info in zipfile.ZipFile(TARGET).infolist()
    )
    print(f"wrote {TARGET} with {len(sizes)} entries")
    for name, size in sizes:
        print(f"  {size:>9,}  {name}")


if __name__ == "__main__":
    main()
