"""Reporting package: one-page outlook and the full evaluation report."""

from src.reporting.evaluation_report import (
    build_report_blocks,
    to_markdown,
    write_markdown,
    write_pdf,
)
from src.reporting.outlook import build_outlook, describe_tone, write_outlook

__all__ = [
    "build_report_blocks",
    "to_markdown",
    "write_markdown",
    "write_pdf",
    "build_outlook",
    "describe_tone",
    "write_outlook",
]
