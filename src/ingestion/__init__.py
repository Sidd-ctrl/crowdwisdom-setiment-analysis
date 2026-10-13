"""Ingestion package: X posts (Apify), prices (yfinance), news, synthetic demo data."""

from src.ingestion.apify_x import ApifyIngestionError, ApifyXClient, normalize_post, normalize_posts
from src.ingestion.news import NewsIngestionError, collect_news, deduplicate_news, normalize_news
from src.ingestion.prices import PriceIngestionError, fetch_prices, price_source
from src.ingestion.synthetic import generate_synthetic_posts, generate_synthetic_prices
from src.ingestion.timeutils import bar_close, bar_floor, build_bar_index, parse_utc_timestamp

__all__ = [
    "ApifyIngestionError",
    "ApifyXClient",
    "normalize_post",
    "normalize_posts",
    "NewsIngestionError",
    "collect_news",
    "deduplicate_news",
    "normalize_news",
    "PriceIngestionError",
    "fetch_prices",
    "price_source",
    "generate_synthetic_posts",
    "generate_synthetic_prices",
    "bar_close",
    "bar_floor",
    "build_bar_index",
    "parse_utc_timestamp",
]
