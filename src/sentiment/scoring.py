"""Post-level sentiment scoring.

VADER is the default backend: it is deterministic, dependency-light and
reproducible, which matters more for an auditable pipeline than a marginal
accuracy gain. FinBERT is available for users who install transformers/torch.
"""

from __future__ import annotations

from config.logging_config import get_logger

log = get_logger(__name__)

# VADER is tuned for general social media, so a small set of market-specific
# terms is added on top of its compound score. The adjustment is deliberately
# capped so VADER remains the dominant signal.
FINANCE_LEXICON = {
    "moon": 0.25,
    "mooning": 0.25,
    "pump": 0.20,
    "pumping": 0.20,
    "breakout": 0.25,
    "bullish": 0.35,
    "rally": 0.25,
    "ath": 0.20,
    "accumulating": 0.20,
    "long": 0.10,
    "dump": -0.30,
    "dumping": -0.30,
    "crash": -0.40,
    "crashed": -0.40,
    "bearish": -0.35,
    "liquidated": -0.35,
    "liquidation": -0.25,
    "rug": -0.45,
    "rugpull": -0.45,
    "short": -0.10,
    "selloff": -0.30,
    "capitulation": -0.30,
}
MAX_LEXICON_ADJUSTMENT = 0.25

POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05


def label_from_score(score: float) -> str:
    if score >= POSITIVE_THRESHOLD:
        return "positive"
    if score <= NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"


def finance_adjustment(text: str) -> float:
    """Sum of finance-term shifts for `text`, clipped to MAX_LEXICON_ADJUSTMENT."""
    tokens = [token.strip(".,!?\"'()[]#$") for token in text.lower().split()]
    total = sum(FINANCE_LEXICON[token] for token in tokens if token in FINANCE_LEXICON)
    return float(max(-MAX_LEXICON_ADJUSTMENT, min(MAX_LEXICON_ADJUSTMENT, total)))


class VaderScorer:
    """VADER compound score plus the capped finance-lexicon adjustment."""

    name = "vader"

    def __init__(self) -> None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self._analyzer = SentimentIntensityAnalyzer()

    def score(self, text: str) -> dict:
        raw = self._analyzer.polarity_scores(text or "")
        compound = max(-1.0, min(1.0, raw["compound"] + finance_adjustment(text or "")))
        return {
            "sentiment_score": compound,
            "sentiment_label": label_from_score(compound),
            "pos": raw["pos"],
            "neg": raw["neg"],
            "neu": raw["neu"],
        }


class FinBertScorer:
    """Optional transformer backend. Loaded lazily and cached."""

    name = "finbert"
    MODEL_ID = "ProsusAI/finbert"

    def __init__(self, model_id: str = MODEL_ID) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "SENTIMENT_BACKEND=finbert needs the optional packages 'transformers' and "
                "'torch'. Install them, or use SENTIMENT_BACKEND=vader."
            ) from exc
        log.info("loading FinBERT model %s (first run downloads weights)", model_id)
        self._pipeline = pipeline("sentiment-analysis", model=model_id, truncation=True, max_length=512)

    def score(self, text: str) -> dict:
        text = (text or "").strip()
        if not text:
            return {"sentiment_score": 0.0, "sentiment_label": "neutral", "pos": 0.0, "neg": 0.0, "neu": 1.0}
        result = self._pipeline(text[:512])[0]
        label = result["label"].lower()
        confidence = float(result["score"])
        positive = confidence if label == "positive" else 0.0
        negative = confidence if label == "negative" else 0.0
        neutral = confidence if label == "neutral" else 0.0
        compound = positive - negative
        return {
            "sentiment_score": compound,
            "sentiment_label": label_from_score(compound),
            "pos": positive,
            "neg": negative,
            "neu": neutral,
        }


def get_scorer(backend: str = "vader"):
    backend = (backend or "vader").lower()
    if backend == "vader":
        return VaderScorer()
    if backend == "finbert":
        return FinBertScorer()
    raise ValueError(f"unknown sentiment backend '{backend}'; expected 'vader' or 'finbert'.")
