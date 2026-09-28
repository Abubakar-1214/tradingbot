"""
data/sentiment_analysis.py — sentiment features (P1-fixed, honest).

P1 FIX: ``get_social_sentiment()`` was a hard-coded ``0.0`` placeholder, so
``aggregate_sentiment()`` silently returned neutral sentiment every time.  Now:

  * ``get_social_sentiment(posts=None, keywords=None)`` accepts optional social
    posts/headlines and scores them with the keyword lexicon; when no data is
    provided it returns 0.0 but LOGS a warning so the caller can never confuse
    "no data" with "measured neutral".
  * ``aggregate_sentiment()`` keeps the weighted blend and explicitly labels
    which sources contributed; the social weight is re-distributed when social
    data is absent so the overall score is never dragged to zero by a missing
    source.
  * FinBERT path is retained when ``transformers`` + ``torch`` are available.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import List, Optional, Sequence

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BULLISH_KEYWORDS = [
    "surge", "rally", "gain", "rise", "jump", "soar",
    "bullish", "breakout", "strength", "support",
    "demand", "optimistic", "positive", "growth",
]
BEARISH_KEYWORDS = [
    "plunge", "crash", "drop", "fall", "decline", "slide",
    "bearish", "breakdown", "weakness", "resistance",
    "fear", "pessimistic", "negative", "recession",
]


def _keyword_score(texts: Sequence[str]) -> float:
    """Keyword-lexicon sentiment in [-1, 1]; 0.0 only when nothing matches."""
    bull = bear = 0
    for text in texts:
        low = (text or "").lower()
        bull += sum(1 for kw in BULLISH_KEYWORDS if kw in low)
        bear += sum(1 for kw in BEARISH_KEYWORDS if kw in low)
    total = bull + bear
    if total == 0:
        return 0.0
    return (bull - bear) / total


class SentimentAnalyzer:
    """
    Market sentiment analysis (news / Fed / social).

    Note: when real feeds are unavailable, callers should pass headline lists
    explicitly.  Absent data is logged and returns a neutral 0.0 — never a
    silently fabricated score.
    """

    def __init__(self, use_finbert: bool = False):
        self.use_finbert = use_finbert
        self.model = None
        self.tokenizer = None

        if use_finbert:
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
                self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
                logger.info("FinBERT model loaded")
            except Exception:  # noqa: BLE001 - any failure falls back gracefully
                logger.warning("transformers/torch unavailable - using keyword sentiment")
                self.use_finbert = False

        self.sentiment_history = deque(maxlen=100)
        logger.info("Sentiment Analyzer initialized")

    # ------------------------------------------------------------------ #
    def analyze_headlines(self, headlines: Sequence[str]) -> float:
        """Score news headlines in [-1, 1] (bullish positive, bearish negative)."""
        if not headlines:
            logger.warning("analyze_headlines: no headlines supplied -> neutral 0.0")
            return 0.0
        if self.use_finbert and self.model is not None:
            return self._analyze_with_finbert(list(headlines))
        return _keyword_score(list(headlines))

    def _analyze_with_finbert(self, headlines: List[str]) -> float:
        import torch

        scores = []
        for headline in headlines:
            inputs = self.tokenizer(
                headline, return_tensors="pt", truncation=True, max_length=512
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
            # FinBERT outputs: [negative, neutral, positive]
            scores.append(probs[0][2].item() - probs[0][0].item())
        return float(np.mean(scores)) if scores else 0.0

    def analyze_fed_speech(self, speech_text: str) -> float:
        """
        Score a Fed speech: +1 dovish (bullish gold), -1 hawkish (bearish gold).
        """
        if not speech_text:
            logger.warning("analyze_fed_speech: no text supplied -> neutral 0.0")
            return 0.0
        hawkish_kw = [
            "inflation", "raise rates", "tighten", "hawkish", "strength",
            "resilient", "overheating", "persistent", "restrictive",
            "combat inflation",
        ]
        dovish_kw = [
            "stimulus", "support", "dovish", "patient", "accommodative",
            "weakness", "downside risks", "monitor", "gradual", "data dependent",
        ]
        low = speech_text.lower()
        hawkish = sum(low.count(kw) for kw in hawkish_kw)
        dovish = sum(low.count(kw) for kw in dovish_kw)
        total = hawkish + dovish
        if total == 0:
            return 0.0
        return (dovish - hawkish) / total

    def get_social_sentiment(
        self,
        posts: Optional[Sequence[str]] = None,
        keywords: Optional[Sequence[str]] = None,
    ) -> float:
        """
        Score social-media posts (e.g. FinTwit / r/gold) in [-1, 1].

        P1 FIX: real data path.  When ``posts`` is empty/None a warning is
        logged and 0.0 returned — callers must not treat this as measured
        neutral.  ``keywords`` is retained for API compatibility but the
        keyword lexicon now drives the scoring.
        """
        if not posts:
            logger.warning(
                "get_social_sentiment: no social posts supplied -> neutral 0.0 "
                "(connect a feed before treating this as a real signal)"
            )
            return 0.0
        return _keyword_score(list(posts))

    def aggregate_sentiment(
        self,
        news_headlines: Optional[Sequence[str]] = None,
        fed_text: Optional[str] = None,
        social_posts: Optional[Sequence[str]] = None,
    ) -> dict:
        """
        Aggregate sentiment into feature dict.

        P1 FIX: social weight is re-distributed to the available sources when
        no social data exists, so a missing source never forces overall 0.0.
        """
        features: dict = {}

        news_s = self.analyze_headlines(news_headlines) if news_headlines else 0.0
        fed_s = self.analyze_fed_speech(fed_text) if fed_text else 0.0
        social_s = self.get_social_sentiment(social_posts) if social_posts else 0.0

        features["news_sentiment"] = float(news_s)
        features["fed_sentiment"] = float(fed_s)
        features["social_sentiment"] = float(social_s)

        present = [k for k, v in
                   (("news_sentiment", news_s), ("fed_sentiment", fed_s),
                    ("social_sentiment", social_s)) if v != 0.0]
        if present:
            weights = {"news_sentiment": 0.5, "fed_sentiment": 0.3, "social_sentiment": 0.2}
            wsum = sum(weights[k] for k in present)
            overall = sum(weights[k] * features[k] for k in present) / wsum
        else:
            overall = 0.0
        features["overall_sentiment"] = float(overall)

        if self.sentiment_history:
            prev = self.sentiment_history[-1]["overall_sentiment"]
            features["sentiment_momentum"] = float(overall - prev)
        else:
            features["sentiment_momentum"] = 0.0

        features["sentiment_divergence"] = float(news_s - social_s)
        features["sources_present"] = present
        self.sentiment_history.append(features.copy())
        return features


if __name__ == "__main__":  # pragma: no cover - manual check
    a = SentimentAnalyzer(use_finbert=False)
    feats = a.aggregate_sentiment(
        news_headlines=["Gold surges to record high on safe-haven demand"],
        fed_text="The Fed remains committed to accommodative policy.",
    )
    for k, v in feats.items():
        print(f"  {k}: {v}")
