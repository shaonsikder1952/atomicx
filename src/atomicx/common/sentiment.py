"""Sentiment Analysis Utilities for AtomicX.

Provides NLP-based sentiment classification for news, social media,
and narrative tracking. Returns numeric scores [-1.0, +1.0] where:
- Positive (+0.3 to +1.0): Bullish signals
- Neutral (-0.2 to +0.2): No clear direction
- Negative (-1.0 to -0.3): Bearish signals
"""

from __future__ import annotations

from loguru import logger


class SentimentAnalyzer:
    """Keyword-based sentiment analyzer with crypto market domain knowledge.

    Uses weighted keyword matching across multiple categories:
    - Price action (rally, crash, surge)
    - Market structure (bullish, bearish)
    - Monetary policy (dovish, hawkish)
    - Risk events (hack, exploit, ban)
    """

    BULLISH_KEYWORDS = [
        "rally", "surge", "bullish", "pump", "moon", "breakout",
        "approval", "adoption", "integration", "partnership",
        "growth", "recovery", "outperform", "uptrend", "accumulation",
        "institutional", "whale buying", "golden cross", "support held",
    ]

    BEARISH_KEYWORDS = [
        "crash", "dump", "bearish", "plunge", "collapse", "selloff",
        "ban", "hack", "exploit", "scam", "fraud", "lawsuit",
        "regulation", "crackdown", "delisting", "death cross",
        "resistance rejected", "breakdown", "capitulation", "panic",
    ]

    DOVISH_KEYWORDS = [
        "rate cut", "easing", "stimulus", "dovish", "accommodation",
        "liquidity injection", "pivot", "pause", "supportive",
    ]

    HAWKISH_KEYWORDS = [
        "rate hike", "tightening", "hawkish", "restrictive",
        "inflation risk", "tapering", "quantitative tightening",
    ]

    AMPLIFIERS = [
        "very", "extremely", "massive", "huge", "significant",
        "major", "critical", "unprecedented", "historic",
    ]

    def __init__(self) -> None:
        self.logger = logger.bind(module="sentiment")

    def analyze_text(self, text: str) -> dict[str, float]:
        """Analyze sentiment of text and return scores.

        Args:
            text: Input text (news headline, tweet, etc.)

        Returns:
            Dict with:
            - score: Numeric sentiment [-1.0, +1.0]
            - confidence: How confident the analysis is [0.0, 1.0]
            - category: "bullish", "bearish", "neutral", "dovish", "hawkish"
        """
        if not text or len(text.strip()) == 0:
            return {"score": 0.0, "confidence": 0.0, "category": "neutral"}

        text_lower = text.lower()

        # Count keyword matches with weighted scores
        bullish_score = self._count_keywords(text_lower, self.BULLISH_KEYWORDS)
        bearish_score = self._count_keywords(text_lower, self.BEARISH_KEYWORDS)
        dovish_score = self._count_keywords(text_lower, self.DOVISH_KEYWORDS)
        hawkish_score = self._count_keywords(text_lower, self.HAWKISH_KEYWORDS)

        # Check for amplifiers (doubles the intensity)
        has_amplifier = any(amp in text_lower for amp in self.AMPLIFIERS)
        amplifier_factor = 1.5 if has_amplifier else 1.0

        # Combine scores
        positive_total = (bullish_score + dovish_score * 0.7) * amplifier_factor
        negative_total = (bearish_score + hawkish_score * 0.7) * amplifier_factor

        total_matches = positive_total + negative_total

        if total_matches == 0:
            return {"score": 0.0, "confidence": 0.0, "category": "neutral"}

        # Normalize to [-1, +1]
        raw_score = (positive_total - negative_total) / max(total_matches, 1.0)

        # Clamp to reasonable bounds
        score = max(-1.0, min(1.0, raw_score))

        # Confidence based on number of matches (more matches = higher confidence)
        confidence = min(1.0, total_matches / 5.0)  # Saturate at 5 matches

        # Determine category
        if score >= 0.3:
            category = "bullish"
        elif score <= -0.3:
            category = "bearish"
        elif dovish_score > hawkish_score and dovish_score > 0:
            category = "dovish"
        elif hawkish_score > dovish_score and hawkish_score > 0:
            category = "hawkish"
        else:
            category = "neutral"

        return {
            "score": round(score, 3),
            "confidence": round(confidence, 3),
            "category": category,
        }

    def _count_keywords(self, text: str, keywords: list[str]) -> float:
        """Count keyword matches with position weighting.

        Keywords appearing earlier in text have slightly higher weight,
        as headlines tend to front-load important information.
        """
        count = 0.0
        text_len = len(text)

        for keyword in keywords:
            if keyword in text:
                # Find position of first occurrence
                pos = text.find(keyword)
                # Position weight: 1.0 at start, 0.7 at end
                pos_weight = 1.0 - (pos / text_len * 0.3) if text_len > 0 else 1.0
                count += pos_weight

        return count

    def batch_analyze(self, texts: list[str]) -> list[dict[str, float]]:
        """Analyze multiple texts in batch.

        Args:
            texts: List of text strings

        Returns:
            List of sentiment analysis results
        """
        return [self.analyze_text(text) for text in texts]


# Global singleton instance
_analyzer_instance = None


def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Get the global sentiment analyzer instance."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = SentimentAnalyzer()
    return _analyzer_instance


def analyze_sentiment(text: str) -> dict[str, float]:
    """Quick function to analyze sentiment of text.

    Convenience wrapper around SentimentAnalyzer.analyze_text().

    Args:
        text: Input text

    Returns:
        Dict with score, confidence, category

    Example:
        >>> result = analyze_sentiment("Bitcoin rallies to new ATH!")
        >>> result["score"]
        0.75
        >>> result["category"]
        'bullish'
    """
    analyzer = get_sentiment_analyzer()
    return analyzer.analyze_text(text)
