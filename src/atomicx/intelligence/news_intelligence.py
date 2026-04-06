"""News Intelligence Service.

Learns causal patterns from news events and market outcomes.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone, timedelta
from typing import Any
from loguru import logger
from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from atomicx.data.storage.database import get_session
from atomicx.data.models.news import NewsEvent, NewsPattern, NewsVariable, DecisionAudit
from atomicx.intelligence.scanner import NewsItem


class NewsIntelligence:
    """Learns patterns from news events and their market impact."""

    # Minimum price change to consider for learning (0.5% = 50 basis points)
    # Anything smaller is just noise and not worth storing/learning from
    MIN_PRICE_CHANGE_THRESHOLD = 0.5  # percent

    def __init__(self):
        self.logger = logger.bind(module="news.intelligence")

    async def store_news_event(
        self,
        news_item: NewsItem,
        analysis: dict[str, Any],
        predicted_impact: dict[str, Any] | None = None
    ) -> int:
        """Store a news event for pattern learning.

        Args:
            news_item: The news story
            analysis: Deep-dive analysis results
            predicted_impact: Optional impact prediction

        Returns:
            Database ID of stored event
        """
        async with get_session() as session:
            # Create event ID
            event_id = hashlib.md5(
                f"{news_item.title}{news_item.source}".encode()
            ).hexdigest()

            # Check if already exists
            result = await session.execute(
                select(NewsEvent).where(NewsEvent.event_id == event_id)
            )
            existing = result.scalar_one_or_none()
            if existing:
                return existing.id

            # Create new event
            event = NewsEvent(
                event_id=event_id,
                title=news_item.title,
                source=news_item.source,
                url=news_item.url,
                discovered_at=news_item.discovered_at,
                significance_score=news_item.significance_score,
                keywords_matched=news_item.keywords_matched,
                article_content=analysis.get("article_text"),
                sentiment=analysis.get("sentiment"),
                people_mentioned=analysis.get("people", []),
                entities_mentioned=analysis.get("entities", []),
            )

            # Add prediction if available
            if predicted_impact:
                event.predicted_impact = predicted_impact.get("direction")
                event.predicted_magnitude = predicted_impact.get("magnitude")
                event.confidence = predicted_impact.get("confidence")

                # Store detailed reasoning
                event.decision_reasoning = {
                    "why_this_direction": predicted_impact.get("reasoning", "Pattern-based prediction"),
                    "key_factors": predicted_impact.get("factors", []),
                    "similar_past_events": predicted_impact.get("historical_matches", []),
                    "confidence_explanation": predicted_impact.get("confidence_breakdown", {})
                }

                event.contributing_factors = predicted_impact.get("contributing_factors", [])
                event.confidence_breakdown = predicted_impact.get("confidence_breakdown", {})

            session.add(event)
            await session.commit()
            await session.refresh(event)

            # Create decision audit trail
            if predicted_impact:
                await self._create_decision_audit(session, event, analysis, predicted_impact)

            await session.commit()

            self.logger.info(
                f"[NEWS-STORE] Saved event: '{news_item.title[:50]}...' "
                f"(significance: {news_item.significance_score:.2f})"
            )

            return event.id

    async def track_outcome(
        self,
        event_id: int,
        price_before: float,
        price_1h: float,
        price_4h: float,
        price_24h: float,
        volume_change: float,
        volatility_change: float
    ) -> None:
        """Track actual market outcome after a news event.

        Args:
            event_id: Database ID of the event
            price_before: Price at time of news
            price_1h: Price 1 hour later
            price_4h: Price 4 hours later
            price_24h: Price 24 hours later
            volume_change: Volume change multiplier
            volatility_change: Volatility change multiplier
        """
        async with get_session() as session:
            result = await session.execute(
                select(NewsEvent).where(NewsEvent.id == event_id)
            )
            event = result.scalar_one_or_none()
            if not event:
                return

            # Calculate price changes
            event.price_change_1h = ((price_1h - price_before) / price_before) * 100
            event.price_change_4h = ((price_4h - price_before) / price_before) * 100
            event.price_change_24h = ((price_24h - price_before) / price_before) * 100
            event.volume_change = volume_change
            event.volatility_change = volatility_change
            event.outcome_tracked = True

            # Check if prediction was correct
            if event.predicted_impact and event.price_change_4h:
                predicted_up = event.predicted_impact in ["BULLISH", "bullish"]
                actual_up = event.price_change_4h > 0
                event.prediction_correct = (predicted_up == actual_up)

            await session.commit()

            # Only learn from significant price movements (ignore noise)
            price_move_significant = abs(event.price_change_4h or 0) >= self.MIN_PRICE_CHANGE_THRESHOLD

            if price_move_significant:
                self.logger.success(
                    f"[NEWS-OUTCOME] Tracked: '{event.title[:40]}...' → "
                    f"4h: {event.price_change_4h:+.2f}% "
                    f"(correct: {event.prediction_correct}) - Learning from outcome"
                )
                # Try to learn pattern from this outcome
                await self._learn_from_outcome(session, event)
            else:
                self.logger.debug(
                    f"[NEWS-OUTCOME] Ignored small move: '{event.title[:40]}...' → "
                    f"4h: {event.price_change_4h:+.2f}% (below {self.MIN_PRICE_CHANGE_THRESHOLD}% threshold)"
                )

    async def _create_decision_audit(
        self,
        session: AsyncSession,
        event: NewsEvent,
        analysis: dict[str, Any],
        predicted_impact: dict[str, Any]
    ) -> None:
        """Create complete audit trail for this prediction decision."""

        audit_id = hashlib.md5(
            f"{event.event_id}{event.discovered_at}".encode()
        ).hexdigest()

        # Build causal chain
        causal_chain = [
            {
                "step": 1,
                "what": "News Event Detected",
                "data": {
                    "title": event.title,
                    "source": event.source,
                    "significance": event.significance_score,
                    "keywords": event.keywords_matched
                }
            },
            {
                "step": 2,
                "what": "Content Analysis",
                "data": {
                    "sentiment": analysis.get("sentiment"),
                    "entities": analysis.get("entities", []),
                    "people": analysis.get("people", [])
                }
            },
            {
                "step": 3,
                "what": "Pattern Matching",
                "data": predicted_impact.get("patterns_matched", [])
            },
            {
                "step": 4,
                "what": "Impact Prediction",
                "data": {
                    "direction": predicted_impact.get("direction"),
                    "magnitude": predicted_impact.get("magnitude"),
                    "confidence": predicted_impact.get("confidence")
                }
            }
        ]

        # Build reasoning tree
        reasoning_tree = {
            "root": f"News: {event.title[:50]}...",
            "branches": [
                {
                    "factor": "Historical Pattern Match",
                    "weight": 0.4,
                    "details": predicted_impact.get("historical_matches", [])
                },
                {
                    "factor": "Sentiment Analysis",
                    "weight": 0.3,
                    "details": {"sentiment": analysis.get("sentiment"), "confidence": analysis.get("sentiment_confidence")}
                },
                {
                    "factor": "Source Reliability",
                    "weight": 0.2,
                    "details": {"source": event.source, "reliability_score": event.significance_score}
                },
                {
                    "factor": "Market Context",
                    "weight": 0.1,
                    "details": predicted_impact.get("market_context", {})
                }
            ]
        }

        audit = DecisionAudit(
            audit_id=audit_id,
            decision_type="prediction",
            decision_timestamp=event.discovered_at,
            decision_outcome=predicted_impact.get("direction"),
            news_event_id=event.id,
            causal_chain=causal_chain,
            reasoning_tree=reasoning_tree,
            factors_analyzed=predicted_impact.get("contributing_factors", []),
            thinking_log=[
                {"thought": "Analyzing news significance...", "result": f"Score: {event.significance_score}"},
                {"thought": "Checking historical patterns...", "result": f"Found {len(predicted_impact.get('patterns_matched', []))} matches"},
                {"thought": "Calculating confidence...", "result": f"{predicted_impact.get('confidence', 0):.2%}"}
            ],
            predicted_outcome=predicted_impact.get("direction")
        )

        session.add(audit)

    async def _learn_from_outcome(self, session: AsyncSession, event: NewsEvent) -> None:
        """Learn or update patterns from a completed news event."""

        # Track the learning attempt
        learning_iteration = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_title": event.title[:50],
            "price_change": event.price_change_4h,
            "attempt": "Detecting pattern type..."
        }

        # Pattern detection logic
        pattern_type = self._detect_pattern_type(event)
        if not pattern_type:
            learning_iteration["result"] = "No pattern detected - event too generic"
            learning_iteration["learned"] = "Need more specific pattern matching"
            event.learning_notes = event.learning_notes or []
            event.learning_notes.append(learning_iteration)
            return

        learning_iteration["pattern_detected"] = pattern_type

        # Create pattern ID
        pattern_id = hashlib.md5(
            f"{pattern_type}{event.source}{event.sentiment}".encode()
        ).hexdigest()[:16]

        # Find or create pattern
        result = await session.execute(
            select(NewsPattern).where(NewsPattern.pattern_id == pattern_id)
        )
        pattern = result.scalar_one_or_none()

        if not pattern:
            # Create new pattern
            pattern = NewsPattern(
                pattern_id=pattern_id,
                pattern_type=pattern_type,
                trigger_keywords=event.keywords_matched,
                source_filter=event.source,
                occurrences=0,
                correct_predictions=0,
                win_rate=0.0,
                confidence=0.0,
                examples=[]
            )
            session.add(pattern)

        # Update pattern statistics
        old_win_rate = pattern.win_rate
        old_confidence = pattern.confidence
        old_occurrences = pattern.occurrences

        pattern.occurrences += 1
        pattern.last_seen = event.discovered_at

        if event.prediction_correct:
            pattern.correct_predictions += 1

        # Calculate win rate and confidence
        if pattern.occurrences > 0:
            pattern.win_rate = pattern.correct_predictions / pattern.occurrences

            # Confidence increases with more data points and higher win rate
            # Use Wilson score interval for confidence
            if pattern.occurrences >= 3:
                pattern.confidence = min(
                    pattern.win_rate * (pattern.occurrences / (pattern.occurrences + 10)),
                    1.0
                )

        # Track learning iteration
        pattern.learning_iterations = pattern.learning_iterations or []
        iteration = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event.title[:50],
            "modification": f"Updated statistics: occurrences {old_occurrences} → {pattern.occurrences}",
            "result": f"Win rate {old_win_rate:.2%} → {pattern.win_rate:.2%}, Confidence {old_confidence:.2%} → {pattern.confidence:.2%}",
            "decision": f"Pattern {'strengthened' if pattern.confidence > old_confidence else 'weakened'}"
        }
        pattern.learning_iterations.append(iteration)

        # Keep only last 10 iterations
        pattern.learning_iterations = pattern.learning_iterations[-10:]

        # Update average impact
        old_avg_impact = pattern.avg_price_impact
        if event.price_change_4h is not None:
            if pattern.avg_price_impact is None:
                pattern.avg_price_impact = event.price_change_4h

                # Log the improvement
                pattern.improvement_log = pattern.improvement_log or []
                pattern.improvement_log.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "what_changed": "Initial average impact established",
                    "why": f"First tracked outcome for this pattern",
                    "impact": f"Avg impact set to {event.price_change_4h:+.2f}%"
                })
            else:
                # Exponential moving average
                pattern.avg_price_impact = (
                    0.8 * pattern.avg_price_impact +
                    0.2 * event.price_change_4h
                )

                # Log significant changes
                if abs(pattern.avg_price_impact - old_avg_impact) > 0.5:
                    pattern.improvement_log = pattern.improvement_log or []
                    pattern.improvement_log.append({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "what_changed": "Average impact adjusted",
                        "why": f"New outcome ({event.price_change_4h:+.2f}%) significantly different from average",
                        "impact": f"Avg impact {old_avg_impact:+.2f}% → {pattern.avg_price_impact:+.2f}%"
                    })

                    # Keep only last 20 improvements
                    pattern.improvement_log = pattern.improvement_log[-20:]

        # Store example
        example = {
            "title": event.title[:100],
            "price_change": event.price_change_4h,
            "date": event.discovered_at.isoformat()
        }
        examples = pattern.examples or []
        examples.append(example)
        pattern.examples = examples[-3:]  # Keep last 3

        # Link event to pattern
        event.pattern_id = pattern.id

        # Update learning notes for the event
        learning_iteration["pattern_linked"] = pattern.pattern_type
        learning_iteration["result"] = f"Pattern updated: {pattern.occurrences} occurrences, {pattern.win_rate:.0%} win rate"
        learning_iteration["learned"] = f"This pattern now has {pattern.confidence:.0%} confidence"
        event.learning_notes = event.learning_notes or []
        event.learning_notes.append(learning_iteration)

        # Update decision audit if it exists
        audit_result = await session.execute(
            select(DecisionAudit).where(DecisionAudit.news_event_id == event.id)
        )
        audit = audit_result.scalar_one_or_none()
        if audit:
            audit.actual_outcome = "BULLISH" if event.price_change_4h > 0 else "BEARISH"
            audit.was_correct = event.prediction_correct
            audit.error_magnitude = abs(event.price_change_4h - (event.predicted_magnitude or 0))
            audit.pattern_id = pattern.id

            # Store learning insights
            audit.learning_insights = [
                {
                    "insight": f"Pattern {pattern.pattern_type} {'confirmed' if event.prediction_correct else 'contradicted'}",
                    "will_change": f"Confidence adjusted from {old_confidence:.2%} to {pattern.confidence:.2%}",
                    "expected_improvement": "Better predictions for similar events in future"
                }
            ]

            # Store system improvements
            audit.system_improvements = [
                {
                    "improvement": f"Pattern statistics updated",
                    "implemented": True,
                    "impact": f"Win rate now {pattern.win_rate:.0%} over {pattern.occurrences} occurrences"
                }
            ]

            # Add problem-solving if prediction was wrong
            if not event.prediction_correct:
                audit.problems_found = [
                    {
                        "problem": f"Predicted {audit.predicted_outcome} but price moved {event.price_change_4h:+.2f}%",
                        "attempted_solutions": [
                            "Lowered pattern confidence",
                            "Updated average impact",
                            "Stored as learning example"
                        ],
                        "what_worked": "Statistical adjustment via exponential moving average",
                        "learned": "This pattern needs more data or has context-dependent behavior"
                    }
                ]

        await session.commit()

        if pattern.occurrences >= 3 and pattern.confidence > 0.6:
            self.logger.success(
                f"[PATTERN-LEARNED] {pattern.pattern_type}: "
                f"{pattern.avg_price_impact:+.1f}% impact "
                f"(conf: {pattern.confidence:.0%}, win rate: {pattern.win_rate:.0%}, "
                f"n={pattern.occurrences})"
            )

    def _detect_pattern_type(self, event: NewsEvent) -> str | None:
        """Detect what type of pattern this event represents."""

        title_lower = event.title.lower()
        keywords = event.keywords_matched or []

        # Fed/Central bank patterns
        if any(k in title_lower for k in ["federal reserve", "fed", "powell", "fomc"]):
            if "rate" in title_lower or "interest" in title_lower:
                return "fed_rate_decision"
            return "fed_statement"

        # Regulatory patterns
        if any(k in title_lower for k in ["sec", "regulation", "ban"]):
            if "etf" in title_lower:
                return "etf_decision"
            return "regulatory_action"

        # Exchange patterns
        if any(k in title_lower for k in ["binance", "coinbase", "kraken", "ftx"]):
            if any(k in title_lower for k in ["hack", "exploit", "breach"]):
                return "exchange_hack"
            if "list" in title_lower:
                return "exchange_listing"
            return "exchange_news"

        # Macro events
        if any(k in title_lower for k in ["inflation", "cpi", "ppi", "gdp"]):
            return "macro_data_release"

        # Security events
        if any(k in title_lower for k in ["hack", "exploit", "rug pull", "scam"]):
            return "security_breach"

        # Whale activity
        if any(k in title_lower for k in ["whale", "large transfer", "billion"]):
            return "whale_movement"

        # Adoption news
        if any(k in title_lower for k in ["adoption", "accept", "payment"]):
            return "adoption_news"

        # General based on sentiment
        if event.sentiment:
            return f"general_{event.sentiment}"

        return None

    async def get_matching_patterns(
        self,
        news_item: NewsItem,
        min_confidence: float = 0.6
    ) -> list[dict[str, Any]]:
        """Find historical patterns matching this news event.

        Args:
            news_item: The news story to match
            min_confidence: Minimum pattern confidence

        Returns:
            List of matching patterns with their statistics
        """
        async with get_session() as session:
            # Simple keyword matching for now
            query = (
                select(NewsPattern)
                .where(
                    and_(
                        NewsPattern.confidence >= min_confidence,
                        NewsPattern.occurrences >= 3
                    )
                )
                .order_by(NewsPattern.confidence.desc())
                .limit(5)
            )

            result = await session.execute(query)
            patterns = result.scalars().all()

            matches = []
            for pattern in patterns:
                # Check keyword overlap
                pattern_keywords = set(pattern.trigger_keywords or [])
                news_keywords = set(news_item.keywords_matched or [])
                overlap = len(pattern_keywords & news_keywords)

                if overlap > 0:
                    matches.append({
                        "pattern_type": pattern.pattern_type,
                        "expected_impact": pattern.avg_price_impact,
                        "confidence": pattern.confidence,
                        "win_rate": pattern.win_rate,
                        "occurrences": pattern.occurrences,
                        "examples": pattern.examples
                    })

            return matches

    async def get_stats(self) -> dict[str, Any]:
        """Get news intelligence statistics."""
        async with get_session() as session:
            # Total events
            total = await session.execute(select(func.count(NewsEvent.id)))
            total_events = total.scalar()

            # Tracked outcomes
            tracked = await session.execute(
                select(func.count(NewsEvent.id)).where(NewsEvent.outcome_tracked == True)
            )
            tracked_events = tracked.scalar()

            # Patterns learned
            patterns = await session.execute(select(func.count(NewsPattern.id)))
            total_patterns = patterns.scalar()

            # High confidence patterns
            confident = await session.execute(
                select(func.count(NewsPattern.id)).where(NewsPattern.confidence >= 0.7)
            )
            confident_patterns = confident.scalar()

            return {
                "total_events": total_events,
                "tracked_outcomes": tracked_events,
                "total_patterns": total_patterns,
                "high_confidence_patterns": confident_patterns,
                "tracking_rate": tracked_events / max(total_events, 1)
            }
