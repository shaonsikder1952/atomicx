"""News Intelligence Database Models.

Tracks news events, outcomes, and learned patterns for causal analysis.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON, Text, ForeignKey, Index
from sqlalchemy.orm import relationship
from atomicx.data.storage.database import Base


class NewsEvent(Base):
    """A news story that was analyzed."""

    __tablename__ = "news_events"

    id = Column(Integer, primary_key=True)
    event_id = Column(String(64), unique=True, index=True, nullable=False)  # Hash of title+source

    # Story metadata
    title = Column(Text, nullable=False)
    source = Column(String(255), nullable=False, index=True)
    url = Column(Text)
    discovered_at = Column(DateTime(timezone=True), nullable=False, index=True)

    # Significance
    significance_score = Column(Float, nullable=False, index=True)
    keywords_matched = Column(JSON)  # List of matched keywords

    # Deep-dive analysis results
    article_content = Column(Text)  # Full article text
    sentiment = Column(String(50), index=True)  # bullish, bearish, neutral, hawkish, dovish
    people_mentioned = Column(JSON)  # List of important people mentioned
    entities_mentioned = Column(JSON)  # Companies, projects, etc.

    # Impact prediction
    predicted_impact = Column(String(50))  # BULLISH, BEARISH, NEUTRAL, VOLATILE
    predicted_magnitude = Column(Float)  # Expected price change %
    confidence = Column(Float)  # 0-1

    # Actual outcome (filled after event)
    outcome_tracked = Column(Boolean, default=False, index=True)
    price_change_1h = Column(Float)
    price_change_4h = Column(Float)
    price_change_24h = Column(Float)
    volatility_change = Column(Float)  # Change in volatility after event
    volume_change = Column(Float)  # Change in volume after event

    # Learning
    prediction_correct = Column(Boolean)  # Was our prediction right?
    pattern_id = Column(Integer, ForeignKey("news_patterns.id"))  # Which pattern matched

    # Causality tracking - WHY decisions were made
    decision_reasoning = Column(JSON)  # Detailed reasoning: {"why_bullish": "...", "key_factors": [...], "similar_past_events": [...]}
    contributing_factors = Column(JSON)  # List of factors: [{"factor": "Fed rate cut", "weight": 0.3, "impact": "+2.5%"}, ...]
    variable_impacts = Column(JSON)  # Which variables affected: {"fed_sentiment": {"before": 0.3, "after": 0.8, "reason": "..."}, ...}
    confidence_breakdown = Column(JSON)  # Why this confidence: {"pattern_match": 0.3, "historical_accuracy": 0.4, "recency": 0.1, ...}

    # Learning iterations - what was tried and learned
    learning_notes = Column(JSON)  # What system learned: [{"iteration": 1, "tried": "...", "result": "...", "learned": "..."}, ...]

    # Relationships
    pattern = relationship("NewsPattern", back_populates="events")
    decision_audit = relationship("DecisionAudit", back_populates="news_event", uselist=False)

    __table_args__ = (
        Index("idx_news_significance_time", significance_score, discovered_at),
        Index("idx_news_outcome", outcome_tracked, discovered_at),
        Index("idx_news_sentiment_time", sentiment, discovered_at),
    )


class NewsPattern(Base):
    """A learned pattern from historical news events."""

    __tablename__ = "news_patterns"

    id = Column(Integer, primary_key=True)
    pattern_id = Column(String(64), unique=True, index=True, nullable=False)

    # Pattern definition
    pattern_type = Column(String(50), nullable=False, index=True)  # person_quote, fed_decision, hack_event, etc.
    trigger_keywords = Column(JSON)  # Keywords that trigger this pattern
    source_filter = Column(String(255))  # Only from certain sources

    # Pattern characteristics
    avg_price_impact = Column(Float)  # Average price change %
    avg_timeframe_hours = Column(Float)  # How long until impact
    volatility_impact = Column(Float)  # How much volatility increases

    # Statistics
    occurrences = Column(Integer, default=0)  # How many times seen
    correct_predictions = Column(Integer, default=0)  # How many times predicted correctly
    win_rate = Column(Float, default=0.0, index=True)  # Prediction accuracy
    avg_magnitude = Column(Float)  # Average absolute price move

    # Pattern strength
    confidence = Column(Float, default=0.0, index=True)  # 0-1, how reliable is this pattern
    last_seen = Column(DateTime(timezone=True), index=True)

    # Pattern details
    examples = Column(JSON)  # Last 3 examples of this pattern
    notes = Column(Text)  # AI-generated notes about this pattern

    # Learning history - what was tried and what worked
    learning_iterations = Column(JSON)  # [{"date": "...", "modification": "...", "result": "...", "decision": "..."}, ...]
    improvement_log = Column(JSON)  # Track improvements: [{"timestamp": "...", "what_changed": "...", "why": "...", "impact": "..."}, ...]
    failed_attempts = Column(JSON)  # What didn't work: [{"tried": "...", "failed_because": "...", "learned": "..."}, ...]

    # Relationships
    events = relationship("NewsEvent", back_populates="pattern")

    __table_args__ = (
        Index("idx_pattern_confidence_win_rate", confidence, win_rate),
        Index("idx_pattern_type_confidence", pattern_type, confidence),
    )


class DecisionAudit(Base):
    """Complete audit trail for every decision made by the system.

    Shows the full causal chain: news → analysis → reasoning → decision → outcome → learning
    """

    __tablename__ = "decision_audits"

    id = Column(Integer, primary_key=True)
    audit_id = Column(String(64), unique=True, index=True, nullable=False)

    # What decision was made
    decision_type = Column(String(50), nullable=False, index=True)  # prediction, trade, variable_update, pattern_learning
    decision_timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    decision_outcome = Column(String(50))  # BULLISH, BEARISH, NEUTRAL, BUY, SELL, HOLD

    # Links to related entities
    news_event_id = Column(Integer, ForeignKey("news_events.id"))
    pattern_id = Column(Integer, ForeignKey("news_patterns.id"))

    # The complete causal chain
    causal_chain = Column(JSON)  # Step-by-step: [{"step": 1, "what": "News detected", "data": {...}}, {"step": 2, "what": "Pattern matched", ...}, ...]

    # WHY this decision was made
    reasoning_tree = Column(JSON)  # Tree structure: {"root": "Fed cuts rates", "branches": [{"factor": "...", "weight": 0.3, "sub_factors": [...]}]}

    # All factors considered
    factors_analyzed = Column(JSON)  # [{"factor": "sentiment", "value": 0.8, "weight": 0.3, "contributed": "+0.24"}, ...]

    # Variables affected and why
    variables_changed = Column(JSON)  # {"var_name": {"old": 0.3, "new": 0.8, "why": "...", "impact_on_system": "..."}, ...}

    # Thinking process
    thinking_log = Column(JSON)  # [{"thought": "Checking historical patterns...", "result": "...", "next_step": "..."}, ...]

    # Problem-solving attempts
    problems_found = Column(JSON)  # [{"problem": "Low confidence", "attempted_solutions": [...], "what_worked": "...", "learned": "..."}, ...]

    # Outcome tracking
    predicted_outcome = Column(String(50))
    actual_outcome = Column(String(50))
    was_correct = Column(Boolean)
    error_magnitude = Column(Float)  # How wrong was it

    # What was learned
    learning_insights = Column(JSON)  # [{"insight": "...", "will_change": "...", "expected_improvement": "..."}, ...]
    system_improvements = Column(JSON)  # [{"improvement": "...", "implemented": true/false, "impact": "..."}, ...]

    # Meta
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    news_event = relationship("NewsEvent", back_populates="decision_audit")
    pattern = relationship("NewsPattern")

    __table_args__ = (
        Index("idx_decision_type_time", decision_type, decision_timestamp),
        Index("idx_decision_correctness", was_correct, decision_timestamp),
    )


class NewsVariable(Base):
    """Dynamic variables created from news patterns.

    These become tradeable signals in the variable engine.
    """

    __tablename__ = "news_variables"

    id = Column(Integer, primary_key=True)
    variable_name = Column(String(255), unique=True, index=True, nullable=False)

    # Variable definition
    pattern_id = Column(Integer, ForeignKey("news_patterns.id"))
    description = Column(Text)

    # Current state
    is_active = Column(Boolean, default=True, index=True)
    current_value = Column(Float)  # Current signal strength
    last_triggered = Column(DateTime(timezone=True))

    # Performance
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    edge = Column(Float, default=0.0)  # Statistical edge
    sharpe_ratio = Column(Float)

    # Meta
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("idx_news_var_active_edge", is_active, edge),
    )
