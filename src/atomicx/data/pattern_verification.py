"""Pattern verification system — detect patterns, store them, verify outcomes.

For every new candle:
1. Run pattern detection logic against current variables
2. Store detected patterns with variables snapshot
3. After N candles (default 20), verify outcome and update record
4. Build historical database: "last 50 times this pattern appeared, what happened?"
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import pandas as pd
from loguru import logger
from sqlalchemy import select, and_, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from atomicx.data.storage.database import get_session_factory, Base
from sqlalchemy import Column, BigInteger, String, DateTime, Numeric, Boolean, Integer, text
from sqlalchemy.dialects.postgresql import JSONB


class PatternLibrary(Base):
    """Pattern detection and outcome tracking table."""
    __tablename__ = "pattern_library"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    pattern_id = Column(String(100), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False, index=True)
    pattern_name = Column(String(100), nullable=False, index=True)
    detected_at = Column(DateTime(timezone=True), nullable=False, index=True)
    variables_snapshot = Column(JSONB, nullable=False)
    regime = Column(String(50), nullable=True)
    confidence_score = Column(Numeric(5, 4), nullable=False)

    # Outcome tracking
    outcome_direction = Column(String(20), nullable=True)
    outcome_return = Column(Numeric(10, 6), nullable=True)
    outcome_verified = Column(Boolean, nullable=False, default=False)
    verification_candles = Column(Integer, nullable=True)
    verified_at = Column(DateTime(timezone=True), nullable=True)

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=text('now()'), nullable=False)
    pattern_metadata = Column(JSONB, nullable=True)


class PatternVerificationService:
    """Detects patterns in real-time and verifies outcomes."""

    # Configuration
    DEFAULT_VERIFICATION_CANDLES = 20  # Verify outcome after 20 candles
    MIN_CONFIDENCE = 0.3  # Minimum confidence to store pattern
    PATTERN_LOOKBACK = 50  # Number of past patterns to return in history

    def __init__(self) -> None:
        self._session_factory = get_session_factory()
        self.logger = logger.bind(module="pattern.verification")

    async def detect_and_store_patterns(
        self,
        symbol: str,
        timeframe: str,
        variables: dict[str, float],
        price: float,
        regime: str | None = None,
    ) -> list[dict[str, Any]]:
        """Detect patterns in current market state and store them.

        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            variables: Current variable values (RSI_14, EMA_9, etc.)
            price: Current price
            regime: Current market regime

        Returns:
            List of detected patterns
        """
        detected_patterns = []

        # Run all pattern detection logic
        patterns = self._detect_all_patterns(variables, price, regime)

        for pattern in patterns:
            if pattern["confidence"] < self.MIN_CONFIDENCE:
                continue

            # Store in database
            pattern_record = await self._store_pattern(
                symbol=symbol,
                timeframe=timeframe,
                pattern_name=pattern["name"],
                variables_snapshot=variables,
                regime=regime,
                confidence=pattern["confidence"],
                metadata=pattern.get("metadata", {}),
            )

            detected_patterns.append(pattern)

        if detected_patterns:
            self.logger.info(
                f"Detected {len(detected_patterns)} patterns for {symbol} {timeframe}"
            )

        return detected_patterns

    def _detect_all_patterns(
        self,
        variables: dict[str, float],
        price: float,
        regime: str | None,
    ) -> list[dict[str, Any]]:
        """Run all pattern detection logic.

        Uses the EXISTING pattern logic from the codebase.
        """
        patterns = []

        # 1. RSI Extreme Patterns
        rsi_14 = variables.get("RSI_14", 50)
        if rsi_14 > 80:
            patterns.append({
                "name": "RSI_OVERBOUGHT_80",
                "confidence": min((rsi_14 - 80) / 20, 1.0),
                "metadata": {"rsi_value": rsi_14, "type": "extreme_value"},
            })
        elif rsi_14 < 20:
            patterns.append({
                "name": "RSI_OVERSOLD_20",
                "confidence": min((20 - rsi_14) / 20, 1.0),
                "metadata": {"rsi_value": rsi_14, "type": "extreme_value"},
            })
        elif rsi_14 > 70:
            patterns.append({
                "name": "RSI_OVERBOUGHT_70",
                "confidence": (rsi_14 - 70) / 30,
                "metadata": {"rsi_value": rsi_14, "type": "extreme_value"},
            })
        elif rsi_14 < 30:
            patterns.append({
                "name": "RSI_OVERSOLD_30",
                "confidence": (30 - rsi_14) / 30,
                "metadata": {"rsi_value": rsi_14, "type": "extreme_value"},
            })

        # 2. Bollinger Band Extremes
        bb_percent = variables.get("BB_PERCENT_B", 0.5)
        if bb_percent > 1.0:
            patterns.append({
                "name": "BB_ABOVE_UPPER_BAND",
                "confidence": min((bb_percent - 1.0), 1.0),
                "metadata": {"bb_percent_b": bb_percent, "type": "extreme_value"},
            })
        elif bb_percent < 0.0:
            patterns.append({
                "name": "BB_BELOW_LOWER_BAND",
                "confidence": min(abs(bb_percent), 1.0),
                "metadata": {"bb_percent_b": bb_percent, "type": "extreme_value"},
            })

        # 3. EMA Trend Structure
        ema_9 = variables.get("EMA_9", 0)
        ema_21 = variables.get("EMA_21", 0)
        ema_50 = variables.get("EMA_50", 0)

        if ema_9 > 0 and ema_21 > 0 and ema_50 > 0:
            if ema_9 > ema_21 > ema_50:
                # Bullish EMA stack
                spread = (ema_9 - ema_50) / ema_50
                patterns.append({
                    "name": "EMA_STACK_BULLISH",
                    "confidence": min(spread * 10, 1.0),  # Wider spread = higher confidence
                    "metadata": {"ema_spread_pct": spread * 100, "type": "trend"},
                })
            elif ema_9 < ema_21 < ema_50:
                # Bearish EMA stack
                spread = (ema_50 - ema_9) / ema_50
                patterns.append({
                    "name": "EMA_STACK_BEARISH",
                    "confidence": min(spread * 10, 1.0),
                    "metadata": {"ema_spread_pct": spread * 100, "type": "trend"},
                })

        # 4. MACD Crossover
        macd_line = variables.get("MACD_LINE", 0)
        macd_signal = variables.get("MACD_SIGNAL", 0)
        macd_hist = variables.get("MACD_HISTOGRAM", 0)

        if macd_hist > 0 and abs(macd_hist) > 0.0001:
            patterns.append({
                "name": "MACD_BULLISH_CROSS",
                "confidence": min(abs(macd_hist) * 1000, 1.0),
                "metadata": {"macd_histogram": macd_hist, "type": "momentum"},
            })
        elif macd_hist < 0 and abs(macd_hist) > 0.0001:
            patterns.append({
                "name": "MACD_BEARISH_CROSS",
                "confidence": min(abs(macd_hist) * 1000, 1.0),
                "metadata": {"macd_histogram": macd_hist, "type": "momentum"},
            })

        # 5. V-Bottom Reversal (RSI oversold + volume spike + MACD turning)
        rel_volume = variables.get("REL_VOLUME", 1.0)
        if rsi_14 < 30 and rel_volume > 2.0 and macd_hist > 0:
            patterns.append({
                "name": "V_BOTTOM_REVERSAL",
                "confidence": min((rel_volume / 5.0) * (30 - rsi_14) / 30, 1.0),
                "metadata": {
                    "rsi": rsi_14,
                    "rel_volume": rel_volume,
                    "macd_hist": macd_hist,
                    "type": "reversal",
                },
            })

        # 6. Regime-Based Patterns
        if regime:
            adx = variables.get("ADX", 20)
            bb_bandwidth = variables.get("BB_BANDWIDTH", 0.05)

            if regime == "trending_volatile" and adx > 30:
                patterns.append({
                    "name": f"REGIME_{regime.upper()}",
                    "confidence": min(adx / 50, 1.0),
                    "metadata": {"adx": adx, "bandwidth": bb_bandwidth, "type": "regime"},
                })

        return patterns

    async def _store_pattern(
        self,
        symbol: str,
        timeframe: str,
        pattern_name: str,
        variables_snapshot: dict[str, float],
        regime: str | None,
        confidence: float,
        metadata: dict[str, Any],
    ) -> str:
        """Store detected pattern in database.

        Returns:
            Pattern ID
        """
        pattern_id = f"{pattern_name}_{symbol}_{uuid.uuid4().hex[:8]}"

        async with self._session_factory() as session:
            stmt = pg_insert(PatternLibrary).values(
                pattern_id=pattern_id,
                symbol=symbol,
                timeframe=timeframe,
                pattern_name=pattern_name,
                detected_at=datetime.now(tz=timezone.utc),
                variables_snapshot=json.loads(json.dumps(variables_snapshot, default=str)),
                regime=regime,
                confidence_score=Decimal(str(confidence)),
                pattern_metadata=metadata,
            )
            await session.execute(stmt)
            await session.commit()

        return pattern_id

    async def verify_pending_patterns(
        self,
        verification_candles: int | None = None,
    ) -> dict[str, Any]:
        """Verify outcomes for patterns that have aged past verification window.

        Args:
            verification_candles: Number of candles to wait before verifying (default 20)

        Returns:
            Stats about verification run
        """
        verify_after = verification_candles or self.DEFAULT_VERIFICATION_CANDLES
        stats = {
            "patterns_checked": 0,
            "patterns_verified": 0,
            "errors": [],
        }

        # Find unverified patterns old enough to verify
        async with self._session_factory() as session:
            result = await session.execute(
                select(PatternLibrary)
                .where(
                    and_(
                        PatternLibrary.outcome_verified == False,
                        PatternLibrary.detected_at < datetime.now(tz=timezone.utc) - timedelta(hours=1)
                    )
                )
                .limit(100)  # Process in batches
            )
            patterns = result.scalars().all()

        stats["patterns_checked"] = len(patterns)

        for pattern in patterns:
            try:
                # Calculate outcome
                outcome = await self._calculate_outcome(
                    pattern.symbol,
                    pattern.timeframe,
                    pattern.detected_at,
                    verify_after,
                )

                if outcome is not None:
                    # Update pattern with outcome
                    async with self._session_factory() as session:
                        stmt = (
                            update(PatternLibrary)
                            .where(PatternLibrary.id == pattern.id)
                            .values(
                                outcome_direction=outcome["direction"],
                                outcome_return=Decimal(str(outcome["return"])),
                                outcome_verified=True,
                                verification_candles=verify_after,
                                verified_at=datetime.now(tz=timezone.utc),
                            )
                        )
                        await session.execute(stmt)
                        await session.commit()

                    stats["patterns_verified"] += 1

            except Exception as e:
                self.logger.error(f"Verification error for pattern {pattern.pattern_id}: {e}")
                stats["errors"].append(str(e))

        if stats["patterns_verified"] > 0:
            self.logger.info(
                f"Verified {stats['patterns_verified']}/{stats['patterns_checked']} patterns"
            )

        # Refresh materialized view periodically
        if stats["patterns_verified"] > 0:
            await self._refresh_performance_view()

        return stats

    async def _calculate_outcome(
        self,
        symbol: str,
        timeframe: str,
        detected_at: datetime,
        candles_ahead: int,
    ) -> dict[str, Any] | None:
        """Calculate outcome N candles after pattern detection."""
        from atomicx.data.storage.models import OHLCV

        # Get entry candle
        async with self._session_factory() as session:
            entry_result = await session.execute(
                select(OHLCV)
                .where(
                    and_(
                        OHLCV.symbol == symbol,
                        OHLCV.timeframe == timeframe,
                        OHLCV.timestamp >= detected_at
                    )
                )
                .order_by(OHLCV.timestamp)
                .limit(1)
            )
            entry_candle = entry_result.scalars().first()

            if not entry_candle:
                return None

            # Get exit candle (N candles later)
            exit_result = await session.execute(
                select(OHLCV)
                .where(
                    and_(
                        OHLCV.symbol == symbol,
                        OHLCV.timeframe == timeframe,
                        OHLCV.timestamp > entry_candle.timestamp
                    )
                )
                .order_by(OHLCV.timestamp)
                .offset(candles_ahead - 1)
                .limit(1)
            )
            exit_candle = exit_result.scalars().first()

            if not exit_candle:
                return None

        # Calculate return
        entry_price = float(entry_candle.close)
        exit_price = float(exit_candle.close)
        price_return = (exit_price - entry_price) / entry_price

        # Determine direction
        if price_return > 0.005:  # > 0.5% = bullish
            direction = "bullish"
        elif price_return < -0.005:  # < -0.5% = bearish
            direction = "bearish"
        else:
            direction = "neutral"

        return {
            "direction": direction,
            "return": price_return,
            "entry_price": entry_price,
            "exit_price": exit_price,
        }

    async def get_pattern_history(
        self,
        pattern_name: str,
        regime: str | None = None,
        symbol: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get historical outcomes for a pattern.

        This is the key function that agents call to answer:
        "Last 50 times RSI_OVERSOLD_30 appeared in trending_volatile regime, what happened?"

        Args:
            pattern_name: Pattern to query
            regime: Optional regime filter
            symbol: Optional symbol filter
            limit: Max results (default 50)

        Returns:
            List of historical pattern occurrences with outcomes
        """
        conditions = [
            PatternLibrary.pattern_name == pattern_name,
            PatternLibrary.outcome_verified == True,
        ]

        if regime:
            conditions.append(PatternLibrary.regime == regime)
        if symbol:
            conditions.append(PatternLibrary.symbol == symbol)

        async with self._session_factory() as session:
            result = await session.execute(
                select(PatternLibrary)
                .where(and_(*conditions))
                .order_by(PatternLibrary.detected_at.desc())
                .limit(limit)
            )
            patterns = result.scalars().all()

        history = []
        for p in patterns:
            history.append({
                "pattern_id": p.pattern_id,
                "symbol": p.symbol,
                "detected_at": p.detected_at.isoformat(),
                "regime": p.regime,
                "confidence": float(p.confidence_score),
                "outcome_direction": p.outcome_direction,
                "outcome_return": float(p.outcome_return) if p.outcome_return else None,
                "variables": p.variables_snapshot,
            })

        return history

    async def get_pattern_performance_stats(
        self,
        pattern_name: str | None = None,
        regime: str | None = None,
    ) -> dict[str, Any]:
        """Get aggregated performance statistics from materialized view."""
        query = "SELECT * FROM pattern_performance WHERE 1=1"
        params = {}

        if pattern_name:
            query += " AND pattern_name = :pattern_name"
            params["pattern_name"] = pattern_name

        if regime:
            query += " AND regime = :regime"
            params["regime"] = regime

        async with self._session_factory() as session:
            result = await session.execute(text(query), params)
            rows = result.fetchall()

        stats = []
        for row in rows:
            stats.append({
                "pattern_name": row[0],
                "regime": row[1],
                "symbol": row[2],
                "timeframe": row[3],
                "total_occurrences": row[4],
                "verified_count": row[5],
                "bullish_outcomes": row[6],
                "bearish_outcomes": row[7],
                "avg_return": float(row[8]) if row[8] else None,
                "return_stddev": float(row[9]) if row[9] else None,
                "avg_confidence": float(row[10]) if row[10] else None,
                "last_seen": row[11].isoformat() if row[11] else None,
            })

        return stats

    async def _refresh_performance_view(self) -> None:
        """Refresh the pattern_performance materialized view."""
        async with self._session_factory() as session:
            await session.execute(text("SELECT refresh_pattern_performance();"))
            await session.commit()

        self.logger.debug("Refreshed pattern_performance materialized view")

    async def get_pattern_stats(self, symbol: str | None = None) -> dict[str, Any]:
        """Get summary pattern statistics for dashboard display.

        Args:
            symbol: Optional symbol filter

        Returns:
            Dict with total_detected, verified_count, win_rate
        """
        from sqlalchemy import func

        async with self._session_factory() as session:
            # Total patterns detected
            query_total = select(func.count(PatternLibrary.id))
            if symbol:
                query_total = query_total.where(PatternLibrary.symbol == symbol)
            result_total = await session.execute(query_total)
            total_detected = result_total.scalar() or 0

            # Total verified
            query_verified = select(func.count(PatternLibrary.id)).where(
                PatternLibrary.outcome_verified == True
            )
            if symbol:
                query_verified = query_verified.where(PatternLibrary.symbol == symbol)
            result_verified = await session.execute(query_verified)
            verified_count = result_verified.scalar() or 0

            # Win rate (bullish + bearish correct outcomes)
            query_correct = select(func.count(PatternLibrary.id)).where(
                and_(
                    PatternLibrary.outcome_verified == True,
                    PatternLibrary.outcome_return > 0  # Positive return = correct prediction
                )
            )
            if symbol:
                query_correct = query_correct.where(PatternLibrary.symbol == symbol)
            result_correct = await session.execute(query_correct)
            correct_count = result_correct.scalar() or 0

            win_rate = (correct_count / verified_count) if verified_count > 0 else 0.0

        return {
            "total_detected": total_detected,
            "verified_count": verified_count,
            "win_rate": win_rate,
        }
