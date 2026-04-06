"""Brain state persistence helpers for pending predictions and outcomes.

Provides Redis-based persistence for pending predictions (0-15 min age)
and database persistence for verified prediction outcomes.

Fixes critical data loss where predictions in-flight are abandoned on restart.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from atomicx.config import get_settings


def _serialize_for_json(obj: Any) -> Any:
    """Recursively serialize objects for JSON storage.

    Converts datetime objects to ISO format strings.
    Handles dicts, lists, and nested structures.
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: _serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_serialize_for_json(item) for item in obj]
    elif hasattr(obj, 'model_dump'):
        # Pydantic model
        return _serialize_for_json(obj.model_dump())
    elif hasattr(obj, '__dict__'):
        # Regular class instance
        return _serialize_for_json(obj.__dict__)
    else:
        return obj


class PredictionPersistence:
    """Manages persistence of pending predictions and verified outcomes."""

    def __init__(self) -> None:
        self._redis_client = None
        self._initialized = False
        self.logger = logger.bind(module="brain.persistence")

    async def initialize(self) -> None:
        """Initialize Redis connection for pending predictions."""
        if self._initialized:
            return

        try:
            import redis.asyncio as redis
            settings = get_settings()

            self._redis_client = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            await self._redis_client.ping()
            self.logger.info("[PERSISTENCE] Redis connected for pending predictions")
            self._initialized = True
        except ImportError:
            self.logger.warning("[PERSISTENCE] redis package not installed, predictions won't persist")
            self._initialized = True
        except Exception as e:
            self.logger.warning(f"[PERSISTENCE] Redis init failed: {e}. Predictions won't persist.")
            self._initialized = True

    async def save_pending_prediction(
        self,
        prediction_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
        confidence: float,
        regime: str,
        packet_json: str,
    ) -> None:
        """Save pending prediction to Redis with 15-minute TTL.

        Args:
            prediction_id: Unique prediction identifier
            symbol: Trading pair
            direction: Predicted direction (bullish/bearish)
            entry_price: Price at prediction time
            confidence: Prediction confidence [0, 1]
            regime: Market regime at prediction time
            packet_json: Full PredictionPacket as JSON string
        """
        if not self._redis_client:
            return

        try:
            key = f"pending:{prediction_id}"
            data = {
                "symbol": symbol,
                "direction": direction,
                "entry_price": str(entry_price),
                "confidence": str(confidence),
                "regime": regime,
                "created_at": datetime.now(tz=timezone.utc).isoformat(),
                "packet_json": packet_json,
            }

            await self._redis_client.hset(key, mapping=data)
            await self._redis_client.expire(key, 900)  # 15 minutes TTL

            self.logger.debug(
                f"[PERSISTENCE] Saved pending prediction {prediction_id} to Redis"
            )
        except Exception as e:
            self.logger.error(
                f"[PERSISTENCE] Failed to save pending prediction {prediction_id}: {e}"
            )

    async def load_pending_predictions(self) -> list[dict[str, Any]]:
        """Load all pending predictions from Redis on startup.

        Returns list of prediction dicts that can be restored to pending_predictions list.
        """
        if not self._redis_client:
            return []

        try:
            keys = []
            async for key in self._redis_client.scan_iter(match="pending:*"):
                keys.append(key)

            predictions = []
            for key in keys:
                data = await self._redis_client.hgetall(key)
                if data:
                    predictions.append({
                        "prediction_id": key.replace("pending:", ""),
                        "symbol": data.get("symbol"),
                        "direction": data.get("direction"),
                        "entry_price": float(data.get("entry_price", 0)),
                        "confidence": float(data.get("confidence", 0)),
                        "regime": data.get("regime"),
                        "created_at": data.get("created_at"),
                        "packet_json": data.get("packet_json"),
                    })

            self.logger.info(
                f"[PERSISTENCE] Loaded {len(predictions)} pending predictions from Redis"
            )
            return predictions
        except Exception as e:
            self.logger.warning(f"[PERSISTENCE] Failed to load pending predictions: {e}")
            return []

    async def remove_pending_prediction(self, prediction_id: str) -> None:
        """Remove pending prediction from Redis after verification."""
        if not self._redis_client:
            return

        try:
            await self._redis_client.delete(f"pending:{prediction_id}")
            self.logger.debug(
                f"[PERSISTENCE] Removed pending prediction {prediction_id} from Redis"
            )
        except Exception as e:
            self.logger.error(
                f"[PERSISTENCE] Failed to remove pending prediction {prediction_id}: {e}"
            )

    async def save_prediction_outcome(
        self,
        prediction_id: str,
        symbol: str,
        timeframe: str,
        predicted_direction: str,
        confidence: float,
        entry_price: float,
        verification_price: float | None,
        was_correct: bool | None,
        actual_return: float | None,
        profit_return: float | None,
        regime: str | None,
        predicted_at: datetime,
        verified_at: datetime | None,
        reasoning: str | None,
        variable_snapshot: dict | None,
        metadata: dict | None = None,
    ) -> None:
        """Save prediction outcome to database.

        Called when prediction is first made (with partial data) and again
        when verified (with complete outcome).
        """
        from atomicx.data.storage.database import get_session_factory
        from atomicx.data.storage.models import PredictionOutcome
        from sqlalchemy.dialects.postgresql import insert

        try:
            # Serialize datetime objects in nested structures for JSON storage
            if variable_snapshot:
                variable_snapshot = _serialize_for_json(variable_snapshot)
            if metadata:
                metadata = _serialize_for_json(metadata)

            session_factory = get_session_factory()
            async with session_factory() as session:
                stmt = insert(PredictionOutcome).values(
                    prediction_id=prediction_id,
                    symbol=symbol,
                    timeframe=timeframe,
                    predicted_direction=predicted_direction,
                    confidence=confidence,
                    entry_price=entry_price,
                    verification_price=verification_price,
                    was_correct=was_correct,
                    actual_return=actual_return,
                    profit_return=profit_return,
                    regime=regime,
                    predicted_at=predicted_at,
                    verified_at=verified_at,
                    reasoning=reasoning,
                    variable_snapshot=variable_snapshot,
                    metadata_=metadata,
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=['prediction_id'],
                    set_={
                        'verification_price': stmt.excluded.verification_price,
                        'was_correct': stmt.excluded.was_correct,
                        'actual_return': stmt.excluded.actual_return,
                        'verified_at': stmt.excluded.verified_at,
                    }
                )
                await session.execute(stmt)
                await session.commit()

                self.logger.debug(
                    f"[PERSISTENCE] Saved prediction outcome {prediction_id} to DB"
                )
        except Exception as e:
            self.logger.error(
                f"[PERSISTENCE] Failed to save prediction outcome {prediction_id}: {e}"
            )

    async def load_recent_outcomes(self, limit: int = 100) -> list[dict[str, Any]]:
        """Load recent verified outcomes from database on startup.

        Used to populate prediction_history list for dashboard.
        """
        from atomicx.data.storage.database import get_session_factory
        from atomicx.data.storage.models import PredictionOutcome
        from sqlalchemy import select

        try:
            session_factory = get_session_factory()
            async with session_factory() as session:
                result = await session.execute(
                    select(PredictionOutcome)
                    .where(PredictionOutcome.verified_at.isnot(None))
                    .order_by(PredictionOutcome.verified_at.desc())
                    .limit(limit)
                )
                outcomes = result.scalars().all()

                history = []
                for outcome in outcomes:
                    history.append({
                        "prediction_id": outcome.prediction_id,
                        "symbol": outcome.symbol,
                        "was_correct": outcome.was_correct,
                        "return": float(outcome.actual_return) if outcome.actual_return else 0.0,
                        "profit_return": float(outcome.profit_return) if outcome.profit_return else 0.0,
                        "regime": outcome.regime,
                        "verified_at": outcome.verified_at,
                    })

                self.logger.info(
                    f"[PERSISTENCE] Loaded {len(history)} recent outcomes from DB"
                )
                return history
        except Exception as e:
            self.logger.warning(f"[PERSISTENCE] Failed to load recent outcomes: {e}")
            return []

    async def scan_orphaned_predictions(self) -> list[dict[str, Any]]:
        """Scan database for predictions with NULL outcomes (orphaned predictions).

        FIX: Critical data loss where predictions made before restart are never verified.
        These "orphaned" predictions skew agent performance metrics permanently.

        Returns list of orphaned predictions that need verification.
        """
        from atomicx.data.storage.database import get_session_factory
        from atomicx.data.storage.models import PredictionOutcome
        from sqlalchemy import select

        try:
            session_factory = get_session_factory()
            async with session_factory() as session:
                # Find predictions older than 15 minutes with NULL was_correct
                result = await session.execute(
                    select(PredictionOutcome)
                    .where(
                        PredictionOutcome.was_correct.is_(None),
                        PredictionOutcome.predicted_at < datetime.now(tz=timezone.utc)
                    )
                    .order_by(PredictionOutcome.predicted_at.asc())
                    .limit(100)  # Process in batches
                )
                orphaned = result.scalars().all()

                orphaned_list = []
                for outcome in orphaned:
                    age_minutes = (datetime.now(tz=timezone.utc) - outcome.predicted_at).total_seconds() / 60
                    orphaned_list.append({
                        "prediction_id": outcome.prediction_id,
                        "symbol": outcome.symbol,
                        "timeframe": outcome.timeframe,
                        "predicted_direction": outcome.predicted_direction,
                        "confidence": float(outcome.confidence),
                        "entry_price": float(outcome.entry_price),
                        "predicted_at": outcome.predicted_at,
                        "regime": outcome.regime,
                        "age_minutes": age_minutes,
                    })

                if orphaned_list:
                    self.logger.warning(
                        f"[ORPHAN-SCAN] Found {len(orphaned_list)} orphaned predictions "
                        f"(age: {orphaned_list[0]['age_minutes']:.0f}-{orphaned_list[-1]['age_minutes']:.0f} min)"
                    )
                else:
                    self.logger.info("[ORPHAN-SCAN] No orphaned predictions found")

                return orphaned_list
        except Exception as e:
            self.logger.error(f"[ORPHAN-SCAN] Failed to scan orphaned predictions: {e}")
            return []

    async def verify_orphaned_prediction(
        self,
        prediction_id: str,
        symbol: str,
        predicted_direction: str,
        entry_price: float,
        current_price: float,
        predicted_at: datetime,
    ) -> tuple[bool, float]:
        """Verify an orphaned prediction against current market price.

        Args:
            prediction_id: Prediction identifier
            symbol: Trading pair
            predicted_direction: bullish or bearish
            entry_price: Price at prediction time
            current_price: Current market price
            predicted_at: When prediction was made

        Returns:
            (was_correct, actual_return) tuple
        """
        # Calculate price movement
        price_change = (current_price - entry_price) / entry_price
        actual_return = price_change

        # Determine correctness based on direction
        if predicted_direction == "bullish":
            was_correct = price_change > 0.001  # >0.1% gain
        elif predicted_direction == "bearish":
            was_correct = price_change < -0.001  # >0.1% loss
        else:
            was_correct = False  # neutral predictions don't count

        age_minutes = (datetime.now(tz=timezone.utc) - predicted_at).total_seconds() / 60

        self.logger.info(
            f"[ORPHAN-VERIFY] {prediction_id}: {predicted_direction} @ ${entry_price:.2f} → ${current_price:.2f} "
            f"({price_change:+.2%}) after {age_minutes:.0f}min = {'✓ CORRECT' if was_correct else '✗ WRONG'}"
        )

        return was_correct, actual_return


# ═══════════════════════════════════════════════════════════════════════════
# REGIME HISTORY PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════


async def save_regime_change(
    symbol: str,
    old_regime: str | None,
    new_regime: str,
    confidence: float,
    drift_score: float | None = None,
    trigger_reason: str | None = None,
    metadata: dict | None = None,
) -> None:
    """Persist regime transition to database.

    Args:
        symbol: Trading pair
        old_regime: Previous regime (None if first detection)
        new_regime: New regime
        confidence: Regime detection confidence
        drift_score: Drift detection score
        trigger_reason: What triggered the regime change
        metadata: Additional context
    """
    from atomicx.data.storage.database import get_session_factory
    from atomicx.data.storage.models import RegimeHistory

    try:
        session_factory = get_session_factory()
        async with session_factory() as session:
            regime_record = RegimeHistory(
                symbol=symbol,
                old_regime=old_regime,
                new_regime=new_regime,
                confidence=confidence,
                drift_score=drift_score,
                trigger_reason=trigger_reason,
                metadata_=metadata,
            )
            session.add(regime_record)
            await session.commit()

            logger.debug(
                f"[PERSISTENCE] Saved regime change {symbol}: {old_regime} → {new_regime}"
            )
    except Exception as e:
        logger.error(f"[PERSISTENCE] Failed to save regime change: {e}")


async def load_last_regime(symbol: str) -> str | None:
    """Load the last known regime for a symbol from database.

    Returns None if no history exists.
    """
    from atomicx.data.storage.database import get_session_factory
    from atomicx.data.storage.models import RegimeHistory
    from sqlalchemy import select

    try:
        session_factory = get_session_factory()
        async with session_factory() as session:
            result = await session.execute(
                select(RegimeHistory)
                .where(RegimeHistory.symbol == symbol)
                .order_by(RegimeHistory.timestamp.desc())
                .limit(1)
            )
            regime_record = result.scalars().first()

            if regime_record:
                logger.debug(
                    f"[PERSISTENCE] Loaded last regime for {symbol}: {regime_record.new_regime}"
                )
                return regime_record.new_regime
            return None
    except Exception as e:
        logger.warning(f"[PERSISTENCE] Failed to load last regime: {e}")
        return None
