"""Base agent class — all agents inherit from this.

Defines the interface every agent must implement, plus common
functionality for performance tracking and dynamic pruning.
"""

from __future__ import annotations

import abc
import asyncio
from collections import deque
from datetime import datetime, timezone
from typing import Dict, Tuple

from loguru import logger
from pydantic import BaseModel, Field

from atomicx.agents.signals import AgentSignal, SignalDirection


# ═══════════════════════════════════════════════════════════════════════════
# BATCHED PERSISTENCE QUEUE (Fix for "Dementia Bug")
# ═══════════════════════════════════════════════════════════════════════════

class AgentPerformanceQueue:
    """Batched persistence queue for agent performance updates.

    FIX: Prevents connection stampede when 50+ agents try to save simultaneously.
    Uses controlled batching with semaphore to limit concurrent DB connections.
    """

    def __init__(self, max_concurrent: int = 10):
        """Initialize queue with concurrency limit."""
        self._queue: deque[Tuple[str, Dict]] = deque()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._processing = False
        self._lock = asyncio.Lock()
        self._background_task: asyncio.Task | None = None
        self._shutdown = False

    def enqueue(self, agent_id: str, performance_data: dict) -> None:
        """Add performance update to queue (non-blocking)."""
        self._queue.append((agent_id, performance_data))

        # Start background processor if not already running
        if self._background_task is None or self._background_task.done():
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    self._background_task = loop.create_task(self._continuous_processor())
            except RuntimeError:
                # No event loop running yet, will be started later
                pass

    async def _continuous_processor(self) -> None:
        """Continuously process queued performance updates in background.

        FIX: Single background task prevents stampede of concurrent batch processors.
        Runs continuously, processing batches as they arrive.
        """
        logger.debug("[AGENT-QUEUE] Background processor started")

        while not self._shutdown:
            try:
                # Check if queue has items
                if self._queue:
                    await self.process_batch()
                else:
                    # Sleep briefly if queue is empty
                    await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"[AGENT-QUEUE] Background processor error: {e}")
                await asyncio.sleep(1.0)  # Back off on error

        logger.debug("[AGENT-QUEUE] Background processor stopped")

    def shutdown(self) -> None:
        """Signal background processor to shutdown."""
        self._shutdown = True
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()

    async def process_batch(self) -> None:
        """Process all queued saves in controlled batches.

        - Takes up to 10 items at a time
        - Uses semaphore to limit concurrent DB connections
        - Handles failures gracefully with logging
        """
        # Prevent concurrent batch processing
        async with self._lock:
            if self._processing:
                return
            self._processing = True

        try:
            processed = 0
            failed = 0

            while self._queue:
                # Take batch of up to 10 items
                batch = []
                for _ in range(min(10, len(self._queue))):
                    if self._queue:
                        batch.append(self._queue.popleft())

                if not batch:
                    break

                # Process batch with concurrency control
                results = await asyncio.gather(
                    *[self._save_with_semaphore(agent_id, data) for agent_id, data in batch],
                    return_exceptions=True
                )

                # Count successes/failures
                for result in results:
                    if isinstance(result, Exception):
                        failed += 1
                    else:
                        processed += 1

            if processed > 0:
                logger.debug(f"[AGENT-QUEUE] Batch complete: {processed} saved, {failed} failed")

        finally:
            self._processing = False

    async def _save_with_semaphore(self, agent_id: str, data: dict) -> None:
        """Save with semaphore to limit concurrent connections."""
        async with self._semaphore:
            await self._do_save(agent_id, data)

    async def _do_save(self, agent_id: str, data: dict) -> None:
        """Actually persist to database."""
        from atomicx.data.storage.database import get_session
        from atomicx.data.storage.models import AgentPerformance
        from sqlalchemy.dialects.postgresql import insert

        try:
            async with get_session() as session:
                stmt = insert(AgentPerformance).values(
                    agent_id=agent_id,
                    total_predictions=data['total_predictions'],
                    correct_predictions=data['correct_predictions'],
                    performance_edge=data['performance_edge'],
                    weight=data['weight'],
                    is_active=data['is_active'],
                    last_prediction_at=datetime.now(tz=timezone.utc),
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=['agent_id'],
                    set_={
                        'total_predictions': stmt.excluded.total_predictions,
                        'correct_predictions': stmt.excluded.correct_predictions,
                        'performance_edge': stmt.excluded.performance_edge,
                        'weight': stmt.excluded.weight,
                        'is_active': stmt.excluded.is_active,
                        'last_prediction_at': stmt.excluded.last_prediction_at,
                        'updated_at': datetime.now(tz=timezone.utc),
                    }
                )
                await session.execute(stmt)
                # Commit is automatic in get_session context manager

        except Exception as e:
            logger.warning(f"[AGENT-QUEUE] Failed to save {agent_id}: {e}")
            raise  # Re-raise for gather to catch


# Global singleton queue
_performance_queue = AgentPerformanceQueue(max_concurrent=10)


class AgentConfig(BaseModel):
    """Configuration for any agent."""

    agent_id: str
    agent_type: str  # atomic, group_leader, super_group, etc.
    name: str
    description: str = ""
    enabled: bool = True
    weight: float = Field(default=1.0, ge=0.0, le=2.0)

    # Performance tracking
    total_predictions: int = 0
    correct_predictions: int = 0
    performance_edge: float = 0.0

    # Pruning
    min_confidence_to_signal: float = 0.40
    auto_prune_after: int = 200  # Min predictions before pruning
    min_edge_threshold: float = 0.02  # 2% edge minimum


class BaseAgent(abc.ABC):
    """Abstract base agent — all agents in the hierarchy inherit from this.

    Provides:
    - Signal generation interface
    - Performance tracking (correct/total predictions)
    - Dynamic pruning (skip if no signal for current query)
    - Weight management
    """

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._active = config.enabled
        self.last_signal: AgentSignal | None = None

    @property
    def agent_id(self) -> str:
        return self.config.agent_id

    @property
    def agent_type(self) -> str:
        return self.config.agent_type

    @property
    def is_active(self) -> bool:
        return self._active and self.config.enabled

    @property
    def should_invert(self) -> bool:
        """Check if agent is so consistently wrong that we should invert its predictions.

        HEDGE FUND TRICK: If an agent is systematically wrong (edge < -20%),
        flip its predictions. Some of the best strategies come from this.
        """
        return (
            self.config.total_predictions >= self.config.auto_prune_after
            and self.config.performance_edge < -0.20
        )

    @abc.abstractmethod
    async def generate_signal(
        self, symbol: str, timeframe: str, context: dict
    ) -> AgentSignal:
        """Generate a directional signal for the given symbol.

        Args:
            symbol: Trading pair, e.g. 'BTC/USDT'
            timeframe: Analysis timeframe, e.g. '4h'
            context: Market context (variable values, causal data, etc.)

        Returns:
            AgentSignal with direction, confidence, and reasoning
        """
        ...

    async def evaluate(
        self, symbol: str, timeframe: str, context: dict
    ) -> AgentSignal | None:
        """Evaluate and return signal, or None if agent should be skipped."""
        if not self.is_active:
            return None

        try:
            signal = await self.generate_signal(symbol, timeframe, context)

            # HEDGE FUND TRICK: Invert consistently wrong agents
            if self.should_invert:
                if signal.direction == SignalDirection.BULLISH:
                    signal.direction = SignalDirection.BEARISH
                    signal.reasoning = f"[INVERTED] {signal.reasoning} (agent edge={self.config.performance_edge:.1%})"
                elif signal.direction == SignalDirection.BEARISH:
                    signal.direction = SignalDirection.BULLISH
                    signal.reasoning = f"[INVERTED] {signal.reasoning} (agent edge={self.config.performance_edge:.1%})"
                logger.info(
                    f"[INVERSION] Agent {self.agent_id} prediction inverted "
                    f"(edge={self.config.performance_edge:.1%} → consistently wrong)"
                )

            # Auto-skip low confidence
            if signal.confidence < self.config.min_confidence_to_signal:
                signal.direction = SignalDirection.SKIP
                self.last_signal = signal
                return signal

            signal.classify_confidence()
            signal.weight = self.config.weight
            self.last_signal = signal
            return signal

        except Exception as e:
            logger.error(f"Agent {self.agent_id} error: {e}")
            return None

    def record_outcome(self, was_correct: bool) -> None:
        """Record a prediction outcome for performance tracking.

        ═══ FIX: Now persists to database after each outcome ═══
        """
        self.config.total_predictions += 1
        if was_correct:
            self.config.correct_predictions += 1

        if self.config.total_predictions > 0:
            win_rate = self.config.correct_predictions / self.config.total_predictions
            self.config.performance_edge = win_rate - 0.5  # Edge over coin flip

        # CRITICAL FIX: Disable agents with strong negative edge
        if (
            self.config.total_predictions >= self.config.auto_prune_after
            and self.config.performance_edge < -0.10  # Worse than -10%
        ):
            self._active = False
            logger.error(
                f"[ANTI-LEARNING] Agent {self.agent_id} DISABLED: "
                f"edge={self.config.performance_edge:.2%} (consistently wrong). "
                f"Consider inverting this agent's predictions."
            )
        # Auto-pruning check for weak positive edge
        elif (
            self.config.total_predictions >= self.config.auto_prune_after
            and self.config.performance_edge < self.config.min_edge_threshold
        ):
            self._active = False
            logger.warning(
                f"Agent {self.agent_id} AUTO-PRUNED: "
                f"edge={self.config.performance_edge:.2%} < "
                f"{self.config.min_edge_threshold:.2%} after "
                f"{self.config.total_predictions} predictions"
            )

        # ═══ FIX: Persist performance to database (batched queue) ═══
        try:
            # Enqueue for batched processing (prevents connection stampede)
            # Background processor handles batching automatically
            _performance_queue.enqueue(
                self.agent_id,
                {
                    'total_predictions': self.config.total_predictions,
                    'correct_predictions': self.config.correct_predictions,
                    'performance_edge': self.config.performance_edge,
                    'weight': self.config.weight,
                    'is_active': self._active,
                }
            )
        except Exception as e:
            logger.warning(f"Failed to enqueue agent {self.agent_id} performance: {e}")

    @property
    def win_rate(self) -> float:
        if self.config.total_predictions == 0:
            return 0.5
        return self.config.correct_predictions / self.config.total_predictions

    # ═══════════════════════════════════════════════════════════════════════════
    # PERSISTENCE METHODS (Fix for agent learning state loss on restart)
    # ═══════════════════════════════════════════════════════════════════════════

    async def save_performance(self) -> None:
        """Persist agent performance to database.

        Called after every prediction outcome. Enables cross-session learning.
        Uses retry logic to handle connection pool exhaustion.
        """
        from atomicx.data.storage.database import get_session
        from atomicx.data.storage.models import AgentPerformance
        from sqlalchemy import select
        from sqlalchemy.dialects.postgresql import insert

        try:
            # Use get_session() which has retry logic built-in
            async with get_session() as session:
                # Upsert (insert or update)
                stmt = insert(AgentPerformance).values(
                    agent_id=self.agent_id,
                    total_predictions=self.config.total_predictions,
                    correct_predictions=self.config.correct_predictions,
                    performance_edge=self.config.performance_edge,
                    weight=self.config.weight,
                    is_active=self._active,
                    last_prediction_at=datetime.now(tz=timezone.utc),
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=['agent_id'],
                    set_={
                        'total_predictions': stmt.excluded.total_predictions,
                        'correct_predictions': stmt.excluded.correct_predictions,
                        'performance_edge': stmt.excluded.performance_edge,
                        'weight': stmt.excluded.weight,
                        'is_active': stmt.excluded.is_active,
                        'last_prediction_at': stmt.excluded.last_prediction_at,
                        'updated_at': datetime.now(tz=timezone.utc),
                    }
                )
                await session.execute(stmt)
                # Commit is automatic in get_session context manager

            # Calculate win rate for logging
            win_rate = (
                self.config.correct_predictions / self.config.total_predictions
                if self.config.total_predictions > 0
                else 0
            )

            logger.info(
                f"[LEARNING] Agent {self.agent_id}: {self.config.correct_predictions}/"
                f"{self.config.total_predictions} correct ({win_rate:.1%}), "
                f"edge={self.config.performance_edge:+.2%}, weight={self.config.weight:.3f}"
            )
        except Exception as e:
            logger.error(f"[PERSISTENCE] Failed to save agent {self.agent_id}: {e}")

    async def load_performance(self) -> bool:
        """Load agent performance from database.

        Called on startup. Returns True if state was restored.
        """
        from atomicx.data.storage.database import get_session_factory
        from atomicx.data.storage.models import AgentPerformance
        from sqlalchemy import select

        try:
            session_factory = get_session_factory()
            async with session_factory() as session:
                result = await session.execute(
                    select(AgentPerformance).where(AgentPerformance.agent_id == self.agent_id)
                )
                perf = result.scalars().first()

                if perf:
                    # Restore performance metrics
                    self.config.total_predictions = perf.total_predictions
                    self.config.correct_predictions = perf.correct_predictions
                    self.config.performance_edge = float(perf.performance_edge)
                    self.config.weight = float(perf.weight)
                    self._active = perf.is_active

                    logger.info(
                        f"[PERSISTENCE] Loaded agent {self.agent_id}: "
                        f"{perf.total_predictions} predictions, "
                        f"edge={perf.performance_edge:.2%}, "
                        f"active={perf.is_active}"
                    )
                    return True
                else:
                    logger.debug(f"[PERSISTENCE] No saved state for agent {self.agent_id}")
                    return False
        except Exception as e:
            logger.warning(f"[PERSISTENCE] Failed to load agent {self.agent_id}: {e}")
            return False
