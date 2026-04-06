"""Batch manager for reducing database connection pressure.

Prevents 50+ agents from hitting the database simultaneously by batching writes.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any
from loguru import logger

from atomicx.data.storage.database import get_session
from atomicx.data.storage.models import AgentPerformance


class BatchManager:
    """Batches agent performance updates to reduce database pressure.

    Instead of 50 agents saving simultaneously:
    - Collects updates in memory
    - Flushes every 5 seconds or when batch size reaches 20
    - Single transaction for entire batch
    """

    def __init__(self):
        self.pending_updates: dict[str, dict[str, Any]] = {}
        self.lock = asyncio.Lock()
        self.flush_task: asyncio.Task | None = None
        self.logger = logger.bind(module="batch_manager")
        self.batch_size = 20  # Flush after 20 updates
        self.flush_interval = 5.0  # Flush every 5 seconds

    async def start(self):
        """Start the periodic flush task."""
        if self.flush_task is None:
            self.flush_task = asyncio.create_task(self._periodic_flush())
            self.logger.info("BatchManager started")

    async def stop(self):
        """Stop the flush task and flush remaining updates."""
        if self.flush_task:
            self.flush_task.cancel()
            try:
                await self.flush_task
            except asyncio.CancelledError:
                pass
        await self.flush()
        self.logger.info("BatchManager stopped")

    async def queue_agent_performance_update(
        self,
        agent_id: str,
        total_predictions: int = 0,
        correct_predictions: int = 0,
        performance_edge: float | None = None,
        weight: float | None = None,
    ):
        """Queue an agent performance update for batch processing.

        Args:
            agent_id: Unique agent identifier
            total_predictions: Number of predictions made
            correct_predictions: Number of correct predictions
            performance_edge: Agent's performance edge (0-1)
            weight: Agent's trust weight
        """
        async with self.lock:
            key = agent_id

            if key in self.pending_updates:
                # Merge with existing pending update
                existing = self.pending_updates[key]
                existing["total_predictions"] += total_predictions
                existing["correct_predictions"] += correct_predictions
                if performance_edge is not None:
                    existing["performance_edge"] = performance_edge
                if weight is not None:
                    existing["weight"] = weight
            else:
                # New update
                self.pending_updates[key] = {
                    "agent_id": agent_id,
                    "total_predictions": total_predictions,
                    "correct_predictions": correct_predictions,
                    "performance_edge": performance_edge,
                    "weight": weight,
                }

            # Flush immediately if batch is full
            if len(self.pending_updates) >= self.batch_size:
                asyncio.create_task(self.flush())

    async def _periodic_flush(self):
        """Periodically flush pending updates."""
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                await self.flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in periodic flush: {e}")

    async def flush(self):
        """Flush all pending updates to database in a single transaction."""
        async with self.lock:
            if not self.pending_updates:
                return

            updates_to_flush = list(self.pending_updates.values())
            self.pending_updates.clear()

        try:
            async with get_session() as session:
                for update in updates_to_flush:
                    # Upsert agent performance
                    perf = await session.get(
                        AgentPerformance,
                        update["agent_id"]
                    )

                    if perf:
                        # Update existing
                        perf.total_predictions += update["total_predictions"]
                        perf.correct_predictions += update["correct_predictions"]
                        if update["performance_edge"] is not None:
                            perf.performance_edge = update["performance_edge"]
                        if update["weight"] is not None:
                            perf.weight = update["weight"]
                        perf.last_prediction_at = datetime.now(timezone.utc)
                    else:
                        # Create new
                        performance_edge = update.get("performance_edge", 0.5)
                        weight = update.get("weight", 1.0)

                        perf = AgentPerformance(
                            agent_id=update["agent_id"],
                            total_predictions=update["total_predictions"],
                            correct_predictions=update["correct_predictions"],
                            performance_edge=performance_edge,
                            weight=weight,
                            last_prediction_at=datetime.now(timezone.utc),
                        )
                        session.add(perf)

                await session.commit()

                self.logger.success(
                    f"[BATCH] Flushed {len(updates_to_flush)} agent performance updates"
                )

        except Exception as e:
            self.logger.error(f"[BATCH] Error flushing updates: {e}")
            # Re-queue failed updates
            async with self.lock:
                for update in updates_to_flush:
                    key = update['agent_id']
                    self.pending_updates[key] = update


# Global singleton
_batch_manager: BatchManager | None = None


def get_batch_manager() -> BatchManager:
    """Get the global batch manager instance."""
    global _batch_manager
    if _batch_manager is None:
        _batch_manager = BatchManager()
    return _batch_manager
