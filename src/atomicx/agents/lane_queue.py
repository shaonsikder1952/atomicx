"""Lane-Aware Multi-Agent Queue System.

Extracted from OpenClaw's battle-tested queue patterns.
Prevents agent stampedes with symbol-specific lanes and priority routing.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Optional

from loguru import logger


class Priority(int, Enum):
    """Task priority levels."""
    URGENT = 0    # Market crash detection, critical alerts
    NORMAL = 1    # Routine analysis, predictions
    LOW = 2       # Historical backtesting, non-time-sensitive

class LaneAwareQueue:
    """Lane-aware FIFO queue with priority routing and per-lane concurrency.

    Extracted from OpenClaw's queue system for multi-agent coordination.

    Key Features:
    - Symbol-specific lanes (BTC/USDT, ETH/USDT get dedicated lanes)
    - Priority tiers (URGENT → NORMAL → LOW)
    - Per-lane concurrency control (prevent collisions)
    - Configurable global concurrency limits
    """

    def __init__(
        self,
        lane_concurrency: int = 1,      # Max concurrent tasks per lane
        main_concurrency: int = 4,      # Max concurrent main-lane tasks
        subagent_concurrency: int = 8,  # Max concurrent subagent tasks
    ):
        """Initialize lane-aware queue.

        Args:
            lane_concurrency: Max concurrent tasks per symbol lane (default: 1)
            main_concurrency: Max concurrent tasks in main lane (default: 4)
            subagent_concurrency: Max concurrent subagent tasks (default: 8)
        """
        # Lane-specific queues: {symbol: {priority: deque}}
        self._lanes: Dict[str, Dict[Priority, deque]] = defaultdict(
            lambda: {
                Priority.URGENT: deque(),
                Priority.NORMAL: deque(),
                Priority.LOW: deque(),
            }
        )

        # Semaphores for concurrency control
        self._lane_semaphores: Dict[str, asyncio.Semaphore] = {}
        self._main_semaphore = asyncio.Semaphore(main_concurrency)
        self._subagent_semaphore = asyncio.Semaphore(subagent_concurrency)

        self._lane_concurrency = lane_concurrency
        self._main_concurrency = main_concurrency
        self._subagent_concurrency = subagent_concurrency

        # Background processors per lane
        self._lane_tasks: Dict[str, asyncio.Task] = {}
        self._shutdown = False

        self.logger = logger.bind(module="agents.lane_queue")
        self.logger.info(
            f"[LANE-QUEUE] Initialized: lane={lane_concurrency}, "
            f"main={main_concurrency}, subagent={subagent_concurrency}"
        )

    def enqueue(
        self,
        task: Callable,
        lane: str = "main",
        priority: Priority = Priority.NORMAL,
        task_id: Optional[str] = None,
    ) -> None:
        """Add task to lane-specific queue.

        Args:
            task: Async callable to execute
            lane: Lane identifier (symbol like "BTC/USDT" or "main")
            priority: Task priority level
            task_id: Optional task identifier for logging
        """
        task_info = {
            "task": task,
            "task_id": task_id or f"{lane}_{priority.name}_{id(task)}",
            "enqueued_at": datetime.now(tz=timezone.utc),
        }

        self._lanes[lane][priority].append(task_info)

        self.logger.debug(
            f"[LANE-QUEUE] Enqueued {task_info['task_id']} → {lane} ({priority.name})"
        )

        # Start lane processor if not running
        if lane not in self._lane_tasks or self._lane_tasks[lane].done():
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    self._lane_tasks[lane] = loop.create_task(
                        self._process_lane(lane)
                    )
            except RuntimeError:
                pass  # No event loop yet

    async def _process_lane(self, lane: str) -> None:
        """Process all tasks in a specific lane.

        Handles priority ordering: URGENT → NORMAL → LOW
        Uses lane-specific semaphore to prevent collisions.
        """
        # Create lane semaphore if needed
        if lane not in self._lane_semaphores:
            if lane == "main":
                semaphore = self._main_semaphore
            elif lane.startswith("subagent_"):
                semaphore = self._subagent_semaphore
            else:
                # Symbol-specific lane
                semaphore = asyncio.Semaphore(self._lane_concurrency)
            self._lane_semaphores[lane] = semaphore

        semaphore = self._lane_semaphores[lane]

        self.logger.debug(f"[LANE-QUEUE] Started processor for lane: {lane}")

        while not self._shutdown:
            try:
                # Check queues in priority order
                task_info = None
                for priority in [Priority.URGENT, Priority.NORMAL, Priority.LOW]:
                    if self._lanes[lane][priority]:
                        task_info = self._lanes[lane][priority].popleft()
                        break

                if task_info is None:
                    # No tasks in lane, sleep briefly
                    await asyncio.sleep(0.1)
                    continue

                # Execute task with semaphore control
                async with semaphore:
                    task = task_info["task"]
                    task_id = task_info["task_id"]
                    enqueued_at = task_info["enqueued_at"]

                    # Calculate queue time
                    queue_time = (datetime.now(tz=timezone.utc) - enqueued_at).total_seconds()

                    self.logger.debug(
                        f"[LANE-QUEUE] Executing {task_id} (queue_time={queue_time:.2f}s)"
                    )

                    try:
                        if asyncio.iscoroutinefunction(task):
                            await task()
                        else:
                            task()

                        self.logger.debug(f"[LANE-QUEUE] ✓ Completed {task_id}")
                    except Exception as e:
                        self.logger.error(f"[LANE-QUEUE] ✗ Failed {task_id}: {e}")

            except Exception as e:
                self.logger.error(f"[LANE-QUEUE] Lane processor error ({lane}): {e}")
                await asyncio.sleep(1.0)  # Back off on error

        self.logger.debug(f"[LANE-QUEUE] Stopped processor for lane: {lane}")

    def get_stats(self) -> dict[str, Any]:
        """Get queue statistics for monitoring.

        Returns:
            Dict with queue depths per lane and priority
        """
        stats = {
            "total_lanes": len(self._lanes),
            "active_processors": sum(1 for t in self._lane_tasks.values() if not t.done()),
            "lanes": {},
        }

        for lane, priorities in self._lanes.items():
            lane_stats = {
                "urgent": len(priorities[Priority.URGENT]),
                "normal": len(priorities[Priority.NORMAL]),
                "low": len(priorities[Priority.LOW]),
                "total": sum(len(q) for q in priorities.values()),
            }
            stats["lanes"][lane] = lane_stats

        return stats

    def shutdown(self) -> None:
        """Shutdown all lane processors gracefully."""
        self._shutdown = True

        # Cancel all lane tasks
        for lane, task in self._lane_tasks.items():
            if not task.done():
                task.cancel()
                self.logger.debug(f"[LANE-QUEUE] Cancelled processor for lane: {lane}")

        self.logger.info("[LANE-QUEUE] Shutdown complete")


# Global singleton instance
_global_lane_queue: Optional[LaneAwareQueue] = None


def get_lane_queue() -> LaneAwareQueue:
    """Get or create global lane-aware queue instance."""
    global _global_lane_queue
    if _global_lane_queue is None:
        _global_lane_queue = LaneAwareQueue(
            lane_concurrency=1,      # 1 task per symbol at a time
            main_concurrency=4,      # 4 concurrent main tasks
            subagent_concurrency=8,  # 8 concurrent subagent tasks
        )
    return _global_lane_queue


def shutdown_lane_queue() -> None:
    """Shutdown global lane queue."""
    global _global_lane_queue
    if _global_lane_queue is not None:
        _global_lane_queue.shutdown()
        _global_lane_queue = None
