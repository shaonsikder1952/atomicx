"""Circuit breaker and data freshness monitoring.

Tracks the health of each data source. If a source fails repeatedly,
the circuit breaker opens and the system degrades gracefully.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from loguru import logger


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failures exceeded threshold — requests blocked
    HALF_OPEN = "half_open"  # Testing if source has recovered


class CircuitBreaker:
    """Per-source circuit breaker with configurable thresholds.

    State transitions:
    CLOSED → OPEN: after `failure_threshold` consecutive failures
    OPEN → HALF_OPEN: after `recovery_timeout` seconds
    HALF_OPEN → CLOSED: on first success
    HALF_OPEN → OPEN: on first failure
    """

    def __init__(
        self,
        source_name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        on_state_change: Any = None,
    ) -> None:
        self.source_name = source_name
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._on_state_change = on_state_change

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: datetime | None = None
        self._last_success_time: datetime | None = None
        self._last_error: str | None = None

    @property
    def state(self) -> CircuitState:
        """Get current state, checking for auto-recovery."""
        if self._state == CircuitState.OPEN and self._last_failure_time:
            elapsed = (datetime.now(tz=timezone.utc) - self._last_failure_time).total_seconds()
            if elapsed >= self._recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                logger.info(
                    f"Circuit breaker [{self.source_name}] → HALF_OPEN (testing recovery)"
                )
        return self._state

    @property
    def is_available(self) -> bool:
        """Check if requests should be allowed."""
        return self.state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)

    def record_success(self) -> None:
        """Record a successful operation."""
        self._failure_count = 0
        self._last_success_time = datetime.now(tz=timezone.utc)

        if self._state != CircuitState.CLOSED:
            old_state = self._state
            self._state = CircuitState.CLOSED
            logger.info(
                f"Circuit breaker [{self.source_name}] → CLOSED (recovered)"
            )
            if self._on_state_change:
                asyncio.create_task(
                    self._on_state_change(self.source_name, old_state, self._state)
                )

    def record_failure(self, error: str | None = None) -> None:
        """Record a failed operation."""
        self._failure_count += 1
        self._last_failure_time = datetime.now(tz=timezone.utc)
        self._last_error = error

        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker [{self.source_name}] → OPEN (recovery failed)"
            )
        elif self._failure_count >= self._failure_threshold:
            old_state = self._state
            self._state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker [{self.source_name}] → OPEN "
                f"(failed {self._failure_count}x, error: {error})"
            )
            if self._on_state_change and old_state != CircuitState.OPEN:
                asyncio.create_task(
                    self._on_state_change(self.source_name, old_state, self._state)
                )

    def to_dict(self) -> dict[str, Any]:
        """Serialize state for monitoring/persistence."""
        return {
            "source_name": self.source_name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "last_failure_time": self._last_failure_time.isoformat() if self._last_failure_time else None,
            "last_success_time": self._last_success_time.isoformat() if self._last_success_time else None,
            "last_error": self._last_error,
        }


class DataFreshnessMonitor:
    """Monitors data freshness across all sources.

    Each source has a staleness threshold. If data is older than the
    threshold, the source is flagged as stale.
    """

    def __init__(self) -> None:
        self._sources: dict[str, dict[str, Any]] = {}
        self._circuit_breakers: dict[str, CircuitBreaker] = {}

    def register_source(
        self,
        name: str,
        staleness_threshold: timedelta = timedelta(minutes=5),
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
    ) -> CircuitBreaker:
        """Register a data source for monitoring."""
        self._sources[name] = {
            "staleness_threshold": staleness_threshold,
            "last_update": None,
            "record_count": 0,
        }
        cb = CircuitBreaker(
            source_name=name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        )
        self._circuit_breakers[name] = cb
        logger.info(f"Registered data source: {name}")
        return cb

    def record_update(self, source_name: str, count: int = 1) -> None:
        """Record that a data source has been updated."""
        if source_name in self._sources:
            self._sources[source_name]["last_update"] = datetime.now(tz=timezone.utc)
            self._sources[source_name]["record_count"] += count
        if source_name in self._circuit_breakers:
            self._circuit_breakers[source_name].record_success()

    def record_error(self, source_name: str, error: str) -> None:
        """Record that a data source has errored."""
        if source_name in self._circuit_breakers:
            self._circuit_breakers[source_name].record_failure(error)

    def get_stale_sources(self) -> list[str]:
        """Get list of sources that have gone stale."""
        now = datetime.now(tz=timezone.utc)
        stale = []
        for name, info in self._sources.items():
            last = info["last_update"]
            threshold = info["staleness_threshold"]
            if last is None or (now - last) > threshold:
                stale.append(name)
        return stale

    def get_status(self) -> dict[str, Any]:
        """Get complete health status of all data sources."""
        now = datetime.now(tz=timezone.utc)
        status = {}
        for name, info in self._sources.items():
            last = info["last_update"]
            cb = self._circuit_breakers.get(name)
            status[name] = {
                "last_update": last.isoformat() if last else None,
                "age_seconds": (now - last).total_seconds() if last else None,
                "is_stale": name in self.get_stale_sources(),
                "record_count": info["record_count"],
                "circuit_breaker": cb.to_dict() if cb else None,
            }
        return status

    def is_source_available(self, source_name: str) -> bool:
        """Check if a source is available (circuit not open)."""
        cb = self._circuit_breakers.get(source_name)
        return cb.is_available if cb else True
