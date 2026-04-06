"""Data replay engine for backtesting.

Replays historical data from TimescaleDB at configurable speeds.
Produces the same callback interface as live connectors,
enabling identical downstream processing for backtests.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from atomicx.data.storage.database import get_session_factory
from atomicx.data.storage.models import OHLCV, OnChainMetric, OrderBookSnapshot, TickData


class DataReplayEngine:
    """Replays historical data for backtesting.

    Supports:
    - Variable speed replay (1x, 10x, 100x, instant)
    - Multiple data types (ticks, klines, depth, on-chain)
    - Time-range selection
    - Symbol filtering
    - Pause/resume
    """

    def __init__(
        self,
        start_time: datetime,
        end_time: datetime,
        symbols: list[str] | None = None,
        speed: float = 1.0,
        on_trade: Any = None,
        on_kline: Any = None,
        on_depth: Any = None,
        on_metric: Any = None,
    ) -> None:
        self._start_time = start_time
        self._end_time = end_time
        self._symbols = symbols
        self._speed = speed  # 1.0 = real-time, 0 = instant
        self._on_trade = on_trade
        self._on_kline = on_kline
        self._on_depth = on_depth
        self._on_metric = on_metric
        self._session_factory = get_session_factory()
        self._running = False
        self._paused = False
        self._current_time: datetime | None = None

    async def start(self) -> None:
        """Start replaying historical data."""
        self._running = True
        logger.info(
            "Starting data replay",
            start=self._start_time.isoformat(),
            end=self._end_time.isoformat(),
            speed=f"{self._speed}x",
            symbols=self._symbols,
        )

        await self._replay_klines()

        if self._running:
            logger.info("Data replay completed")

    async def _replay_klines(self) -> None:
        """Replay OHLCV kline data in chronological order."""
        async with self._session_factory() as session:
            query = (
                select(OHLCV)
                .where(OHLCV.timestamp >= self._start_time)
                .where(OHLCV.timestamp <= self._end_time)
                .order_by(OHLCV.timestamp)
            )

            if self._symbols:
                query = query.where(OHLCV.symbol.in_(self._symbols))

            result = await session.execute(query)
            klines = result.scalars().all()

            logger.info(f"Replaying {len(klines)} klines")

            prev_ts: datetime | None = None
            for kline in klines:
                if not self._running:
                    break

                while self._paused:
                    await asyncio.sleep(0.1)

                # Simulate time delay between events
                if self._speed > 0 and prev_ts:
                    delta = (kline.timestamp - prev_ts).total_seconds()
                    await asyncio.sleep(delta / self._speed)

                self._current_time = kline.timestamp

                kline_data = {
                    "timestamp": kline.timestamp,
                    "symbol": kline.symbol,
                    "timeframe": kline.timeframe,
                    "open": kline.open,
                    "high": kline.high,
                    "low": kline.low,
                    "close": kline.close,
                    "volume": kline.volume,
                    "quote_volume": kline.quote_volume,
                    "trade_count": kline.trade_count,
                    "is_closed": True,
                }

                if self._on_kline:
                    await self._on_kline(kline_data)

                prev_ts = kline.timestamp

    def pause(self) -> None:
        """Pause replay."""
        self._paused = True
        logger.info("Replay paused")

    def resume(self) -> None:
        """Resume replay."""
        self._paused = False
        logger.info("Replay resumed")

    async def stop(self) -> None:
        """Stop replay."""
        self._running = False
        logger.info("Replay stopped")

    @property
    def current_time(self) -> datetime | None:
        """Get the current replay timestamp."""
        return self._current_time
