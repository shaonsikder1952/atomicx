"""Data persistence service — writes market data to TimescaleDB.

Consumes data from connectors (via callbacks) and batch-inserts to database.
Buffers writes for efficiency (configurable flush interval).
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from loguru import logger
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from atomicx.data.storage.database import get_session_factory
from atomicx.data.storage.models import (
    OHLCV,
    DataFreshness,
    FundingRate,
    OnChainMetric,
    OrderBookSnapshot,
    TickData,
)


class DataPersistenceService:
    """Buffers and batch-writes market data to TimescaleDB.

    Uses upsert (INSERT ON CONFLICT) for idempotent writes.
    Configurable buffer size and flush interval.
    """

    def __init__(
        self,
        buffer_size: int = 500,
        flush_interval: float = 5.0,
    ) -> None:
        self._buffer_size = buffer_size
        self._flush_interval = flush_interval
        self._buffers: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._session_factory = get_session_factory()
        self._running = False
        self._flush_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the persistence service with periodic flush."""
        self._running = True
        self._flush_task = asyncio.create_task(self._periodic_flush())
        logger.info("Data persistence service started")

    async def _periodic_flush(self) -> None:
        """Periodically flush all buffers."""
        while self._running:
            await asyncio.sleep(self._flush_interval)
            await self.flush_all()

    async def write_trade(self, trade: dict[str, Any]) -> None:
        """Buffer a trade for batch writing."""
        self._buffers["trades"].append(trade)
        if len(self._buffers["trades"]) >= self._buffer_size:
            await self._flush_trades()

    async def write_kline(self, kline: dict[str, Any]) -> None:
        """Buffer a kline for batch writing (only closed candles)."""
        if kline.get("is_closed", False):
            self._buffers["klines"].append(kline)
            if len(self._buffers["klines"]) >= self._buffer_size:
                await self._flush_klines()

    async def write_depth(self, depth: dict[str, Any]) -> None:
        """Buffer a depth snapshot for batch writing."""
        self._buffers["depth"].append(depth)
        if len(self._buffers["depth"]) >= self._buffer_size:
            await self._flush_depth()

    async def write_onchain_metric(self, metric: dict[str, Any]) -> None:
        """Buffer an on-chain metric for batch writing."""
        self._buffers["onchain"].append(metric)
        if len(self._buffers["onchain"]) >= self._buffer_size:
            await self._flush_onchain()

    async def flush_all(self) -> None:
        """Flush all buffers to database."""
        for flush_fn in [
            self._flush_trades,
            self._flush_klines,
            self._flush_depth,
            self._flush_onchain,
        ]:
            try:
                await flush_fn()
            except Exception as e:
                logger.error(f"Flush error: {e}")

    async def _flush_trades(self) -> None:
        """Batch insert trades to TimescaleDB."""
        trades = self._buffers["trades"]
        if not trades:
            return
        self._buffers["trades"] = []

        async with self._session_factory() as session:
            try:
                await session.execute(
                    TickData.__table__.insert(),
                    [
                        {
                            "timestamp": t["timestamp"],
                            "symbol": t["symbol"],
                            "price": t["price"],
                            "quantity": t["quantity"],
                            "is_buyer_maker": t.get("is_buyer_maker"),
                            "trade_id": t.get("trade_id"),
                        }
                        for t in trades
                    ],
                )
                await session.commit()
                logger.debug(f"Flushed {len(trades)} trades to DB")
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to flush trades: {e}")
                self._buffers["trades"].extend(trades)  # Re-queue

    async def _flush_klines(self) -> None:
        """Batch upsert klines to TimescaleDB."""
        klines = self._buffers["klines"]
        if not klines:
            return
        self._buffers["klines"] = []

        async with self._session_factory() as session:
            try:
                for k in klines:
                    stmt = pg_insert(OHLCV).values(
                        timestamp=k["timestamp"],
                        symbol=k["symbol"],
                        timeframe=k.get("timeframe", "1m"),
                        open=k["open"],
                        high=k["high"],
                        low=k["low"],
                        close=k["close"],
                        volume=k["volume"],
                        quote_volume=k.get("quote_volume"),
                        trade_count=k.get("trade_count"),
                    )
                    stmt = stmt.on_conflict_do_update(
                        constraint="uq_ohlcv_ts_sym_tf",
                        set_={
                            "high": stmt.excluded.high,
                            "low": stmt.excluded.low,
                            "close": stmt.excluded.close,
                            "volume": stmt.excluded.volume,
                            "quote_volume": stmt.excluded.quote_volume,
                            "trade_count": stmt.excluded.trade_count,
                        },
                    )
                    await session.execute(stmt)
                await session.commit()
                logger.debug(f"Flushed {len(klines)} klines to DB")
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to flush klines: {e}")

    async def _flush_depth(self) -> None:
        """Batch insert depth snapshots to TimescaleDB."""
        snapshots = self._buffers["depth"]
        if not snapshots:
            return
        self._buffers["depth"] = []

        async with self._session_factory() as session:
            try:
                await session.execute(
                    OrderBookSnapshot.__table__.insert(),
                    [
                        {
                            "timestamp": d["timestamp"],
                            "symbol": d["symbol"],
                            "bids": [[str(p), str(q)] for p, q in d.get("bids", [])],
                            "asks": [[str(p), str(q)] for p, q in d.get("asks", [])],
                            "bid_total_volume": d.get("bid_total_volume"),
                            "ask_total_volume": d.get("ask_total_volume"),
                            "spread": d.get("spread"),
                            "mid_price": d.get("mid_price"),
                        }
                        for d in snapshots
                    ],
                )
                await session.commit()
                logger.debug(f"Flushed {len(snapshots)} depth snapshots to DB")
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to flush depth: {e}")

    async def _flush_onchain(self) -> None:
        """Batch insert on-chain metrics to TimescaleDB."""
        metrics = self._buffers["onchain"]
        if not metrics:
            return
        self._buffers["onchain"] = []

        async with self._session_factory() as session:
            try:
                await session.execute(
                    OnChainMetric.__table__.insert(),
                    [
                        {
                            "timestamp": m["timestamp"],
                            "symbol": m["symbol"],
                            "metric_name": m["metric_name"],
                            "value": m["value"],
                            "source": m["source"],
                            "metadata": m.get("metadata"),
                        }
                        for m in metrics
                    ],
                )
                await session.commit()
                logger.debug(f"Flushed {len(metrics)} on-chain metrics to DB")
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to flush on-chain metrics: {e}")

    async def stop(self) -> None:
        """Flush remaining data and stop."""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
        await self.flush_all()
        logger.info("Data persistence service stopped")
