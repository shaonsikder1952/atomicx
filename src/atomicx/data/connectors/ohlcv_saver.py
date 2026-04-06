"""Real-time OHLCV persistence from WebSocket streams.

Receives kline/candlestick data from BinanceWebSocketConnector and writes
to TimescaleDB for historical analysis and backtesting.

INSTITUTIONAL FIX: Now also persists CVD and liquidation events for
anti-spoofing detection and Pain Map analysis.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from loguru import logger
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from atomicx.data.storage.database import get_session_factory
from atomicx.data.storage.models import OHLCV, Liquidation, CumulativeDelta


class OHLCVSaver:
    """Persists real-time OHLCV candles to TimescaleDB.

    Features:
    - Batched writes (reduces DB load)
    - Upsert on conflict (handles duplicate candles)
    - Async non-blocking writes
    - Only saves closed candles (is_closed=True)
    """

    def __init__(self, batch_size: int = 10, flush_interval: float = 5.0) -> None:
        self._session_factory = get_session_factory()

        # M3 OPTIMIZATION: Tune batch size for unified memory
        try:
            from atomicx.common.hardware import get_optimizer
            optimizer = get_optimizer()
            if optimizer.is_apple_silicon:
                # Estimate 0.01MB per OHLCV record
                optimal_batch = optimizer.get_optimal_batch_size(
                    item_size_mb=0.01,
                    memory_budget_pct=0.10  # Use 10% of available memory
                )
                batch_size = optimal_batch
                self.logger.info(
                    f"[M3-OPTIMIZER] Batch size tuned for unified memory: {batch_size}"
                )
        except Exception:
            pass  # Fall back to default if optimizer unavailable

        self._batch: list[dict[str, Any]] = []
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._flush_task: asyncio.Task | None = None
        self._running = False

        # INSTITUTIONAL FIX: Separate batches for CVD and liquidations
        self._cvd_batch: list[dict[str, Any]] = []
        self._liquidation_batch: list[dict[str, Any]] = []
        self._last_cvd: dict[str, float] = {}  # symbol -> last CVD value (for period_delta)

        self.logger = logger.bind(module="data.ohlcv_saver")

    async def start(self) -> None:
        """Start background flush task."""
        self._running = True
        self._flush_task = asyncio.create_task(self._periodic_flush())
        self.logger.info(
            f"OHLCV Saver started (batch_size={self._batch_size}, "
            f"flush_interval={self._flush_interval}s)"
        )

    async def stop(self) -> None:
        """Stop saver and flush remaining data."""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final flush all batches
        if self._batch:
            await self._flush_batch()
        if self._cvd_batch:
            await self._flush_cvd_batch()
        if self._liquidation_batch:
            await self._flush_liquidation_batch()

        self.logger.info("OHLCV Saver stopped")

    async def on_kline(self, kline: dict[str, Any]) -> None:
        """Callback for WebSocket kline events.

        Args:
            kline: Kline data from BinanceWebSocketConnector
                {
                    'timestamp': datetime,
                    'symbol': 'BTCUSDT',
                    'timeframe': '1m',
                    'open': Decimal,
                    'high': Decimal,
                    'low': Decimal,
                    'close': Decimal,
                    'volume': Decimal,
                    'quote_volume': Decimal,
                    'trade_count': int,
                    'is_closed': bool
                }
        """
        # Only save completed candles
        if not kline.get("is_closed", False):
            return

        # Add to batch
        candle = {
            "timestamp": kline["timestamp"],
            "symbol": self._normalize_symbol(kline["symbol"]),
            "timeframe": kline["timeframe"],
            "open": kline["open"],
            "high": kline["high"],
            "low": kline["low"],
            "close": kline["close"],
            "volume": kline["volume"],
            "quote_volume": kline.get("quote_volume"),
            "trade_count": kline.get("trade_count"),
        }

        self._batch.append(candle)

        # Flush if batch is full
        if len(self._batch) >= self._batch_size:
            await self._flush_batch()

    def _normalize_symbol(self, binance_symbol: str) -> str:
        """Convert Binance 'BTCUSDT' to internal 'BTC/USDT' format."""
        if binance_symbol.endswith("USDT") and len(binance_symbol) > 4:
            base = binance_symbol[:-4]
            return f"{base}/USDT"
        return binance_symbol

    async def _periodic_flush(self) -> None:
        """Flush all batches periodically even if not full."""
        while self._running:
            await asyncio.sleep(self._flush_interval)
            if self._batch:
                await self._flush_batch()
            if self._cvd_batch:
                await self._flush_cvd_batch()
            if self._liquidation_batch:
                await self._flush_liquidation_batch()

    async def _flush_batch(self) -> None:
        """Write batch to database using binary COPY protocol (institutional-grade).

        INSTITUTIONAL FIX: Uses PostgreSQL COPY FROM STDIN (10-50x faster than INSERT).
        For conflicts, falls back to upsert logic.
        """
        if not self._batch:
            return

        batch = self._batch.copy()
        self._batch.clear()

        try:
            # ═══ INSTITUTIONAL: Binary COPY Protocol ═══
            # This is the fastest way to bulk insert into PostgreSQL
            # Used by all professional trading systems
            try:
                await self._flush_batch_copy(batch)
                self.logger.debug(
                    f"[OHLCV-SAVER] Flushed {len(batch)} candles via COPY protocol"
                )
                return
            except Exception as copy_error:
                # COPY failed (likely due to conflicts), fall back to upsert
                self.logger.warning(
                    f"[OHLCV-SAVER] COPY failed ({type(copy_error).__name__}), "
                    f"falling back to upsert"
                )

            # ═══ Fallback: Upsert for handling conflicts ═══
            async with self._session_factory() as session:
                stmt = insert(OHLCV).values(batch)
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_ohlcv_ts_sym_tf",
                    set_={
                        "open": stmt.excluded.open,
                        "high": stmt.excluded.high,
                        "low": stmt.excluded.low,
                        "close": stmt.excluded.close,
                        "volume": stmt.excluded.volume,
                        "quote_volume": stmt.excluded.quote_volume,
                        "trade_count": stmt.excluded.trade_count,
                    }
                )
                await session.execute(stmt)
                await session.commit()

            self.logger.debug(
                f"[OHLCV-SAVER] Flushed {len(batch)} candles via upsert"
            )

        except Exception as e:
            self.logger.error(f"[OHLCV-SAVER] Failed to flush batch: {e}")
            # Re-add to batch for retry
            self._batch.extend(batch)

    async def _flush_batch_copy(self, batch: list[dict]) -> None:
        """Flush batch using PostgreSQL COPY protocol (binary format).

        This is 10-50x faster than INSERT statements.
        Institutional trading systems use this for high-frequency data.
        """
        if not batch:
            return

        # Get raw asyncpg connection from SQLAlchemy engine
        from atomicx.data.storage.database import get_engine

        engine = get_engine()
        async with engine.begin() as conn:
            # Get the underlying asyncpg connection
            raw_conn = await conn.get_raw_connection()
            asyncpg_conn = raw_conn.driver_connection

            # Convert batch dicts to tuples for COPY
            # Column order must match the table definition
            records = [
                (
                    row["timestamp"],
                    row["symbol"],
                    row["timeframe"],
                    row["open"],
                    row["high"],
                    row["low"],
                    row["close"],
                    row["volume"],
                    row.get("quote_volume"),
                    row.get("trade_count"),
                )
                for row in batch
            ]

            # Use asyncpg's copy_records_to_table (binary COPY protocol)
            await asyncpg_conn.copy_records_to_table(
                "ohlcv",
                records=records,
                columns=[
                    "timestamp",
                    "symbol",
                    "timeframe",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "quote_volume",
                    "trade_count",
                ],
            )

    async def on_cvd(self, symbol: str, cvd_value: float) -> None:
        """Callback for CVD updates from WebSocket.

        Args:
            symbol: Internal format symbol (BTC/USDT)
            cvd_value: Current cumulative volume delta
        """
        # Calculate period delta
        last_cvd = self._last_cvd.get(symbol, 0.0)
        period_delta = cvd_value - last_cvd
        self._last_cvd[symbol] = cvd_value

        # Add to batch
        cvd_record = {
            "timestamp": datetime.now(tz=timezone.utc),
            "symbol": symbol,
            "cvd_value": cvd_value,
            "period_delta": period_delta,
            "reset_flag": False,
        }

        self._cvd_batch.append(cvd_record)

        # Flush if batch is full
        if len(self._cvd_batch) >= self._batch_size:
            await self._flush_cvd_batch()

    async def on_liquidation(self, liquidation: dict[str, Any]) -> None:
        """Callback for liquidation events from WebSocket.

        Args:
            liquidation: Liquidation data from BinanceWebSocketConnector
                {
                    'timestamp': datetime,
                    'symbol': 'BTC/USDT',  # Internal format
                    'side': 'SELL',  # SELL = long liq, BUY = short liq
                    'price': Decimal,
                    'quantity': Decimal,
                    'order_type': str
                }
        """
        qty = float(liquidation["quantity"])
        price = float(liquidation["price"])
        notional = qty * price

        liq_record = {
            "timestamp": liquidation["timestamp"],
            "symbol": liquidation["symbol"],
            "side": liquidation["side"],
            "price": liquidation["price"],
            "quantity": liquidation["quantity"],
            "notional_usd": notional,
            "order_type": liquidation.get("order_type"),
        }

        self._liquidation_batch.append(liq_record)

        # Flush if batch is full
        if len(self._liquidation_batch) >= self._batch_size:
            await self._flush_liquidation_batch()

    async def _flush_cvd_batch(self) -> None:
        """Write CVD batch using binary COPY protocol (institutional-grade)."""
        if not self._cvd_batch:
            return

        batch = self._cvd_batch.copy()
        self._cvd_batch.clear()

        try:
            # ═══ INSTITUTIONAL: Binary COPY Protocol ═══
            await self._flush_cvd_batch_copy(batch)
            self.logger.debug(
                f"[CVD-SAVER] Flushed {len(batch)} CVD records via COPY protocol"
            )

        except Exception as e:
            self.logger.error(f"[CVD-SAVER] Failed to flush batch: {e}")
            # Re-add to batch for retry
            self._cvd_batch.extend(batch)

    async def _flush_cvd_batch_copy(self, batch: list[dict]) -> None:
        """Flush CVD batch using PostgreSQL COPY protocol (binary format)."""
        if not batch:
            return

        from atomicx.data.storage.database import get_engine

        engine = get_engine()
        async with engine.begin() as conn:
            raw_conn = await conn.get_raw_connection()
            asyncpg_conn = raw_conn.driver_connection

            records = [
                (
                    row["timestamp"],
                    row["symbol"],
                    row["cvd_value"],
                    row["period_delta"],
                    row["reset_flag"],
                )
                for row in batch
            ]

            await asyncpg_conn.copy_records_to_table(
                "cumulative_delta",
                records=records,
                columns=["timestamp", "symbol", "cvd_value", "period_delta", "reset_flag"],
            )

    async def _flush_liquidation_batch(self) -> None:
        """Write liquidation batch using binary COPY protocol (institutional-grade)."""
        if not self._liquidation_batch:
            return

        batch = self._liquidation_batch.copy()
        self._liquidation_batch.clear()

        try:
            # ═══ INSTITUTIONAL: Binary COPY Protocol ═══
            await self._flush_liquidation_batch_copy(batch)
            self.logger.debug(
                f"[LIQUIDATION-SAVER] Flushed {len(batch)} liquidations via COPY protocol"
            )

        except Exception as e:
            self.logger.error(f"[LIQUIDATION-SAVER] Failed to flush batch: {e}")
            # Re-add to batch for retry
            self._liquidation_batch.extend(batch)

    async def _flush_liquidation_batch_copy(self, batch: list[dict]) -> None:
        """Flush liquidation batch using PostgreSQL COPY protocol (binary format)."""
        if not batch:
            return

        from atomicx.data.storage.database import get_engine

        engine = get_engine()
        async with engine.begin() as conn:
            raw_conn = await conn.get_raw_connection()
            asyncpg_conn = raw_conn.driver_connection

            records = [
                (
                    row["timestamp"],
                    row["symbol"],
                    row["side"],
                    row["price"],
                    row["quantity"],
                    row["notional_usd"],
                    row.get("order_type"),
                )
                for row in batch
            ]

            await asyncpg_conn.copy_records_to_table(
                "liquidations",
                records=records,
                columns=[
                    "timestamp",
                    "symbol",
                    "side",
                    "price",
                    "quantity",
                    "notional_usd",
                    "order_type",
                ],
            )
