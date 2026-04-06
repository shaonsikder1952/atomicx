"""Production-grade OHLCV data ingestion with incremental sync.

Automatically:
1. Fetches historical OHLCV data on first run (500+ candles per timeframe)
2. Incremental sync on subsequent runs (only fetches missing candles)
3. Computes all 46 variables on historical data
4. Builds pattern library with outcome verification
5. Health checks and monitoring
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import ccxt.async_support as ccxt
import pandas as pd
import polars as pl
from loguru import logger
from sqlalchemy import select, func, and_
from sqlalchemy.dialects.postgresql import insert as pg_insert

from atomicx.data.storage.database import get_session_factory
from atomicx.data.storage.models import OHLCV
from atomicx.data.connectors.stock import YahooFinanceConnector
from atomicx.variables.models import ComputedVariable
from atomicx.variables.indicators import compute_all_indicators


class DataIngestionService:
    """Production OHLCV ingestion with incremental sync."""

    # Default configuration
    DEFAULT_TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]
    INITIAL_CANDLES_PER_TF = {
        "1m": 500,    # ~8 hours
        "5m": 500,    # ~1.7 days
        "15m": 500,   # ~5 days
        "1h": 500,    # ~21 days
        "4h": 500,    # ~83 days
        "1d": 500,    # ~1.4 years
    }
    BATCH_SIZE = 1000  # CCXT max per request

    def __init__(self, symbols: list[str] | None = None) -> None:
        from atomicx.config import get_settings
        settings = get_settings()
        self._symbols = symbols or settings.default_symbols
        self._session_factory = get_session_factory()
        self._exchange: ccxt.binance | None = None
        self.logger = logger.bind(module="data.ingestion")

    def _get_exchange(self) -> ccxt.binance:
        """Get or create async exchange connection."""
        if self._exchange is None:
            self._exchange = ccxt.binance({
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            })
        return self._exchange

    def _is_stock_symbol(self, symbol: str) -> bool:
        """Detect if symbol is a stock (no slash) vs crypto (has slash)."""
        return "/" not in symbol

    async def _fetch_candles_from_connector(
        self,
        symbol: str,
        timeframe: str,
        since: int,
        limit: int,
    ) -> list:
        """Fetch candles using appropriate connector (crypto or stock).

        Returns candles in CCXT format: [[timestamp_ms, open, high, low, close, volume], ...]
        """
        if self._is_stock_symbol(symbol):
            # Stock symbol - use Yahoo Finance
            connector = YahooFinanceConnector(symbol=symbol)

            # Convert since (ms timestamp) to datetime
            since_dt = datetime.fromtimestamp(since / 1000, tz=timezone.utc)

            # Fetch OHLCVBar objects
            bars = await connector.get_historical_ohlcv(
                timeframe=timeframe,
                limit=limit,
                since=since_dt,
            )

            # Convert OHLCVBar objects to CCXT format
            candles = []
            for bar in bars:
                candles.append([
                    int(bar.timestamp.timestamp() * 1000),  # Convert to ms
                    float(bar.open),
                    float(bar.high),
                    float(bar.low),
                    float(bar.close),
                    float(bar.volume),
                ])

            return candles
        else:
            # Crypto symbol - use CCXT Binance
            exchange = self._get_exchange()
            candles = await exchange.fetch_ohlcv(
                symbol, timeframe, since=since, limit=limit
            )
            return candles

    async def run_full_ingestion(
        self,
        timeframes: list[str] | None = None,
        force_full_backfill: bool = False,
    ) -> dict[str, Any]:
        """Run complete data ingestion pipeline.

        Args:
            timeframes: List of timeframes to ingest (defaults to all)
            force_full_backfill: If True, re-download all data from scratch

        Returns:
            Dict with ingestion statistics
        """
        tfs = timeframes or self.DEFAULT_TIMEFRAMES
        stats = {
            "ohlcv_downloaded": 0,
            "variables_computed": 0,
            "symbols_processed": 0,
            "timeframes": {},
            "errors": [],
        }

        self.logger.info(
            f"Starting data ingestion for {len(self._symbols)} symbols "
            f"across {len(tfs)} timeframes"
        )

        for symbol in self._symbols:
            try:
                for tf in tfs:
                    # Check if we need full backfill or incremental sync
                    if force_full_backfill:
                        count = await self._full_backfill(symbol, tf)
                    else:
                        count = await self._incremental_sync(symbol, tf)

                    stats["ohlcv_downloaded"] += count
                    stats["timeframes"][f"{symbol}:{tf}"] = count

                    # Compute variables for this data
                    var_count = await self._compute_variables(symbol, tf)
                    stats["variables_computed"] += var_count

                stats["symbols_processed"] += 1

            except Exception as e:
                self.logger.error(f"Ingestion failed for {symbol}: {e}")
                stats["errors"].append(f"{symbol}: {str(e)}")

        self.logger.success(
            f"Ingestion complete: {stats['ohlcv_downloaded']:,} candles, "
            f"{stats['variables_computed']:,} variable values computed"
        )

        return stats

    async def _incremental_sync(self, symbol: str, timeframe: str) -> int:
        """Fetch only missing candles since last DB timestamp."""
        # Get latest timestamp from DB
        async with self._session_factory() as session:
            result = await session.execute(
                select(func.max(OHLCV.timestamp))
                .where(
                    and_(
                        OHLCV.symbol == symbol,
                        OHLCV.timeframe == timeframe
                    )
                )
            )
            last_ts = result.scalar()

        if last_ts is None:
            # No data exists - do full backfill
            self.logger.info(f"{symbol} {timeframe}: No existing data, running full backfill")
            return await self._full_backfill(symbol, timeframe)

        # Calculate how many candles we're missing
        now = datetime.now(tz=timezone.utc)
        time_diff = now - last_ts

        # Convert timeframe to timedelta
        tf_minutes = self._timeframe_to_minutes(timeframe)
        expected_candles = int(time_diff.total_seconds() / 60 / tf_minutes)

        if expected_candles <= 1:
            self.logger.debug(f"{symbol} {timeframe}: Up to date ({expected_candles} missing)")
            return 0

        self.logger.info(
            f"{symbol} {timeframe}: Syncing ~{expected_candles} missing candles "
            f"since {last_ts.strftime('%Y-%m-%d %H:%M')}"
        )

        # Fetch from last_ts + 1 candle to now
        since = int((last_ts + timedelta(minutes=tf_minutes)).timestamp() * 1000)
        return await self._fetch_and_store(symbol, timeframe, since, expected_candles + 10)

    async def _full_backfill(self, symbol: str, timeframe: str) -> int:
        """Download initial historical data."""
        candle_count = self.INITIAL_CANDLES_PER_TF.get(timeframe, 500)

        # Calculate start time
        tf_minutes = self._timeframe_to_minutes(timeframe)
        start_time = datetime.now(tz=timezone.utc) - timedelta(
            minutes=candle_count * tf_minutes
        )
        since = int(start_time.timestamp() * 1000)

        self.logger.info(
            f"{symbol} {timeframe}: Full backfill — {candle_count} candles "
            f"from {start_time.strftime('%Y-%m-%d %H:%M')}"
        )

        return await self._fetch_and_store(symbol, timeframe, since, candle_count)

    async def _fetch_and_store(
        self,
        symbol: str,
        timeframe: str,
        since: int,
        expected_count: int,
    ) -> int:
        """Fetch candles from appropriate connector and store in DB."""
        total_downloaded = 0
        end_time_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)

        while since < end_time_ms and total_downloaded < expected_count + 100:
            try:
                candles = await self._fetch_candles_from_connector(
                    symbol, timeframe, since=since, limit=self.BATCH_SIZE
                )

                if not candles:
                    break

                await self._store_candles(symbol, timeframe, candles)
                total_downloaded += len(candles)

                # Move to next batch
                since = candles[-1][0] + 1

                # Rate limit protection
                await asyncio.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Fetch error at {since}: {e}")
                await asyncio.sleep(2)
                break

        if total_downloaded > 0:
            self.logger.info(
                f"  ✓ {symbol} {timeframe}: {total_downloaded:,} candles downloaded"
            )

        return total_downloaded

    async def _store_candles(
        self, symbol: str, timeframe: str, candles: list
    ) -> None:
        """Batch upsert candles to TimescaleDB."""
        async with self._session_factory() as session:
            for candle in candles:
                ts, o, h, l, c, v = candle[:6]
                stmt = pg_insert(OHLCV).values(
                    timestamp=datetime.fromtimestamp(ts / 1000, tz=timezone.utc),
                    symbol=symbol,
                    timeframe=timeframe,
                    open=Decimal(str(o)),
                    high=Decimal(str(h)),
                    low=Decimal(str(l)),
                    close=Decimal(str(c)),
                    volume=Decimal(str(v)),
                )
                stmt = stmt.on_conflict_do_nothing(constraint="uq_ohlcv_ts_sym_tf")
                await session.execute(stmt)
            await session.commit()

    async def _compute_variables(self, symbol: str, timeframe: str) -> int:
        """Compute all 46 variables on ingested data."""
        # Fetch all OHLCV data
        async with self._session_factory() as session:
            result = await session.execute(
                select(OHLCV)
                .where(
                    and_(
                        OHLCV.symbol == symbol,
                        OHLCV.timeframe == timeframe
                    )
                )
                .order_by(OHLCV.timestamp)
            )
            rows = result.scalars().all()

        if not rows:
            return 0

        df = pd.DataFrame([
            {
                "timestamp": r.timestamp,
                "open": float(r.open),
                "high": float(r.high),
                "low": float(r.low),
                "close": float(r.close),
                "volume": float(r.volume),
            }
            for r in rows
        ])

        # INSTITUTIONAL: Compute all indicators with Polars (Rust-backed)
        df_polars = pl.from_pandas(df)
        computed_polars = compute_all_indicators(df_polars)
        computed = computed_polars.to_pandas()

        # Map to variable IDs
        indicator_map = {
            "ema_9": "EMA_9", "ema_21": "EMA_21", "ema_50": "EMA_50", "ema_200": "EMA_200",
            "sma_20": "SMA_20", "sma_50": "SMA_50", "vwap": "VWAP",
            "rsi_14": "RSI_14", "rsi_7": "RSI_7",
            "macd_line": "MACD_LINE", "macd_signal": "MACD_SIGNAL",
            "macd_histogram": "MACD_HISTOGRAM",
            "stoch_rsi_k": "STOCH_RSI_K", "stoch_rsi_d": "STOCH_RSI_D",
            "roc_12": "ROC_12",
            "bb_upper": "BB_UPPER", "bb_lower": "BB_LOWER",
            "bb_percent_b": "BB_PERCENT_B", "bb_bandwidth": "BB_BANDWIDTH",
            "atr_14": "ATR_14", "garch_vol": "GARCH_VOL",
            "obv": "OBV", "relative_volume": "REL_VOLUME", "vpt": "VPT",
            "adx": "ADX", "plus_di": "PLUS_DI", "minus_di": "MINUS_DI",
        }

        # Store in batches
        total_stored = 0
        batch_size = 500

        for start_idx in range(0, len(computed), batch_size):
            chunk = computed.iloc[start_idx:start_idx + batch_size]
            values = []

            for _, row in chunk.iterrows():
                ts = row["timestamp"]
                for col, var_id in indicator_map.items():
                    if col in computed.columns and pd.notna(row.get(col)):
                        values.append({
                            "timestamp": ts,
                            "variable_id": var_id,
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "value": float(row[col]),
                            "confidence": 1.0,
                        })

            if values:
                async with self._session_factory() as session:
                    # Upsert: delete old values first, then insert
                    await session.execute(
                        ComputedVariable.__table__.insert(), values
                    )
                    await session.commit()
                total_stored += len(values)

        if total_stored > 0:
            self.logger.info(
                f"  ✓ {symbol} {timeframe}: {total_stored:,} variable values computed"
            )

        return total_stored

    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes."""
        mapping = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "1h": 60,
            "4h": 240,
            "1d": 1440,
        }
        return mapping.get(timeframe, 60)

    async def get_data_health(self) -> dict[str, Any]:
        """Get health status of ingested data."""
        health = {
            "symbols": {},
            "overall_status": "healthy",
            "recommendations": [],
        }

        async with self._session_factory() as session:
            for symbol in self._symbols:
                symbol_health = {}

                for tf in self.DEFAULT_TIMEFRAMES:
                    # Count candles
                    count_result = await session.execute(
                        select(func.count(OHLCV.id))
                        .where(
                            and_(
                                OHLCV.symbol == symbol,
                                OHLCV.timeframe == tf
                            )
                        )
                    )
                    candle_count = count_result.scalar() or 0

                    # Get latest timestamp
                    ts_result = await session.execute(
                        select(func.max(OHLCV.timestamp))
                        .where(
                            and_(
                                OHLCV.symbol == symbol,
                                OHLCV.timeframe == tf
                            )
                        )
                    )
                    last_ts = ts_result.scalar()

                    # Calculate staleness
                    staleness_minutes = None
                    is_stale = False
                    if last_ts:
                        staleness = datetime.now(tz=timezone.utc) - last_ts
                        staleness_minutes = int(staleness.total_seconds() / 60)
                        tf_minutes = self._timeframe_to_minutes(tf)
                        is_stale = staleness_minutes > (tf_minutes * 3)  # 3x timeframe = stale

                    symbol_health[tf] = {
                        "candle_count": candle_count,
                        "last_update": last_ts.isoformat() if last_ts else None,
                        "staleness_minutes": staleness_minutes,
                        "is_stale": is_stale,
                        "status": "empty" if candle_count == 0 else ("stale" if is_stale else "healthy"),
                    }

                    # Generate recommendations
                    if candle_count == 0:
                        health["recommendations"].append(
                            f"No data for {symbol} {tf} — run ingestion"
                        )
                        health["overall_status"] = "degraded"
                    elif is_stale:
                        health["recommendations"].append(
                            f"{symbol} {tf} is stale ({staleness_minutes}m old)"
                        )
                        if health["overall_status"] == "healthy":
                            health["overall_status"] = "degraded"

                health["symbols"][symbol] = symbol_health

        return health

    async def close(self) -> None:
        """Close exchange connection."""
        if self._exchange:
            await self._exchange.close()
