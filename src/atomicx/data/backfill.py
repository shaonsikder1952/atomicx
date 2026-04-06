"""Historical data backfill service.

Downloads months/years of OHLCV data from Binance REST API
and on-chain data from CoinGecko, then computes all variables
on the historical data to populate the computed_variables table.

Usage:
    python scripts/backfill_historical.py --symbols BTC/USDT ETH/USDT SOL/USDT --days 365
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import ccxt.async_support as ccxt
import pandas as pd
from loguru import logger

from atomicx.data.storage.database import get_session_factory
from atomicx.data.storage.models import OHLCV
from atomicx.variables.indicators import compute_all_indicators


class HistoricalBackfillService:
    """Downloads historical OHLCV data from Binance.

    Binance free API returns up to 1000 candles per request.
    This service pages through the entire history in batches.
    """

    def __init__(self, symbols: list[str] | None = None) -> None:
        from atomicx.config import get_settings
        settings = get_settings()
        self._symbols = symbols or settings.default_symbols
        self._session_factory = get_session_factory()
        self._exchange: ccxt.binance | None = None

    async def _get_exchange(self) -> ccxt.binance:
        """Get or create the exchange connection."""
        if self._exchange is None:
            self._exchange = ccxt.binance({
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            })
        return self._exchange

    async def backfill_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        days: int = 180,
        batch_size: int = 1000,
    ) -> int:
        """Download historical OHLCV data for a symbol.

        Args:
            symbol: Trading pair, e.g. 'BTC/USDT'
            timeframe: Candle timeframe, e.g. '1m', '1h', '1d'
            days: Number of days of history to download
            batch_size: Candles per API request (max 1000)

        Returns:
            Total number of candles downloaded
        """
        exchange = await self._get_exchange()
        end_time = datetime.now(tz=timezone.utc)
        start_time = end_time - timedelta(days=days)
        since = int(start_time.timestamp() * 1000)

        total_downloaded = 0
        logger.info(
            f"Backfilling {symbol} {timeframe} — {days} days "
            f"({start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')})"
        )

        while since < int(end_time.timestamp() * 1000):
            try:
                candles = await exchange.fetch_ohlcv(
                    symbol, timeframe, since=since, limit=batch_size
                )
                if not candles:
                    break

                await self._store_candles(symbol, timeframe, candles)
                total_downloaded += len(candles)

                # Move to next batch
                since = candles[-1][0] + 1  # 1ms after last candle

                if total_downloaded % 10000 == 0:
                    logger.info(f"  {symbol}: {total_downloaded:,} candles downloaded...")

            except Exception as e:
                logger.error(f"Backfill error at {since}: {e}")
                await asyncio.sleep(2)  # Rate limit protection

        logger.info(f"✓ {symbol} {timeframe}: {total_downloaded:,} candles downloaded")
        return total_downloaded

    async def _store_candles(
        self, symbol: str, timeframe: str, candles: list
    ) -> None:
        """Batch insert candles to TimescaleDB with upsert."""
        from sqlalchemy.dialects.postgresql import insert as pg_insert

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

    async def backfill_all(
        self,
        timeframes: list[str] | None = None,
        days: int = 180,
    ) -> dict[str, int]:
        """Backfill all symbols across all timeframes.

        Returns:
            Dict of symbol:timeframe → candle count
        """
        tfs = timeframes or ["1m", "5m", "15m", "1h", "4h", "1d"]
        results = {}

        for symbol in self._symbols:
            for tf in tfs:
                # Adjust days for higher timeframes (no need for 180 days of 1m)
                tf_days = days
                if tf == "1m":
                    tf_days = min(days, 30)  # 30 days of 1m = ~43k candles
                elif tf == "5m":
                    tf_days = min(days, 90)
                elif tf in ("15m", "1h"):
                    tf_days = min(days, 180)

                count = await self.backfill_ohlcv(symbol, tf, tf_days)
                results[f"{symbol}:{tf}"] = count

        return results

    async def close(self) -> None:
        """Close the exchange connection."""
        if self._exchange:
            await self._exchange.close()


class VariableBackfillService:
    """Computes all 46 variables on historical OHLCV data.

    This populates the computed_variables table so the causal discovery
    engine has enough data to work with.
    """

    def __init__(self) -> None:
        self._session_factory = get_session_factory()

    async def backfill_variables(
        self,
        symbol: str,
        timeframe: str = "1h",
        batch_size: int = 500,
    ) -> int:
        """Compute and store all variables for historical data.

        Processes in batches to avoid memory issues.
        """
        from sqlalchemy import select, func
        from atomicx.variables.models import ComputedVariable
        from atomicx.variables.types import VariableValue, VariableTimeframe

        async with self._session_factory() as session:
            # Get total count
            count_result = await session.execute(
                select(func.count(OHLCV.id)).where(
                    OHLCV.symbol == symbol, OHLCV.timeframe == timeframe
                )
            )
            total = count_result.scalar() or 0

        if total == 0:
            logger.warning(f"No OHLCV data for {symbol} {timeframe}")
            return 0

        logger.info(f"Computing variables for {symbol} {timeframe} — {total:,} candles")

        # Fetch all data and compute
        df = await self._fetch_all_ohlcv(symbol, timeframe)
        if df.empty:
            return 0

        # Compute all indicators
        computed = compute_all_indicators(df)

        # Map columns to variable IDs
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

        total_stored = 0

        # Process in chunks to avoid memory overload
        for start_idx in range(0, len(computed), batch_size):
            chunk = computed.iloc[start_idx : start_idx + batch_size]
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
                    await session.execute(
                        ComputedVariable.__table__.insert(), values
                    )
                    await session.commit()
                total_stored += len(values)

            if start_idx % 5000 == 0 and start_idx > 0:
                logger.info(f"  {symbol}: {start_idx:,}/{len(computed):,} rows processed...")

        logger.info(f"✓ {symbol} {timeframe}: {total_stored:,} variable values stored")
        return total_stored

    async def _fetch_all_ohlcv(
        self, symbol: str, timeframe: str
    ) -> pd.DataFrame:
        """Fetch all OHLCV data for a symbol/timeframe."""
        from sqlalchemy import select

        async with self._session_factory() as session:
            result = await session.execute(
                select(OHLCV)
                .where(OHLCV.symbol == symbol, OHLCV.timeframe == timeframe)
                .order_by(OHLCV.timestamp)
            )
            rows = result.scalars().all()

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame([
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
