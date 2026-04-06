"""Variable computation engine — orchestrates indicator computation from live data.

Consumes OHLCV data from Kafka/DB and computes all registered variables,
storing results back to TimescaleDB and making them available to the
agent hierarchy via the registry API.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pandas as pd
import polars as pl
from loguru import logger
from sqlalchemy import select

from atomicx.data.storage.database import get_session_factory
from atomicx.data.storage.models import OHLCV
from atomicx.variables.catalog import get_default_variables
from atomicx.variables.indicators import compute_all_indicators
from atomicx.variables.registry import VariableRegistry
from atomicx.variables.types import VariableDefinition, VariableStatus, VariableValue, VariableTimeframe
from atomicx.common.cache import get_sensory_cache


class VariableComputeEngine:
    """Orchestrates variable computation across all symbols and timeframes.

    Runs on a configurable schedule (e.g., every minute for 1m data, every
    hour for 1h data). For each computation cycle:
    1. Fetch latest OHLCV data from TimescaleDB
    2. Compute all technical indicators
    3. Store computed values in the registry
    4. Publish to Kafka for downstream agents
    """

    def __init__(
        self,
        registry: VariableRegistry | None = None,
        symbols: list[str] | None = None,
    ) -> None:
        from atomicx.config import get_settings
        settings = get_settings()
        self.registry = registry or VariableRegistry()
        self._symbols = symbols or settings.default_symbols
        self._session_factory = get_session_factory()
        self._cache = get_sensory_cache()
        self._running = False
        self.logger = logger.bind(module="variables.engine")


    async def initialize(self) -> None:
        """Register all default variables in the registry."""
        variables = get_default_variables()
        await self.registry.register_batch(variables)
        logger.info(f"Initialized {len(variables)} variables in registry")

    async def start(self, compute_interval: float = 60.0) -> None:
        """Start the computation loop."""
        self._running = True
        await self.initialize()

        logger.info(
            "Variable compute engine started",
            symbols=self._symbols,
            interval=compute_interval,
        )

        while self._running:
            try:
                await self._compute_cycle()
            except Exception as e:
                logger.error(f"Computation cycle error: {e}")

            if self._running:
                await asyncio.sleep(compute_interval)

    async def _compute_cycle(self) -> None:
        """Run one computation cycle across all symbols."""
        for symbol in self._symbols:
            try:
                await self._compute_for_symbol(symbol)
            except Exception as e:
                logger.error(f"Error computing variables for {symbol}: {e}")

    async def _compute_for_symbol(self, symbol: str) -> None:
        """Compute all variables for a single symbol.

        INSTITUTIONAL FIX: Uses Polars for 5-10x faster indicator computation.
        EVENT LOOP FIX: Runs CPU-intensive operations in thread pool to prevent blocking.
        """
        # Fetch recent OHLCV data (enough for longest lookback)
        df_pandas = await self._fetch_ohlcv(symbol, timeframe="1m", limit=250)
        if df_pandas.empty:
            return

        # INSTITUTIONAL + EVENT LOOP FIX: Run CPU-intensive Polars computation in thread pool
        def _compute_indicators_sync(df_pandas):
            df_polars = pl.from_pandas(df_pandas)
            computed_polars = compute_all_indicators(df_polars)
            return computed_polars.to_pandas()

        computed = await asyncio.to_thread(_compute_indicators_sync, df_pandas)

        now = datetime.now(tz=timezone.utc)

        # Map indicator columns to variable IDs and store
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
            "close": "PRICE",
        }

        values = []
        latest = computed.iloc[-1] if len(computed) > 0 else None
        if latest is None:
            return

        for col, var_id in indicator_map.items():
            if col in computed.columns and pd.notna(latest.get(col)):
                values.append(
                    VariableValue(
                        variable_id=var_id,
                        symbol=symbol,
                        timestamp=now,
                        value=float(latest[col]),
                        timeframe=VariableTimeframe.M1,
                    )
                )

        # FIX: Bridge sensory cache data into registry (microstructure metrics)
        # This ensures background loop stores OB_IMBALANCE, CVD, FUNDING_RATE, etc.
        sensory = self._cache.get_all(symbol)
        sensory_vars = []
        for key, value in sensory.items():
            if isinstance(value, (int, float)):
                sensory_vars.append(key)
                values.append(
                    VariableValue(
                        variable_id=key,
                        symbol=symbol,
                        timestamp=now,
                        value=float(value),
                        timeframe=VariableTimeframe.M1,
                    )
                )

        if sensory_vars:
            self.logger.info(f"[SENSORY-BRIDGE] Adding {len(sensory_vars)} sensory variables for {symbol}: {sensory_vars}")

            # Special logging for CVD (anti-spoofing metric)
            cvd = sensory.get("CVD")
            if cvd is not None:
                self.logger.debug(f"[CVD] {symbol} cumulative delta: {cvd:.2f}")

        if values:
            await self.registry.store_values_batch(values)
            sensory_count = len(sensory)
            logger.debug(f"Computed {len(values)} variables for {symbol} ({sensory_count} from sensory cache)")

    async def _fetch_ohlcv(
        self, symbol: str, timeframe: str = "1m", limit: int = 250
    ) -> pd.DataFrame:
        """Fetch recent OHLCV data from TimescaleDB (preferred) or CCXT (fallback).

        Priority:
        1. TimescaleDB historical data (fast, reliable)
        2. CCXT live API (slower, rate-limited)
        3. Empty DataFrame (degraded mode)
        """
        # TRY 1: Database
        try:
            async with self._session_factory() as session:
                result = await session.execute(
                    select(OHLCV)
                    .where(OHLCV.symbol == symbol)
                    .where(OHLCV.timeframe == timeframe)
                    .order_by(OHLCV.timestamp.desc())
                    .limit(limit)
                )
                rows = result.scalars().all()

            if rows:
                data = [
                    {
                        "timestamp": r.timestamp,
                        "open": float(r.open),
                        "high": float(r.high),
                        "low": float(r.low),
                        "close": float(r.close),
                        "volume": float(r.volume),
                    }
                    for r in reversed(rows)
                ]
                self.logger.debug(
                    f"[DATA SOURCE] {symbol} {timeframe}: Loaded {len(rows)} candles from TimescaleDB"
                )
                return pd.DataFrame(data)
            else:
                self.logger.info(
                    f"[DATA SOURCE] {symbol} {timeframe}: Database empty, falling back to CCXT"
                )

        except Exception as e:
            self.logger.warning(
                f"[DATA SOURCE] {symbol} {timeframe}: Database error — {e}. "
                f"Falling back to CCXT"
            )

        # TRY 2: CCXT Live API
        try:
            import ccxt
            exchange = ccxt.binance({
                "enableRateLimit": True,
                "options": {"defaultType": "spot"}
            })
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1m', limit=limit)

            if hasattr(exchange, 'close'):
                exchange.close()

            df = pd.DataFrame(
                ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

            self.logger.info(
                f"[DATA SOURCE] {symbol} {timeframe}: Fetched {len(df)} candles from CCXT live API"
            )
            return df

        except Exception as ex:
            self.logger.error(
                f"[DATA SOURCE] {symbol} {timeframe}: CCXT fallback failed — {ex}. "
                f"Operating in degraded mode."
            )
            return pd.DataFrame()

    async def compute_snapshot(self, symbol: str) -> dict[str, float]:
        """Compute and return current values for all variables (on-demand).

        Used by agents that need real-time variable snapshots.

        INSTITUTIONAL FIX: Uses Polars for high-performance computation.
        EVENT LOOP FIX: Runs CPU-intensive operations in thread pool to prevent blocking.
        """
        df_pandas = await self._fetch_ohlcv(symbol, limit=250)
        if df_pandas.empty:
            return {}

        # INSTITUTIONAL + EVENT LOOP FIX: Run CPU-intensive Polars computation in thread pool
        # This prevents blocking the async event loop during indicator calculation
        def _compute_indicators_sync(df_pandas):
            df_polars = pl.from_pandas(df_pandas)
            computed_polars = compute_all_indicators(df_polars)
            return computed_polars.to_pandas()

        computed = await asyncio.to_thread(_compute_indicators_sync, df_pandas)

        latest = computed.iloc[-1]

        snapshot = {}
        if "close" in computed.columns:
            snapshot["PRICE"] = float(latest["close"])
            
        for col in computed.columns:
            if col not in ("timestamp", "open", "high", "low", "close", "volume"):
                val = latest.get(col)
                if pd.notna(val):
                    snapshot[col.upper()] = float(val)

        # Merge WebSocket sensory data (microstructure)
        sensory = self._cache.get_all(symbol)
        for k, v in sensory.items():
            snapshot[k] = v

        if "close" in computed.columns:
            snapshot["RECENT_PRICES"] = computed["close"].tail(20).tolist()
        else:
            snapshot["RECENT_PRICES"] = []

        # Note: FUNDING_RATE and OB_IMBALANCE require live WebSocket data.
        # We leave them absent (not injected) so downstream modules can
        # distinguish 'no data' from 'neutral' and act accordingly.
        # The strategic layer already handles missing values gracefully.

        return snapshot

    async def stop(self) -> None:
        """Stop the computation engine."""
        self._running = False
        logger.info("Variable compute engine stopped")
