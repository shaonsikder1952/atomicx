"""Yahoo Finance connector for stocks and commodities.

Supports:
- Stocks: AAPL, TSLA, GOOGL, etc.
- Commodity Futures: CL=F (oil), GC=F (gold), etc.

Uses yfinance library (free, no API key required).
Data is 15-minute delayed for free tier, which is fine for our use case.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import AsyncIterator

import yfinance as yf
from loguru import logger

from atomicx.data.connectors.base import (
    DataConnector,
    OHLCVBar,
    RealtimeTick,
    SymbolNotFoundError,
    RateLimitError,
)


class YahooFinanceConnector(DataConnector):
    """Yahoo Finance data connector for stocks and commodities.

    Polling-based (no WebSocket), fetches data every 60 seconds.
    """

    # Timeframe mapping: internal → yfinance
    TIMEFRAME_MAP = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "1h": "1h",
        "4h": "1h",  # No direct 4h support, will resample
        "1d": "1d",
    }

    def __init__(self, symbol: str):
        super().__init__(symbol)
        self._ticker = yf.Ticker(symbol)
        self._polling_task: asyncio.Task | None = None
        self._last_price: Decimal | None = None

    async def get_historical_ohlcv(
        self,
        timeframe: str,
        limit: int = 500,
        since: datetime | None = None
    ) -> list[OHLCVBar]:
        """Fetch historical OHLCV from Yahoo Finance.

        Args:
            timeframe: 1m, 5m, 15m, 1h, 4h, 1d
            limit: Number of candles (max 730 for intraday)
            since: Start timestamp

        Returns:
            List of OHLCV bars
        """
        try:
            # Map timeframe
            yf_interval = self.TIMEFRAME_MAP.get(timeframe)
            if not yf_interval:
                raise ValueError(f"Unsupported timeframe: {timeframe}")

            # Calculate period
            if since:
                # Use specific start date
                start_date = since.strftime("%Y-%m-%d")
                end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                period = None
            else:
                # Use period (max days for intraday data)
                if timeframe in ("1m", "5m", "15m"):
                    period = "7d"  # Yahoo limit for 1m/5m/15m
                elif timeframe in ("1h", "4h"):
                    period = "60d"  # Yahoo limit for 1h
                else:
                    period = "730d"  # 2 years for daily
                start_date = None
                end_date = None

            # Fetch data
            self.logger.debug(
                f"Fetching {timeframe} data for {self.symbol}, "
                f"period={period}, start={start_date}, limit={limit}"
            )

            # Run synchronous yfinance call in thread pool
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None,
                lambda: self._ticker.history(
                    period=period,
                    interval=yf_interval,
                    start=start_date,
                    end=end_date,
                    prepost=False,  # Regular trading hours only
                    actions=False,  # Skip dividends/splits
                )
            )

            if df.empty:
                raise SymbolNotFoundError(f"No data found for {self.symbol}")

            # Convert to OHLCVBar objects
            bars = []
            for timestamp, row in df.iterrows():
                # Skip rows with NaN values
                if row.isna().any():
                    continue

                bars.append(OHLCVBar(
                    timestamp=timestamp.to_pydatetime().replace(tzinfo=timezone.utc),
                    open=Decimal(str(row['Open'])),
                    high=Decimal(str(row['High'])),
                    low=Decimal(str(row['Low'])),
                    close=Decimal(str(row['Close'])),
                    volume=Decimal(str(row['Volume'])),
                    quote_volume=None,  # Yahoo doesn't provide quote volume
                    trade_count=None,   # Yahoo doesn't provide trade count
                ))

            # Apply limit (Yahoo returns all data in period)
            if limit and len(bars) > limit:
                bars = bars[-limit:]

            self.logger.success(
                f"Fetched {len(bars)} {timeframe} candles for {self.symbol}"
            )

            return bars

        except Exception as e:
            if "No data found" in str(e) or "No timezone found" in str(e):
                raise SymbolNotFoundError(f"Symbol {self.symbol} not found") from e
            elif "rate limit" in str(e).lower():
                raise RateLimitError(f"Yahoo Finance rate limit exceeded") from e
            else:
                self.logger.error(f"Failed to fetch historical data: {e}")
                raise

    async def subscribe_realtime(self) -> AsyncIterator[RealtimeTick]:
        """Subscribe to real-time price updates via polling.

        Polls Yahoo Finance every 60 seconds for latest price.
        """
        self.logger.info(f"Starting real-time polling for {self.symbol} (60s interval)")

        while self._is_running:
            try:
                price = await self.get_current_price()

                yield RealtimeTick(
                    timestamp=datetime.now(timezone.utc),
                    price=price,
                    volume=None,  # Not available in fast quote
                )

                # Wait 60 seconds before next poll
                await asyncio.sleep(60)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Polling error: {e}. Retrying in 60s...")
                await asyncio.sleep(60)

    async def validate_symbol(self) -> bool:
        """Check if symbol exists on Yahoo Finance."""
        try:
            # Try to fetch ticker info
            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(None, lambda: self._ticker.info)

            # If we get info back, symbol exists
            if info and 'symbol' in info:
                return True
            return False

        except Exception as e:
            self.logger.debug(f"Symbol validation failed for {self.symbol}: {e}")
            return False

    async def get_current_price(self) -> Decimal:
        """Get current market price from fast quote."""
        try:
            loop = asyncio.get_event_loop()

            # Use fast_info for quick price lookup
            fast_info = await loop.run_in_executor(
                None,
                lambda: self._ticker.fast_info
            )

            # Get last price
            if hasattr(fast_info, 'last_price') and fast_info.last_price:
                price = Decimal(str(fast_info.last_price))
                self._last_price = price
                return price

            # Fallback to regular info
            info = await loop.run_in_executor(None, lambda: self._ticker.info)
            if 'currentPrice' in info:
                price = Decimal(str(info['currentPrice']))
                self._last_price = price
                return price
            elif 'regularMarketPrice' in info:
                price = Decimal(str(info['regularMarketPrice']))
                self._last_price = price
                return price

            raise ValueError(f"No price data available for {self.symbol}")

        except Exception as e:
            self.logger.error(f"Failed to get current price: {e}")
            # Return last known price if available
            if self._last_price:
                self.logger.warning(f"Using cached price: ${self._last_price}")
                return self._last_price
            raise

    def get_supported_timeframes(self) -> list[str]:
        """Yahoo Finance supported timeframes."""
        return ["1m", "5m", "15m", "1h", "4h", "1d"]

    async def start(self) -> None:
        """Start the connector (begin polling)."""
        self._is_running = True
        self.logger.success(f"Yahoo Finance connector started for {self.symbol}")

    async def stop(self) -> None:
        """Stop the connector."""
        self._is_running = False

        if self._polling_task and not self._polling_task.done():
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass

        self.logger.info(f"Yahoo Finance connector stopped for {self.symbol}")
