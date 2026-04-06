"""Binance connector implementation using CCXT.

Provides crypto market data from Binance exchange via CCXT library.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import AsyncIterator

import ccxt.async_support as ccxt

from atomicx.data.connectors.base import (
    DataConnector,
    OHLCVBar,
    RealtimeTick,
    SymbolNotFoundError,
    RateLimitError,
)


class BinanceConnector(DataConnector):
    """Binance exchange connector using CCXT.

    Implements DataConnector interface for crypto assets on Binance.
    Uses CCXT library for REST API access.
    """

    def __init__(self, symbol: str):
        """Initialize Binance connector.

        Args:
            symbol: Crypto pair in format "BTC/USDT"
        """
        super().__init__(symbol)
        self._exchange: ccxt.binance | None = None
        self._last_price: Decimal | None = None

    def _get_exchange(self) -> ccxt.binance:
        """Get or create CCXT exchange instance."""
        if self._exchange is None:
            self._exchange = ccxt.binance({
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            })
        return self._exchange

    async def get_historical_ohlcv(
        self,
        timeframe: str,
        limit: int = 500,
        since: datetime | None = None
    ) -> list[OHLCVBar]:
        """Fetch historical OHLCV data from Binance.

        Args:
            timeframe: Candle timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles to fetch (max 1000)
            since: Start timestamp

        Returns:
            List of OHLCV bars
        """
        exchange = self._get_exchange()

        try:
            # Convert since to milliseconds timestamp
            since_ms = int(since.timestamp() * 1000) if since else None

            # Fetch candles using CCXT
            candles = await exchange.fetch_ohlcv(
                self.symbol,
                timeframe,
                since=since_ms,
                limit=limit
            )

            # Convert CCXT format to OHLCVBar
            bars = []
            for candle in candles:
                timestamp_ms, open_price, high, low, close, volume = candle

                bars.append(OHLCVBar(
                    timestamp=datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc),
                    open=Decimal(str(open_price)),
                    high=Decimal(str(high)),
                    low=Decimal(str(low)),
                    close=Decimal(str(close)),
                    volume=Decimal(str(volume)),
                    quote_volume=None,
                    trade_count=None,
                ))

            self.logger.success(f"Fetched {len(bars)} candles for {self.symbol} {timeframe}")
            return bars

        except ccxt.NetworkError as e:
            raise RuntimeError(f"Network error fetching OHLCV: {e}") from e
        except ccxt.ExchangeError as e:
            if "Invalid symbol" in str(e):
                raise SymbolNotFoundError(f"Symbol {self.symbol} not found on Binance") from e
            raise RuntimeError(f"Exchange error: {e}") from e

    async def subscribe_realtime(self) -> AsyncIterator[RealtimeTick]:
        """Subscribe to real-time price updates.

        Note: This is a polling implementation. For WebSocket streaming,
        use BinanceWebSocketConnector instead.

        Yields:
            RealtimeTick objects every 5 seconds
        """
        import asyncio

        exchange = self._get_exchange()

        while self._is_running:
            try:
                ticker = await exchange.fetch_ticker(self.symbol)
                price = Decimal(str(ticker['last']))
                volume = Decimal(str(ticker.get('baseVolume', 0)))

                yield RealtimeTick(
                    timestamp=datetime.now(timezone.utc),
                    price=price,
                    volume=volume,
                )

                await asyncio.sleep(5)  # Poll every 5 seconds

            except Exception as e:
                self.logger.error(f"Error fetching real-time price: {e}")
                await asyncio.sleep(5)

    async def validate_symbol(self) -> bool:
        """Check if symbol exists on Binance.

        Returns:
            True if symbol is valid
        """
        exchange = self._get_exchange()

        try:
            await exchange.load_markets()
            return self.symbol in exchange.markets
        except Exception as e:
            self.logger.error(f"Error validating symbol: {e}")
            return False

    async def get_current_price(self) -> Decimal:
        """Get current market price for symbol.

        Returns:
            Current price
        """
        exchange = self._get_exchange()

        try:
            ticker = await exchange.fetch_ticker(self.symbol)
            price = Decimal(str(ticker['last']))
            self._last_price = price
            return price
        except ccxt.ExchangeError as e:
            if "Invalid symbol" in str(e):
                raise SymbolNotFoundError(f"Symbol {self.symbol} not found") from e
            raise RuntimeError(f"Error fetching price: {e}") from e

    def get_supported_timeframes(self) -> list[str]:
        """Get supported timeframes for Binance.

        Returns:
            List of timeframe strings
        """
        return ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]

    async def start(self) -> None:
        """Start the connector."""
        self._is_running = True
        # Load markets on startup for validation
        exchange = self._get_exchange()
        await exchange.load_markets()
        self.logger.info(f"Binance connector started for {self.symbol}")

    async def stop(self) -> None:
        """Stop the connector and clean up."""
        self._is_running = False
        if self._exchange:
            await self._exchange.close()
            self._exchange = None
        self.logger.info(f"Binance connector stopped for {self.symbol}")

    def normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to Binance format.

        Args:
            symbol: Symbol like "BTC/USDT"

        Returns:
            Symbol in Binance format (same as input for CCXT)
        """
        return symbol  # CCXT handles format conversion

    def denormalize_symbol(self, symbol: str) -> str:
        """Convert Binance symbol to internal format.

        Args:
            symbol: Binance symbol

        Returns:
            Internal format symbol
        """
        return symbol  # CCXT uses standard "BTC/USDT" format
