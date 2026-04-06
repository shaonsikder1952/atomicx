"""Base data connector abstraction for universal multi-asset support.

Defines the interface that all data connectors must implement.
Connectors: Binance (crypto), Yahoo Finance (stocks/commodities), Alpha Vantage (forex).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from typing import Any, AsyncIterator
from dataclasses import dataclass

from loguru import logger


@dataclass
class OHLCVBar:
    """Standardized OHLCV candlestick data."""
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    quote_volume: Decimal | None = None
    trade_count: int | None = None


@dataclass
class RealtimeTick:
    """Standardized real-time price tick."""
    timestamp: datetime
    price: Decimal
    volume: Decimal | None = None


class DataConnector(ABC):
    """Abstract base class for all data connectors.

    Each connector implements this interface to provide data from different sources:
    - Binance: Crypto (WebSocket + REST)
    - Yahoo Finance: Stocks + Commodities (REST polling)
    - Alpha Vantage: Forex (REST polling)
    """

    def __init__(self, symbol: str):
        """Initialize connector for a specific symbol.

        Args:
            symbol: The asset symbol (e.g., "BTC/USDT", "AAPL", "CL=F", "EURUSD")
        """
        self.symbol = symbol
        self.logger = logger.bind(connector=self.__class__.__name__, symbol=symbol)
        self._is_running = False

    @abstractmethod
    async def get_historical_ohlcv(
        self,
        timeframe: str,
        limit: int = 500,
        since: datetime | None = None
    ) -> list[OHLCVBar]:
        """Fetch historical OHLCV data.

        Args:
            timeframe: Candle timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles to fetch
            since: Start timestamp (if None, fetch most recent)

        Returns:
            List of OHLCV bars in chronological order

        Raises:
            ValueError: If symbol doesn't exist or timeframe not supported
            RuntimeError: If API error or rate limit exceeded
        """
        pass

    @abstractmethod
    async def subscribe_realtime(self) -> AsyncIterator[RealtimeTick]:
        """Subscribe to real-time price updates.

        Yields real-time price ticks as they arrive.
        For WebSocket connectors: continuous stream.
        For polling connectors: updates every 60 seconds.

        Yields:
            RealtimeTick objects with current price

        Raises:
            RuntimeError: If connection fails
        """
        pass

    @abstractmethod
    async def validate_symbol(self) -> bool:
        """Check if symbol exists and is tradeable.

        Returns:
            True if symbol is valid, False otherwise
        """
        pass

    @abstractmethod
    async def get_current_price(self) -> Decimal:
        """Get current market price (single request, no subscription).

        Returns:
            Current price as Decimal

        Raises:
            ValueError: If symbol doesn't exist
            RuntimeError: If API error
        """
        pass

    @abstractmethod
    def get_supported_timeframes(self) -> list[str]:
        """Get list of supported timeframes for this connector.

        Returns:
            List of timeframe strings (e.g., ["1m", "5m", "1h", "1d"])
        """
        pass

    @abstractmethod
    async def start(self) -> None:
        """Start the connector (e.g., establish WebSocket connection)."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the connector and clean up resources."""
        pass

    def normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to connector-specific format.

        Override in subclasses if needed.
        E.g., "BTC/USDT" → "BTCUSDT" for Binance

        Args:
            symbol: Symbol in internal format

        Returns:
            Symbol in connector-specific format
        """
        return symbol

    def denormalize_symbol(self, symbol: str) -> str:
        """Convert connector-specific symbol back to internal format.

        Override in subclasses if needed.
        E.g., "BTCUSDT" → "BTC/USDT" for internal use

        Args:
            symbol: Symbol in connector-specific format

        Returns:
            Symbol in internal format
        """
        return symbol


class ConnectorError(Exception):
    """Base exception for connector errors."""
    pass


class SymbolNotFoundError(ConnectorError):
    """Symbol doesn't exist on this exchange/API."""
    pass


class RateLimitError(ConnectorError):
    """API rate limit exceeded."""
    pass


class ConnectionError(ConnectorError):
    """Connection to data source failed."""
    pass
