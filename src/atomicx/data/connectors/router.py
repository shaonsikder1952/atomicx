"""Connector router - auto-detects asset type and routes to correct data source.

Universal routing system for crypto, stocks, commodities, and forex.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Type

from loguru import logger

from atomicx.data.connectors.base import DataConnector


class AssetType(Enum):
    """Supported asset types."""
    CRYPTO = "crypto"
    STOCK = "stock"
    COMMODITY = "commodity"
    FOREX = "forex"
    UNKNOWN = "unknown"


class ConnectorRouter:
    """Routes symbols to appropriate data connectors based on asset type.

    Examples:
        BTC/USDT → Crypto → Binance
        AAPL → Stock → Yahoo Finance
        CL=F → Commodity → Yahoo Finance
        EURUSD → Forex → Alpha Vantage
    """

    def __init__(self):
        self.logger = logger.bind(module="connector.router")
        self._connector_cache: dict[str, DataConnector] = {}

    def detect_asset_type(self, symbol: str) -> tuple[AssetType, str]:
        """Detect asset type from symbol format.

        Args:
            symbol: Asset symbol

        Returns:
            Tuple of (AssetType, normalized_symbol)

        Examples:
            "BTC/USDT" → (AssetType.CRYPTO, "BTC/USDT")
            "AAPL" → (AssetType.STOCK, "AAPL")
            "CL=F" → (AssetType.COMMODITY, "CL=F")
            "EURUSD" → (AssetType.FOREX, "EURUSD")
        """
        symbol_upper = symbol.upper().strip()

        # Crypto: Has "/" separator and ends with USDT/BUSD/USD/BTC/ETH
        if "/" in symbol_upper:
            if re.match(r"^[A-Z0-9]+/(USDT|BUSD|USD|BTC|ETH|BNB)$", symbol_upper):
                return (AssetType.CRYPTO, symbol_upper)

        # Commodity: Ends with =F (futures contract)
        if symbol_upper.endswith("=F"):
            return (AssetType.COMMODITY, symbol_upper)

        # Forex: Exactly 6 characters, no separators (e.g., EURUSD, GBPJPY)
        if len(symbol_upper) == 6 and symbol_upper.isalpha():
            return (AssetType.FOREX, symbol_upper)

        # Stock: 1-5 uppercase letters (e.g., AAPL, TSLA, GOOGL)
        if 1 <= len(symbol_upper) <= 5 and symbol_upper.isalpha():
            return (AssetType.STOCK, symbol_upper)

        # Unknown - default to stock and let connector validate
        self.logger.warning(f"Unknown asset type for {symbol}, defaulting to STOCK")
        return (AssetType.STOCK, symbol_upper)

    def get_connector_class(self, asset_type: AssetType) -> Type[DataConnector]:
        """Get the connector class for a given asset type.

        Args:
            asset_type: The asset type

        Returns:
            DataConnector subclass

        Raises:
            ValueError: If asset type not supported yet
        """
        if asset_type == AssetType.CRYPTO:
            # Use existing Binance connector (needs adaptation to new interface)
            from atomicx.data.connectors.binance import BinanceConnector
            return BinanceConnector

        elif asset_type == AssetType.STOCK:
            # Yahoo Finance for stocks
            from atomicx.data.connectors.stock import YahooFinanceConnector
            return YahooFinanceConnector

        elif asset_type == AssetType.COMMODITY:
            # Yahoo Finance also handles commodity futures
            from atomicx.data.connectors.stock import YahooFinanceConnector
            return YahooFinanceConnector

        elif asset_type == AssetType.FOREX:
            # Alpha Vantage for forex (will implement later)
            raise NotImplementedError(
                "Forex support (Alpha Vantage) not yet implemented. "
                "Coming in Phase 4. For now, only crypto/stocks/commodities supported."
            )

        else:
            raise ValueError(f"Unsupported asset type: {asset_type}")

    def get_data_source_name(self, asset_type: AssetType) -> str:
        """Get the data source name for storage.

        Args:
            asset_type: The asset type

        Returns:
            Data source identifier (e.g., "binance", "yahoo", "alpha_vantage")
        """
        if asset_type == AssetType.CRYPTO:
            return "binance"
        elif asset_type in (AssetType.STOCK, AssetType.COMMODITY):
            return "yahoo"
        elif asset_type == AssetType.FOREX:
            return "alpha_vantage"
        else:
            return "unknown"

    async def get_connector(self, symbol: str) -> DataConnector:
        """Get or create a connector for the given symbol.

        Automatically detects asset type and routes to correct connector.

        Args:
            symbol: Asset symbol

        Returns:
            Initialized DataConnector instance

        Raises:
            ValueError: If asset type not supported
            NotImplementedError: If connector not yet implemented
        """
        # Check cache first
        if symbol in self._connector_cache:
            return self._connector_cache[symbol]

        # Detect asset type
        asset_type, normalized_symbol = self.detect_asset_type(symbol)
        self.logger.info(f"Detected {symbol} as {asset_type.value}")

        # Get connector class
        connector_class = self.get_connector_class(asset_type)

        # Create and cache connector
        connector = connector_class(normalized_symbol)
        self._connector_cache[symbol] = connector

        self.logger.success(
            f"Created {connector_class.__name__} for {symbol} ({asset_type.value})"
        )

        return connector

    async def validate_symbol(self, symbol: str) -> tuple[bool, str | None]:
        """Validate if symbol exists on its respective exchange/API.

        Args:
            symbol: Asset symbol

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            connector = await self.get_connector(symbol)
            is_valid = await connector.validate_symbol()

            if is_valid:
                return (True, None)
            else:
                return (False, f"Symbol {symbol} not found on exchange")

        except NotImplementedError as e:
            return (False, str(e))
        except Exception as e:
            self.logger.error(f"Failed to validate {symbol}: {e}")
            return (False, f"Validation error: {str(e)}")

    def clear_cache(self, symbol: str | None = None) -> None:
        """Clear connector cache.

        Args:
            symbol: Specific symbol to clear, or None to clear all
        """
        if symbol:
            self._connector_cache.pop(symbol, None)
            self.logger.debug(f"Cleared connector cache for {symbol}")
        else:
            self._connector_cache.clear()
            self.logger.debug("Cleared all connector caches")


# Global singleton router
_router_instance: ConnectorRouter | None = None


def get_connector_router() -> ConnectorRouter:
    """Get the global connector router instance."""
    global _router_instance
    if _router_instance is None:
        _router_instance = ConnectorRouter()
    return _router_instance
