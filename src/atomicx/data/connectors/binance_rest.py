"""Binance REST API connector for historical data.

Used by dashboard to fetch:
- Historical OHLCV data (candlesticks)
- Order book snapshots
- Recent trades
"""

from __future__ import annotations

import aiohttp
from typing import List, Dict, Any
from loguru import logger
from datetime import datetime


class BinanceRestAPI:
    """Binance REST API client for historical data."""

    def __init__(self):
        self.base_url = "https://api.binance.com"
        self.session: aiohttp.ClientSession | None = None

    async def _ensure_session(self):
        """Ensure aiohttp session exists."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def close(self):
        """Close the session."""
        if self.session and not self.session.closed:
            await self.session.close()

    async def get_historical_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 200
    ) -> List[Dict[str, Any]]:
        """Get historical candlestick data.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Timeframe ("1m", "5m", "15m", "1h", "4h", "1d", "1w")
            limit: Number of candles (max 1000)

        Returns:
            List of candle dictionaries with OHLCV data
        """
        await self._ensure_session()

        url = f"{self.base_url}/api/v3/klines"
        params = {
            "symbol": symbol.replace("/", "").upper(),  # BTC/USDT → BTCUSDT
            "interval": interval,
            "limit": min(limit, 1000)  # Binance max is 1000
        }

        try:
            async with self.session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"[BINANCE-REST] Klines error {response.status}: {error_text}")
                    return []

                data = await response.json()

                # Format: [timestamp, open, high, low, close, volume, close_time, ...]
                klines = []
                for candle in data:
                    klines.append({
                        "timestamp": int(candle[0]),  # Open time
                        "open": float(candle[1]),
                        "high": float(candle[2]),
                        "low": float(candle[3]),
                        "close": float(candle[4]),
                        "volume": float(candle[5]),
                        "close_time": int(candle[6]),
                        "quote_volume": float(candle[7]),
                        "trades": int(candle[8])
                    })

                logger.info(f"[BINANCE-REST] Fetched {len(klines)} candles for {symbol} @ {interval}")
                return klines

        except asyncio.TimeoutError:
            logger.error(f"[BINANCE-REST] Timeout fetching klines for {symbol}")
            return []
        except Exception as e:
            logger.error(f"[BINANCE-REST] Error fetching klines: {e}")
            return []

    async def get_orderbook_snapshot(
        self,
        symbol: str,
        limit: int = 50
    ) -> Dict[str, Any]:
        """Get order book snapshot (L2 depth).

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            limit: Depth levels (5, 10, 20, 50, 100, 500, 1000, 5000)

        Returns:
            Dictionary with bids and asks
        """
        await self._ensure_session()

        url = f"{self.base_url}/api/v3/depth"
        params = {
            "symbol": symbol.replace("/", "").upper(),
            "limit": limit
        }

        try:
            async with self.session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"[BINANCE-REST] Order book error {response.status}: {error_text}")
                    return {"bids": [], "asks": []}

                data = await response.json()

                # Format bids/asks as [price, quantity]
                bids = [[float(price), float(qty)] for price, qty in data.get("bids", [])]
                asks = [[float(price), float(qty)] for price, qty in data.get("asks", [])]

                logger.debug(f"[BINANCE-REST] Fetched order book for {symbol}: {len(bids)} bids, {len(asks)} asks")

                return {
                    "symbol": symbol,
                    "bids": bids,
                    "asks": asks,
                    "last_update_id": data.get("lastUpdateId")
                }

        except asyncio.TimeoutError:
            logger.error(f"[BINANCE-REST] Timeout fetching order book for {symbol}")
            return {"bids": [], "asks": []}
        except Exception as e:
            logger.error(f"[BINANCE-REST] Error fetching order book: {e}")
            return {"bids": [], "asks": []}

    async def get_recent_trades(
        self,
        symbol: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get recent trades.

        Args:
            symbol: Trading pair
            limit: Number of trades (max 1000)

        Returns:
            List of recent trades
        """
        await self._ensure_session()

        url = f"{self.base_url}/api/v3/trades"
        params = {
            "symbol": symbol.replace("/", "").upper(),
            "limit": min(limit, 1000)
        }

        try:
            async with self.session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"[BINANCE-REST] Recent trades error {response.status}: {error_text}")
                    return []

                data = await response.json()

                trades = []
                for trade in data:
                    trades.append({
                        "id": trade["id"],
                        "price": float(trade["price"]),
                        "qty": float(trade["qty"]),
                        "time": trade["time"],
                        "is_buyer_maker": trade["isBuyerMaker"]
                    })

                return trades

        except Exception as e:
            logger.error(f"[BINANCE-REST] Error fetching recent trades: {e}")
            return []

    async def get_ticker_24h(self, symbol: str) -> Dict[str, Any]:
        """Get 24h ticker statistics.

        Args:
            symbol: Trading pair

        Returns:
            24h stats including price change, volume, etc.
        """
        await self._ensure_session()

        url = f"{self.base_url}/api/v3/ticker/24hr"
        params = {"symbol": symbol.replace("/", "").upper()}

        try:
            async with self.session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"[BINANCE-REST] Ticker error {response.status}: {error_text}")
                    return {}

                data = await response.json()

                return {
                    "symbol": symbol,
                    "price_change": float(data.get("priceChange", 0)),
                    "price_change_percent": float(data.get("priceChangePercent", 0)),
                    "last_price": float(data.get("lastPrice", 0)),
                    "high_price": float(data.get("highPrice", 0)),
                    "low_price": float(data.get("lowPrice", 0)),
                    "volume": float(data.get("volume", 0)),
                    "quote_volume": float(data.get("quoteVolume", 0))
                }

        except Exception as e:
            logger.error(f"[BINANCE-REST] Error fetching ticker: {e}")
            return {}


# Singleton instance
_binance_rest_instance: BinanceRestAPI | None = None


def get_binance_rest() -> BinanceRestAPI:
    """Get singleton Binance REST API instance."""
    global _binance_rest_instance
    if _binance_rest_instance is None:
        _binance_rest_instance = BinanceRestAPI()
    return _binance_rest_instance
