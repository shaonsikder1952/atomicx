"""CoinGecko REST API poller for on-chain and market overview data.

Handles:
- Market data (price, market cap, volume, supply)
- Trending coins
- Global market stats
- On-chain proxy metrics (where available via free API)

Rate-limited to respect CoinGecko's free tier (10-30 req/min).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import httpx
from loguru import logger

from atomicx.config import get_settings


class CoinGeckoPoller:
    """Polls CoinGecko API for on-chain and market data.

    Respects rate limits with built-in throttling.
    Publishes data via callback for Kafka/DB consumers.
    """

    # CoinGecko coin IDs for common symbols
    SYMBOL_TO_ID: dict[str, str] = {
        "BTC/USDT": "bitcoin",
        "ETH/USDT": "ethereum",
        "SOL/USDT": "solana",
        "BNB/USDT": "binancecoin",
        "XRP/USDT": "ripple",
        "ADA/USDT": "cardano",
        "AVAX/USDT": "avalanche-2",
        "DOT/USDT": "polkadot",
        "DOGE/USDT": "dogecoin",
        "MATIC/USDT": "matic-network",
    }

    def __init__(
        self,
        symbols: list[str] | None = None,
        poll_interval: int | None = None,
        on_market_data: Any = None,
        on_global_data: Any = None,
    ) -> None:
        settings = get_settings()
        self._base_url = settings.coingecko_api_url
        self._symbols = symbols or settings.default_symbols
        self._poll_interval = poll_interval or settings.data_poll_interval_seconds
        self._on_market_data = on_market_data
        self._on_global_data = on_global_data
        self._client: httpx.AsyncClient | None = None
        self._running = False
        self._request_delay = 2.0  # Seconds between requests (rate limit protection)

    def _get_coin_ids(self) -> list[str]:
        """Convert trading symbols to CoinGecko coin IDs."""
        ids = []
        for sym in self._symbols:
            coin_id = self.SYMBOL_TO_ID.get(sym)
            if coin_id:
                ids.append(coin_id)
            else:
                # Try to derive from symbol
                base = sym.split("/")[0].lower()
                ids.append(base)
        return ids

    async def start(self) -> None:
        """Start the polling loop."""
        self._running = True
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=30.0,
            headers={"Accept": "application/json"},
        )

        logger.info(
            "Starting CoinGecko poller",
            symbols=self._symbols,
            interval=self._poll_interval,
        )

        while self._running:
            try:
                await self._poll_cycle()
            except Exception as e:
                logger.error(f"CoinGecko poll error: {e}")

            if self._running:
                await asyncio.sleep(self._poll_interval)

    async def _poll_cycle(self) -> None:
        """Execute one full polling cycle."""
        # 1. Fetch market data for tracked coins
        await self._fetch_market_data()
        await asyncio.sleep(self._request_delay)

        # 2. Fetch global market overview
        await self._fetch_global_data()

    async def _fetch_market_data(self) -> None:
        """Fetch current market data for all tracked coins."""
        if not self._client:
            return

        coin_ids = self._get_coin_ids()
        ids_str = ",".join(coin_ids)

        try:
            response = await self._client.get(
                "/coins/markets",
                params={
                    "vs_currency": "usd",
                    "ids": ids_str,
                    "order": "market_cap_desc",
                    "sparkline": "false",
                    "price_change_percentage": "1h,24h,7d",
                },
            )
            response.raise_for_status()
            data = response.json()

            now = datetime.now(tz=timezone.utc)
            for coin in data:
                metrics = self._extract_market_metrics(coin, now)
                if self._on_market_data:
                    for metric in metrics:
                        await self._on_market_data(metric)

            logger.debug(f"Fetched market data for {len(data)} coins")

        except httpx.HTTPError as e:
            logger.error(f"CoinGecko market data fetch failed: {e}")

    def _extract_market_metrics(
        self, coin: dict, timestamp: datetime
    ) -> list[dict[str, Any]]:
        """Extract individual metrics from CoinGecko market response."""
        symbol = coin.get("symbol", "").upper() + "/USDT"
        source = "coingecko"

        metrics = []
        metric_fields = {
            "market_cap": coin.get("market_cap"),
            "total_volume_24h": coin.get("total_volume"),
            "circulating_supply": coin.get("circulating_supply"),
            "total_supply": coin.get("total_supply"),
            "max_supply": coin.get("max_supply"),
            "ath_change_percentage": coin.get("ath_change_percentage"),
            "price_change_24h_pct": coin.get("price_change_percentage_24h"),
            "price_change_1h_pct": coin.get(
                "price_change_percentage_1h_in_currency"
            ),
            "price_change_7d_pct": coin.get(
                "price_change_percentage_7d_in_currency"
            ),
            "market_cap_rank": coin.get("market_cap_rank"),
        }

        for metric_name, value in metric_fields.items():
            if value is not None:
                metrics.append(
                    {
                        "timestamp": timestamp,
                        "symbol": symbol,
                        "metric_name": metric_name,
                        "value": Decimal(str(value)),
                        "source": source,
                        "metadata": {"coin_id": coin.get("id")},
                    }
                )

        return metrics

    async def _fetch_global_data(self) -> None:
        """Fetch global crypto market stats."""
        if not self._client:
            return

        try:
            response = await self._client.get("/global")
            response.raise_for_status()
            data = response.json().get("data", {})

            global_metrics = {
                "timestamp": datetime.now(tz=timezone.utc),
                "total_market_cap_usd": data.get("total_market_cap", {}).get("usd"),
                "total_volume_24h_usd": data.get("total_volume", {}).get("usd"),
                "btc_dominance": data.get("market_cap_percentage", {}).get("btc"),
                "eth_dominance": data.get("market_cap_percentage", {}).get("eth"),
                "active_cryptocurrencies": data.get("active_cryptocurrencies"),
                "market_cap_change_24h_pct": data.get(
                    "market_cap_change_percentage_24h_usd"
                ),
            }

            if self._on_global_data:
                await self._on_global_data(global_metrics)

            logger.debug("Fetched global market data")

        except httpx.HTTPError as e:
            logger.error(f"CoinGecko global data fetch failed: {e}")

    async def stop(self) -> None:
        """Stop the polling loop and close the HTTP client."""
        self._running = False
        if self._client:
            await self._client.aclose()
        logger.info("CoinGecko poller stopped")
