"""Funding Rate Tracker for Leverage Signals.

Tracks perpetual swap funding rates across exchanges.
High positive funding = overleveraged longs (bearish)
High negative funding = overleveraged shorts (bullish)
"""

from __future__ import annotations

import asyncio
import aiohttp
from dataclasses import dataclass
from typing import Dict, Optional
from loguru import logger


@dataclass
class FundingRate:
    """Funding rate snapshot."""
    exchange: str
    symbol: str
    rate: float  # Hourly rate (e.g., 0.01 = 1%)
    rate_8h: float  # 8-hour annualized rate
    timestamp: float
    open_interest: Optional[float] = None


class FundingRateTracker:
    """Track funding rates for leverage analysis.

    Usage:
        tracker = FundingRateTracker()
        await tracker.start()

        # Get current funding signal
        signal = await tracker.get_signal()

        if signal["rate"] > 0.1:
            logger.warning("Overleveraged longs - potential long squeeze")
    """

    def __init__(self):
        self.funding_rates: Dict[str, FundingRate] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_running = False

        logger.info("[FUNDING] Initialized funding rate tracker")

    async def start(self):
        """Start tracking funding rates."""
        self.is_running = True
        self.session = aiohttp.ClientSession()

        logger.info("[FUNDING] Started tracking")

        # Start tracking loop
        asyncio.create_task(self._tracking_loop())

    async def stop(self):
        """Stop tracking."""
        self.is_running = False
        if self.session:
            await self.session.close()

        logger.info("[FUNDING] Stopped tracking")

    async def _tracking_loop(self):
        """Main tracking loop."""
        while self.is_running:
            try:
                await self._fetch_funding_rates()
                await asyncio.sleep(60)  # Update every minute
            except Exception as e:
                logger.error(f"[FUNDING] Error in tracking loop: {e}")
                await asyncio.sleep(300)

    async def _fetch_funding_rates(self):
        """Fetch funding rates from multiple exchanges."""
        if not self.session:
            return

        # Binance
        await self._fetch_binance_funding()

        # Add other exchanges as needed

    async def _fetch_binance_funding(self):
        """Fetch Binance funding rate."""
        try:
            url = "https://fapi.binance.com/fapi/v1/fundingRate"
            params = {"symbol": "BTCUSDT", "limit": 1}

            async with self.session.get(url, params=params, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    if data:
                        latest = data[0]
                        rate = float(latest["fundingRate"])

                        funding = FundingRate(
                            exchange="binance",
                            symbol="BTC/USDT",
                            rate=rate,
                            rate_8h=rate * 8,  # Convert to 8h rate
                            timestamp=float(latest["fundingTime"]) / 1000,
                        )

                        self.funding_rates["binance_BTCUSDT"] = funding

                        logger.debug(f"[FUNDING] Binance: {rate:.4%}")

        except Exception as e:
            logger.error(f"[FUNDING] Binance fetch error: {e}")

    async def get_signal(self) -> dict:
        """Get trading signal from funding rates.

        Returns:
            Signal dictionary
        """
        if not self.funding_rates:
            return {"direction": "neutral", "confidence": 0.0, "rate": 0.0}

        # Average funding rate across exchanges
        rates = [fr.rate for fr in self.funding_rates.values()]
        avg_rate = sum(rates) / len(rates)

        # Interpretation:
        # > 0.1% = Very high positive funding = Overleveraged longs = BEARISH
        # > 0.05% = High positive funding = Moderately bearish
        # < -0.05% = High negative funding = Moderately bullish
        # < -0.1% = Very high negative funding = Overleveraged shorts = BULLISH

        if avg_rate > 0.001:  # > 0.1%
            return {
                "direction": "bearish",
                "confidence": min(avg_rate * 1000, 0.9),
                "rate": avg_rate,
                "reason": "overleveraged_longs",
            }
        elif avg_rate > 0.0005:  # > 0.05%
            return {
                "direction": "bearish",
                "confidence": 0.6,
                "rate": avg_rate,
                "reason": "high_long_funding",
            }
        elif avg_rate < -0.001:  # < -0.1%
            return {
                "direction": "bullish",
                "confidence": min(abs(avg_rate) * 1000, 0.9),
                "rate": avg_rate,
                "reason": "overleveraged_shorts",
            }
        elif avg_rate < -0.0005:  # < -0.05%
            return {
                "direction": "bullish",
                "confidence": 0.6,
                "rate": avg_rate,
                "reason": "high_short_funding",
            }

        return {
            "direction": "neutral",
            "confidence": 0.3,
            "rate": avg_rate,
            "reason": "balanced_funding",
        }
