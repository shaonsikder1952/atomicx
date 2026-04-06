"""Stablecoin Flow Analyzer for Capital Movement.

Tracks stablecoin minting/burning and exchange flows.
Heavy minting = capital entering crypto = BULLISH
Heavy burning = capital exiting crypto = BEARISH
"""

from __future__ import annotations

import asyncio
import aiohttp
from dataclasses import dataclass
from typing import Dict, List, Optional
from loguru import logger


@dataclass
class StablecoinFlow:
    """Stablecoin flow snapshot."""
    stablecoin: str  # USDT, USDC, BUSD
    total_supply: float
    supply_change_24h: float
    exchange_inflow_24h: float
    exchange_outflow_24h: float
    net_flow: float  # Positive = entering, Negative = exiting
    timestamp: float


class StablecoinFlowAnalyzer:
    """Analyze stablecoin flows for capital movement signals.

    Usage:
        analyzer = StablecoinFlowAnalyzer()
        await analyzer.start()

        signal = await analyzer.get_signal()

        if signal["net_flow"] > 1e9:  # $1B+ inflow
            logger.info("Massive capital entering crypto - bullish")
    """

    def __init__(self):
        self.stablecoin_data: Dict[str, StablecoinFlow] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_running = False

        logger.info("[STABLES] Initialized stablecoin flow analyzer")

    async def start(self):
        """Start analyzing stablecoin flows."""
        self.is_running = True
        self.session = aiohttp.ClientSession()

        logger.info("[STABLES] Started analyzing")

        asyncio.create_task(self._analysis_loop())

    async def stop(self):
        """Stop analyzing."""
        self.is_running = False
        if self.session:
            await self.session.close()

        logger.info("[STABLES] Stopped analyzing")

    async def _analysis_loop(self):
        """Main analysis loop."""
        while self.is_running:
            try:
                await self._fetch_stablecoin_data()
                await asyncio.sleep(3600)  # Update hourly
            except Exception as e:
                logger.error(f"[STABLES] Error in analysis loop: {e}")
                await asyncio.sleep(1800)

    async def _fetch_stablecoin_data(self):
        """Fetch stablecoin data from various sources."""
        if not self.session:
            return

        # Fetch for major stablecoins
        await self._fetch_usdt_data()
        await self._fetch_usdc_data()

    async def _fetch_usdt_data(self):
        """Fetch USDT supply and flow data."""
        try:
            # CoinGecko API for supply
            url = "https://api.coingecko.com/api/v3/coins/tether"

            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    market_data = data.get("market_data", {})

                    total_supply = market_data.get("circulating_supply", 0)

                    # Calculate 24h change (simplified - would need historical data)
                    supply_change = 0.0  # Placeholder

                    # Exchange flows (would need blockchain analysis)
                    inflow = 0.0  # Placeholder
                    outflow = 0.0  # Placeholder

                    import time

                    flow = StablecoinFlow(
                        stablecoin="USDT",
                        total_supply=total_supply,
                        supply_change_24h=supply_change,
                        exchange_inflow_24h=inflow,
                        exchange_outflow_24h=outflow,
                        net_flow=inflow - outflow,
                        timestamp=time.time(),
                    )

                    self.stablecoin_data["USDT"] = flow

                    logger.debug(f"[STABLES] USDT supply: ${total_supply/1e9:.2f}B")

        except Exception as e:
            logger.error(f"[STABLES] USDT fetch error: {e}")

    async def _fetch_usdc_data(self):
        """Fetch USDC supply and flow data."""
        try:
            # CoinGecko API
            url = "https://api.coingecko.com/api/v3/coins/usd-coin"

            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    market_data = data.get("market_data", {})

                    total_supply = market_data.get("circulating_supply", 0)
                    supply_change = 0.0  # Placeholder

                    import time

                    flow = StablecoinFlow(
                        stablecoin="USDC",
                        total_supply=total_supply,
                        supply_change_24h=supply_change,
                        exchange_inflow_24h=0.0,
                        exchange_outflow_24h=0.0,
                        net_flow=0.0,
                        timestamp=time.time(),
                    )

                    self.stablecoin_data["USDC"] = flow

                    logger.debug(f"[STABLES] USDC supply: ${total_supply/1e9:.2f}B")

        except Exception as e:
            logger.error(f"[STABLES] USDC fetch error: {e}")

    async def get_signal(self) -> dict:
        """Get trading signal from stablecoin flows.

        Returns:
            Signal dictionary
        """
        if not self.stablecoin_data:
            return {"direction": "neutral", "confidence": 0.0, "net_flow": 0.0}

        # Calculate aggregate net flow
        total_net_flow = sum(flow.net_flow for flow in self.stablecoin_data.values())

        # Calculate supply changes
        total_supply_change = sum(
            flow.supply_change_24h for flow in self.stablecoin_data.values()
        )

        # Interpretation:
        # Heavy minting (> $500M) = BULLISH (capital entering)
        # Heavy burning (< -$500M) = BEARISH (capital exiting)
        # Heavy exchange inflow (> $1B) = BULLISH (ready to buy)
        # Heavy exchange outflow (> $1B) = BEARISH (moving to cold storage)

        if total_supply_change > 5e8:  # > $500M minted
            return {
                "direction": "bullish",
                "confidence": 0.8,
                "net_flow": total_net_flow,
                "supply_change": total_supply_change,
                "reason": "heavy_minting",
            }
        elif total_supply_change < -5e8:  # > $500M burned
            return {
                "direction": "bearish",
                "confidence": 0.8,
                "net_flow": total_net_flow,
                "supply_change": total_supply_change,
                "reason": "heavy_burning",
            }
        elif total_net_flow > 1e9:  # > $1B net inflow
            return {
                "direction": "bullish",
                "confidence": 0.7,
                "net_flow": total_net_flow,
                "supply_change": total_supply_change,
                "reason": "heavy_exchange_inflow",
            }
        elif total_net_flow < -1e9:  # > $1B net outflow
            return {
                "direction": "bearish",
                "confidence": 0.6,
                "net_flow": total_net_flow,
                "supply_change": total_supply_change,
                "reason": "heavy_exchange_outflow",
            }

        return {
            "direction": "neutral",
            "confidence": 0.4,
            "net_flow": total_net_flow,
            "supply_change": total_supply_change,
            "reason": "balanced_flows",
        }
