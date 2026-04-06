"""Options Flow Scanner for Institutional Positioning.

Tracks Bitcoin options flow for institutional positioning signals.
High put/call ratio = bearish, High call/put ratio = bullish
Gamma exposure indicates dealer hedging pressure.
"""

from __future__ import annotations

import asyncio
import aiohttp
from dataclasses import dataclass
from typing import Dict, List, Optional
from loguru import logger


@dataclass
class OptionsFlow:
    """Options market snapshot."""
    symbol: str
    put_call_ratio: float
    total_volume: float
    put_volume: float
    call_volume: float
    gamma_exposure: float  # Dealer gamma exposure
    max_pain: float  # Price where most options expire worthless
    timestamp: float


class OptionsFlowScanner:
    """Scanner for options market data.

    Usage:
        scanner = OptionsFlowScanner()
        await scanner.start()

        signal = await scanner.get_signal()

        if signal["put_call_ratio"] > 1.5:
            logger.info("Heavy put buying - bearish positioning")
    """

    def __init__(self):
        self.options_data: Dict[str, OptionsFlow] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_running = False

        logger.info("[OPTIONS] Initialized options flow scanner")

    async def start(self):
        """Start scanning options flow."""
        self.is_running = True
        self.session = aiohttp.ClientSession()

        logger.info("[OPTIONS] Started scanning")

        asyncio.create_task(self._scanning_loop())

    async def stop(self):
        """Stop scanning."""
        self.is_running = False
        if self.session:
            await self.session.close()

        logger.info("[OPTIONS] Stopped scanning")

    async def _scanning_loop(self):
        """Main scanning loop."""
        while self.is_running:
            try:
                await self._fetch_options_data()
                await asyncio.sleep(300)  # Update every 5 minutes
            except Exception as e:
                logger.error(f"[OPTIONS] Error in scanning loop: {e}")
                await asyncio.sleep(600)

    async def _fetch_options_data(self):
        """Fetch options data from Deribit (main BTC options exchange)."""
        if not self.session:
            return

        try:
            # Deribit public API
            url = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
            params = {"currency": "BTC", "kind": "option"}

            async with self.session.get(url, params=params, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    result = data.get("result", [])

                    self._process_options_data(result)

        except Exception as e:
            logger.error(f"[OPTIONS] Deribit fetch error: {e}")

    def _process_options_data(self, options_data: List[dict]):
        """Process options data and calculate metrics."""
        import time

        if not options_data:
            return

        # Separate calls and puts
        calls = [opt for opt in options_data if opt.get("instrument_name", "").endswith("C")]
        puts = [opt for opt in options_data if opt.get("instrument_name", "").endswith("P")]

        # Calculate volumes
        call_volume = sum(opt.get("volume", 0) for opt in calls)
        put_volume = sum(opt.get("volume", 0) for opt in puts)
        total_volume = call_volume + put_volume

        # Put/Call ratio
        put_call_ratio = put_volume / call_volume if call_volume > 0 else 1.0

        # Simplified gamma exposure (would need full calculations in production)
        gamma_exposure = (call_volume - put_volume) / total_volume if total_volume > 0 else 0.0

        # Max pain (simplified - would need full OI calculation)
        max_pain = 50000.0  # Placeholder

        flow = OptionsFlow(
            symbol="BTC",
            put_call_ratio=put_call_ratio,
            total_volume=total_volume,
            put_volume=put_volume,
            call_volume=call_volume,
            gamma_exposure=gamma_exposure,
            max_pain=max_pain,
            timestamp=time.time(),
        )

        self.options_data["BTC"] = flow

        logger.debug(f"[OPTIONS] P/C ratio: {put_call_ratio:.2f}, Gamma: {gamma_exposure:.2f}")

    async def get_signal(self) -> dict:
        """Get trading signal from options flow.

        Returns:
            Signal dictionary
        """
        if "BTC" not in self.options_data:
            return {"direction": "neutral", "confidence": 0.0, "put_call_ratio": 1.0}

        flow = self.options_data["BTC"]

        # Interpretation:
        # P/C > 1.5 = Heavy put buying = BEARISH
        # P/C > 1.2 = Moderate put buying = Moderately bearish
        # P/C < 0.7 = Heavy call buying = BULLISH
        # P/C < 0.8 = Moderate call buying = Moderately bullish

        if flow.put_call_ratio > 1.5:
            return {
                "direction": "bearish",
                "confidence": 0.8,
                "put_call_ratio": flow.put_call_ratio,
                "reason": "heavy_put_buying",
            }
        elif flow.put_call_ratio > 1.2:
            return {
                "direction": "bearish",
                "confidence": 0.6,
                "put_call_ratio": flow.put_call_ratio,
                "reason": "moderate_put_buying",
            }
        elif flow.put_call_ratio < 0.7:
            return {
                "direction": "bullish",
                "confidence": 0.8,
                "put_call_ratio": flow.put_call_ratio,
                "reason": "heavy_call_buying",
            }
        elif flow.put_call_ratio < 0.8:
            return {
                "direction": "bullish",
                "confidence": 0.6,
                "put_call_ratio": flow.put_call_ratio,
                "reason": "moderate_call_buying",
            }

        return {
            "direction": "neutral",
            "confidence": 0.4,
            "put_call_ratio": flow.put_call_ratio,
            "reason": "balanced_options",
        }
