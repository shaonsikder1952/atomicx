"""Mempool Monitoring for Whale Transaction Detection.

Monitors Bitcoin/Ethereum mempool for large pending transactions.
Provides 5-15 minute edge when whales move funds to exchanges.
"""

from __future__ import annotations

import asyncio
import aiohttp
from dataclasses import dataclass
from typing import List, Optional
from loguru import logger


@dataclass
class MempoolTransaction:
    """Large mempool transaction."""
    txid: str
    from_address: str
    to_address: str
    amount_btc: float
    amount_usd: float
    fee: float
    detected_at: float
    is_exchange_deposit: bool
    is_exchange_withdrawal: bool
    whale_category: str  # "mega", "large", "medium"


class MempoolMonitor:
    """Monitor mempool for significant transactions.

    Usage:
        monitor = MempoolMonitor(threshold_btc=100)
        await monitor.start()

        # Get recent large transactions
        large_txs = await monitor.get_large_transactions()

        for tx in large_txs:
            if tx.is_exchange_deposit:
                logger.warning(f"Whale depositing {tx.amount_btc} BTC to exchange - possible sell")
    """

    # Known exchange addresses (partial list)
    EXCHANGE_ADDRESSES = {
        # Binance
        "bc1qm34lsc65zpw79lxes69zkqmk6ee3ewf0j77s3h": "binance",
        "1NDyJtNTjmwk5xPNhjgAMu4HDHigtobu1s": "binance",
        # Coinbase
        "bc1qgdjqv0av3q56jvd82tkdjpy7gdp9ut8tlqmgrpmv24sq90ecnvqqjwvw97": "coinbase",
        # Kraken
        "bc1qj5swkkkk50ymyeqx2em906jfft86ptd4xs8wwf": "kraken",
        # Add more as needed
    }

    def __init__(
        self,
        threshold_btc: float = 100.0,  # Minimum BTC to track
        api_endpoint: str = "https://mempool.space/api",
    ):
        self.threshold_btc = threshold_btc
        self.api_endpoint = api_endpoint
        self.tracked_transactions: List[MempoolTransaction] = []
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_running = False

        logger.info(f"[MEMPOOL] Initialized monitor (threshold={threshold_btc} BTC)")

    async def start(self):
        """Start monitoring mempool."""
        self.is_running = True
        self.session = aiohttp.ClientSession()

        logger.info("[MEMPOOL] Started monitoring")

        # Start monitoring loop
        asyncio.create_task(self._monitor_loop())

    async def stop(self):
        """Stop monitoring."""
        self.is_running = False
        if self.session:
            await self.session.close()

        logger.info("[MEMPOOL] Stopped monitoring")

    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                await self._fetch_and_process_mempool()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"[MEMPOOL] Error in monitor loop: {e}")
                await asyncio.sleep(60)

    async def _fetch_and_process_mempool(self):
        """Fetch current mempool and process large transactions."""
        if not self.session:
            return

        try:
            # Fetch mempool transactions
            url = f"{self.api_endpoint}/mempool/recent"
            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    self._process_transactions(data)

        except asyncio.TimeoutError:
            logger.warning("[MEMPOOL] API request timed out")
        except Exception as e:
            logger.error(f"[MEMPOOL] Fetch error: {e}")

    def _process_transactions(self, transactions: List[dict]):
        """Process transactions and identify large ones."""
        import time

        for tx in transactions:
            # Extract transaction details
            txid = tx.get("txid", "")
            vout = tx.get("vout", [])

            # Calculate total output
            total_btc = sum(output.get("value", 0) for output in vout) / 1e8  # Satoshi to BTC

            if total_btc >= self.threshold_btc:
                # Parse addresses
                addresses = [output.get("scriptpubkey_address") for output in vout]
                to_address = addresses[0] if addresses else "unknown"

                # Check if exchange interaction
                is_deposit = any(addr in self.EXCHANGE_ADDRESSES for addr in addresses)
                is_withdrawal = False  # Would need to check inputs

                # Categorize whale
                if total_btc >= 1000:
                    category = "mega"
                elif total_btc >= 500:
                    category = "large"
                else:
                    category = "medium"

                # Create transaction record
                mempool_tx = MempoolTransaction(
                    txid=txid,
                    from_address="unknown",  # Would parse from inputs
                    to_address=to_address,
                    amount_btc=total_btc,
                    amount_usd=total_btc * 50000,  # Approximate
                    fee=tx.get("fee", 0) / 1e8,
                    detected_at=time.time(),
                    is_exchange_deposit=is_deposit,
                    is_exchange_withdrawal=is_withdrawal,
                    whale_category=category,
                )

                # Add to tracked list
                self.tracked_transactions.append(mempool_tx)

                # Keep only recent (last 1000)
                if len(self.tracked_transactions) > 1000:
                    self.tracked_transactions = self.tracked_transactions[-1000:]

                logger.info(
                    f"[MEMPOOL] {category.upper()} whale: {total_btc:.2f} BTC "
                    f"{'→ EXCHANGE' if is_deposit else ''}"
                )

    async def get_large_transactions(
        self, time_window_seconds: float = 300
    ) -> List[MempoolTransaction]:
        """Get large transactions from recent time window.

        Args:
            time_window_seconds: Time window (default: 5 minutes)

        Returns:
            List of large transactions
        """
        import time

        cutoff_time = time.time() - time_window_seconds

        recent_txs = [
            tx for tx in self.tracked_transactions
            if tx.detected_at >= cutoff_time
        ]

        return recent_txs

    def get_signal(self) -> dict:
        """Get trading signal from mempool analysis.

        Returns:
            Signal dictionary
        """
        recent_txs = asyncio.run(self.get_large_transactions(300))  # Last 5 minutes

        if not recent_txs:
            return {"direction": "neutral", "confidence": 0.0, "reason": "no_activity"}

        # Count exchange deposits vs withdrawals
        deposits = sum(1 for tx in recent_txs if tx.is_exchange_deposit)
        withdrawals = sum(1 for tx in recent_txs if tx.is_exchange_withdrawal)

        # Heavy deposit activity = bearish (selling pressure)
        if deposits > withdrawals + 2:
            return {
                "direction": "bearish",
                "confidence": min(deposits / 10, 0.8),
                "reason": f"{deposits} whale deposits to exchanges",
            }

        # Heavy withdrawal activity = bullish (accumulation)
        elif withdrawals > deposits + 2:
            return {
                "direction": "bullish",
                "confidence": min(withdrawals / 10, 0.8),
                "reason": f"{withdrawals} whale withdrawals from exchanges",
            }

        return {"direction": "neutral", "confidence": 0.3, "reason": "balanced_flow"}
