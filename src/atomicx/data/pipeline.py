"""Data pipeline orchestrator — wires all components together.

This is the main entry point for the data pipeline service.
It starts all connectors, the persistence service, and the Kafka publisher,
and wires data flow between them.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

from loguru import logger

from atomicx.config import get_settings
from atomicx.data.connectors.binance_ws import BinanceWebSocketConnector
from atomicx.data.connectors.circuit_breaker import DataFreshnessMonitor
from atomicx.data.connectors.coingecko import CoinGeckoPoller
from atomicx.data.connectors.kafka_publisher import KafkaPublisher
from atomicx.data.storage.persistence import DataPersistenceService


class DataPipelineOrchestrator:
    """Orchestrates the entire data pipeline.

    Wires together:
    1. Binance WebSocket → Kafka + TimescaleDB
    2. CoinGecko Poller → Kafka + TimescaleDB
    3. Circuit breaker monitoring across all sources
    """

    def __init__(self) -> None:
        settings = get_settings()

        # Data freshness monitor
        self.monitor = DataFreshnessMonitor()

        # Register data sources with circuit breakers
        self._binance_cb = self.monitor.register_source(
            "binance_ws",
            staleness_threshold=timedelta(seconds=30),
            failure_threshold=3,
            recovery_timeout=30.0,
        )
        self._coingecko_cb = self.monitor.register_source(
            "coingecko",
            staleness_threshold=timedelta(minutes=5),
            failure_threshold=5,
            recovery_timeout=120.0,
        )

        # Kafka publisher
        self.kafka = KafkaPublisher()

        # Persistence service
        self.persistence = DataPersistenceService(
            buffer_size=500,
            flush_interval=5.0,
        )

        # Binance WebSocket connector
        self.binance = BinanceWebSocketConnector(
            symbols=settings.default_symbols,
            on_trade=self._handle_trade,
            on_kline=self._handle_kline,
            on_depth=self._handle_depth,
        )

        # CoinGecko poller
        self.coingecko = CoinGeckoPoller(
            symbols=settings.default_symbols,
            on_market_data=self._handle_market_data,
            on_global_data=self._handle_global_data,
        )

    # ── Event Handlers ───────────────────────────────────────

    async def _handle_trade(self, trade: dict) -> None:
        """Process incoming trade event."""
        self.monitor.record_update("binance_ws")
        await asyncio.gather(
            self.kafka.publish_trade(trade),
            self.persistence.write_trade(trade),
            return_exceptions=True,
        )

    async def _handle_kline(self, kline: dict) -> None:
        """Process incoming kline event."""
        self.monitor.record_update("binance_ws")
        await asyncio.gather(
            self.kafka.publish_kline(kline),
            self.persistence.write_kline(kline),
            return_exceptions=True,
        )

    async def _handle_depth(self, depth: dict) -> None:
        """Process incoming depth event."""
        self.monitor.record_update("binance_ws")
        await asyncio.gather(
            self.kafka.publish_depth(depth),
            self.persistence.write_depth(depth),
            return_exceptions=True,
        )

    async def _handle_market_data(self, metric: dict) -> None:
        """Process on-chain/market metric from CoinGecko."""
        self.monitor.record_update("coingecko")
        await asyncio.gather(
            self.kafka.publish_onchain_metric(metric),
            self.persistence.write_onchain_metric(metric),
            return_exceptions=True,
        )

    async def _handle_global_data(self, data: dict) -> None:
        """Process global market data from CoinGecko."""
        self.monitor.record_update("coingecko")
        await self.kafka.publish_global_data(data)

    # ── Lifecycle ────────────────────────────────────────────

    async def start(self) -> None:
        """Start all pipeline components."""
        logger.info("═══ Starting AtomicX Data Pipeline ═══")

        # Start infrastructure services
        try:
            await self.kafka.start()
        except Exception as e:
            logger.warning(f"Kafka not available (will retry): {e}")

        await self.persistence.start()

        # Start data connectors as concurrent tasks
        tasks = [
            asyncio.create_task(self.binance.start(), name="binance_ws"),
            asyncio.create_task(self.coingecko.start(), name="coingecko"),
            asyncio.create_task(self._health_check_loop(), name="health_check"),
        ]

        logger.info("═══ AtomicX Data Pipeline Running ═══")

        # Wait for any task to complete (or fail)
        done, pending = await asyncio.wait(
            tasks, return_when=asyncio.FIRST_EXCEPTION
        )

        # If a task failed, log and cancel others
        for task in done:
            if task.exception():
                logger.error(
                    f"Pipeline task {task.get_name()} failed: {task.exception()}"
                )

        for task in pending:
            task.cancel()

    async def _health_check_loop(self) -> None:
        """Periodic health check of all data sources."""
        while True:
            await asyncio.sleep(30)
            stale = self.monitor.get_stale_sources()
            if stale:
                logger.warning(f"Stale data sources: {stale}")

            status = self.monitor.get_status()
            for name, info in status.items():
                cb = info.get("circuit_breaker", {})
                if cb and cb.get("state") in ("open",):
                    logger.error(
                        f"Circuit OPEN for {name} — "
                        f"errors: {cb.get('failure_count')}, "
                        f"last: {cb.get('last_error')}"
                    )

    async def stop(self) -> None:
        """Stop all pipeline components gracefully."""
        logger.info("Stopping AtomicX Data Pipeline...")
        await asyncio.gather(
            self.binance.stop(),
            self.coingecko.stop(),
            self.kafka.stop(),
            self.persistence.stop(),
            return_exceptions=True,
        )
        logger.info("AtomicX Data Pipeline stopped")
