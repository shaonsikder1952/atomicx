"""Kafka message publisher for the data pipeline.

Publishes market data events to Kafka topics for downstream services.
Topics: market-ticks, market-klines, market-depth, onchain-metrics
"""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal
from typing import Any

from aiokafka import AIOKafkaProducer
from loguru import logger

from atomicx.config import get_settings


class DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal and datetime types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class KafkaPublisher:
    """Publishes market data events to Kafka topics."""

    # Topic definitions
    TOPIC_TICKS = "market-ticks"
    TOPIC_KLINES = "market-klines"
    TOPIC_DEPTH = "market-depth"
    TOPIC_ONCHAIN = "onchain-metrics"
    TOPIC_GLOBAL = "market-global"

    def __init__(self) -> None:
        settings = get_settings()
        self._bootstrap_servers = settings.kafka_bootstrap_servers
        self._producer: AIOKafkaProducer | None = None

    async def start(self) -> None:
        """Start the Kafka producer."""
        self._producer = AIOKafkaProducer(
            bootstrap_servers=self._bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, cls=DecimalEncoder).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            acks="all",
            compression_type="gzip",
        )
        await self._producer.start()
        logger.info("Kafka publisher started", servers=self._bootstrap_servers)

    async def publish_trade(self, trade: dict) -> None:
        """Publish a trade event to the ticks topic."""
        if self._producer:
            await self._producer.send(
                self.TOPIC_TICKS,
                value=trade,
                key=trade.get("symbol", "UNKNOWN"),
            )

    async def publish_kline(self, kline: dict) -> None:
        """Publish a kline event to the klines topic."""
        if self._producer:
            await self._producer.send(
                self.TOPIC_KLINES,
                value=kline,
                key=kline.get("symbol", "UNKNOWN"),
            )

    async def publish_depth(self, depth: dict) -> None:
        """Publish an order book depth snapshot to the depth topic."""
        if self._producer:
            await self._producer.send(
                self.TOPIC_DEPTH,
                value=depth,
                key=depth.get("symbol", "UNKNOWN"),
            )

    async def publish_onchain_metric(self, metric: dict) -> None:
        """Publish an on-chain metric to the onchain topic."""
        if self._producer:
            await self._producer.send(
                self.TOPIC_ONCHAIN,
                value=metric,
                key=f"{metric.get('symbol', '')}:{metric.get('metric_name', '')}",
            )

    async def publish_global_data(self, data: dict) -> None:
        """Publish global market data to the global topic."""
        if self._producer:
            await self._producer.send(
                self.TOPIC_GLOBAL,
                value=data,
                key="global",
            )

    async def stop(self) -> None:
        """Stop the Kafka producer."""
        if self._producer:
            await self._producer.stop()
        logger.info("Kafka publisher stopped")
