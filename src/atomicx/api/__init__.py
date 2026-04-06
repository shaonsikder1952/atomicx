"""AtomicX API — FastAPI application.

Provides REST endpoints for:
- Health status & data freshness
- Historical data queries
- Pipeline control
- Prediction output (future phases)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query
from loguru import logger
from sqlalchemy import func, select

from atomicx.common.logging import setup_logging
from atomicx.data.storage.database import get_session_factory
from atomicx.data.storage.models import OHLCV, DataFreshness, OnChainMetric, TickData


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle handler."""
    setup_logging()
    logger.info("AtomicX API starting")
    yield
    logger.info("AtomicX API shutting down")


app = FastAPI(
    title="AtomicX — Causal Intelligence Engine",
    description="Universal Reality Modeling and Prediction Engine",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Health & Status ───────────────────────────────────────────


@app.get("/health")
async def health_check():
    """Basic health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "version": "0.1.0",
    }


@app.get("/api/v1/status/data-freshness")
async def data_freshness():
    """Get data freshness status for all sources."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(select(DataFreshness))
        sources = result.scalars().all()
        return {
            "sources": [
                {
                    "name": s.source_name,
                    "last_update": s.last_update.isoformat() if s.last_update else None,
                    "status": s.status,
                    "error_count": s.error_count,
                }
                for s in sources
            ]
        }


# ── Market Data Queries ──────────────────────────────────────


@app.get("/api/v1/ohlcv/{symbol}")
async def get_ohlcv(
    symbol: str,
    timeframe: str = Query(default="1m", description="1m, 5m, 15m, 1h, 4h, 1d"),
    start: datetime | None = Query(default=None, description="Start time (ISO 8601)"),
    end: datetime | None = Query(default=None, description="End time (ISO 8601)"),
    limit: int = Query(default=500, le=5000, description="Max rows to return"),
):
    """Query OHLCV candlestick data."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        query = (
            select(OHLCV)
            .where(OHLCV.symbol == symbol.upper())
            .where(OHLCV.timeframe == timeframe)
            .order_by(OHLCV.timestamp.desc())
            .limit(limit)
        )
        if start:
            query = query.where(OHLCV.timestamp >= start)
        if end:
            query = query.where(OHLCV.timestamp <= end)

        result = await session.execute(query)
        candles = result.scalars().all()

        return {
            "symbol": symbol.upper(),
            "timeframe": timeframe,
            "count": len(candles),
            "data": [
                {
                    "timestamp": c.timestamp.isoformat(),
                    "open": str(c.open),
                    "high": str(c.high),
                    "low": str(c.low),
                    "close": str(c.close),
                    "volume": str(c.volume),
                }
                for c in reversed(candles)
            ],
        }


@app.get("/api/v1/metrics/{symbol}")
async def get_metrics(
    symbol: str,
    metric_name: str | None = Query(default=None),
    limit: int = Query(default=100, le=1000),
):
    """Query on-chain and market metrics."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        query = (
            select(OnChainMetric)
            .where(OnChainMetric.symbol == symbol.upper())
            .order_by(OnChainMetric.timestamp.desc())
            .limit(limit)
        )
        if metric_name:
            query = query.where(OnChainMetric.metric_name == metric_name)

        result = await session.execute(query)
        metrics = result.scalars().all()

        return {
            "symbol": symbol.upper(),
            "count": len(metrics),
            "data": [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "metric": m.metric_name,
                    "value": str(m.value),
                    "source": m.source,
                }
                for m in metrics
            ],
        }


@app.get("/api/v1/stats")
async def get_pipeline_stats():
    """Get pipeline statistics."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        tick_count = await session.execute(select(func.count(TickData.id)))
        ohlcv_count = await session.execute(select(func.count(OHLCV.id)))
        metric_count = await session.execute(select(func.count(OnChainMetric.id)))

        return {
            "tick_count": tick_count.scalar() or 0,
            "ohlcv_count": ohlcv_count.scalar() or 0,
            "metric_count": metric_count.scalar() or 0,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
