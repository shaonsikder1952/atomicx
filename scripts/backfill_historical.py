"""Run historical data backfill, variable computation, and pattern discovery.

Usage:
    # Start Docker services first
    docker compose up -d

    # Run migrations
    alembic upgrade head

    # Run backfill
    python scripts/backfill_historical.py --symbols BTC/USDT ETH/USDT SOL/USDT --days 180
"""

from __future__ import annotations

import argparse
import asyncio

from loguru import logger

from atomicx.common.logging import setup_logging
from atomicx.data.backfill import HistoricalBackfillService, VariableBackfillService
from atomicx.variables.patterns import PatternDiscoveryEngine


async def main(symbols: list[str], days: int) -> None:
    """Run the full backfill pipeline."""
    setup_logging()
    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║  AtomicX — Historical Data Backfill      ║")
    logger.info("╚══════════════════════════════════════════╝")

    # Step 1: Download OHLCV data from Binance
    logger.info("═══ Step 1: Downloading OHLCV from Binance ═══")
    backfill = HistoricalBackfillService(symbols=symbols)
    try:
        results = await backfill.backfill_all(
            timeframes=["1h", "4h", "1d"],
            days=days,
        )
        for key, count in results.items():
            logger.info(f"  {key}: {count:,} candles")
    finally:
        await backfill.close()

    # Step 2: Compute all variables on historical data
    logger.info("═══ Step 2: Computing Variables ═══")
    var_backfill = VariableBackfillService()
    for symbol in symbols:
        for tf in ["1h", "4h", "1d"]:
            await var_backfill.backfill_variables(symbol, tf)

    # Step 3: Discover patterns
    logger.info("═══ Step 3: Discovering Patterns ═══")
    discovery = PatternDiscoveryEngine()
    for symbol in symbols:
        patterns = await discovery.run_full_discovery(
            symbol=symbol,
            timeframe="1h",
            output_path=f"PATTERNS_{symbol.replace('/', '_')}.md",
        )
        logger.info(f"  {symbol}: {len(patterns)} patterns found")

    logger.info("═══ Backfill Complete ═══")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AtomicX Historical Backfill")
    parser.add_argument(
        "--symbols", nargs="+", default=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        help="Trading pairs to backfill",
    )
    parser.add_argument(
        "--days", type=int, default=180,
        help="Number of days of history to download",
    )
    args = parser.parse_args()
    asyncio.run(main(args.symbols, args.days))
