"""Startup orchestrator — runs data ingestion pipeline before brain loop.

Ensures the system has historical data loaded before agents start making predictions.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from atomicx.data.ingestion import DataIngestionService
from atomicx.data.pattern_verification import PatternVerificationService
from atomicx.variables.patterns import PatternDiscoveryEngine


class DataStartupOrchestrator:
    """Orchestrates complete data pipeline startup."""

    def __init__(self, symbols: list[str] | None = None) -> None:
        from atomicx.config import get_settings
        settings = get_settings()
        self._symbols = symbols or settings.default_symbols
        self.ingestion = DataIngestionService(symbols=self._symbols)
        self.pattern_verifier = PatternVerificationService()
        self.pattern_discovery = PatternDiscoveryEngine()
        self.logger = logger.bind(module="data.startup")

    async def run_startup_pipeline(
        self,
        force_full_backfill: bool = False,
        run_pattern_discovery: bool = False,
    ) -> dict[str, Any]:
        """Run complete startup data pipeline.

        Args:
            force_full_backfill: Re-download all historical data
            run_pattern_discovery: Run full pattern discovery analysis

        Returns:
            Health status and statistics
        """
        self.logger.info("=" * 80)
        self.logger.info("ATOMICX DATA PIPELINE STARTUP")
        self.logger.info("=" * 80)

        start_time = datetime.now(tz=timezone.utc)
        results = {
            "started_at": start_time.isoformat(),
            "ingestion": {},
            "pattern_verification": {},
            "pattern_discovery": {},
            "health_check": {},
            "status": "success",
            "errors": [],
        }

        # Step 1: Data Ingestion (OHLCV + Variables)
        self.logger.info("\n[1/4] Running OHLCV & Variable Ingestion...")
        try:
            ingestion_stats = await self.ingestion.run_full_ingestion(
                force_full_backfill=force_full_backfill
            )
            results["ingestion"] = ingestion_stats

            if ingestion_stats["errors"]:
                self.logger.warning(
                    f"Ingestion completed with {len(ingestion_stats['errors'])} errors"
                )
                results["status"] = "partial"

        except Exception as e:
            self.logger.error(f"Ingestion failed: {e}")
            results["errors"].append(f"Ingestion: {str(e)}")
            results["status"] = "failed"
            return results

        # Step 2: Pattern Discovery (Optional - expensive operation)
        if run_pattern_discovery:
            self.logger.info("\n[2/4] Running Pattern Discovery Analysis...")
            try:
                for symbol in self._symbols:
                    patterns = await self.pattern_discovery.run_full_discovery(
                        symbol=symbol,
                        timeframe="1h",
                        output_path=f"patterns_{symbol.replace('/', '_')}_study.md"
                    )
                    results["pattern_discovery"][symbol] = {
                        "patterns_found": len(patterns),
                    }
                    self.logger.info(
                        f"  ✓ {symbol}: {len(patterns)} patterns discovered"
                    )

            except Exception as e:
                self.logger.error(f"Pattern discovery failed: {e}")
                results["errors"].append(f"Pattern discovery: {str(e)}")
                results["status"] = "partial"
        else:
            self.logger.info("\n[2/4] Skipping pattern discovery (set run_pattern_discovery=True to enable)")

        # Step 3: Verify Pending Patterns
        self.logger.info("\n[3/4] Verifying Pending Pattern Outcomes...")
        try:
            verify_stats = await self.pattern_verifier.verify_pending_patterns()
            results["pattern_verification"] = verify_stats

            if verify_stats["patterns_verified"] > 0:
                self.logger.success(
                    f"  ✓ Verified {verify_stats['patterns_verified']} patterns"
                )
            else:
                self.logger.info("  No pending patterns to verify")

        except Exception as e:
            self.logger.error(f"Pattern verification failed: {e}")
            results["errors"].append(f"Pattern verification: {str(e)}")
            # Not critical - continue

        # Step 4: Health Check
        self.logger.info("\n[4/4] Running Health Check...")
        try:
            health = await self.ingestion.get_data_health()
            results["health_check"] = health

            if health["overall_status"] == "healthy":
                self.logger.success("  ✓ All data sources healthy")
            elif health["overall_status"] == "degraded":
                self.logger.warning(f"  ⚠ Data health degraded: {len(health['recommendations'])} issues")
                for rec in health["recommendations"][:5]:
                    self.logger.warning(f"    - {rec}")
            else:
                self.logger.error(f"  ✗ Data health critical")
                results["status"] = "failed"

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            results["errors"].append(f"Health check: {str(e)}")

        # Summary
        elapsed = (datetime.now(tz=timezone.utc) - start_time).total_seconds()
        results["completed_at"] = datetime.now(tz=timezone.utc).isoformat()
        results["duration_seconds"] = elapsed

        self.logger.info("\n" + "=" * 80)
        self.logger.info("STARTUP PIPELINE COMPLETE")
        self.logger.info("=" * 80)
        self.logger.info(f"Status: {results['status'].upper()}")
        self.logger.info(f"Duration: {elapsed:.1f}s")

        if "ingestion" in results:
            ing = results["ingestion"]
            self.logger.info(
                f"OHLCV: {ing.get('ohlcv_downloaded', 0):,} candles | "
                f"Variables: {ing.get('variables_computed', 0):,} computed"
            )

        if "pattern_verification" in results:
            pv = results["pattern_verification"]
            self.logger.info(
                f"Patterns: {pv.get('patterns_verified', 0)} verified"
            )

        if results["errors"]:
            self.logger.warning(f"Errors: {len(results['errors'])}")
            for err in results["errors"]:
                self.logger.warning(f"  - {err}")

        self.logger.info("=" * 80 + "\n")

        return results

    async def run_health_check_only(self) -> dict[str, Any]:
        """Quick health check without running ingestion."""
        self.logger.info("Running data health check...")
        health = await self.ingestion.get_data_health()

        # Print summary
        self.logger.info(f"Overall status: {health['overall_status'].upper()}")

        for symbol, tf_data in health["symbols"].items():
            for tf, stats in tf_data.items():
                status_emoji = "✓" if stats["status"] == "healthy" else ("⚠" if stats["status"] == "stale" else "✗")
                self.logger.info(
                    f"  {status_emoji} {symbol} {tf}: {stats['candle_count']:,} candles "
                    f"(last update: {stats['staleness_minutes']}m ago)" if stats['staleness_minutes']
                    else f"  {status_emoji} {symbol} {tf}: {stats['candle_count']:,} candles (no data)"
                )

        if health["recommendations"]:
            self.logger.warning(f"\nRecommendations ({len(health['recommendations'])}):")
            for rec in health["recommendations"]:
                self.logger.warning(f"  - {rec}")

        return health

    async def close(self) -> None:
        """Cleanup resources."""
        await self.ingestion.close()


async def run_startup_with_args() -> dict[str, Any]:
    """CLI entry point for running startup pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="AtomicX Data Startup Pipeline")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Symbols to process (default: from config)",
    )
    parser.add_argument(
        "--force-backfill",
        action="store_true",
        help="Force full historical backfill",
    )
    parser.add_argument(
        "--pattern-discovery",
        action="store_true",
        help="Run pattern discovery analysis (expensive)",
    )
    parser.add_argument(
        "--health-only",
        action="store_true",
        help="Only run health check",
    )

    args = parser.parse_args()

    orchestrator = DataStartupOrchestrator(symbols=args.symbols)

    try:
        if args.health_only:
            return await orchestrator.run_health_check_only()
        else:
            return await orchestrator.run_startup_pipeline(
                force_full_backfill=args.force_backfill,
                run_pattern_discovery=args.pattern_discovery,
            )
    finally:
        await orchestrator.close()


if __name__ == "__main__":
    asyncio.run(run_startup_with_args())
